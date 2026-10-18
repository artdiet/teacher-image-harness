// Files on the computer running Claude: where finished pictures are saved, and which
// pictures may be sent to the server (to edit, or to keep as a reference).

import { constants } from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { now } from "./client.js";
import { expandHome } from "./config.js";

export const MAX_INPUT_BYTES = 20 * 1024 * 1024; // the server's default upload cap
// A healthy disk answers in milliseconds. A file operation still going after this long is on a
// slow or stuck one (a network drive that went away, a disk waking up), or waits for a macOS
// question about folder access that nobody has answered yet.
export const SLOW_FILE_MS = 1_500;

export class UserError extends Error {}

// The pictures folder (or its disk) won't take the file. Every picture fails alike, and
// asking the image server again doesn't help until that is fixed.
export class SaveError extends UserError {}

const SIGNATURES = [
  ["image/png", (b) => b.length > 8 && b.readUInt32BE(0) === 0x89504e47 && b.readUInt32BE(4) === 0x0d0a1a0a],
  ["image/jpeg", (b) => b.length > 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff],
  ["image/webp", (b) => b.length > 12 && b.toString("latin1", 0, 4) === "RIFF" && b.toString("latin1", 8, 12) === "WEBP"],
  ["image/gif", (b) => b.length > 6 && /^GIF8[79]a$/.test(b.toString("latin1", 0, 6))],
];

export function sniffImage(bytes) {
  return SIGNATURES.find(([, test]) => test(bytes))?.[0] ?? null;
}

export function isInside(root, target) {
  const rel = path.relative(root, target);
  return rel !== "" && rel !== ".." && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel);
}

// Windows reserves device names, also with an extension ("con.txt", "COM1.v2", "lpt³.x"): a file
// called that can't be created, so the first part of such a name gets a "_".
const DEVICE_NAME = /^(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³])(?=\s*(\.|$))/i;

// The first `max` UTF-16 units of `text`, never cutting a character that takes two of them
// (half of one can't be written in a file name).
function cut(text, max) {
  if (text.length <= max) return text;
  const last = text.charCodeAt(max - 1);
  return text.slice(0, last >= 0xd800 && last <= 0xdbff ? max - 1 : max);
}

// A file or folder name with no path separators, leading dots or characters Windows rejects.
// Letters keep their combining marks (the vowel signs of Hindi, Thai or Tamil, Arabic harakat),
// which NFC doesn't merge into them.
export function safeName(text, fallback = "image") {
  const name = cut(String(text ?? "").normalize("NFC")
    .replace(/\.png$/i, "")
    .replace(/[^\p{L}\p{M}\p{N} ._()'-]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[.\s-]+/, ""), 80)
    .replace(/[.\s]+$/, "")
    .replace(DEVICE_NAME, "$1_");
  return name || fallback;
}

export function nameFromPrompt(prompt) {
  const words = String(prompt).toLowerCase().match(/[\p{L}\p{M}\p{N}]+/gu) ?? [];
  let name = "";
  for (const word of words) {
    const next = name ? `${name}-${word}` : word;
    if (next.length > 48) {
      // A first word longer than that (Chinese or Thai written without spaces, a long
      // compound) is cut short rather than dropped, which would name every picture "image".
      if (!name) name = cut(word, 48);
      break;
    }
    name = next;
  }
  return safeName(name, "image");
}

// A subfolder of the pictures folder. A full path counts only if it points inside that folder:
// the extension never saves anywhere else (the "Save pictures in" setting chooses where).
export function outputFolder(outputDir, folder) {
  let wanted = String(folder ?? "").trim();
  const full = expandHome(wanted);
  if (path.isAbsolute(full) || path.win32.isAbsolute(full)) {
    const resolved = path.resolve(full);
    const rel = path.relative(outputDir, resolved);
    if (rel !== "" && !isInside(outputDir, resolved)) {
      throw new UserError(`Pictures are saved inside ${outputDir}, so the folder is a name inside it, like ` +
        "\"Bear book\" (not a full path). To save pictures somewhere else, choose another folder for " +
        "\"Save pictures in\" in the extension settings.");
    }
    wanted = rel;
  }
  const parts = wanted.split(/[\\/]+/).map((p) => safeName(p, "")).filter(Boolean).slice(0, 3);
  const dir = path.join(outputDir, ...parts);
  if (parts.length && !isInside(outputDir, dir)) throw new UserError("That folder name isn't allowed.");
  return dir;
}

const NOT_ALLOWED = new Set(["EACCES", "EPERM"]);
const NO_ROOM = new Set(["ENOSPC", "EDQUOT", "EFBIG"]);
const IN_THE_WAY = new Set(["EEXIST", "ENOTDIR"]); // from creating the folder: a file has its name
const BAD_NAME = new Set(["ENAMETOOLONG", "EINVAL"]);
const OTHER_FOLDER = "choose another folder for \"Save pictures in\" in the extension settings.";

const exists = (p) => fs.access(p).then(() => true, () => false);

// On a Mac, a folder on a disk that isn't plugged in can't be created: making it would mean
// creating /Volumes/<disk>, which only the system may do (EACCES).
async function onMissingDisk(dir) {
  const [, top, disk] = path.resolve(dir).split(path.sep);
  return process.platform === "darwin" && top === "Volumes" && Boolean(disk) &&
    !(await exists(path.join("/Volumes", disk)));
}

// Why a picture couldn't be saved, in words for the teacher (other errors pass through).
async function unwritable(err, dir) {
  const code = err?.code;
  // A drive that isn't connected: ENOENT on Windows (no such drive letter), EACCES on a Mac.
  if (code === "ENOENT" || ((NOT_ALLOWED.has(code) || code === "EROFS") && await onMissingDisk(dir))) {
    return new SaveError(`The folder for pictures, ${dir}, can't be created: is the drive it is on connected? ` +
      `Connect it, or ${OTHER_FOLDER}`);
  }
  // A locked memory card, a disk image or a shared drive that only allows reading: no
  // permission in System Settings changes that.
  if (code === "EROFS") {
    return new SaveError(`Pictures can't be saved in ${dir}: the disk it is on is read-only; ${OTHER_FOLDER}`);
  }
  if (NOT_ALLOWED.has(code)) {
    return new SaveError(`Claude isn't allowed to save pictures in ${dir}. Choose another folder for ` +
      "\"Save pictures in\" in the extension settings, or on a Mac allow Claude in System Settings > " +
      "Privacy & Security > Files and Folders.");
  }
  if (NO_ROOM.has(code)) {
    return new SaveError(`There's no room to save pictures in ${dir}: the disk is full. Free up some space, ` +
      `or ${OTHER_FOLDER}`);
  }
  if (IN_THE_WAY.has(code)) {
    return new SaveError(`Pictures can't be saved in ${dir}: a file is in the way (it has the name of that ` +
      `folder, or of a folder above it). Use another folder name, or ${OTHER_FOLDER}`);
  }
  if (BAD_NAME.has(code)) {
    return new SaveError(`Pictures can't be saved in ${dir} under that name: it is too long or not allowed on ` +
      `that disk. Use a shorter file or folder name, or ${OTHER_FOLDER}`);
  }
  return err;
}

// Creates `file` with these bytes; false if the name is taken (never overwrites). A write that
// fails partway (a full disk) removes the file it started, so no cut-off picture is left under
// the real name.
async function writeNew(file, bytes) {
  let handle;
  try {
    handle = await fs.open(file, "wx");
  } catch (err) {
    if (err.code === "EEXIST") return false;
    throw err;
  }
  try {
    await handle.writeFile(bytes);
    await handle.close();
  } catch (err) {
    await handle.close().catch(() => {});
    await fs.unlink(file).catch(() => {});
    throw err;
  }
  return true;
}

// Never overwrites: "owl.png", then "owl (2).png", ...
export async function savePng(dir, baseName, bytes) {
  try {
    await fs.mkdir(dir, { recursive: true });
    for (let i = 1; i < 1000; i++) {
      const file = path.join(dir, i === 1 ? `${baseName}.png` : `${baseName} (${i}).png`);
      if (await writeNew(file, bytes)) return file;
    }
  } catch (err) {
    throw await unwritable(err, dir);
  }
  throw new SaveError(`Too many files named ${baseName} in ${dir}. Choose another file name.`);
}

// Windows' "Copy as path" puts quotes around a path, and some apps hand over file:// URLs.
export function plainPath(raw) {
  let p = String(raw).trim();
  if (p.length > 1 && "\"'“‘".includes(p[0]) && "\"'”’".includes(p.at(-1))) {
    p = p.slice(1, -1).trim();
  }
  if (/^file:\/\//i.test(p)) {
    try {
      p = fileURLToPath(p);
    } catch {
      // not a file on this computer; the full-path check explains
    }
  }
  return expandHome(p);
}

// The real path of each folder pictures may be sent from, asked for once per folder: a folder
// on a drive that stopped answering would otherwise hold another of Node's few file threads on
// every check. One that couldn't be resolved is asked for again next time (it may exist by then).
const realRoots = new Map(); // folder -> { done, since, real } (real: undefined until known, null if none)

function realRoot(root) {
  let entry = realRoots.get(root);
  if (!entry) {
    entry = { since: now(), real: undefined };
    entry.done = fs.realpath(root).then((real) => {
      entry.real = real;
    }, () => {
      entry.real = null;
      realRoots.delete(root);
    });
    realRoots.set(root, entry);
  }
  return entry;
}

// Resolves when `promise` settles or after `ms`, whichever comes first.
function within(promise, ms) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    const done = () => {
      clearTimeout(timer);
      resolve();
    };
    promise.then(done, done);
  });
}

// Inside a folder pictures may be sent from. The target is normally a real path (no links in
// it), so lying inside a folder's path as written proves it without touching the disk; a path
// that couldn't be resolved is compared the same way, only to choose the advice for a refusal.
// Only then are the folders' real paths compared (on a Mac, /tmp is really /private/tmp): one
// still unknown after SLOW_FILE_MS, its drive slow or stuck, counts as no match.
async function isAllowed(target, roots) {
  if (roots.some((root) => isInside(path.resolve(root), target))) return true;
  const entries = roots.map(realRoot);
  const inside = (entry) => Boolean(entry.real) && isInside(entry.real, target);
  const unknown = entries.filter((entry) => entry.real === undefined);
  const wait = Math.max(0, ...unknown.map((entry) => SLOW_FILE_MS - (now() - entry.since)));
  if (!entries.some(inside) && wait > 0) {
    // Done at the first folder that holds the target, without waiting for the others.
    const found = Promise.any(unknown.map((entry) => entry.done.then(() => inside(entry) || Promise.reject())));
    await within(found, wait);
  }
  return entries.some(inside);
}

function outsideRoots(file) {
  // IMAGE_INPUT_ROOTS is only for manual setups: the extension's settings page doesn't show it.
  return new UserError(`Pictures can only be sent from your home folder (or the folder Teach Image saves ` +
    `pictures in). Copy ${path.basename(file)} into your home folder first, for example into Pictures, and ` +
    "use that copy. (Advanced: a manual setup can allow more folders with IMAGE_INPUT_ROOTS.)");
}

// The folder can be listed and entered, so a refusal comes from the file itself.
const canOpenFolder = (dir) => fs.access(dir, constants.R_OK | constants.X_OK).then(() => true, () => false);

// Copying the picture somewhere Claude can read is the other way out, unless it is already there.
function copyHint(file) {
  const home = os.homedir();
  if (!isInside(home, file)) return " Or copy the picture into your home folder, for example into Pictures, and use that copy.";
  if (isInside(path.join(home, "Pictures"), file)) return "";
  return " Or copy the picture into your Pictures folder and use that copy.";
}

// Why a picture couldn't be read, in words for the teacher (other errors pass through).
async function unreadable(err, filePath, target, roots) {
  if (err?.code === "ENOENT" || err?.code === "ENOTDIR") {
    return new UserError(`There's no file at ${filePath} on this computer.`);
  }
  // Too long, links that lead round in a circle, or characters this disk doesn't allow.
  if (err?.code === "ENAMETOOLONG" || err?.code === "ELOOP" || err?.code === "EINVAL") {
    return new UserError(`Claude can't open ${filePath}: that isn't a usable path to a picture file. Give the ` +
      "full path of the picture itself, like ~/Pictures/photo.jpg");
  }
  if (err?.code !== "EACCES" && err?.code !== "EPERM") return err;
  // Unlocking a folder is pointless if pictures may not be sent from it anyway.
  if (!(await isAllowed(target, roots))) return outsideRoots(target);
  if (err.code === "EACCES" && await canOpenFolder(path.dirname(target))) {
    // Claude runs as the teacher's own account, so Preview or Finder can't open it either:
    // only giving the account access (or a copy from the owner) helps.
    return new UserError(`This account isn't allowed to open ${filePath}: the picture's own permissions don't ` +
      "allow it. Give your account Read access to it (on a Mac: select it in Finder, choose File > Get Info, " +
      "and under Sharing & Permissions set your account to Read only or Read & Write), or ask whoever owns " +
      "the file for a copy. Then try again.");
  }
  // On a Mac, protected folders (Desktop, Documents, Downloads, other disks) need the person's
  // permission for Claude, which macOS asks for only once.
  return new UserError(`Claude isn't allowed to read the folder ${filePath} is in. On a Mac, allow it in ` +
    `System Settings > Privacy & Security > Files and Folders (or Full Disk Access) > Claude.${copyHint(target)}`);
}

async function readImage(real, filePath) {
  const stat = await fs.stat(real);
  if (!stat.isFile()) throw new UserError(`${filePath} isn't a file.`);
  if (stat.size > MAX_INPUT_BYTES) throw new UserError(`${path.basename(real)} is over 20 MB.`);
  const bytes = await fs.readFile(real);
  const type = sniffImage(bytes);
  if (!type) throw new UserError(`${path.basename(real)} isn't a PNG, JPEG, WebP or GIF picture.`);
  return { bytes, type, name: path.basename(real) };
}

export async function readInputImage(filePath, roots) {
  const expanded = plainPath(filePath);
  if (!path.isAbsolute(expanded)) {
    throw new UserError(`"${filePath}" isn't a full path. Give the whole path, like ~/Pictures/photo.jpg`);
  }
  let real;
  try {
    real = await fs.realpath(expanded);
  } catch (err) {
    throw await unreadable(err, filePath, path.resolve(expanded), roots);
  }
  if (!(await isAllowed(real, roots))) throw outsideRoots(real);
  try {
    return await readImage(real, filePath);
  } catch (err) {
    throw await unreadable(err, filePath, real, roots);
  }
}
