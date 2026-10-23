// The MCP tools. Each call goes to the image server's job API; finished pictures are saved
// on this computer as full-size PNGs, and the tool result carries a small preview plus the
// facts (saved path, image_id, seed, prompt) in text, since the model may not see an image
// until its next turn.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

import pkg from "../package.json" with { type: "json" };
import { ImageServerError, OutOfTime, now, sleep, timedOut } from "./client.js";
import path from "node:path";

import {
  SLOW_FILE_MS, SaveError, UserError, nameFromPrompt, outputFolder, plainPath, readInputImage, safeName, savePng,
  sniffImage,
} from "./files.js";

export const VERSION = pkg.version;

// Claude Desktop stops a local tool call at about 60 s. Everything one call does (uploading,
// waiting, downloading) fits in this budget; a picture that takes longer is handed back as a
// job_id for get_image_job.
export const CALL_BUDGET_MS = 50_000;
const FINISH_RESERVE_MS = 8_000; // waiting stops this long before the deadline, to save the PNGs
const PREVIEW_MIN_MS = 3_000; // previews only if this much time is left once the PNGs are saved
const PREVIEW_PHASE_MS = 5_000; // and only this long: they are optional, the saved PNGs are not
const CANCEL_TIMEOUT_MS = 5_000;
// A picture on this computer that is still being read when this little time is left isn't sent:
// the upload couldn't finish anyway, and the slow read (not the server) is what to report.
const UPLOAD_MIN_MS = 1_500;
// Writing a picture always gets this long, even when a download finished just before the
// deadline, so that a healthy folder isn't called slow. It may run a little past the call's
// budget, still well within Claude Desktop's ~60 s.
const WRITE_MIN_MS = SLOW_FILE_MS;

// Claude Desktop rejects tool results over ~150k characters; previews are base64 (4/3 larger).
const PREVIEW_TOTAL = 75_000;
const PREVIEW_MAX = 70_000;
const PREVIEW_MIN = 8_000;
const REMEMBERED_JOBS = 50;

const ASPECTS = {
  square: [1024, 1024],
  portrait: [896, 1152],
  landscape: [1152, 896],
  wide: [1344, 768],
  tall: [768, 1344],
};
const FINISHED = new Set(["done", "failed", "cancelled"]);

const INSTRUCTIONS = `Teach Image makes pictures on the teacher's own Mac with a local image model (no per-picture cost).
- Choose a style: icon for worksheet icons and clip art, coloring for coloring pages, storybook for picture-book illustrations, none to use the prompt exactly as written.
- Describe the subject plainly (who or what, pose, setting, details); the style adds the art direction. Avoid asking for words in a picture: spelling is unreliable, so put text in the document instead.
- Each finished picture is saved as a full-size PNG on this computer. The result gives the saved path and a smaller preview; tell the user where the file is.
- To change a picture, call edit_image with its image_id. To keep a character the same across pages, save it with save_reference and pass its reference_id to later calls.
- The same prompt with the same seed gives the same picture again.
- If a call returns a job_id instead of a picture, call get_image_job with it.`;

const imageId = z.string().regex(/^img_[0-9a-f]{16}$/, "an image_id looks like img_ followed by 16 characters");
const referenceId = z.string().regex(/^ref_[0-9a-f]{16}$/, "a reference_id looks like ref_ followed by 16 characters");
const jobId = z.string().regex(/^job_[0-9a-f]{16}$/, "a job_id looks like job_ followed by 16 characters");

const style = z.enum(["icon", "coloring", "storybook", "none"]).describe(
  "icon: flat worksheet icon on white. coloring: black-and-white line-art coloring page. " +
  "storybook: children's picture-book illustration (soft watercolor unless the prompt names another medium). " +
  "none: the prompt is used exactly as written.");
const aspect = z.enum(Object.keys(ASPECTS)).describe(
  "square 1024x1024, portrait 896x1152 (page), landscape 1152x896, wide 1344x768, tall 768x1344");
const options = {
  size: z.string().regex(/^\d{3,4}x\d{3,4}$/, "size looks like 1024x768").optional()
    .describe("Exact WIDTHxHEIGHT in pixels (256 to 2048 per side); the server snaps to multiples of 16 and " +
      "scales very large sizes down to about 1.5 megapixels. Overrides aspect"),
  n: z.number().int().min(1).max(4).default(1).describe("How many variations to make (each uses the next seed)"),
  seed: z.number().int().min(0).max(4294967295).optional()
    .describe("Reuse the seed from an earlier result to repeat or slightly vary that picture"),
  reference_ids: z.array(referenceId).max(10).optional()
    .describe("Saved references (characters, style samples) the picture should match; see list_references"),
  transparent: z.boolean().default(false)
    .describe("Make the white background transparent (good for icons placed on worksheets)"),
  model: z.string().max(64).optional().describe("Leave empty for the default; image_server_status lists the models"),
  file_name: z.string().max(100).optional()
    .describe("Name for the saved PNG (default: from the prompt). Existing files are never overwritten."),
  folder: z.string().max(200).optional()
    .describe("Subfolder inside the pictures folder, e.g. \"Bear book\" (a name, not a full path)"),
};

const WRITES = { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: false };
const READS = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false };

function text(message) {
  return { content: [{ type: "text", text: message }] };
}

function problem(message) {
  return { isError: true, content: [{ type: "text", text: message }] };
}

// This computer's local time with its zone ("2026-10-18 08:01 EDT"): Claude may run in the cloud,
// on another clock, so a bare time would be ambiguous.
function when(seconds) {
  if (!seconds) return "unknown time";
  const d = new Date(seconds * 1000);
  const pad = (v) => String(v).padStart(2, "0");
  const zone = new Intl.DateTimeFormat("en-US", { timeZoneName: "short" }).formatToParts(d)
    .find((part) => part.type === "timeZoneName")?.value;
  const local = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  return zone ? `${local} ${zone}` : local;
}

function clip(value, max) {
  const s = String(value ?? "");
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

function sizeFields({ size, aspect: shape }) {
  if (size) return { size };
  if (shape) return { width: ASPECTS[shape][0], height: ASPECTS[shape][1] };
  return {};
}

// The server splits a multipart tags field on commas, so a comma inside a tag would make two.
function cleanTags(tags) {
  return (tags ?? []).map((tag) => tag.replace(/,/g, " ").replace(/\s+/g, " ").trim()).filter(Boolean);
}

// Claude Desktop stops a call that ran too long with a reason like "Request timed out" (a
// string or an Error); a person stopping it gives a different one.
export function looksLikeTimeout(reason) {
  return /timed? ?out/i.test(String(reason));
}

// Whether `promise` settles within `ms`. Stopping the call rejects. Either way `promise` keeps
// going, and its failure is handled here (an unhandled one would end the whole process).
function settlesWithin(promise, ms, signal) {
  return new Promise((resolve, reject) => {
    let timer;
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const done = (settled) => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      resolve(settled);
    };
    promise.then(() => done(true), () => done(true));
    if (signal?.aborted) return onAbort();
    timer = setTimeout(done, Math.max(0, ms), false);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

// A file operation that never returns (a folder on a network drive that went away, a disk that
// doesn't wake up, a macOS question about folder access that nobody answers) holds one of the
// few threads Node shares between all file work and looking up the server's name, for as long
// as it hangs. Once they are all taken, nothing can be saved or reached any more. So while a
// write into a folder, or a read of a picture, is still going, no other is started there.
// One set for the whole process, like those threads.
const writing = new Map(); // folder -> the save going on there
const reading = new Map(); // picture -> the read going on

// Starts `start()` as the file operation at `key` once no other is going on there. One going on
// in another call usually ends within milliseconds, so it is waited for, but only until it
// counts as slow; then null is returned and nothing is started. An operation is
// { done, since, outlived }, where `outlived` means the call that started it stopped waiting for
// it (so it is slow, and nothing waits for it again).
async function startAt(map, key, signal, start) {
  for (let going = map.get(key); going; going = map.get(key)) {
    const wait = going.outlived ? 0 : SLOW_FILE_MS - (now() - going.since);
    if (wait <= 0 || !(await settlesWithin(going.done, wait, signal))) return null;
  }
  // Nothing is going on at `key` from here until `op` is in the map: no await in between.
  const op = { done: start(), since: now(), outlived: false };
  map.set(key, op);
  const clear = () => {
    if (map.get(key) === op) map.delete(key);
  };
  op.done.then(clear, clear);
  return op;
}

// Whether a write into `dir` is slow: left behind by an earlier call, or going on for a while.
function slowWrite(dir) {
  const going = writing.get(dir);
  return Boolean(going) && (going.outlived || now() - going.since >= SLOW_FILE_MS);
}

// What makes saving in a folder slow, in the order a teacher can check it.
const SLOW_CAUSES = "if a message asked whether Claude may use that folder, allow it; if the folder is on a " +
  "network or external drive, check that it is connected.";

// Reading a picture took most of the call's time: a file that has to download first (iCloud, a
// network drive), a disk waking up, or a question about folder access left unanswered.
function slowRead(sourcePath) {
  return new UserError(`Reading ${sourcePath} on this computer took too long, so it wasn't sent. If a message ` +
    "asked whether Claude may open that folder, allow it; a picture kept in iCloud or on a network drive may " +
    "need to download first. Then try again.");
}

function summary(result) {
  const seconds = result.images.reduce((sum, image) => sum + (image.seconds ?? 0), 0);
  const first = result.images[0];
  return result.images.length > 1
    ? `Made ${result.images.length} pictures in ${seconds.toFixed(1)} s (${result.model}, ${first.width}x${first.height}).`
    : `Made 1 picture in ${seconds.toFixed(1)} s (${result.model}, ${first.width}x${first.height}, seed ${first.seed}).`;
}

// What a result says about its previews: why any are missing, and that the PNGs are full
// size only when they were saved.
function previewNote({ blocks, failed, skipped }, count, saved) {
  const one = count === 1;
  if (blocks.length === count) {
    const note = one ? "The preview below is downscaled" : "Previews below are downscaled";
    return saved ? `${note}; the saved ${one ? "PNG is" : "PNGs are"} full size.` : `${note}.`;
  }
  const notes = [];
  if (failed) {
    notes.push(one ? "The preview couldn't be loaded this time." : `${failed} of the ${count} previews couldn't be loaded this time.`);
  }
  if (skipped) {
    notes.push(one ? "The preview was left out to keep this result quick."
      : "Some previews were left out to keep this result small and quick.");
  }
  if (saved) notes.push(one ? "The saved PNG is complete." : "The saved PNGs are complete.");
  notes.push("get_image shows a picture's preview.");
  return notes.join(" ");
}

const noPreviews = (count) => ({ blocks: [], failed: 0, skipped: count });

export function createServer(config, api, {
  log = (...args) => console.error("[teach-image]", ...args),
  callBudgetMs = CALL_BUDGET_MS,
  finishReserveMs = FINISH_RESERVE_MS,
  previewMinMs = PREVIEW_MIN_MS,
  previewPhaseMs = PREVIEW_PHASE_MS,
} = {}) {
  const server = new McpServer({ name: "teach-image", title: "Teach Image", version: VERSION },
    { instructions: INSTRUCTIONS });
  const saveTo = new Map(); // job_id -> where its pictures go, for jobs collected by a later call
  const savedSoFar = new Map(); // job_id -> Map(image_id -> path), until all its pictures are saved
  const collected = new Map(); // job_id -> finished tool result, so a job is only saved once
  const saving = new Map(); // job_id -> the save in progress, so two calls never save a picture twice

  function remember(map, key, value) {
    map.set(key, value);
    if (map.size > REMEMBERED_JOBS) map.delete(map.keys().next().value);
  }

  // Each tool call gets one deadline; the client fits every request inside it.
  function begin(extra) {
    const start = now();
    return { extra, signal: extra.signal, start, deadline: start + callBudgetMs };
  }

  // Request options for a call's requests: stopped with the call, never past its deadline.
  const inCall = (call) => ({ signal: call.signal, call });

  const timeLeft = (call) => call.deadline - now();

  function explain(err) {
    if (err instanceof UserError || err instanceof ImageServerError) return err.message;
    log("tool failed:", err);
    // A file system error that files.js has no words for is still about this computer's files.
    if (typeof err?.code === "string" && typeof err?.syscall === "string") {
      return `A file or folder on this computer couldn't be used: ${err.message}`;
    }
    return `Something went wrong in the Teach Image extension: ${err?.message ?? err}`;
  }

  async function guarded(extra, work) {
    try {
      return await work(begin(extra));
    } catch (err) {
      if (extra.signal.aborted) return problem("Cancelled.");
      return problem(explain(err));
    }
  }

  async function progress(extra, state, job) {
    const token = extra._meta?.progressToken;
    const value = Math.round((job.progress ?? 0) * 100);
    if (token === undefined || value <= state.last) return;
    state.last = value;
    const message = job.status === "queued" ? "Waiting for the image server" : `Drawing (${value}%)`;
    await extra.sendNotification({
      method: "notifications/progress",
      params: { progressToken: token, progress: value, total: 100, message },
    }).catch(() => {});
  }

  // Polls until the job finishes or `until`. A poll that fails for a passing reason (no answer,
  // a timeout, a 5xx while the server restarts) is tried again; other failures are thrown.
  // Returns the last job state seen (null if none) and, if the last poll failed, why. The last
  // poll is at `until` itself, so a picture finishing just then is still collected.
  async function wait(id, call, until) {
    const state = { last: -1 };
    let delay = 500;
    let job = null;
    let trouble = null;
    for (;;) {
      try {
        job = await api.job(id, inCall(call));
        trouble = null;
        if (FINISHED.has(job.status)) return { job, trouble };
        await progress(call.extra, state, job);
      } catch (err) {
        if (call.signal.aborted || !err?.transient) throw err;
        trouble = err;
        if (err instanceof OutOfTime) return { job, trouble };
      }
      const left = until - now();
      if (left <= 0) return { job, trouble };
      await sleep(Math.min(delay, left), call.signal);
      delay = Math.min(delay * 1.5, 2000);
    }
  }

  // Not finished yet, or no word from the server in time: either way the job_id goes back.
  function notYet(id, { job, trouble }) {
    if (trouble && !(trouble instanceof OutOfTime)) {
      return text(`Lost touch with the image server while waiting for the picture: ${trouble.message}\n` +
        `The picture may still be drawing. job_id: ${id}\n` +
        "Call get_image_job with this job_id in a minute to collect it.");
    }
    if (!job) {
      return text(`The picture was handed to the image server, but this call ran out of time before it could ` +
        `check on it. job_id: ${id}\nCall get_image_job with this job_id to collect it.`);
    }
    const where = job.status === "queued" ? "is waiting its turn on the image server"
      : `is still drawing (${Math.round((job.progress ?? 0) * 100)}%)`;
    return text(`The picture ${where}. job_id: ${id}\n` +
      "Call get_image_job with this job_id to collect it (it waits up to about 40 seconds).");
  }

  // The server answered about the job, but not with its state.
  function lostTrack(id, err) {
    if (err?.status === 404 && !err.foreign) {
      // A job id survives server restarts (docs/API.md), so a 404 means this server never had it.
      return problem(`The image server has no job ${id}. Check the job_id; if the server's pictures were ` +
        "cleared or the server address changed, this picture is gone. list_recent_images shows the pictures it has.");
    }
    return problem(`${explain(err)}\nThe picture was requested as job_id ${id}: call get_image_job with it ` +
      "later to collect it.");
  }

  // Previews are optional, so they get a few seconds at most. One that can't be loaded is left
  // out (`failed`); once time or the size budget runs out, the rest are left out (`skipped`).
  async function previews(images, call) {
    const budget = Math.max(PREVIEW_MIN, Math.min(PREVIEW_MAX, Math.floor(PREVIEW_TOTAL / images.length)));
    const phase = { ...call, deadline: Math.min(call.deadline, now() + previewPhaseMs) };
    const shown = { blocks: [], failed: 0, skipped: 0 };
    let total = 0;
    for (const [i, image] of images.entries()) {
      let preview;
      try {
        preview = await api.imagePreview(image.image_id, budget, inCall(phase));
      } catch (err) {
        if (call.signal.aborted) throw err;
        if (timedOut(err)) {
          shown.skipped = images.length - i;
          break;
        }
        shown.failed++; // a missing preview never spoils a finished picture
        continue;
      }
      // Judged by its bytes, not its label: a page from something else on the network, or a
      // broken file, must not reach Claude as a picture.
      const { data } = preview;
      const type = sniffImage(data);
      if (!type) {
        shown.failed++;
        continue;
      }
      if (total + data.length > PREVIEW_TOTAL + PREVIEW_MIN) {
        shown.skipped = images.length - i;
        break;
      }
      total += data.length;
      shown.blocks.push({ type: "image", data: data.toString("base64"), mimeType: type });
    }
    return shown;
  }

  // Not every PNG was saved: the deadline came first, or saving in `slowFolder` is slow (a
  // write there still going, from this call or an earlier one). The pictures are safe on the
  // server, so this isn't an error: a later get_image_job saves only what is still missing.
  function finishedOnServer(job, files, slowFolder) {
    const { images } = job.result;
    const what = images.length > 1 ? `The ${images.length} pictures are` : "The picture is";
    let first = `${what} finished on the image server, but this call ran out of time before ` +
      `${images.length > 1 ? "all of them were" : "it was"} saved on this computer.`;
    // Also said when this call returned at once: an earlier write there hasn't finished.
    if (slowFolder) first = `${what} finished on the image server, but saving in ${slowFolder} is slow: ${SLOW_CAUSES}`;
    const lines = [first];
    for (const [i, image] of images.entries()) {
      const where = files.get(image.image_id);
      if (where) lines.push(`${i + 1}. Saved to: ${where} (image_id ${image.image_id})`);
    }
    const rest = images.length === 1 ? "it and see a preview" : `${files.size ? "the rest" : "them"} and see previews`;
    lines.push(`job_id: ${job.job_id}`, `${slowFolder ? "Then call" : "Call"} get_image_job with this job_id to save ${rest}.`);
    return text(lines.join("\n"));
  }

  // The full-size picture. Something else on the network (a captive portal, a proxy) can answer
  // in the server's place, and its page must never be saved as a .png.
  async function fullPng(id, call) {
    const { data } = await api.imageFull(id, inCall(call));
    if (sniffImage(data) !== "image/png") {
      throw new ImageServerError("What came back for this picture isn't a PNG picture (something else on the " +
        "network may have answered instead of the image server).");
    }
    return data;
  }

  // Where a finished job's pictures go: as asked when it was made or collected, else the
  // pictures folder, named after the prompt.
  function destination(job) {
    const target = saveTo.get(job.job_id) ?? {};
    return {
      dir: target.dir ?? config.outputDir,
      base: target.base ?? nameFromPrompt(job.result.params?.prompt ?? "image"),
    };
  }

  // Downloads and saves the job's PNGs that aren't saved yet (under the job's save lock). The
  // full PNGs come first: they are what the teacher keeps; previews follow only with time to
  // spare. `outOfTime`: the deadline came first, or (with `slowFolder`) saving there is slow;
  // what was saved so far stays in savedSoFar, and a file still being written is added there
  // once it is done. `writes` collects the file writes, so the save lock can wait for one that
  // outlasts the call.
  async function savePictures(job, { dir, base }, call, writes) {
    const id = job.job_id;
    const { result } = job;
    const files = savedSoFar.get(id) ?? new Map();
    remember(savedSoFar, id, files);
    const several = result.images.length > 1;
    const failed = new Map(); // image_id -> why it wasn't saved
    let blocked = null; // the folder or disk refused a picture: it refuses the rest alike
    for (const [i, image] of result.images.entries()) {
      if (files.has(image.image_id)) continue;
      if (blocked) {
        failed.set(image.image_id, blocked);
        continue;
      }
      try {
        const data = await fullPng(image.image_id, call);
        const name = several ? `${base}-${i + 1}` : base;
        const save = await startAt(writing, dir, call.signal, () => savePng(dir, name, data));
        if (!save) return { files, failed, outOfTime: true, slowFolder: dir }; // an earlier write there hangs
        const write = save.done.then((file) => { files.set(image.image_id, file); });
        writes.push(write);
        // A folder on a slow or vanished drive can take longer than the call has left: the
        // write then carries on by itself.
        if (!(await settlesWithin(write, Math.max(WRITE_MIN_MS, timeLeft(call)), call.signal))) {
          save.outlived = true;
          return { files, failed, outOfTime: true, slowFolder: dir };
        }
        await write; // its error, if it failed
      } catch (err) {
        if (call.signal.aborted) throw err;
        if (err instanceof OutOfTime) return { files, failed, outOfTime: true };
        failed.set(image.image_id, err);
        if (err instanceof SaveError) blocked = err;
      }
    }
    return { files, failed };
  }

  // The result for a finished job: where each picture was saved (or why not), its facts, and
  // previews when there is time. Remembered once every picture is saved.
  async function report(job, { files, failed }, call) {
    const id = job.job_id;
    const { result } = job;
    const several = result.images.length > 1;
    // A folder or disk problem must be fixed first; a failed download can just be tried again.
    const local = [...failed.values()].find((err) => err instanceof SaveError);
    const lines = [summary(result)];
    for (const [i, image] of result.images.entries()) {
      const where = files.get(image.image_id);
      const err = failed.get(image.image_id);
      const why = err?.message ?? String(err);
      if (!several) {
        lines.push(`${where ? `Saved to: ${where}` : `Not saved on this computer: ${why}`}\nimage_id: ${image.image_id}`);
        continue;
      }
      const note = where ?? (local && why === local.message ? "not saved on this computer" : `not saved on this computer (${why})`);
      lines.push(`${i + 1}. ${note}\n   image_id ${image.image_id}, seed ${image.seed}`);
    }
    if (local && several) lines.push(local.message);
    if (failed.size) {
      const which = failed.size > 1 ? "The pictures not saved are" : "The picture not saved is";
      lines.push(local
        ? `${which} still on the image server: once that is fixed, call get_image_job with job_id ${id} to save ` +
          `${failed.size > 1 ? "them" : "it"}.`
        : `${which} still on the image server: call get_image_job with job_id ${id} to try saving again.`);
    }
    lines.push("Use an image_id with edit_image to change a picture, or save_reference to reuse a character.");
    if (result.params?.full_prompt) lines.push(`Prompt the model received: "${clip(result.params.full_prompt, 1200)}"`);
    if (result.ignored?.length) lines.push(`Not used by this model: ${result.ignored.join(", ")}.`);
    const shown = timeLeft(call) >= previewMinMs ? await previews(result.images, call) : noPreviews(result.images.length);
    lines.push(previewNote(shown, result.images.length, !failed.size));
    const done = { content: [{ type: "text", text: lines.join("\n") }, ...shown.blocks] };
    if (!failed.size) {
      remember(collected, id, done);
      saveTo.delete(id);
      savedSoFar.delete(id);
    }
    return done;
  }

  // One save per job at a time: a second call for the same job waits for the first, but never
  // past its own time, then saves only what is still missing. The lock covers the downloads
  // only, never the optional previews. A write that hangs holds the lock as long as it hangs,
  // so a slow write into the job's folder (or the folder of the save holding the lock) is
  // reported at once instead of being waited for.
  async function finish(job, call) {
    const id = job.job_id;
    if (job.status === "failed") return problem(`The picture couldn't be made: ${job.error ?? "unknown error"}.`);
    if (job.status === "cancelled") return text(`Job ${id} was cancelled; no picture was made.`);
    const soFar = () => savedSoFar.get(id) ?? new Map();
    // Every picture saved already (by a write that ended after its call did): nothing to write.
    const unsaved = () => job.result.images.some((image) => !soFar().has(image.image_id));
    for (;;) {
      // Saved by another call, also while this one waited for it: that call's result. Looked at
      // first, because saving the last picture forgets the job's folder and what was saved so far.
      if (collected.has(id)) return collected.get(id);
      const lock = saving.get(id);
      const slow = unsaved() && [destination(job).dir, lock?.dir].find((dir) => dir && slowWrite(dir));
      if (slow) return finishedOnServer(job, soFar(), slow);
      if (!lock) break;
      const left = timeLeft(call) - finishReserveMs;
      // Out of time while the other save goes on at a healthy pace (a slow write would have been
      // reported just above), so the folder isn't blamed for the wait.
      if (left <= 0) return finishedOnServer(job, soFar());
      // Looked at again after a while: a write that the other save starts meanwhile may hang.
      await settlesWithin(lock.done, Math.min(left, SLOW_FILE_MS), call.signal);
    }
    // Where the pictures go is read only now: a call collecting the job meanwhile may have
    // chosen another folder.
    const where = destination(job);
    const writes = [];
    const work = savePictures(job, where, call, writes);
    // The lock lasts until every file write of this save has ended, also one still going after
    // the call ran out of time, so the next save neither repeats nor misses that picture.
    const lock = {
      dir: where.dir,
      done: work.then(() => {}, () => {}).then(() => Promise.allSettled(writes)).then(() => {}),
    };
    saving.set(id, lock);
    lock.done.then(() => {
      if (saving.get(id) === lock) saving.delete(id);
    });
    const saved = await work;
    return saved.outOfTime ? finishedOnServer(job, saved.files, saved.slowFolder) : report(job, saved, call);
  }

  // Waits for a job, then saves its pictures. The job exists on the server by now, so every
  // outcome names its job_id, except when the user stops the call.
  async function collect(id, call, waitSeconds) {
    const until = Math.min(call.start + waitSeconds * 1000, call.deadline - finishReserveMs);
    let polled;
    try {
      polled = await wait(id, call, until);
    } catch (err) {
      if (!call.signal.aborted) return lostTrack(id, err);
      // Stopped by the person: stop the picture too (also when collecting it with
      // get_image_job). When Claude Desktop stops a call for taking too long, the job keeps
      // going, so the picture can still be collected.
      if (!looksLikeTimeout(call.signal.reason)) {
        await api.cancelJob(id, { timeoutMs: CANCEL_TIMEOUT_MS }).catch(() => {});
      }
      throw err;
    }
    if (!polled.job || !FINISHED.has(polled.job.status)) return notYet(id, polled);
    try {
      return await finish(polled.job, call);
    } catch (err) {
      if (call.signal.aborted) throw err;
      return problem(`${explain(err)}\nThe picture is finished on the image server: call get_image_job with ` +
        `job_id ${id} to save it.`);
    }
  }

  async function submit(job, target, call) {
    if (call.signal.aborted) return problem("Cancelled.");
    // Not stopped by the call's signal: once the server has made the job, this call must know
    // its id, to cancel it if the call was stopped meanwhile.
    let created;
    try {
      created = await api.submitJob(job, { call });
    } catch (err) {
      // The request went out but its answer was lost (a timeout, a dropped connection): the
      // server may have made the job anyway, and asking again at once would make it twice. Such
      // a picture is only listed once it is finished, so an empty list right away proves nothing.
      if (call.signal.aborted || !err?.lostAnswer) throw err;
      return problem(`${err.message}\nThe image server may have started the picture anyway: it may still be ` +
        "waiting its turn or being drawn there. Wait a minute, then check list_recent_images (finished pictures " +
        "appear there) before asking for it again, so it isn't made twice.");
    }
    if (call.signal.aborted) {
      await api.cancelJob(created.job_id, { timeoutMs: CANCEL_TIMEOUT_MS }).catch(() => {});
      return problem("Cancelled.");
    }
    remember(saveTo, created.job_id, target);
    return collect(created.job_id, call, config.waitSeconds);
  }

  // Reads a picture on this computer to send, within the call's time (the read counts toward
  // it). A read still going when too little is left for the upload is reported as such, and so
  // is a slow read of the same picture still going from an earlier call, without another.
  async function upload(sourcePath, call) {
    const read = await startAt(reading, path.resolve(plainPath(sourcePath)), call.signal,
      () => readInputImage(sourcePath, config.inputRoots));
    if (!read) throw slowRead(sourcePath);
    if (!(await settlesWithin(read.done, timeLeft(call) - UPLOAD_MIN_MS, call.signal))) {
      read.outlived = true;
      throw slowRead(sourcePath);
    }
    const file = await read.done;
    const form = new FormData();
    form.append("file", new Blob([file.bytes], { type: file.type }), file.name);
    return { file, form };
  }

  // Where a result's pictures go. The folder is checked first, before anything is sent.
  function target(args, prompt, dir = outputFolder(config.outputDir, args.folder)) {
    return { dir, base: args.file_name ? safeName(args.file_name) : nameFromPrompt(prompt) };
  }

  // get_image's copy of a picture: what to say about it, and whether the result ends there (the
  // copy is still being written when the time is up, so there is none left for the preview).
  async function saveCopy(id, { dir, base }, call) {
    const busy = `The copy wasn't saved: an earlier picture is still being saved in ${dir}, which is slow: ` +
      `${SLOW_CAUSES} Then ask for the copy again.`;
    if (slowWrite(dir)) return { line: busy }; // nothing downloaded for nothing
    const data = await fullPng(id, call);
    const save = await startAt(writing, dir, call.signal, () => savePng(dir, base, data));
    if (!save) return { line: busy };
    if (!(await settlesWithin(save.done, Math.max(WRITE_MIN_MS, timeLeft(call)), call.signal))) {
      save.outlived = true;
      return {
        end: true,
        line: `The copy is still being saved in ${dir}, which is slow: ${SLOW_CAUSES} It may still appear ` +
          "there; look before asking for another copy.",
      };
    }
    return { line: `Saved to: ${await save.done}` };
  }

  server.registerTool("generate_image", {
    title: "Generate picture",
    description: "Make new pictures from a description on the teacher's image server. Waits up to about " +
      "40 seconds, then returns where the full-size PNG was saved on this computer, its image_id and seed, " +
      "and a preview. A picture that takes longer comes back as a job_id for get_image_job.",
    inputSchema: {
      prompt: z.string().min(1).max(4000)
        .describe("What to draw: subject, pose, setting, details. The style adds the art direction."),
      style: style.default("none"),
      aspect: aspect.default("square"),
      ...options,
    },
    annotations: WRITES,
  }, (args, extra) => guarded(extra, (call) => submit({
    prompt: args.prompt, style: args.style, n: args.n, transparent: args.transparent,
    seed: args.seed, model: args.model, reference_ids: args.reference_ids, ...sizeFields(args),
  }, target(args, args.prompt), call)));

  server.registerTool("edit_image", {
    title: "Edit picture",
    description: "Change an existing picture: one made earlier (image_id) or a picture file on this computer " +
      "(source_path), e.g. \"give the owl a red scarf\" or \"turn this photo into a coloring page\". " +
      "Keeps the source's shape unless aspect or size is given. Returns like generate_image.",
    inputSchema: {
      instruction: z.string().min(1).max(4000).describe("What to change or make from the source picture"),
      image_id: imageId.optional().describe("A picture made earlier (from a result or list_recent_images)"),
      source_path: z.string().max(1024).optional()
        .describe("Or a picture file on this computer: full path, e.g. ~/Pictures/photo.jpg (PNG, JPEG, WebP, GIF)"),
      style: style.default("none"),
      aspect: aspect.optional(),
      ...options,
    },
    annotations: WRITES,
  }, (args, extra) => guarded(extra, async (call) => {
    if (Boolean(args.image_id) === Boolean(args.source_path)) {
      throw new UserError("Give either image_id (a picture made earlier) or source_path (a file), not both.");
    }
    // A folder that isn't allowed is refused before the picture is uploaded for nothing.
    const dir = outputFolder(config.outputDir, args.folder);
    let source = args.image_id;
    let base = args.instruction;
    if (args.source_path) {
      const { file, form } = await upload(args.source_path, call);
      source = (await api.uploadImage(form, inCall(call))).image_id;
      base = `${file.name.replace(/\.[^.]+$/, "")} ${args.instruction}`;
    }
    return submit({
      prompt: args.instruction, style: args.style, n: args.n, transparent: args.transparent,
      seed: args.seed, model: args.model, reference_ids: args.reference_ids, source_image_id: source,
      ...sizeFields(args),
    }, target(args, base, dir), call);
  }));

  server.registerTool("get_image_job", {
    title: "Collect picture",
    description: "Check on a picture that wasn't finished when generate_image or edit_image returned, waiting " +
      "for it up to about 40 seconds. When it is done, saves it and returns it like generate_image. Calling it " +
      "again for a job already saved returns the earlier result and doesn't save another copy (within one " +
      "Claude Desktop session).",
    inputSchema: {
      job_id: jobId,
      wait_seconds: z.number().int().min(0).max(40).optional()
        .describe("How long to wait for it to finish (default about 40)"),
      file_name: options.file_name,
      folder: options.folder,
    },
    annotations: WRITES,
  }, (args, extra) => guarded(extra, async (call) => {
    if (args.file_name || args.folder) {
      const earlier = saveTo.get(args.job_id);
      remember(saveTo, args.job_id, {
        dir: args.folder !== undefined ? outputFolder(config.outputDir, args.folder) : earlier?.dir,
        // Without a file_name (and none given earlier), files are named from the job's prompt.
        base: args.file_name ? safeName(args.file_name) : earlier?.base,
      });
    }
    if (collected.has(args.job_id)) return collected.get(args.job_id);
    return collect(args.job_id, call, args.wait_seconds ?? config.waitSeconds);
  }));

  server.registerTool("cancel_image_job", {
    title: "Cancel picture",
    description: "Stop a picture that is waiting or being drawn (for example, after changing your mind).",
    inputSchema: { job_id: jobId },
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  }, (args, extra) => guarded(extra, async (call) => {
    let job;
    try {
      job = await api.cancelJob(args.job_id, inCall(call));
    } catch (err) {
      if (err?.status === 404 && !err.foreign) {
        return problem(`The image server has no job ${args.job_id}, so there is nothing to cancel. Check the job_id.`);
      }
      throw err;
    }
    switch (job.status) {
      case "queued":
      case "running":
        return text(`Cancelling job ${job.job_id}.`);
      case "cancelled":
        // Also the answer for a job this call just dropped from the queue (docs/API.md).
        return text(`Job ${job.job_id} is cancelled; no picture will be made.`);
      case "done":
        return text(`Job ${job.job_id} had already finished: the picture was made anyway (get_image_job ` +
          "collects it if it wasn't saved yet).");
      default:
        return text(`Job ${job.job_id} had already ${job.status}.`);
    }
  }));

  server.registerTool("get_image", {
    title: "Show picture",
    description: "Show a picture made earlier (preview plus its prompt, seed and settings). Set save_copy to " +
      "also save the full-size PNG on this computer.",
    inputSchema: {
      image_id: imageId,
      save_copy: z.boolean().default(false).describe("Also save the full-size PNG on this computer"),
      file_name: options.file_name,
      folder: options.folder,
    },
    annotations: WRITES,
  }, (args, extra) => guarded(extra, async (call) => {
    const dir = args.save_copy ? outputFolder(config.outputDir, args.folder) : null;
    const meta = await api.imageMeta(args.image_id, inCall(call));
    const lines = [`image_id: ${meta.id} (made ${when(meta.created_at)})`];
    if (meta.source === "upload") lines.push(`Uploaded picture${meta.filename ? `: ${meta.filename}` : ""}`);
    if (meta.width) lines.push(`Size: ${meta.width}x${meta.height}`);
    if (meta.prompt) lines.push(`Prompt: "${clip(meta.prompt, 1200)}" (style ${meta.style ?? "none"})`);
    if (meta.model) lines.push(`Model: ${meta.model}, seed ${meta.seed}`);
    if (meta.source_image_id) lines.push(`Edited from: ${meta.source_image_id}`);
    if (meta.reference_ids?.length) lines.push(`References: ${meta.reference_ids.join(", ")}`);
    if (args.save_copy) {
      // An upload is named after its file, without the extension ("class photo", not "class photo png").
      const where = target(args, meta.prompt ?? meta.filename?.replace(/\.[^.]+$/, "") ?? meta.id, dir);
      const copy = await saveCopy(args.image_id, where, call);
      lines.push(copy.line);
      if (copy.end) return text(lines.join("\n"));
    }
    const shown = timeLeft(call) >= previewMinMs ? await previews([{ image_id: meta.id }], call) : noPreviews(1);
    if (shown.failed) lines.push("The preview couldn't be loaded this time.");
    else if (shown.skipped) lines.push("There wasn't time to load the preview; ask again to see it.");
    return { content: [{ type: "text", text: lines.join("\n") }, ...shown.blocks] };
  }));

  server.registerTool("list_recent_images", {
    title: "Recent pictures",
    description: "List recent pictures on the image server, newest first, with their image_id, prompt and seed.",
    inputSchema: { limit: z.number().int().min(1).max(50).default(10).describe("How many to list") },
    annotations: READS,
  }, (args, extra) => guarded(extra, async (call) => {
    const { images } = await api.images(args.limit, inCall(call));
    if (!images.length) return text("No pictures on the image server yet.");
    return text(images.map((m) => {
      const what = m.source === "upload" ? `uploaded ${m.filename ?? "picture"}`
        : `"${clip(m.prompt, 100)}" (${m.style ?? "none"}, seed ${m.seed}, ${m.model})`;
      return `${m.id}  ${when(m.created_at)}  ${m.width}x${m.height}  ${what}`;
    }).join("\n"));
  }));

  server.registerTool("list_references", {
    title: "List references",
    description: "List saved reference pictures (characters, style samples) to pass as reference_ids.",
    inputSchema: { tag: z.string().max(40).optional().describe("Only references with this tag") },
    annotations: READS,
  }, (args, extra) => guarded(extra, async (call) => {
    // Cleaned like the tags save_reference stores, so a tag finds what was saved with it.
    const [tag] = cleanTags(args.tag === undefined ? [] : [args.tag]);
    const { references } = await api.references(tag, inCall(call));
    if (!references.length) {
      return text(tag ? `No references tagged "${tag}".` : "No references saved yet. Use save_reference to add one.");
    }
    return text(references.map((r) =>
      `${r.id}  "${r.name}"${r.tags.length ? `  tags: ${r.tags.join(", ")}` : ""}  ${r.width}x${r.height}`).join("\n"));
  }));

  server.registerTool("save_reference", {
    title: "Save reference",
    description: "Keep a picture as a reference (a character sheet or style sample) so later pictures can match " +
      "it: pass the returned reference_id in reference_ids. Use a picture made earlier (image_id) or a file on " +
      "this computer (source_path).",
    inputSchema: {
      name: z.string().min(1).max(120).describe("Short name, e.g. \"Benny the bear, front view\""),
      image_id: imageId.optional(),
      source_path: z.string().max(1024).optional().describe("Full path to a PNG, JPEG, WebP or GIF on this computer"),
      tags: z.array(z.string().min(1).max(40)).max(20).optional().describe("Tags for finding it later, e.g. [\"bear book\"]"),
    },
    annotations: WRITES,
  }, (args, extra) => guarded(extra, async (call) => {
    if (Boolean(args.image_id) === Boolean(args.source_path)) {
      throw new UserError("Give either image_id (a picture made earlier) or source_path (a file), not both.");
    }
    const tags = cleanTags(args.tags);
    let record;
    if (args.image_id) {
      record = await api.addReference({ from_image_id: args.image_id, name: args.name, tags }, inCall(call));
    } else {
      const { form } = await upload(args.source_path, call);
      form.append("name", args.name);
      form.append("tags", tags.join(","));
      record = await api.addReference(form, inCall(call));
    }
    return text(`Saved reference "${record.name}": reference_id ${record.id}` +
      `${record.tags.length ? ` (tags: ${record.tags.join(", ")})` : ""}. ` +
      "Pass it in reference_ids to keep later pictures consistent with it.");
  }));

  server.registerTool("delete_reference", {
    title: "Delete reference",
    description: "Delete a saved reference picture. Pictures already made from it are kept.",
    inputSchema: { reference_id: referenceId },
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  }, (args, extra) => guarded(extra, async (call) => {
    await api.deleteReference(args.reference_id, inCall(call));
    return text(`Deleted reference ${args.reference_id}.`);
  }));

  server.registerTool("image_server_status", {
    title: "Image server status",
    description: "Check that the image server is reachable and list its models, the loaded model, the queue, " +
      "and where pictures are saved on this computer. Use it first when another tool reports a connection problem.",
    inputSchema: {},
    annotations: READS,
  }, (args, extra) => guarded(extra, async (call) => {
    const lines = [`Image server: ${config.serverUrl}`, `Pictures are saved in: ${config.outputDir}`];
    const notTheServer = (reachable) => problem([...lines, reachable, "Check the server address in the extension " +
      "settings (on the server Mac, scripts/enable_lan.sh prints it)."].join("\n"));
    let health;
    try {
      health = await api.health(inCall(call));
    } catch (err) {
      if (!(err instanceof ImageServerError)) throw err;
      // A page from another program at this address (Ollama's 404, a login page, ...).
      if (err.foreign) return notTheServer("Reachable: something answers at this address, but it isn't the image server.");
      return problem([...lines, err.message].join("\n"));
    }
    if (health?.status !== "ok" || typeof health.version !== "string") {
      // Another web app (Open WebUI also has a /health) answers at this address, or a 1.0 image
      // server, whose /health says only {"status": "ok"} and which has no job API.
      return notTheServer(health?.status === "ok"
        ? "Reachable: something answers at this address, but not this version of the image server: it is an " +
          "older image server (update it on the server Mac with scripts/install_image_server.sh) or another program."
        : "Reachable: something answers at this address, but it isn't the image server.");
    }
    lines.push(`Reachable: yes (version ${health.version}, up ${Math.round((health.uptime_s ?? 0) / 60)} min)`);
    lines.push(`Loaded model: ${health.loaded_model ?? "none (the first picture takes longer while one loads)"}`);
    // The whole server's queue, Open WebUI's pictures included (jobs_active counts only the job API's).
    lines.push(`Pictures waiting or being drawn (all apps, Open WebUI included): ${health.queue_depth ?? "unknown"}`);
    try {
      const { models } = await api.models(inCall(call));
      lines.push("Models:");
      for (const m of models) {
        const roles = [m.default && "default", m.edit_default && "default for edits"].filter(Boolean);
        // The model's limit counts the picture being edited too (docs/API.md).
        const edits = `edits with up to ${m.max_reference_images} pictures (source + references)`;
        lines.push(`- ${m.id}${roles.length ? ` (${roles.join(", ")})` : ""}: ` +
          `${m.supports_edit ? edits : "new pictures only"}, ${m.license}`);
      }
      lines.push("Token: accepted");
    } catch (err) {
      if (!(err instanceof ImageServerError)) throw err;
      // Only a refused or missing token is the token's fault; anything else is reported as it is.
      const token = err.status === 401 || err.code === "NO_TOKEN";
      lines.push(token ? `Token: ${err.message}` : `Models: couldn't be listed (${err.message})`);
    }
    return text(lines.join("\n"));
  }));

  return server;
}
