// Settings come from environment variables. In the Claude Desktop extension they are filled
// from the extension's settings page (manifest.json user_config); in claude_desktop_config.json
// they go in the server's "env" block.
//
//   IMAGE_SERVER_URL     http://<image-server-mac>:7865 (default http://127.0.0.1:7865)
//   IMAGE_SERVER_TOKEN   this computer's token (scripts/client_tokens.sh add <name> on the server)
//   IMAGE_OUTPUT_DIR     where full-size PNGs are saved (default: see defaultOutputDir)
//   IMAGE_WAIT_SECONDS   how long a tool call waits for a picture before handing back a job id (5 to 40)
//   IMAGE_INPUT_ROOTS    extra folders pictures may be sent from (the home and output folders always are)

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const DEFAULT_SERVER_URL = "http://127.0.0.1:7865";
const SERVER_PORT = "7865"; // the image server's port
// Local MCP calls time out at 60 s in Claude Desktop; a call has 50 s in all (server.js), and
// saving the pictures after the wait needs some of it.
export const DEFAULT_WAIT_SECONDS = 40;
const MIN_WAIT_SECONDS = 5;
const MAX_WAIT_SECONDS = 40;

const TOKEN_PROBLEM = "The API token contains characters that can't be sent. Copy it again from the server Mac " +
  "and paste it into the extension settings.";

// An optional extension setting left empty can arrive as "" or as its unexpanded placeholder.
function setting(env, name) {
  const value = String(env[name] ?? "").trim();
  return value && !value.startsWith("${") ? value : "";
}

export function expandHome(p, home = os.homedir()) {
  if (p === "~") return home;
  if (p.startsWith("~/") || p.startsWith("~\\")) return path.join(home, p.slice(2));
  return p;
}

function isDir(p) {
  try {
    return fs.statSync(p).isDirectory();
  } catch {
    return false;
  }
}

// Inside the Claude folder Cowork works in, when there is one, so tasks can use the pictures.
export function defaultOutputDir(home = os.homedir()) {
  for (const folder of [path.join(home, "Documents", "Claude"), path.join(home, "Claude")]) {
    if (isDir(folder)) return path.join(folder, "Generated Images");
  }
  return path.join(home, "Pictures", "Teach Image");
}

// Whether the address names its port. URL can't tell: it drops a default port, so
// "http://mac.local:80" and "http://mac.local" both have an empty port.
function namesPort(address) {
  const host = /^[a-z][a-z0-9+.-]*:\/\/([^/?#]*)/i.exec(address)?.[1] ?? "";
  return /:\d+$/.test(host.replace(/^\[[^\]]*\]/, "")); // [IPv6] addresses are full of colons
}

function parseServerUrl(raw) {
  const address = raw.includes("://") ? raw : `http://${raw}`;
  let url;
  try {
    url = new URL(address);
  } catch {
    return { problem: `The image server address "${raw}" isn't a URL. It should look like http://192.168.1.20:7865` };
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return { problem: `The image server address must start with http:// or https:// (got ${url.protocol}).` };
  }
  if (url.username || url.password) {
    return { problem: "Put the token in the API token setting, not in the server address." };
  }
  // The server listens on IPv4 only, and Node 18 tries localhost's IPv6 address (::1) first,
  // which nothing answers.
  if (url.hostname === "localhost") url.hostname = "127.0.0.1";
  // An http:// address without a port has most likely just left it out: nothing serves the
  // image server on port 80. (An https:// one is taken as given.)
  if (url.protocol === "http:" && !namesPort(address)) url.port = SERVER_PORT;
  // Open WebUI is set up with .../v1; this proxy talks to the server root.
  const base = url.pathname.replace(/\/+$/, "").replace(/\/v1$/, "");
  return { serverUrl: `${url.origin}${base}` };
}

// A pasted token can pick up spaces, line breaks and invisible characters (zero-width spaces,
// byte-order marks). None belong in a token, and an HTTP header can't carry most of them.
function parseToken(raw) {
  const token = raw.replace(/[\u200b-\u200d\u2060\ufeff]/g, "").trim()
    .replace(/^Bearer\s+/i, "")
    .replace(/[\s\u200b-\u200d\u2060\ufeff]/g, "");
  return { token, problem: /^[\x21-\x7e]*$/.test(token) ? null : TOKEN_PROBLEM };
}

export function loadConfig(env = process.env) {
  const { serverUrl, problem } = parseServerUrl(setting(env, "IMAGE_SERVER_URL") || DEFAULT_SERVER_URL);
  const wait = Number.parseInt(setting(env, "IMAGE_WAIT_SECONDS"), 10);
  const roots = setting(env, "IMAGE_INPUT_ROOTS").split(path.delimiter).filter(Boolean);
  const outputDir = path.resolve(expandHome(setting(env, "IMAGE_OUTPUT_DIR")) || defaultOutputDir());
  const { token, problem: tokenProblem } = parseToken(setting(env, "IMAGE_SERVER_TOKEN"));
  return {
    serverUrl: serverUrl ?? DEFAULT_SERVER_URL,
    problem: problem ?? tokenProblem ?? null,
    token,
    outputDir,
    waitSeconds: Number.isFinite(wait)
      ? Math.min(MAX_WAIT_SECONDS, Math.max(MIN_WAIT_SECONDS, wait)) : DEFAULT_WAIT_SECONDS,
    // Pictures saved by this extension can always be sent back to edit.
    inputRoots: [os.homedir(), outputDir, ...roots.map((r) => path.resolve(expandHome(r)))],
  };
}
