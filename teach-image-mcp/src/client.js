// REST client for the image server's job API (docs/API.md). Every failure becomes an
// ImageServerError whose message is written for the person using Claude.

const MAX_DOWNLOAD = 64 * 1024 * 1024;
const MIN_REQUEST_MS = 1_000; // a request started with less time left than this can't finish anyway
const REFUSED_RETRY_MS = 3_000; // a restarting server refuses connections for a few seconds
// Longer than undici's 10 s connect timeout, so a Mac that is asleep or gone is reported as
// unreachable (UND_ERR_CONNECT_TIMEOUT), not as a server that didn't answer.
const HEALTH_TIMEOUT_MS = 12_000;
const TIMEOUTS = new Set(["TimeoutError", "UND_ERR_HEADERS_TIMEOUT", "UND_ERR_BODY_TIMEOUT"]);
// Something at the address took the connection, then closed it: usually the server restarting.
const DROPPED = new Set(["ECONNRESET", "EPIPE", "UND_ERR_SOCKET"]);
// Nothing at the address took the connection, so the request never reached the server.
const NOT_SENT = new Set(["ECONNREFUSED", "ENOTFOUND", "EAI_AGAIN", "EHOSTUNREACH", "ENETUNREACH",
  "EHOSTDOWN", "UND_ERR_CONNECT_TIMEOUT"]);
// A secure (https) connection that failed before any request was sent: the image server speaks
// plain http (wrong version number), or a certificate wasn't trusted.
const TLS_FAILURE = /^(ERR_SSL_|ERR_TLS_|EPROTO$)|CERT|^UNABLE_TO_/;
// FastAPI's answers for a route that doesn't exist: the address leads to another web app (or a
// path inside one), not to the image server, whose own 404s name what is missing.
const NO_SUCH_ROUTE = new Map([[404, "Not Found"], [405, "Method Not Allowed"]]);
const DETAIL_MAX = 300; // the server's details are short; a longer one is cut
const CHECK_ADDRESS = "Check the server address in the extension settings (on the server Mac, " +
  "scripts/enable_lan.sh prints it).";
// A loopback address is right only on the server Mac itself; it is also the setting's default,
// so another computer may still have it.
const ELSEWHERE = "If the image server runs on another computer, put that Mac's address in the extension " +
  "settings (scripts/enable_lan.sh on it prints it).";

// Claude also runs on Windows and Linux, where "this Mac" would be wrong.
const thisComputer = () => (process.platform === "darwin" ? "this Mac" : "this computer");

// Deadlines use a clock that never jumps: a system clock set back (or forward) mid-call
// must not stretch or cut a call's time.
export const now = () => performance.now();

export class ImageServerError extends Error {
  // status: the HTTP status, when the server answered. code: the network error (ECONNREFUSED,
  // ...) when it didn't. transient: asking again a little later may work (no answer, a
  // timeout, a 5xx), unlike an answer such as "wrong token" that will be the same next time.
  // lostAnswer: the request went out but no answer came back, so the server may have acted
  // on it (made a job) anyway. foreign: the answer came from something that isn't the image
  // server (a wrong address).
  constructor(message, { status, code, transient = false, lostAnswer = false, foreign = false } = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.transient = transient;
    this.lostAnswer = lostAnswer;
    this.foreign = foreign;
  }
}

// The tool call's deadline came first: too little time was left to start a request, or the
// answer didn't arrive before it (lostAnswer).
export class OutOfTime extends ImageServerError {
  constructor(config, { lostAnswer = false } = {}) {
    super(`Ran out of time waiting for the image server at ${config.serverUrl}. It may be busy or the ` +
      "network slow; try again in a minute.", { transient: true, lostAnswer });
  }
}

// A request that ran out of time, whether at the call's deadline or at its own timeout.
export function timedOut(err) {
  return err instanceof OutOfTime || TIMEOUTS.has(err?.code);
}

export function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

// AbortSignal.any arrived in Node 20.3; Claude Desktop's bundled Node may be older.
function anySignal(signals) {
  if (typeof AbortSignal.any === "function") return AbortSignal.any(signals);
  const controller = new AbortController();
  for (const signal of signals) {
    if (signal.aborted) {
      controller.abort(signal.reason);
      break;
    }
    signal.addEventListener("abort", () => controller.abort(signal.reason), { once: true });
  }
  return controller.signal;
}

function isLoopback(serverUrl) {
  const host = new URL(serverUrl).hostname;
  return host === "localhost" || host === "[::1]" || host.startsWith("127.");
}

// fetch's own timeout is a DOMException whose numeric legacy code (23) means nothing to anyone,
// so look at names first, and only show string codes (ECONNREFUSED, UND_ERR_SOCKET, ...).
function failureCode(err) {
  if (err?.name === "TimeoutError" || err?.cause?.name === "TimeoutError") return "TimeoutError";
  for (const code of [err?.cause?.code, err?.code]) {
    if (typeof code === "string") return code;
  }
  return err?.cause?.message || err?.message || "error";
}

// No connection, or a connection lost. The advice depends on what happened: a refused or
// dropped connection means something at the address answered, so the Mac is awake and
// reachable. `reached`: an earlier request of this call got an answer from the same address.
function unreachable(config, code, reached) {
  const server = `The image server at ${config.serverUrl}`;
  if (code === "ECONNREFUSED") {
    if (reached) return `${server} stopped accepting connections (ECONNREFUSED); it may be restarting. Try again in a few seconds.`;
    if (isLoopback(config.serverUrl)) {
      return `${server} isn't accepting connections (ECONNREFUSED). It may be restarting (try again in a few ` +
        `seconds) or not running: start it on ${thisComputer()} with scripts/install_image_server.sh. ${ELSEWHERE}`;
    }
    return `${server} isn't accepting connections (ECONNREFUSED). It may be restarting (try again in a few ` +
      "seconds) or stopped, LAN access may be off, or the address may be out of date. On the server Mac, " +
      "scripts/doctor.sh checks it, and scripts/enable_lan.sh turns on LAN access and prints the address.";
  }
  if (DROPPED.has(code)) return `${server} closed the connection (${code}); it may be restarting. Try again in a moment.`;
  if (isLoopback(config.serverUrl)) {
    return `Can't reach the image server at ${config.serverUrl} (${code}). Is it running on ${thisComputer()}? ` +
      `Start it with scripts/install_image_server.sh. ${ELSEWHERE}`;
  }
  if (reached) {
    return `Lost the connection to the image server at ${config.serverUrl} (${code}). Check that the Mac ` +
      "running it is still awake and on the network.";
  }
  return `Can't reach the image server at ${config.serverUrl} (${code}). Check that the Mac running it is ` +
    "awake and on the same network, and that the server address in this extension's settings is right. " +
    "On a Mac, Claude also needs permission to reach the local network: System Settings > Privacy & Security > " +
    "Local Network > Claude.";
}

// A connection that was made but got no answer. A restarting server refuses connections and a
// busy one still answers, so neither explains this; a firewall holding connections does.
function silence(config, timeoutMs, reached) {
  const server = `The image server at ${config.serverUrl}`;
  const seconds = Math.round(timeoutMs / 1000);
  if (reached) return `${server} stopped answering (nothing for ${seconds} s). Try again in a minute.`;
  if (isLoopback(config.serverUrl)) {
    return `${server} didn't answer within ${seconds} s. If this keeps happening, run scripts/doctor.sh on ` +
      `${thisComputer()}. ${ELSEWHERE}`;
  }
  return `${server} didn't answer within ${seconds} s. If this keeps happening, the server Mac's firewall may ` +
    "be holding connections from this computer: run scripts/check_lan.sh on that Mac.";
}

function notTheServer(config, what, { status } = {}) {
  return new ImageServerError(`Something at ${config.serverUrl} ${what}, but it doesn't look like the image ` +
    `server. ${CHECK_ADDRESS}`, { status, foreign: true });
}

// A request that got no answer. `clamped`: its timeout was the tool call's deadline.
function noAnswer(config, err, { clamped, timeoutMs, reached }) {
  const code = failureCode(err);
  // fetch is told never to follow redirects, and the image server never sends one: the answer
  // came from something else (a login page, another web app). Nothing reached the server.
  if (/redirect/i.test(String(err?.cause?.message ?? err?.message))) {
    return notTheServer(config, "redirected the request elsewhere");
  }
  if (config.serverUrl.startsWith("https:") && TLS_FAILURE.test(code)) {
    return new ImageServerError(`Couldn't make a secure (https) connection to ${config.serverUrl} (${code}). ` +
      `The image server uses plain http: change the address to http://${new URL(config.serverUrl).host} in ` +
      "the extension settings.", { code });
  }
  if (!TIMEOUTS.has(code)) {
    return new ImageServerError(unreachable(config, code, reached),
      { code, transient: true, lostAnswer: !NOT_SENT.has(code) });
  }
  if (clamped) return new OutOfTime(config, { lostAnswer: true });
  return new ImageServerError(silence(config, timeoutMs, reached), { code, transient: true, lostAnswer: true });
}

function clip(text, max = DETAIL_MAX) {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

// The image server's own words about a failure: {"detail": "..."}, or FastAPI's list of
// validation errors. Null for an answer in any other form (a page from something else).
function serverDetail(text) {
  let detail;
  try {
    detail = JSON.parse(text)?.detail;
  } catch {
    return null;
  }
  if (typeof detail === "string") return clip(detail);
  if (!Array.isArray(detail)) return null;
  // [{loc: ["body", "prompt"], msg: "..."}]
  return clip(detail.map((d) => {
    const where = (d?.loc ?? []).filter((part) => part !== "body").join(".");
    return where ? `${where}: ${d?.msg}` : String(d?.msg ?? JSON.stringify(d));
  }).join("; "));
}

function sentence(text) {
  const s = text.trim();
  return `${s.charAt(0).toUpperCase()}${s.slice(1)}${/[.!?…]$/.test(s) ? "" : "."}`;
}

// What a refused request means for the person using Claude, or null when the answer didn't
// come from the image server. `detail`: the server's own words (null if the answer had none);
// `text`: the answer as it came.
function statusMessage(status, detail, text, config) {
  if (status === 401) {
    return "The image server didn't accept this computer's token. Check the API token in the extension " +
      "settings (on the server Mac, scripts/client_tokens.sh add <name> makes a new one).";
  }
  if (status === 413) return "That picture is larger than the image server accepts. Try a smaller copy of it.";
  if (status === 503) {
    // Busy (too many pictures waiting), restarting, or too many open connections from this
    // computer: the server's own words say which. It is an answer, so nothing was lost: the
    // request wasn't carried out (no job was made) and isn't repeated here.
    return detail ? sentence(detail) : "The image server is busy with other pictures. Try again in a minute.";
  }
  if (detail === null) {
    // Plain text: one of the two refusals the server makes before its app runs (docs/API.md),
    // its web framework's answer to a crash, or a page from something else at this address.
    if (status === 400 && /invalid host header/i.test(text)) {
      return `The image server doesn't accept the address ${new URL(config.serverUrl).host}. On the server ` +
        "Mac, run scripts/enable_lan.sh (it lists the Mac's name and IP), or add this address to " +
        "IMAGE_SERVER_ALLOWED_HOSTS in .env and restart the server.";
    }
    if (status === 403 && /private network/i.test(text)) {
      return "The image server only answers computers on the local network, and refused this one.";
    }
    if (status >= 500) {
      return `Image server error ${status}. Try again in a minute; if it keeps happening, run scripts/doctor.sh ` +
        "on the server Mac.";
    }
    return null;
  }
  if (NO_SUCH_ROUTE.get(status) === detail) return null;
  if (status === 400) return detail || "The image server rejected the request.";
  return detail ? `Image server error ${status}: ${detail}` : `Image server error ${status}.`;
}

async function failure(res, config) {
  const text = await res.text().catch(() => "");
  const message = statusMessage(res.status, serverDetail(text), text, config);
  // Never echoed: a page from something else can be long, and says nothing about the server.
  if (message === null) return notTheServer(config, `answered (HTTP ${res.status})`, { status: res.status });
  return new ImageServerError(message, { status: res.status, transient: res.status >= 500 });
}

async function answer(res, bytes, config) {
  if (bytes) {
    if (Number(res.headers.get("content-length") ?? 0) > MAX_DOWNLOAD) {
      throw new ImageServerError("The image server sent an unexpectedly large file.");
    }
    return { data: Buffer.from(await res.arrayBuffer()), type: res.headers.get("content-type") ?? "" };
  }
  if (res.status === 204) return null;
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    throw notTheServer(config, `answered (HTTP ${res.status})`, { status: res.status });
  }
}

export function createClient(config, { fetchImpl = globalThis.fetch, refusedRetryMs = REFUSED_RETRY_MS } = {}) {
  // `call` ({ deadline, signal }) is shared by the requests of one tool call: none runs past
  // the deadline, and if the first one finds the connection refused (the server is
  // restarting, so it never saw the request and even a POST is safe to repeat), it is tried
  // once more. `call.reached` records that the server answered one of them.
  async function request(method, route, opts = {}) {
    const { call, signal } = opts;
    const first = Boolean(call) && !call.started;
    if (call) call.started = true;
    try {
      return await attempt(method, route, opts);
    } catch (err) {
      if (!first || err?.code !== "ECONNREFUSED") throw err;
      if (call.deadline - now() < refusedRetryMs + MIN_REQUEST_MS) throw err;
      // Stopping the call ends the pause even for a request that is otherwise not stopped
      // with it (POST /jobs): nothing has reached the server yet, so nothing is left behind.
      await sleep(refusedRetryMs, signal ?? call.signal);
      return attempt(method, route, opts);
    }
  }

  async function attempt(method, route, { json, form, query, signal, call, auth = true, timeoutMs = 15_000, bytes = false }) {
    if (config.problem) throw new ImageServerError(config.problem, { code: "SETTINGS" });
    if (auth && !config.token) {
      throw new ImageServerError("No API token is set. Add this computer's token in the extension settings " +
        "(on the server Mac: scripts/client_tokens.sh add <name>).", { code: "NO_TOKEN" });
    }
    const url = new URL(config.serverUrl + route);
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value !== undefined && value !== null) url.searchParams.set(key, String(value));
    }
    // A new connection for every request: Node 26's fetch holds back a request on a kept-alive
    // connection that sat idle for half a second or more until about 4 s after the previous
    // one (Node 18 to 24 don't), which would slow every poll for a picture.
    const headers = { accept: bytes ? "image/*" : "application/json", connection: "close" };
    if (auth) headers.authorization = `Bearer ${config.token}`;
    let body;
    if (json !== undefined) {
      headers["content-type"] = "application/json";
      body = JSON.stringify(json);
    } else if (form !== undefined) {
      // Sent as finished bytes, not as FormData: Node 24's fetch crashes the whole process (an
      // uncaught undici error) when a FormData body meets a signal that is already stopped.
      const encoded = new Response(form);
      headers["content-type"] = encoded.headers.get("content-type");
      body = await encoded.blob();
    }
    // A call stopped meanwhile sends nothing more.
    if (signal?.aborted) throw signal.reason;
    const left = call ? call.deadline - now() : Infinity;
    if (left < MIN_REQUEST_MS) throw new OutOfTime(config);
    // The timeout covers reading the body too, so a slow download also stops at the deadline.
    // (AbortSignal.timeout takes whole milliseconds; the clock has fractions.)
    const timeout = AbortSignal.timeout(Math.floor(Math.min(timeoutMs, left)));
    let res;
    try {
      res = await fetchImpl(url, {
        method, headers, body, redirect: "error",
        signal: signal ? anySignal([signal, timeout]) : timeout,
      });
      if (call) call.reached = true;
      if (res.ok) return await answer(res, bytes, config);
    } catch (err) {
      if (signal?.aborted) throw signal.reason ?? err;
      if (err instanceof ImageServerError) throw err;
      throw noAnswer(config, err, { clamped: left < timeoutMs, timeoutMs, reached: Boolean(call?.reached || res) });
    }
    throw await failure(res, config);
  }

  const id = (value) => encodeURIComponent(value);
  return {
    health: (opts) => request("GET", "/health", { auth: false, timeoutMs: HEALTH_TIMEOUT_MS, ...opts }),
    models: (opts) => request("GET", "/models", opts),
    submitJob: (job, opts) => request("POST", "/jobs", { json: job, ...opts }),
    job: (jobId, opts) => request("GET", `/jobs/${id(jobId)}`, opts),
    cancelJob: (jobId, opts) => request("DELETE", `/jobs/${id(jobId)}`, opts),
    uploadImage: (form, opts) => request("POST", "/images", { form, timeoutMs: 60_000, ...opts }),
    images: (limit, opts) => request("GET", "/images", { query: { limit }, ...opts }),
    imageMeta: (imageId, opts) => request("GET", `/images/${id(imageId)}/meta`, opts),
    imageFull: (imageId, opts) => request("GET", `/images/${id(imageId)}`,
      { query: { size: "full" }, bytes: true, timeoutMs: 60_000, ...opts }),
    imagePreview: (imageId, budget, opts) => request("GET", `/images/${id(imageId)}`,
      { query: { size: "preview", budget }, bytes: true, timeoutMs: 30_000, ...opts }),
    references: (tag, opts) => request("GET", "/references", { query: { tag }, ...opts }),
    addReference: (body, opts) => request("POST", "/references",
      body instanceof FormData ? { form: body, timeoutMs: 60_000, ...opts } : { json: body, ...opts }),
    deleteReference: (refId, opts) => request("DELETE", `/references/${id(refId)}`, opts),
  };
}
