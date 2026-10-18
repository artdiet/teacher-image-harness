// A stand-in for the image server's job API, enough to exercise the MCP proxy.
//
//   mock.trouble(match, what)  makes chosen requests misbehave (late, an error status, a dropped connection)
//   mock.restart(ms)           refuses connections for a while, like the server restarting
//   mock.state.requests        every request as "METHOD /path?query", in order

import http from "node:http";

export const TOKEN = "test-token-0123456789abcdef";
export const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==", "base64");

const hex = () => Array.from({ length: 16 }, () => "0123456789abcdef"[Math.floor(Math.random() * 16)]).join("");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// A preview: WebP's signature followed by filler (the extension checks pictures by their bytes).
function webp(size) {
  const bytes = Buffer.alloc(size, 1);
  bytes.write("RIFF", 0, "latin1");
  bytes.writeUInt32LE(size - 8, 4);
  bytes.write("WEBPVP8 ", 8, "latin1");
  return bytes;
}

export async function startMock({ jobMs = 50, hostCheck = null } = {}) {
  const state = {
    jobMs, jobs: new Map(), images: new Map(), references: new Map(),
    submitted: [], cancelled: [], uploads: [], previewBudgets: [], fullDownloads: 0,
    requests: [], troubles: [],
  };

  // The next requests whose "METHOD /path?query" matches (a RegExp, or a string it starts with)
  // misbehave, `times` times, after letting `skip` of them through:
  //   { delayMs }                  answer that much later (normally, unless combined with the rest)
  //   { status, body, headers }    answer with this instead (body: object -> JSON, string -> text)
  //   { reset: true }              drop the connection without an answer
  function trouble(match, { times = 1, skip = 0, ...what } = {}) {
    state.troubles.push({ match, times, skip, ...what });
  }

  function troubleFor(line) {
    for (const t of state.troubles) {
      if (t.times <= 0 || !(typeof t.match === "string" ? line.startsWith(t.match) : t.match.test(line))) continue;
      if (t.skip > 0) {
        t.skip--;
        return null;
      }
      t.times--;
      return t;
    }
    return null;
  }

  function jobView(job) {
    const elapsed = Date.now() - job.created;
    if (job.status === "queued" || job.status === "running") {
      if (job.cancelRequested) job.status = "cancelled";
      else if (elapsed >= state.jobMs) finishJob(job);
      else {
        job.status = elapsed > state.jobMs / 4 ? "running" : "queued";
        job.progress = Math.min(0.99, elapsed / state.jobMs);
      }
    }
    return { job_id: job.id, status: job.status, progress: job.progress, params: job.params,
      created_at: job.created / 1000, error: job.error ?? null, result: job.result ?? null };
  }

  function finishJob(job) {
    if (job.params.prompt.includes("FAIL")) {
      job.status = "failed";
      job.error = "the model rejected the input";
      return;
    }
    const images = [];
    for (let i = 0; i < job.params.n; i++) {
      const id = `img_${hex()}`;
      const seed = job.params.seed + i;
      state.images.set(id, { id, created_at: Date.now() / 1000, ...job.params, seed, job_id: job.id });
      images.push({ image_id: id, seed, width: job.params.width, height: job.params.height, seconds: 1.5 });
    }
    job.status = "done";
    job.progress = 1;
    job.result = { image_id: images[0].image_id, seed: images[0].seed, images, model: job.params.model, params: job.params };
  }

  function send(res, status, body, headers = {}) {
    if (Buffer.isBuffer(body)) {
      res.writeHead(status, { "content-length": body.length, ...headers });
      return res.end(body);
    }
    const data = body === undefined ? "" : JSON.stringify(body);
    res.writeHead(status, { "content-type": "application/json", ...headers });
    return res.end(data);
  }

  const server = http.createServer(async (req, res) => {
    // A new connection for every request, whatever the client asks for. Node 26's fetch (not
    // 18 to 24) holds back a request on a kept-alive connection that sat idle for half a second
    // or more, against any server; the extension asks for "connection: close" for that reason,
    // and this keeps tests that call the client directly quick too. With no pooled
    // connections, restart() also reliably means "connection refused".
    res.setHeader("connection", "close");
    const chunks = [];
    try {
      for await (const chunk of req) chunks.push(chunk);
    } catch {
      return; // the client went away mid-request (a stopped upload)
    }
    const raw = Buffer.concat(chunks);
    const url = new URL(req.url, "http://mock");
    const route = `${req.method} ${url.pathname}`;
    state.requests.push(`${req.method} ${req.url}`);
    const odd = troubleFor(`${req.method} ${req.url}`);
    if (odd?.delayMs) await sleep(odd.delayMs);
    if (odd?.reset) return req.socket.destroy();
    if (odd?.status) {
      if (typeof odd.body !== "string") return send(res, odd.status, odd.body, odd.headers);
      res.writeHead(odd.status, { "content-type": "text/plain", ...odd.headers });
      return res.end(odd.body);
    }
    if (hostCheck && !hostCheck(req.headers.host)) {
      res.writeHead(400, { "content-type": "text/plain" });
      return res.end("Invalid host header");
    }
    if (route === "GET /health") {
      return send(res, 200, { status: "ok", version: "2.0.0", loaded_model: null, queue_depth: 0,
        jobs_active: [...state.jobs.values()].filter((j) => j.status === "queued").length, memory: {}, uptime_s: 125 });
    }
    if (req.headers.authorization !== `Bearer ${TOKEN}`) return send(res, 401, { detail: "missing or invalid token" });

    if (route === "GET /models") {
      return send(res, 200, { models: [
        { id: "flux2-klein-4b", license: "apache-2.0", supports_edit: true, max_reference_images: 4,
          default: true, edit_default: true },
        { id: "z-image-turbo", license: "apache-2.0", supports_edit: false, max_reference_images: 0,
          default: false, edit_default: false },
      ] });
    }
    if (route === "POST /jobs") {
      const body = JSON.parse(raw.toString());
      state.submitted.push(body);
      if (body.model === "nope") return send(res, 400, { detail: "unknown model 'nope'; available: flux2-klein-4b" });
      const [width, height] = body.size ? body.size.split("x").map(Number) : [body.width ?? 1024, body.height ?? 1024];
      const job = {
        id: `job_${hex()}`, created: Date.now(), status: "queued", progress: 0,
        params: { prompt: body.prompt, style: body.style ?? "none", full_prompt: `${body.prompt}, styled`,
          model: body.model ?? "flux2-klein-4b", width, height, seed: body.seed ?? 42, n: body.n ?? 1,
          reference_ids: body.reference_ids ?? [], source_image_id: body.source_image_id ?? null },
      };
      state.jobs.set(job.id, job);
      return send(res, 202, { job_id: job.id, status: "queued", queue_depth: 1 });
    }
    let m;
    if ((m = route.match(/^(GET|DELETE) \/jobs\/(job_[0-9a-f]{16})$/))) {
      const job = state.jobs.get(m[2]);
      if (!job) return send(res, 404, { detail: "no such job" });
      if (m[1] === "DELETE") {
        state.cancelled.push(job.id);
        // Like jobs.py: a job still waiting is dropped at once and the answer says so; a running
        // one stops at its next step, so its status flips on a later read.
        const view = jobView(job);
        if (view.status === "queued") {
          job.status = "cancelled";
          return send(res, 200, jobView(job));
        }
        if (view.status === "running") job.cancelRequested = true;
        return send(res, 200, view);
      }
      return send(res, 200, jobView(job));
    }
    if (route === "POST /images") {
      const type = req.headers["content-type"] ?? "";
      if (!type.startsWith("multipart/form-data") || !raw.includes(PNG.subarray(0, 8))) {
        return send(res, 400, { detail: "send the picture as multipart field 'file'" });
      }
      const filename = /filename="([^"]+)"/.exec(raw.toString("latin1"))?.[1];
      state.uploads.push({ filename, bytes: raw.length });
      const id = `img_${hex()}`;
      state.images.set(id, { id, created_at: Date.now() / 1000, source: "upload", filename, width: 1, height: 1 });
      return send(res, 201, { image_id: id, width: 1, height: 1 });
    }
    if (route === "GET /images") {
      const limit = Number(url.searchParams.get("limit") ?? 20);
      return send(res, 200, { images: [...state.images.values()].reverse().slice(0, limit) });
    }
    if ((m = route.match(/^GET \/images\/(img_[0-9a-f]{16})(\/meta)?$/))) {
      const meta = state.images.get(m[1]);
      if (!meta) return send(res, 404, { detail: "no such image" });
      if (m[2]) return send(res, 200, meta);
      if (url.searchParams.get("size") === "preview") {
        const budget = Number(url.searchParams.get("budget") ?? 70000);
        state.previewBudgets.push(budget);
        return send(res, 200, webp(Math.min(budget, 5000)), { "content-type": "image/webp" });
      }
      state.fullDownloads++;
      return send(res, 200, PNG, { "content-type": "image/png" });
    }
    if (route === "POST /references") {
      let record;
      if ((req.headers["content-type"] ?? "").startsWith("application/json")) {
        const body = JSON.parse(raw.toString());
        if (!state.images.has(body.from_image_id)) return send(res, 404, { detail: "no image" });
        record = { name: body.name, tags: body.tags, source: `image:${body.from_image_id}` };
      } else {
        const text = raw.toString("latin1");
        const field = (name) => new RegExp(`name="${name}"\\r\\n\\r\\n([^\\r]*)`).exec(text)?.[1] ?? "";
        record = { name: field("name"), tags: field("tags").split(",").filter(Boolean), source: "upload" };
      }
      const id = `ref_${hex()}`;
      state.references.set(id, { id, ...record, width: 1, height: 1, created_at: Date.now() / 1000 });
      return send(res, 201, state.references.get(id));
    }
    if (route === "GET /references") {
      const tag = url.searchParams.get("tag");
      return send(res, 200, { references: [...state.references.values()].filter((r) => !tag || r.tags.includes(tag)) });
    }
    if ((m = route.match(/^DELETE \/references\/(ref_[0-9a-f]{16})$/))) {
      if (!state.references.delete(m[1])) return send(res, 404, { detail: "no such reference" });
      res.writeHead(204);
      return res.end();
    }
    return send(res, 404, { detail: "Not Found" });
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();

  // Stop listening for `ms` (connections are refused), then listen again; resolves when back.
  function restart(ms) {
    server.close();
    return new Promise((resolve) => setTimeout(() => server.listen(port, "127.0.0.1", resolve), ms));
  }

  return {
    url: `http://127.0.0.1:${port}`, state, trouble, restart,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}
