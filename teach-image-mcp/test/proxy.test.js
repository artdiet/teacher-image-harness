import assert from "node:assert/strict";
import fs from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { after, before, beforeEach, describe, test } from "node:test";
import { pathToFileURL } from "node:url";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import { createClient, now } from "../src/client.js";
import { loadConfig } from "../src/config.js";
import { nameFromPrompt, outputFolder, readInputImage, safeName, savePng, sniffImage } from "../src/files.js";
import { createServer, looksLikeTimeout } from "../src/server.js";
import { PNG, TOKEN, startMock } from "./mock-server.js";

let mock;
let tmp;

const RESTARTING = "the image server is restarting; try again in a few seconds";
const FULL_PNG = /^GET \/images\/img_[0-9a-f]{16}\?size=full/;

// options: client settings (fetchImpl, refusedRetryMs) and server timings (callBudgetMs, ...).
async function connect(overrides = {}, { fetchImpl, refusedRetryMs = 200, ...timing } = {}) {
  const config = {
    serverUrl: mock.url, problem: null, token: TOKEN, outputDir: path.join(tmp, "out"),
    waitSeconds: 5, inputRoots: [tmp], ...overrides,
  };
  const server = createServer(config, createClient(config, { fetchImpl, refusedRetryMs }), { log: () => {}, ...timing });
  const [serverSide, clientSide] = InMemoryTransport.createLinkedPair();
  await server.connect(serverSide);
  const client = new Client({ name: "test", version: "1.0.0" });
  await client.connect(clientSide);
  return { client, config };
}

const call = (client, name, args = {}, opts) => client.callTool({ name, arguments: args }, undefined, opts);
const textOf = (result) => result.content.filter((c) => c.type === "text").map((c) => c.text).join("\n");
const imagesOf = (result) => result.content.filter((c) => c.type === "image");
const jobIdOf = (result) => /job_[0-9a-f]{16}/.exec(textOf(result))?.[0];
const lastJob = () => [...mock.state.jobs.keys()].at(-1);
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const files = (dir) => fs.readdir(dir).then((names) => names.sort(), () => []);
const noFolderPermissions = (process.platform === "win32" || process.getuid?.() === 0) && "needs folder permissions";
const networkError = (code) => new TypeError("fetch failed", { cause: Object.assign(new Error(code), { code }) });

// How long `work` takes, on a clock that a test may not stub.
async function timed(work) {
  const started = performance.now();
  const result = await work;
  return { result, ms: performance.now() - started };
}

async function waitFor(check, ms = 3000) {
  const end = Date.now() + ms;
  while (!check()) {
    if (Date.now() > end) assert.fail("timed out waiting");
    await delay(20);
  }
}

// waitFor, for a check that has to look at the disk.
async function eventually(check, ms = 3000) {
  const end = Date.now() + ms;
  while (!(await check())) {
    if (Date.now() > end) assert.fail("timed out waiting");
    await delay(20);
  }
}

// A port nothing listens on: connections to it are refused.
async function closedPort() {
  const probe = net.createServer();
  await new Promise((resolve) => probe.listen(0, "127.0.0.1", resolve));
  const { port } = probe.address();
  await new Promise((resolve) => probe.close(resolve));
  return port;
}

before(async () => {
  mock = await startMock();
});
after(() => mock.close());
beforeEach(async () => {
  tmp = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), "teach-image-")));
  mock.state.jobMs = 50;
  mock.state.troubles = [];
  mock.state.requests = [];
});

describe("tools", () => {
  test("lists the tools with annotations and server instructions", async () => {
    const { client } = await connect();
    const { tools } = await client.listTools();
    assert.deepEqual(tools.map((t) => t.name).sort(), [
      "cancel_image_job", "delete_reference", "edit_image", "generate_image", "get_image", "get_image_job",
      "image_server_status", "list_recent_images", "list_references", "save_reference"]);
    const byName = Object.fromEntries(tools.map((t) => [t.name, t]));
    assert.equal(byName.delete_reference.annotations.destructiveHint, true);
    assert.equal(byName.list_references.annotations.readOnlyHint, true);
    assert.ok(tools.every((t) => !t.outputSchema), "no outputSchema (Claude clients mishandle it)");
    assert.match(client.getInstructions(), /edit_image/);
    const size = "Exact WIDTHxHEIGHT in pixels (256 to 2048 per side); the server snaps to multiples of 16 and " +
      "scales very large sizes down to about 1.5 megapixels. Overrides aspect";
    assert.equal(byName.generate_image.inputSchema.properties.size.description, size);
    assert.equal(byName.edit_image.inputSchema.properties.size.description, size);
    assert.equal(byName.get_image_job.inputSchema.properties.wait_seconds.maximum, 40);
    assert.match(byName.get_image_job.description, /Calling it again for a job already saved returns the earlier result and doesn't save another copy \(within one Claude Desktop session\)\.$/);
  });

  test("generate_image saves the PNG, never overwrites, and returns facts plus a preview", async () => {
    const { client, config } = await connect();
    const args = { prompt: "A friendly owl reading a book", style: "icon", aspect: "portrait", seed: 7 };
    const first = await call(client, "generate_image", args);
    assert.ok(!first.isError, textOf(first));
    const job = mock.state.submitted.at(-1);
    assert.equal(job.style, "icon");
    assert.deepEqual([job.width, job.height], [896, 1152]);
    const text = textOf(first);
    const saved = path.join(config.outputDir, "a-friendly-owl-reading-a-book.png");
    assert.match(text, new RegExp(`Saved to: ${saved.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
    assert.match(text, /image_id: img_[0-9a-f]{16}/);
    assert.match(text, /seed 7/);
    assert.match(text, /Prompt the model received: "A friendly owl reading a book, styled"/);
    assert.deepEqual(await fs.readFile(saved), PNG);
    const [preview] = imagesOf(first);
    assert.equal(preview.mimeType, "image/webp");
    assert.equal(mock.state.previewBudgets.at(-1), 70000);

    const second = await call(client, "generate_image", args);
    assert.match(textOf(second), /a-friendly-owl-reading-a-book \(2\)\.png/);
  });

  test("several pictures share the preview budget and get numbered files", async () => {
    const { client, config } = await connect();
    const result = await call(client, "generate_image",
      { prompt: "apple", n: 4, file_name: "../../Fruit: apple?", folder: "Unit 3/../Fruit" });
    assert.ok(!result.isError, textOf(result));
    assert.equal(imagesOf(result).length, 4);
    assert.deepEqual(mock.state.previewBudgets.slice(-4), [18750, 18750, 18750, 18750]);
    const files = await fs.readdir(path.join(config.outputDir, "Unit 3", "Fruit"));
    assert.deepEqual(files.sort(), ["Fruit apple-1.png", "Fruit apple-2.png", "Fruit apple-3.png", "Fruit apple-4.png"]);
    assert.match(textOf(result), /Made 4 pictures/);
  });

  test("a slow picture comes back as a job_id, collected once by get_image_job", async () => {
    mock.state.jobMs = 2000;
    const { client, config } = await connect({ waitSeconds: 0.3 });
    const pending = await call(client, "generate_image", { prompt: "slow turtle", file_name: "turtle" });
    const jobId = jobIdOf(pending);
    assert.ok(jobId, textOf(pending));
    assert.match(textOf(pending), /get_image_job/);
    const done = await call(client, "get_image_job", { job_id: jobId, wait_seconds: 10 });
    assert.match(textOf(done), /turtle\.png/);
    const downloads = mock.state.fullDownloads;
    const again = await call(client, "get_image_job", { job_id: jobId });
    assert.equal(textOf(again), textOf(done));
    assert.equal(mock.state.fullDownloads, downloads, "not downloaded or saved twice");
    assert.deepEqual(await fs.readdir(config.outputDir), ["turtle.png"]);
  });

  test("two calls collecting the same job save its picture once", async () => {
    mock.state.jobMs = 600;
    mock.trouble(FULL_PNG, { delayMs: 300, times: 2 });
    const { client, config } = await connect({ waitSeconds: 0.2 });
    const pending = await call(client, "generate_image", { prompt: "popular puffin" });
    const jobId = jobIdOf(pending);
    assert.ok(jobId, textOf(pending));
    const [a, b] = await Promise.all([1, 2].map(() => call(client, "get_image_job", { job_id: jobId, wait_seconds: 10 })));
    assert.equal(textOf(a), textOf(b));
    assert.deepEqual(await files(config.outputDir), ["popular-puffin.png"]);
  });

  test("get_image_job with only a folder names the files from the job's prompt", async () => {
    mock.state.jobMs = 1000;
    const { client: first } = await connect({ waitSeconds: 0.2 });
    const pending = await call(first, "generate_image", { prompt: "Tidy desk with a plant" });
    const jobId = jobIdOf(pending);
    assert.ok(jobId, textOf(pending));
    // A new extension process (Claude Desktop restarted) doesn't know where this job was meant to go.
    const { client: second, config } = await connect();
    const done = await call(second, "get_image_job", { job_id: jobId, folder: "Unit 5", wait_seconds: 10 });
    assert.ok(!done.isError, textOf(done));
    assert.deepEqual(await files(path.join(config.outputDir, "Unit 5")), ["tidy-desk-with-a-plant.png"]);
  });

  test("progress notifications are sent while waiting", async () => {
    mock.state.jobMs = 1500;
    const { client } = await connect();
    const seen = [];
    const result = await call(client, "generate_image", { prompt: "progress please" },
      { onprogress: (p) => seen.push(p.progress) });
    assert.ok(!result.isError, textOf(result));
    assert.ok(seen.length >= 1, "at least one progress update");
    assert.deepEqual(seen, [...seen].sort((a, b) => a - b), "progress only goes up");
  });

  test("a failed job is reported as an error", async () => {
    const { client } = await connect();
    const result = await call(client, "generate_image", { prompt: "please FAIL" });
    assert.equal(result.isError, true);
    assert.match(textOf(result), /couldn't be made: the model rejected the input/);
  });

  test("server validation errors come through", async () => {
    const { client } = await connect();
    const result = await call(client, "generate_image", { prompt: "x", model: "nope" });
    assert.equal(result.isError, true);
    assert.match(textOf(result), /unknown model 'nope'/);
  });

  test("edit_image uploads a local picture and edits it", async () => {
    const { client } = await connect();
    const photo = path.join(tmp, "class photo.png");
    await fs.writeFile(photo, PNG);
    const result = await call(client, "edit_image", { instruction: "turn it into a coloring page", source_path: photo,
      style: "coloring" });
    assert.ok(!result.isError, textOf(result));
    assert.equal(mock.state.uploads.at(-1).filename, "class photo.png");
    const job = mock.state.submitted.at(-1);
    assert.match(job.source_image_id, /^img_/);
    assert.equal(job.width, undefined, "keeps the source's shape unless asked");
    assert.match(textOf(result), /class-photo-turn-it-into-a-coloring-page\.png/);
  });

  test("edit_image refuses files outside allowed folders, non-images and ambiguous sources", async () => {
    const { client } = await connect({ inputRoots: [path.join(tmp, "allowed")] });
    await fs.mkdir(path.join(tmp, "allowed"));
    const outside = path.join(tmp, "outside.png");
    await fs.writeFile(outside, PNG);
    let result = await call(client, "edit_image", { instruction: "x", source_path: outside });
    assert.match(textOf(result), /home folder/);
    const link = path.join(tmp, "allowed", "link.png");
    await fs.symlink(outside, link);
    result = await call(client, "edit_image", { instruction: "x", source_path: link });
    assert.match(textOf(result), /home folder/, "symlinks out of an allowed folder are refused");
    const notes = path.join(tmp, "allowed", "notes.png");
    await fs.writeFile(notes, "%PDF-1.7 definitely not a picture");
    result = await call(client, "edit_image", { instruction: "x", source_path: notes });
    assert.match(textOf(result), /isn't a PNG, JPEG, WebP or GIF/);
    result = await call(client, "edit_image", { instruction: "x", source_path: "relative/photo.png" });
    assert.match(textOf(result), /isn't a full path/);
    result = await call(client, "edit_image", { instruction: "x" });
    assert.match(textOf(result), /either image_id/);
    assert.equal(mock.state.uploads.filter((u) => u.filename !== "class photo.png").length, 0);
  });

  test("references: save from an image and from a file, list by tag, delete", async () => {
    const { client } = await connect();
    const made = await call(client, "generate_image", { prompt: "Benny the bear" });
    const imageId = /img_[0-9a-f]{16}/.exec(textOf(made))[0];
    let result = await call(client, "save_reference", { name: "Benny, front", image_id: imageId, tags: ["bear book"] });
    const refId = /ref_[0-9a-f]{16}/.exec(textOf(result))[0];
    const sheet = path.join(tmp, "sheet.png");
    await fs.writeFile(sheet, PNG);
    result = await call(client, "save_reference", { name: "Style sample", source_path: sheet, tags: ["style", "bear book"] });
    assert.match(textOf(result), /Saved reference "Style sample"/);
    result = await call(client, "list_references", { tag: "style" });
    assert.match(textOf(result), /Style sample/);
    assert.doesNotMatch(textOf(result), /Benny/);
    result = await call(client, "delete_reference", { reference_id: refId });
    assert.match(textOf(result), /Deleted/);
    result = await call(client, "delete_reference", { reference_id: refId });
    assert.equal(result.isError, true);
  });

  test("commas inside a tag become spaces, whichever way the reference is saved", async () => {
    const { client } = await connect();
    const made = await call(client, "generate_image", { prompt: "tagged fox" });
    const imageId = /img_[0-9a-f]{16}/.exec(textOf(made))[0];
    const tags = ["red, blue", "big,small"];
    let result = await call(client, "save_reference", { name: "From an image", image_id: imageId, tags });
    assert.match(textOf(result), /tags: red blue, big small/);
    const sheet = path.join(tmp, "fox.png");
    await fs.writeFile(sheet, PNG);
    result = await call(client, "save_reference", { name: "From a file", source_path: sheet, tags });
    assert.match(textOf(result), /tags: red blue, big small/);
    for (const ref of [...mock.state.references.values()].slice(-2)) assert.deepEqual(ref.tags, ["red blue", "big small"]);
    // Listing by the tag as it was given finds both: the filter is cleaned the same way.
    result = await call(client, "list_references", { tag: "red, blue" });
    assert.match(textOf(result), /From an image/);
    assert.match(textOf(result), /From a file/);
    result = await call(client, "list_references", { tag: "green,  grey" });
    assert.equal(textOf(result), "No references tagged \"green grey\".");
  });

  test("get_image shows a stored picture and can save a copy", async () => {
    const { client, config } = await connect();
    const made = await call(client, "generate_image", { prompt: "red apple", folder: "fruit" });
    const imageId = /img_[0-9a-f]{16}/.exec(textOf(made))[0];
    const result = await call(client, "get_image", { image_id: imageId, save_copy: true });
    assert.match(textOf(result), /Prompt: "red apple"/);
    assert.equal(imagesOf(result).length, 1);
    assert.ok((await fs.readdir(config.outputDir)).includes("red-apple.png"));
    const list = await call(client, "list_recent_images", { limit: 5 });
    assert.match(textOf(list), new RegExp(imageId));
  });

  test("a preview that fails never spoils a finished picture, and is called missing, not left out", async () => {
    mock.trouble(/size=preview/, { reset: true });
    const { client, config } = await connect();
    const made = await call(client, "generate_image", { prompt: "brave mouse" });
    assert.ok(!made.isError, textOf(made));
    assert.equal(imagesOf(made).length, 0);
    assert.match(textOf(made), /The preview couldn't be loaded this time\. The saved PNG is complete\./);
    assert.doesNotMatch(textOf(made), /small and quick|Some previews/);
    assert.deepEqual(await files(config.outputDir), ["brave-mouse.png"]);
    const imageId = /img_[0-9a-f]{16}/.exec(textOf(made))[0];
    mock.trouble(/size=preview/, { status: 500, body: { detail: "preview failed" } });
    const shown = await call(client, "get_image", { image_id: imageId });
    assert.ok(!shown.isError, textOf(shown));
    assert.match(textOf(shown), /Prompt: "brave mouse"/);
    assert.match(textOf(shown), /preview couldn't be loaded/);
    mock.trouble(/size=preview/, { status: 200, body: "<html>Please log in</html>", skip: 1 });
    const pair = await call(client, "generate_image", { prompt: "two mice", n: 2 });
    assert.ok(!pair.isError, textOf(pair));
    assert.equal(imagesOf(pair).length, 1);
    assert.match(textOf(pair), /1 of the 2 previews couldn't be loaded this time\. The saved PNGs are complete\./);
    assert.doesNotMatch(textOf(pair), /small and quick/);
    // Labelled as a picture but not one: never handed to Claude as an image.
    mock.trouble(/size=preview/, { status: 200, body: Buffer.from("not really a webp"), headers: { "content-type": "image/webp" } });
    const labelled = await call(client, "get_image", { image_id: imageId });
    assert.ok(!labelled.isError, textOf(labelled));
    assert.equal(imagesOf(labelled).length, 0);
    assert.match(textOf(labelled), /preview couldn't be loaded/);
  });

  test("edit_image refuses a folder outside the pictures folder before uploading anything", async () => {
    const { client } = await connect();
    const photo = path.join(tmp, "class photo.png");
    await fs.writeFile(photo, PNG);
    const result = await call(client, "edit_image",
      { instruction: "add a hat", source_path: photo, folder: "/Users/teacher/Desktop/Bear book" });
    assert.equal(result.isError, true);
    assert.match(textOf(result), /like "Bear book" \(not a full path\)/);
    assert.deepEqual(mock.state.requests, [], "nothing uploaded, nothing drawn");
  });

  test("get_image names a copy of an uploaded picture after the file, without its extension", async () => {
    const { client, config } = await connect();
    const photo = path.join(tmp, "class photo.png");
    await fs.writeFile(photo, PNG);
    const edited = await call(client, "edit_image", { instruction: "add a hat", source_path: photo });
    assert.ok(!edited.isError, textOf(edited));
    const uploaded = mock.state.submitted.at(-1).source_image_id;
    const copy = await call(client, "get_image", { image_id: uploaded, save_copy: true });
    assert.ok(!copy.isError, textOf(copy));
    assert.match(textOf(copy), /Uploaded picture: class photo\.png/);
    assert.match(textOf(copy), /Saved to: \S+class-photo\.png$/m);
    assert.ok((await files(config.outputDir)).includes("class-photo.png"));
  });

  test("cancel_image_job says what became of the job", async () => {
    mock.state.jobMs = 10_000;
    const { client } = await connect({ waitSeconds: 0.2 });
    // Still waiting its turn: the server drops it at once, and its answer already says cancelled.
    const waiting = jobIdOf(await call(client, "generate_image", { prompt: "never started" }));
    assert.ok(waiting);
    let result = await call(client, "cancel_image_job", { job_id: waiting });
    assert.equal(textOf(result), `Job ${waiting} is cancelled; no picture will be made.`);
    result = await call(client, "cancel_image_job", { job_id: waiting });
    assert.equal(textOf(result), `Job ${waiting} is cancelled; no picture will be made.`);
    // Being drawn: it stops at its next step.
    mock.state.jobMs = 1000;
    const drawing = jobIdOf(await call(client, "generate_image", { prompt: "half drawn" }));
    assert.ok(drawing);
    await delay(400); // the mock draws after a quarter of the job's time
    assert.equal(textOf(await call(client, "cancel_image_job", { job_id: drawing })), `Cancelling job ${drawing}.`);
    mock.state.jobMs = 50;
    const { client: patient } = await connect();
    const made = await call(patient, "generate_image", { prompt: "already made" });
    assert.ok(!made.isError, textOf(made));
    assert.match(textOf(await call(patient, "cancel_image_job", { job_id: lastJob() })),
      new RegExp(`^Job ${lastJob()} had already finished: the picture was made anyway`));
  });

  test("ids are validated before reaching the server", async () => {
    const { client } = await connect();
    const result = await call(client, "get_image", { image_id: "../v1/unload" });
    assert.equal(result.isError, true);
    assert.match(textOf(result), /image_id looks like/);
  });

  test("a full path as the folder counts only inside the pictures folder", async () => {
    const { client, config } = await connect();
    const inside = await call(client, "generate_image",
      { prompt: "bear cub", folder: path.join(config.outputDir, "Bear book") });
    assert.ok(!inside.isError, textOf(inside));
    assert.deepEqual(await files(path.join(config.outputDir, "Bear book")), ["bear-cub.png"]);
    const before = mock.state.submitted.length;
    for (const folder of ["/Users/teacher/Desktop/Bear book", "~/Desktop/Bear book", "C:\\Users\\t\\Desktop\\Bear book"]) {
      const result = await call(client, "generate_image", { prompt: "bear cub", folder });
      assert.equal(result.isError, true, folder);
      assert.match(textOf(result), /saved inside .*like "Bear book" \(not a full path\).*"Save pictures in"/s);
    }
    assert.equal(mock.state.submitted.length, before, "refused before anything was drawn");
    assert.deepEqual(await files(config.outputDir), ["Bear book"], "no mirrored folders like Users/teacher/Desktop");
  });

  test("a pictures folder Claude can't save in is explained, and a retry is suggested only once that is fixed", {
    skip: noFolderPermissions,
  }, async () => {
    const locked = path.join(tmp, "locked");
    await fs.mkdir(locked, { mode: 0o500 });
    try {
      const { client } = await connect({ outputDir: path.join(locked, "out") });
      const downloads = mock.state.fullDownloads;
      const pair = await call(client, "generate_image", { prompt: "nowhere to go", n: 2 });
      assert.ok(!pair.isError, textOf(pair));
      const text = textOf(pair);
      assert.match(text, /1\. not saved on this computer\n/);
      assert.match(text, /Claude isn't allowed to save pictures in \S+locked\S+out\. Choose another folder for "Save pictures in"/);
      assert.match(text, /still on the image server: once that is fixed, call get_image_job with job_id job_\w+ to save them\./);
      assert.match(text, /Previews below are downscaled\.$/m);
      assert.doesNotMatch(text, /EACCES|try saving again|full size|saved PNGs/);
      assert.equal(mock.state.fullDownloads - downloads, 1, "the second picture isn't downloaded for nothing");
      const one = await call(client, "generate_image", { prompt: "still nowhere" });
      assert.match(textOf(one), /Not saved on this computer: Claude isn't allowed to save pictures in/);
      assert.match(textOf(one), /once that is fixed, call get_image_job with job_id job_\w+ to save it\./);
      const imageId = /img_[0-9a-f]{16}/.exec(textOf(one))[0];
      const copy = await call(client, "get_image", { image_id: imageId, save_copy: true });
      assert.equal(copy.isError, true);
      assert.match(textOf(copy), /^Claude isn't allowed to save pictures in/);
    } finally {
      await fs.chmod(locked, 0o700);
    }
  });
});

describe("one time budget per call", () => {
  test("waiting stops before the budget runs out and hands back the job_id", async () => {
    mock.state.jobMs = 10_000;
    const { client } = await connect({ waitSeconds: 40 }, { callBudgetMs: 1500, finishReserveMs: 500 });
    const started = Date.now();
    const result = await call(client, "generate_image", { prompt: "slow owl" });
    const elapsed = Date.now() - started;
    assert.ok(!result.isError, textOf(result));
    assert.ok(elapsed < 1500, `returned after ${elapsed} ms`);
    assert.match(textOf(result), new RegExp(`job_id: ${lastJob()}`));
    assert.match(textOf(result), /get_image_job/);
    assert.ok(!mock.state.cancelled.includes(lastJob()), "the job keeps going");
  });

  test("upload time counts toward the budget", async () => {
    mock.state.jobMs = 10_000;
    mock.trouble("POST /images", { delayMs: 900 });
    const { client } = await connect({ waitSeconds: 40 }, { callBudgetMs: 2000, finishReserveMs: 500 });
    const photo = path.join(tmp, "photo.png");
    await fs.writeFile(photo, PNG);
    const started = Date.now();
    const result = await call(client, "edit_image", { instruction: "add a hat", source_path: photo });
    const elapsed = Date.now() - started;
    assert.ok(!result.isError, textOf(result));
    assert.match(textOf(result), /job_id: job_/);
    // Counted from the upload's end, the wait would have run on to about 2.2 s.
    assert.ok(elapsed >= 900 && elapsed < 2000, `returned after ${elapsed} ms`);
  });

  test("previews are left out when little time is left after saving", async () => {
    mock.trouble(FULL_PNG, { delayMs: 800 });
    const { client, config } = await connect({}, { callBudgetMs: 2500, finishReserveMs: 500, previewMinMs: 1500 });
    const result = await call(client, "generate_image", { prompt: "quick fox" });
    assert.ok(!result.isError, textOf(result));
    assert.equal(imagesOf(result).length, 0);
    assert.equal(mock.state.requests.filter((r) => r.includes("size=preview")).length, 0, "no preview asked for");
    assert.match(textOf(result), /The preview was left out to keep this result quick\. The saved PNG is complete\./);
    assert.deepEqual(await files(config.outputDir), ["quick-fox.png"]);
  });

  test("previews that hang hold back a saved result for a few seconds at most", async () => {
    mock.trouble(/size=preview/, { delayMs: 5000, times: 4 });
    const { client, config } = await connect({}, { previewPhaseMs: 1200 });
    const { result, ms } = await timed(call(client, "generate_image", { prompt: "hurried hens", n: 4 }));
    assert.ok(!result.isError, textOf(result));
    assert.ok(ms < 2500, `returned after ${ms} ms`); // the job takes ~0.5 s, the previews get 1.2 s
    assert.equal(imagesOf(result).length, 0);
    assert.match(textOf(result), /Some previews were left out to keep this result small and quick\. The saved PNGs are complete\./);
    assert.equal((await files(config.outputDir)).length, 4);
    assert.equal(mock.state.requests.filter((r) => r.includes("size=preview")).length, 1, "no more tries once one timed out");
  });

  test("a call waiting for another call's save never runs past its own budget", async () => {
    mock.state.jobMs = 60_000;
    const { client, config } = await connect({ waitSeconds: 0.2 }, { callBudgetMs: 2000, finishReserveMs: 500 });
    const jobId = jobIdOf(await call(client, "generate_image", { prompt: "shared job" }));
    assert.ok(jobId);
    mock.state.jobMs = 0; // done from now on
    // A asks first, but its first look at the job is answered late; B, a moment later, sees it
    // done at once, takes the save and is stuck downloading until B's own deadline.
    mock.trouble(/^GET \/jobs\//, { delayMs: 400 });
    mock.trouble(FULL_PNG, { delayMs: 2500 });
    const a = timed(call(client, "get_image_job", { job_id: jobId }));
    await delay(100);
    const b = timed(call(client, "get_image_job", { job_id: jobId }));
    const [first, second] = await Promise.all([a, b]);
    assert.ok(first.ms < 2000, `A returned after ${first.ms} ms, past its 2000 ms budget`);
    for (const { result } of [first, second]) {
      assert.ok(!result.isError, textOf(result));
      assert.match(textOf(result), /The picture is finished on the image server/);
      assert.match(textOf(result), new RegExp(`job_id: ${jobId}`));
    }
    const later = await call(client, "get_image_job", { job_id: jobId });
    assert.match(textOf(later), /Saved to: .*shared-job\.png/);
    assert.deepEqual(await files(config.outputDir), ["shared-job.png"]);
  });

  test("the budget holds when the system clock is set back during a call", async () => {
    mock.state.jobMs = 10_000;
    const { client } = await connect({ waitSeconds: 40 }, { callBudgetMs: 1500, finishReserveMs: 500 });
    const realNow = Date.now;
    const step = setTimeout(() => { Date.now = () => realNow() - 3_600_000; }, 300);
    try {
      const { result, ms } = await timed(call(client, "generate_image", { prompt: "clock set back" }));
      assert.ok(!result.isError, textOf(result));
      assert.ok(ms < 1500, `returned after ${ms} ms`);
      assert.match(textOf(result), /job_id: job_/);
    } finally {
      clearTimeout(step);
      Date.now = realNow;
    }
  });

  test("pictures not all saved in time: the job_id comes back, and get_image_job saves only the rest", async () => {
    mock.trouble(FULL_PNG, { delayMs: 1200 });
    const { client, config } = await connect({}, { callBudgetMs: 2500, finishReserveMs: 500, previewMinMs: 0 });
    const first = await call(client, "generate_image", { prompt: "twin owls", n: 2 });
    assert.ok(!first.isError, textOf(first));
    assert.equal(jobIdOf(first), lastJob());
    assert.match(textOf(first), /2 pictures are finished on the image server/);
    assert.match(textOf(first), /twin-owls-1\.png/);
    assert.match(textOf(first), /get_image_job with this job_id to save the rest/);
    assert.deepEqual(await files(config.outputDir), ["twin-owls-1.png"]);

    const rest = await call(client, "get_image_job", { job_id: lastJob() });
    assert.ok(!rest.isError, textOf(rest));
    assert.match(textOf(rest), /1\. .*twin-owls-1\.png/);
    assert.match(textOf(rest), /2\. .*twin-owls-2\.png/);
    assert.equal(imagesOf(rest).length, 2);
    assert.deepEqual(await files(config.outputDir), ["twin-owls-1.png", "twin-owls-2.png"]);
    const again = await call(client, "get_image_job", { job_id: lastJob() });
    assert.equal(textOf(again), textOf(rest));
    assert.deepEqual(await files(config.outputDir), ["twin-owls-1.png", "twin-owls-2.png"], "nothing saved twice");
  });

  test("a pictures folder too slow to write to hands back the job_id in time, and nothing is saved twice", async () => {
    const realOpen = fs.open;
    let opens = 0;
    fs.open = async (...args) => {
      if (opens++ === 0) await delay(2500); // the first file waits for a drive to wake up
      return realOpen(...args);
    };
    try {
      const { client, config } = await connect({}, { callBudgetMs: 2500, finishReserveMs: 500 });
      const { result, ms } = await timed(call(client, "generate_image", { prompt: "sleepy drive", n: 2 }));
      assert.ok(!result.isError, textOf(result));
      assert.ok(ms < 2700, `returned after ${ms} ms, past its 2500 ms budget`);
      assert.match(textOf(result), /^The 2 pictures are finished on the image server, but saving in \S+out is slow: if a message asked whether Claude may use that folder, allow it; if the folder is on a network or external drive, check that it is connected\.\n/);
      assert.match(textOf(result), new RegExp(`job_id: ${lastJob()}\nThen call get_image_job with this job_id to save them`));
      // While that write is still going, collecting the job says so at once instead of waiting
      // for it, and starts no other write there.
      const early = await timed(call(client, "get_image_job", { job_id: lastJob() }));
      assert.ok(!early.result.isError, textOf(early.result));
      assert.ok(early.ms < 1000, `returned after ${early.ms} ms`);
      assert.match(textOf(early.result), /but saving in \S+out is slow: if a message asked whether Claude may use that folder/);
      assert.match(textOf(early.result), new RegExp(`job_id: ${lastJob()}\n`));
      assert.equal(opens, 1);
      // The write finishes by itself; collecting the job then saves only the other picture.
      await eventually(async () => (await files(config.outputDir)).includes("sleepy-drive-1.png"));
      const rest = await call(client, "get_image_job", { job_id: lastJob() });
      assert.ok(!rest.isError, textOf(rest));
      assert.match(textOf(rest), /1\. \S+sleepy-drive-1\.png/);
      assert.match(textOf(rest), /2\. \S+sleepy-drive-2\.png/);
      await delay(300); // a second save of picture 1 would have landed by now, as "sleepy-drive-1 (2).png"
      assert.deepEqual(await files(config.outputDir), ["sleepy-drive-1.png", "sleepy-drive-2.png"]);
    } finally {
      fs.open = realOpen;
    }
  });

  test("a folder whose write hangs gets no other write: later pictures and copies for it say so at once", async () => {
    const realOpen = fs.open;
    let opens = 0;
    fs.open = async (...args) => {
      if (opens++ === 0) await delay(3500); // a network drive that went away
      return realOpen(...args);
    };
    try {
      const { client, config } = await connect({}, { callBudgetMs: 2500, finishReserveMs: 500, previewMinMs: 0 });
      const stuck = await call(client, "generate_image", { prompt: "first page" });
      assert.match(textOf(stuck), /but saving in \S+out is slow/);
      // Another picture for that folder is drawn, but neither downloaded nor written beside the stuck write.
      const downloads = mock.state.fullDownloads;
      const next = await timed(call(client, "generate_image", { prompt: "second page" }));
      const second = lastJob();
      assert.ok(!next.result.isError, textOf(next.result));
      assert.ok(next.ms < 1500, `returned after ${next.ms} ms`); // the job is done at the second look, after 0.5 s
      assert.match(textOf(next.result), /^The picture is finished on the image server, but saving in \S+out is slow: if a message asked whether Claude may use that folder, allow it; if the folder is on a network or external drive, check that it is connected\.\n/);
      assert.match(textOf(next.result), new RegExp(`job_id: ${second}\nThen call get_image_job with this job_id to save it and see a preview\\.$`));
      // A copy for that folder: not saved, and said so at once; the rest of get_image still works.
      const imageId = [...mock.state.images.keys()].at(-1);
      const copy = await timed(call(client, "get_image", { image_id: imageId, save_copy: true }));
      assert.ok(!copy.result.isError, textOf(copy.result));
      assert.ok(copy.ms < 1000, `returned after ${copy.ms} ms`);
      assert.match(textOf(copy.result), /Prompt: "second page"/);
      assert.match(textOf(copy.result), /\nThe copy wasn't saved: an earlier picture is still being saved in \S+out, which is slow: if a message asked whether Claude may use that folder, allow it; .* Then ask for the copy again\./);
      assert.equal(imagesOf(copy.result).length, 1, "the preview is still shown");
      assert.equal(mock.state.fullDownloads, downloads, "nothing downloaded for a folder whose write hangs");
      assert.equal(opens, 1, "no second write started there");
      // Another folder isn't held up.
      const elsewhere = await call(client, "generate_image", { prompt: "other folder", folder: "Unit 2" });
      assert.ok(!elsewhere.isError, textOf(elsewhere));
      assert.deepEqual(await files(path.join(config.outputDir, "Unit 2")), ["other-folder.png"]);
      // Once the stuck write has ended, the waiting picture is saved.
      await eventually(async () => (await files(config.outputDir)).includes("first-page.png"), 4000);
      const later = await call(client, "get_image_job", { job_id: second });
      assert.match(textOf(later), /Saved to: \S+second-page\.png/);
    } finally {
      fs.open = realOpen;
    }
  });

  test("a write that starts hanging while another call waits to save there is waited for only briefly", async () => {
    const realOpen = fs.open;
    let opens = 0;
    fs.open = async (...args) => {
      if (opens++ === 0) await delay(3000); // a drive waking up: slow, but it gets there
      return realOpen(...args);
    };
    try {
      const { client, config } = await connect();
      const first = timed(call(client, "generate_image", { prompt: "slow start" }));
      await delay(600); // "slow start" is being written by now
      const next = await timed(call(client, "generate_image", { prompt: "close behind" }));
      // It found that write going on, waited for it until it counted as slow, and started none beside it.
      assert.ok(!next.result.isError, textOf(next.result));
      assert.match(textOf(next.result), /^The picture is finished on the image server, but saving in \S+out is slow/);
      assert.ok(next.ms > 1000 && next.ms < 2500, `returned after ${next.ms} ms`);
      assert.equal(opens, 1);
      // The first call, with its own time left, saw its picture saved.
      const { result } = await first;
      assert.match(textOf(result), /Saved to: \S+slow-start\.png/);
      const later = await call(client, "get_image_job", { job_id: lastJob() });
      assert.match(textOf(later), /Saved to: \S+close-behind\.png/);
      assert.deepEqual(await files(config.outputDir), ["close-behind.png", "slow-start.png"]);
    } finally {
      fs.open = realOpen;
    }
  });

  test("a call waiting for another call's save of the same job notices when that save's write hangs", async () => {
    mock.state.jobMs = 60_000;
    const { client, config } = await connect({ waitSeconds: 0.2 }, { callBudgetMs: 8000, finishReserveMs: 500 });
    const jobId = jobIdOf(await call(client, "generate_image", { prompt: "shared slow drive" }));
    assert.ok(jobId);
    mock.state.jobMs = 0; // done from now on
    const realOpen = fs.open;
    let opens = 0;
    fs.open = async (...args) => {
      if (opens++ === 0) await delay(3000);
      return realOpen(...args);
    };
    try {
      // A takes the save and starts writing; B, a moment later, waits for A's save to end. Once
      // A's write counts as slow, B says so instead of waiting out its own time for it.
      const a = timed(call(client, "get_image_job", { job_id: jobId }));
      await delay(100);
      const b = await timed(call(client, "get_image_job", { job_id: jobId }));
      assert.ok(!b.result.isError, textOf(b.result));
      assert.match(textOf(b.result), /^The picture is finished on the image server, but saving in \S+out is slow/);
      assert.ok(b.ms < 2500, `returned after ${b.ms} ms`);
      const { result } = await a;
      assert.match(textOf(result), /Saved to: \S+shared-slow-drive\.png/);
      assert.equal(opens, 1);
      assert.deepEqual(await files(config.outputDir), ["shared-slow-drive.png"]);
    } finally {
      fs.open = realOpen;
    }
  });

  test("a call whose wait for another call's save runs out during a healthy write doesn't call the folder slow", async () => {
    mock.state.jobMs = 60_000;
    const { client, config } = await connect({ waitSeconds: 0.2 }, { callBudgetMs: 3000, finishReserveMs: 500 });
    const jobId = jobIdOf(await call(client, "generate_image", { prompt: "healthy folder" }));
    assert.ok(jobId);
    mock.state.jobMs = 0; // done from now on
    // A takes the save. Its download is slow (the network), then its write takes 1.2 s, fine for a
    // disk. B, waiting for A's save, runs out of time at about 2.6 s, in the middle of that write.
    mock.trouble(FULL_PNG, { delayMs: 1800 });
    const realOpen = fs.open;
    const write = {};
    fs.open = async (...args) => {
      write.start = performance.now();
      await delay(1200);
      write.end = performance.now();
      return realOpen(...args);
    };
    try {
      const a = call(client, "get_image_job", { job_id: jobId });
      await delay(100);
      const b = await call(client, "get_image_job", { job_id: jobId });
      assert.ok(write.start < performance.now() && write.end === undefined, "B ran out of time during A's write");
      assert.ok(!b.isError, textOf(b));
      assert.match(textOf(b), new RegExp("^The picture is finished on the image server, but this call ran out of " +
        `time before it was saved on this computer\\.\\njob_id: ${jobId}\\nCall get_image_job with this job_id`));
      assert.doesNotMatch(textOf(b), /slow/);
      assert.match(textOf(await a), /Saved to: \S+healthy-folder\.png/);
    } finally {
      fs.open = realOpen;
    }
    assert.deepEqual(await files(config.outputDir), ["healthy-folder.png"]);
  });

  test("a call whose wait for another call's save runs out during a slow write names the folder", async () => {
    mock.state.jobMs = 60_000;
    const { client, config } = await connect({ waitSeconds: 0.2 }, { callBudgetMs: 3000, finishReserveMs: 500 });
    const jobId = jobIdOf(await call(client, "generate_image", { prompt: "tired drive" }));
    assert.ok(jobId);
    mock.state.jobMs = 0; // done from now on
    // A's write starts at about 0.6 s and takes 2.5 s. B, waiting for A's save, finds it young at
    // its look at about 1.6 s and waits on; at its last look, at about 2.6 s, the write is slow.
    mock.trouble(FULL_PNG, { delayMs: 600 });
    const realOpen = fs.open;
    fs.open = async (...args) => {
      await delay(2500);
      return realOpen(...args);
    };
    try {
      const a = call(client, "get_image_job", { job_id: jobId });
      await delay(100);
      const b = await timed(call(client, "get_image_job", { job_id: jobId }));
      assert.ok(!b.result.isError, textOf(b.result));
      assert.ok(b.ms > 2000 && b.ms < 3000, `returned after ${b.ms} ms`);
      assert.match(textOf(b.result), new RegExp("^The picture is finished on the image server, but saving in \\S+out " +
        "is slow: if a message asked whether Claude may use that folder, allow it; [^\\n]*\\n" +
        `job_id: ${jobId}\\nThen call get_image_job with this job_id`));
      assert.match(textOf(await a), /saving in \S+out is slow/);
    } finally {
      fs.open = realOpen;
    }
    // The write ends by itself; collecting the job then reports the picture saved, once.
    await eventually(async () => (await files(config.outputDir)).includes("tired-drive.png"));
    assert.match(textOf(await call(client, "get_image_job", { job_id: jobId })), /Saved to: \S+tired-drive\.png/);
    assert.deepEqual(await files(config.outputDir), ["tired-drive.png"]);
  });

  test("a call that waited for another call's save of the same job reports that save, even while the main folder is slow", async () => {
    mock.state.jobMs = 60_000;
    const { client, config } = await connect({ waitSeconds: 0.2 },
      { callBudgetMs: 8000, finishReserveMs: 500, previewMinMs: 60_000 });
    const jobId = jobIdOf(await call(client, "generate_image", { prompt: "jay", folder: "Unit 3" }));
    assert.ok(jobId);
    mock.state.jobMs = 0; // done from now on
    const realOpen = fs.open;
    let release;
    const stuck = new Promise((resolve) => { release = resolve; });
    fs.open = async (file, ...rest) => {
      if (path.dirname(file) === config.outputDir) await stuck; // only the main pictures folder is stuck
      return realOpen(file, ...rest);
    };
    try {
      const other = call(client, "generate_image", { prompt: "kay" }); // its write into the main folder hangs
      await delay(1700); // and counts as slow by now
      // A takes the save of "jay" into Unit 3 and B waits for it. With no previews to load, A
      // forgets the job's folder as soon as the picture is saved, before B looks again.
      mock.trouble(FULL_PNG, { delayMs: 800 });
      const a = call(client, "get_image_job", { job_id: jobId });
      await delay(100);
      const b = call(client, "get_image_job", { job_id: jobId });
      for (const result of await Promise.all([a, b])) {
        assert.ok(!result.isError, textOf(result));
        assert.match(textOf(result), /\nSaved to: .+Unit 3.jay\.png\n/);
        assert.doesNotMatch(textOf(result), /slow/);
      }
      assert.deepEqual(await files(path.join(config.outputDir, "Unit 3")), ["jay.png"]);
      release();
      assert.match(textOf(await other), /Saved to: \S+kay\.png/);
    } finally {
      release();
      fs.open = realOpen;
    }
  });

  test("collecting a job into another folder while the job's earlier save hangs says so at once", async () => {
    mock.state.jobMs = 60_000;
    const { client, config } = await connect({ waitSeconds: 0.2 }, { callBudgetMs: 2500, finishReserveMs: 500 });
    const jobId = jobIdOf(await call(client, "generate_image", { prompt: "moved" }));
    assert.ok(jobId);
    mock.state.jobMs = 0; // done from now on
    const realOpen = fs.open;
    let release;
    const stuck = new Promise((resolve) => { release = resolve; });
    fs.open = async (file, ...rest) => {
      if (path.dirname(file) === config.outputDir) await stuck; // the main pictures folder's drive went away
      return realOpen(file, ...rest);
    };
    try {
      assert.match(textOf(await call(client, "get_image_job", { job_id: jobId })), /but saving in \S+out is slow/);
      // That save keeps the job until its write ends, so the job can't be saved in another folder
      // meanwhile: said at once, naming the stuck folder, instead of waiting out the call's time.
      const { result, ms } = await timed(call(client, "get_image_job", { job_id: jobId, folder: "Elsewhere" }));
      assert.ok(!result.isError, textOf(result));
      assert.ok(ms < 1000, `returned after ${ms} ms`);
      assert.match(textOf(result), new RegExp("^The picture is finished on the image server, but saving in \\S+out " +
        `is slow: [^\\n]*\\njob_id: ${jobId}\\nThen call get_image_job with this job_id`));
      assert.deepEqual(await files(path.join(config.outputDir, "Elsewhere")), []);
    } finally {
      release();
      fs.open = realOpen;
    }
    // Once the write ends, the picture is saved where that save put it, and only there.
    await eventually(async () => (await files(config.outputDir)).includes("moved.png"));
    assert.match(textOf(await call(client, "get_image_job", { job_id: jobId })), /Saved to: \S+out.moved\.png/);
    assert.deepEqual(await files(path.join(config.outputDir, "Elsewhere")), []);
  });

  test("a picture whose write ended after its call is reported saved, even while another write there hangs", async () => {
    const realOpen = fs.open;
    let opens = 0;
    fs.open = async (...args) => {
      const delays = [2600, 4000]; // "alpha" lands just after its call ended; "beta" hangs for a while
      await delay(delays[opens++] ?? 0);
      return realOpen(...args);
    };
    try {
      const { client, config } = await connect({}, { callBudgetMs: 2500, finishReserveMs: 500 });
      const alpha = await call(client, "generate_image", { prompt: "alpha" });
      const alphaJob = jobIdOf(alpha);
      assert.match(textOf(alpha), /saving in \S+out is slow/);
      await eventually(async () => (await files(config.outputDir)).includes("alpha.png"));
      assert.match(textOf(await call(client, "generate_image", { prompt: "beta" })), /saving in \S+out is slow/);
      const done = await call(client, "get_image_job", { job_id: alphaJob });
      assert.ok(!done.isError, textOf(done));
      assert.match(textOf(done), /^Made 1 picture.*\nSaved to: \S+alpha\.png\n/);
      assert.doesNotMatch(textOf(done), /slow/);
    } finally {
      fs.open = realOpen;
    }
  });

  test("pictures saved at the same moment into one folder take turns, and both are saved", async () => {
    mock.state.jobMs = 300;
    const { client, config } = await connect({ waitSeconds: 0.2 });
    const jobs = [];
    for (const prompt of ["left page", "right page"]) jobs.push(jobIdOf(await call(client, "generate_image", { prompt })));
    assert.ok(jobs.every(Boolean));
    await delay(400); // both are done now
    const realOpen = fs.open;
    fs.open = async (...args) => {
      await delay(200); // each write takes a moment, so the two overlap
      return realOpen(...args);
    };
    try {
      const results = await Promise.all(jobs.map((job_id) => call(client, "get_image_job", { job_id })));
      for (const result of results) {
        assert.ok(!result.isError, textOf(result));
        assert.match(textOf(result), /Saved to: /);
        assert.doesNotMatch(textOf(result), /slow/);
      }
      assert.deepEqual(await files(config.outputDir), ["left-page.png", "right-page.png"]);
    } finally {
      fs.open = realOpen;
    }
  });

  test("a picture downloaded just before the deadline still gets time to be written", async () => {
    mock.trouble(FULL_PNG, { delayMs: 1200 }); // done about 0.3 s before the deadline
    const realOpen = fs.open;
    fs.open = async (...args) => {
      await delay(600); // a healthy disk that takes a moment
      return realOpen(...args);
    };
    try {
      const { client, config } = await connect({}, { callBudgetMs: 2000, finishReserveMs: 500 });
      const { result, ms } = await timed(call(client, "generate_image", { prompt: "just in time" }));
      assert.ok(!result.isError, textOf(result));
      assert.match(textOf(result), /Saved to: \S+just-in-time\.png/);
      assert.doesNotMatch(textOf(result), /slow|job_id/);
      assert.ok(ms < 2000 + 1500, `returned after ${ms} ms`);
      assert.deepEqual(await files(config.outputDir), ["just-in-time.png"]);
    } finally {
      fs.open = realOpen;
    }
  });

  test("a copy downloaded just before the deadline still gets time to be written", async () => {
    const { client, config } = await connect({}, { callBudgetMs: 2000, finishReserveMs: 500 });
    const made = await call(client, "generate_image", { prompt: "copy me" });
    const imageId = /img_[0-9a-f]{16}/.exec(textOf(made))[0];
    mock.trouble(FULL_PNG, { delayMs: 1700 }); // done about 0.3 s before the deadline
    const realOpen = fs.open;
    fs.open = async (...args) => {
      await delay(600); // a healthy disk that takes a moment
      return realOpen(...args);
    };
    try {
      const { result, ms } = await timed(call(client, "get_image", { image_id: imageId, save_copy: true }));
      assert.ok(!result.isError, textOf(result));
      assert.match(textOf(result), /\nSaved to: \S+copy-me \(2\)\.png\n/);
      assert.doesNotMatch(textOf(result), /slow/);
      assert.ok(ms < 2000 + 1500, `returned after ${ms} ms`);
    } finally {
      fs.open = realOpen;
    }
    assert.deepEqual(await files(config.outputDir), ["copy-me (2).png", "copy-me.png"]);
  });

  test("a picture that finishes just as the wait ends is collected in the same call", async () => {
    mock.state.jobMs = 1600; // looks at 0, 0.5 and 1.25 s find it unfinished; the wait ends at 2 s
    const { client, config } = await connect({ waitSeconds: 2 });
    const result = await call(client, "generate_image", { prompt: "photo finish" });
    assert.ok(!result.isError, textOf(result));
    assert.match(textOf(result), /^Made 1 picture/);
    assert.deepEqual(await files(config.outputDir), ["photo-finish.png"]);
  });

  test("a copy that can't be saved in time doesn't hold up get_image", async () => {
    const { client, config } = await connect({}, { callBudgetMs: 2500, finishReserveMs: 500 });
    const made = await call(client, "generate_image", { prompt: "slow copy" });
    const imageId = /img_[0-9a-f]{16}/.exec(textOf(made))[0];
    const realOpen = fs.open;
    fs.open = async (...args) => {
      await delay(3000);
      return realOpen(...args);
    };
    try {
      const { result, ms } = await timed(call(client, "get_image", { image_id: imageId, save_copy: true }));
      assert.ok(!result.isError, textOf(result));
      assert.ok(ms < 2700, `returned after ${ms} ms, past its 2500 ms budget`);
      assert.match(textOf(result), /Prompt: "slow copy"/);
      assert.match(textOf(result), /The copy is still being saved in \S+out, which is slow: if a message asked whether Claude may use that folder, allow it; if the folder is on a network or external drive, check that it is connected\. It may still appear there/);
    } finally {
      fs.open = realOpen;
    }
    await delay(1500); // the write carries on and lands under a free name
    assert.deepEqual(await files(config.outputDir), ["slow-copy (2).png", "slow-copy.png"]);
  });

  test("a picture that takes too long to read is reported as that, within the budget", async () => {
    const photo = path.join(tmp, "cloud photo.png");
    await fs.writeFile(photo, PNG);
    const { client } = await connect({}, { callBudgetMs: 2000, finishReserveMs: 500 });
    const realReadFile = fs.readFile;
    let reads = 0;
    fs.readFile = async (...args) => {
      reads++;
      await delay(3000); // a file that has to download first
      return realReadFile(...args);
    };
    try {
      // The second call finds the first one's read still going: it says so at once, and doesn't
      // start another (each one stuck would hold one of Node's few file threads).
      for (const [tool, args, limit] of [["edit_image", { instruction: "add a hat", source_path: photo }, 1000],
        ["save_reference", { name: "Hat", source_path: `"${photo}"` }, 300]]) {
        const { result, ms } = await timed(call(client, tool, args));
        assert.equal(result.isError, true, tool);
        assert.ok(ms < limit, `${tool} returned after ${ms} ms`);
        assert.match(textOf(result), /^Reading \S+cloud photo\.png"? on this computer took too long, so it wasn't sent\. If a message asked whether Claude may open that folder, allow it/);
        assert.doesNotMatch(textOf(result), /image server/);
      }
      assert.equal(reads, 1);
    } finally {
      fs.readFile = realReadFile;
    }
    assert.deepEqual(mock.state.requests, [], "nothing was sent");
  });

  test("two calls sending the same picture at once both send it", async () => {
    const photo = path.join(tmp, "shared photo.png");
    await fs.writeFile(photo, PNG);
    const { client } = await connect();
    const realReadFile = fs.readFile;
    fs.readFile = async (...args) => {
      await delay(300); // a large photo: the two reads overlap
      return realReadFile(...args);
    };
    try {
      const [edit, reference] = await Promise.all([
        call(client, "edit_image", { instruction: "add a hat", source_path: photo }),
        call(client, "save_reference", { name: "Hat", source_path: photo }),
      ]);
      assert.ok(!edit.isError, textOf(edit));
      assert.ok(!reference.isError, textOf(reference));
    } finally {
      fs.readFile = realReadFile;
    }
    assert.equal(mock.state.requests.filter((r) => /^POST \/(images|references)/.test(r)).length, 2);
  });

  test("a download cut off by the deadline is handed back, then saved once", async () => {
    mock.trouble(FULL_PNG, { delayMs: 3000 });
    const { client, config } = await connect({}, { callBudgetMs: 2000, finishReserveMs: 500 });
    const started = Date.now();
    const first = await call(client, "generate_image", { prompt: "late lark" });
    const elapsed = Date.now() - started;
    assert.ok(!first.isError, textOf(first));
    assert.ok(elapsed < 2300, `returned after ${elapsed} ms`);
    assert.match(textOf(first), /The picture is finished on the image server/);
    assert.match(textOf(first), new RegExp(`job_id: ${lastJob()}`));
    assert.deepEqual(await files(config.outputDir), []);
    const later = await call(client, "get_image_job", { job_id: lastJob() });
    assert.match(textOf(later), /Saved to: .*late-lark\.png/);
    assert.deepEqual(await files(config.outputDir), ["late-lark.png"]);
  });
});

describe("a job_id is never lost", () => {
  test("brief trouble while waiting (dropped connection, restarting, 500) is ridden out", async () => {
    mock.state.jobMs = 300;
    mock.trouble(/^GET \/jobs\//, { reset: true });
    mock.trouble(/^GET \/jobs\//, { status: 503, body: { detail: RESTARTING }, headers: { "retry-after": "5" } });
    mock.trouble(/^GET \/jobs\//, { status: 500, body: { detail: "internal hiccup" } });
    const { client, config } = await connect();
    const result = await call(client, "generate_image", { prompt: "steady heron" });
    assert.ok(!result.isError, textOf(result));
    assert.deepEqual(await files(config.outputDir), ["steady-heron.png"]);
    assert.equal(mock.state.requests.filter((r) => r.startsWith("GET /jobs/")).length, 4);
  });

  test("trouble that outlasts the wait still hands back the job_id", async () => {
    mock.state.jobMs = 10_000;
    mock.trouble(/^GET \/jobs\//, { status: 503, body: { detail: RESTARTING }, times: 100 });
    const { client } = await connect({ waitSeconds: 1 });
    const result = await call(client, "generate_image", { prompt: "patient crane" });
    assert.ok(!result.isError, textOf(result));
    assert.match(textOf(result), /The image server is restarting/);
    assert.match(textOf(result), new RegExp(`job_id: ${lastJob()}`));
    assert.match(textOf(result), /get_image_job/);
  });

  test("a job the server doesn't know points to list_recent_images, not to a restart", async () => {
    mock.trouble(/^GET \/jobs\//, { status: 404, body: { detail: "no such job" } });
    const { client } = await connect();
    const result = await call(client, "generate_image", { prompt: "vanished vole" });
    assert.equal(result.isError, true);
    assert.match(textOf(result), new RegExp(`The image server has no job ${lastJob()}\\. Check the job_id`));
    assert.match(textOf(result), /list_recent_images/);
    assert.doesNotMatch(textOf(result), /restart/, "a job id survives restarts (docs/API.md)");
    const unknown = await call(client, "get_image_job", { job_id: "job_0123456789abcdef" });
    assert.match(textOf(unknown), /has no job job_0123456789abcdef/);
  });

  test("a picture request whose answer is lost warns before asking again", async () => {
    mock.trouble("POST /jobs", { delayMs: 1500 });
    const { client } = await connect({}, { callBudgetMs: 1200, finishReserveMs: 300 });
    const before = mock.state.jobs.size;
    const late = await call(client, "generate_image", { prompt: "lost in the post" });
    assert.equal(late.isError, true);
    // Only finished pictures are listed, so checking at once would find nothing and make it twice.
    assert.match(textOf(late), /\nThe image server may have started the picture anyway: it may still be waiting its turn or being drawn there\. Wait a minute, then check list_recent_images \(finished pictures appear there\) before asking for it again, so it isn't made twice\.$/);
    await waitFor(() => mock.state.jobs.size === before + 1); // it did
    mock.trouble("POST /jobs", { reset: true });
    const dropped = await call(client, "generate_image", { prompt: "dropped in the post" });
    assert.equal(dropped.isError, true);
    assert.match(textOf(dropped), /closed the connection/);
    assert.match(textOf(dropped), /check list_recent_images/);
    // A refused connection never reached the server: nothing to warn about.
    const { client: closed } = await connect({ serverUrl: `http://127.0.0.1:${await closedPort()}` },
      { refusedRetryMs: 50 });
    const refused = await call(closed, "generate_image", { prompt: "never sent" });
    assert.equal(refused.isError, true);
    assert.doesNotMatch(textOf(refused), /may have started/);
  });

  test("a dropped download is blamed on the dropped connection, not on a stopped server", async () => {
    mock.trouble(FULL_PNG, { reset: true });
    const { client } = await connect();
    const result = await call(client, "generate_image", { prompt: "dropped download" });
    assert.ok(!result.isError, textOf(result));
    assert.match(textOf(result), /Not saved on this computer: The image server at \S+ closed the connection \(\w+\); it may be restarting/);
    assert.match(textOf(result), /call get_image_job with job_id job_\w+ to try saving again/);
    assert.doesNotMatch(textOf(result), /Is it running on this Mac|full size/);
  });

  test("a restart while waiting isn't blamed on a stopped server or the address", async () => {
    mock.state.jobMs = 5000;
    const { client } = await connect({ waitSeconds: 1 });
    let back;
    const timer = setTimeout(() => { back = mock.restart(1500); }, 200);
    try {
      const result = await call(client, "generate_image", { prompt: "restart midway" });
      assert.ok(!result.isError, textOf(result));
      assert.match(textOf(result), /stopped accepting connections \(ECONNREFUSED\); it may be restarting/);
      assert.doesNotMatch(textOf(result), /install_image_server|enable_lan|Local Network/);
      assert.match(textOf(result), new RegExp(`job_id: ${lastJob()}`));
    } finally {
      clearTimeout(timer);
      await back;
    }
  });

  test("other errors after the job exists still name the job_id", async () => {
    mock.trouble(/^GET \/jobs\//, { status: 401, body: { detail: "missing or invalid token" } });
    const { client } = await connect();
    const result = await call(client, "generate_image", { prompt: "locked out lynx" });
    assert.equal(result.isError, true);
    assert.match(textOf(result), /didn't accept this computer's token/);
    assert.match(textOf(result), new RegExp(`job_id ${lastJob()}: call get_image_job`));
  });

  test("a full-size download that isn't a PNG is never saved as one", async () => {
    // e.g. a school network's sign-in page answering in the server's place
    mock.trouble(FULL_PNG, { status: 200, body: "<html>Please sign in to the school network</html>" });
    const { client, config } = await connect();
    const first = await call(client, "generate_image", { prompt: "portal parrot" });
    assert.ok(!first.isError, textOf(first));
    assert.match(textOf(first), /Not saved on this computer: What came back for this picture isn't a PNG picture/);
    assert.match(textOf(first), new RegExp(`call get_image_job with job_id ${lastJob()} to try saving again`));
    assert.doesNotMatch(textOf(first), /full size/);
    assert.deepEqual(await files(config.outputDir), []);
    const imageId = /img_[0-9a-f]{16}/.exec(textOf(first))[0];
    mock.trouble(FULL_PNG, { status: 200, body: "<html>Please sign in to the school network</html>" });
    const copy = await call(client, "get_image", { image_id: imageId, save_copy: true });
    assert.equal(copy.isError, true);
    assert.match(textOf(copy), /^What came back for this picture isn't a PNG picture/);
    assert.deepEqual(await files(config.outputDir), []);
    const later = await call(client, "get_image_job", { job_id: lastJob() });
    assert.match(textOf(later), /Saved to: \S+portal-parrot\.png/);
    assert.deepEqual(await fs.readFile(path.join(config.outputDir, "portal-parrot.png")), PNG);
  });

  test("a failed download names the job_id, and get_image_job saves the picture later", async () => {
    mock.trouble(FULL_PNG, { status: 500, body: { detail: "disk hiccup" } });
    const { client, config } = await connect();
    const first = await call(client, "generate_image", { prompt: "shy otter" });
    assert.ok(!first.isError, textOf(first));
    assert.match(textOf(first), /Not saved on this computer: Image server error 500: disk hiccup/);
    assert.match(textOf(first), new RegExp(`get_image_job with job_id ${lastJob()}`));
    assert.deepEqual(await files(config.outputDir), []);
    const later = await call(client, "get_image_job", { job_id: lastJob() });
    assert.match(textOf(later), /Saved to: .*shy-otter\.png/);
    assert.deepEqual(await files(config.outputDir), ["shy-otter.png"]);
  });
});

describe("stopping a call", () => {
  test("stopping a call cancels the job on the server", async () => {
    mock.state.jobMs = 5000;
    const { client } = await connect();
    const controller = new AbortController();
    const pending = call(client, "generate_image", { prompt: "never mind" }, { signal: controller.signal });
    await delay(400);
    controller.abort();
    await assert.rejects(pending);
    await waitFor(() => mock.state.cancelled.includes(lastJob()));
  });

  test("a call stopped while the job is being submitted cancels the job it made", async () => {
    mock.state.jobMs = 5000;
    mock.trouble("POST /jobs", { delayMs: 500 });
    const { client } = await connect();
    const before = mock.state.jobs.size;
    const controller = new AbortController();
    const pending = call(client, "generate_image", { prompt: "changed my mind" }, { signal: controller.signal });
    await delay(150);
    controller.abort();
    await assert.rejects(pending);
    await waitFor(() => mock.state.jobs.size === before + 1 && mock.state.cancelled.includes(lastJob()));
  });

  test("a call that timed out leaves its job running, to be collected later", async () => {
    mock.state.jobMs = 1200;
    const { client } = await connect();
    await assert.rejects(call(client, "generate_image", { prompt: "slow snail" }, { timeout: 400 }), /timed out/i);
    const jobId = lastJob();
    await delay(300);
    assert.ok(!mock.state.cancelled.includes(jobId), "not cancelled");
    const later = await call(client, "get_image_job", { job_id: jobId, wait_seconds: 10 });
    assert.match(textOf(later), /slow-snail\.png/);
  });

  test("stopping get_image_job cancels the picture too; a timeout doesn't", async () => {
    mock.state.jobMs = 5000;
    const { client } = await connect({ waitSeconds: 0.2 });
    const stopped = jobIdOf(await call(client, "generate_image", { prompt: "collect, then stop" }));
    const controller = new AbortController();
    const collecting = call(client, "get_image_job", { job_id: stopped, wait_seconds: 10 }, { signal: controller.signal });
    await delay(300);
    controller.abort();
    await assert.rejects(collecting);
    await waitFor(() => mock.state.cancelled.includes(stopped));
    const slow = jobIdOf(await call(client, "generate_image", { prompt: "collect, then time out" }));
    await assert.rejects(call(client, "get_image_job", { job_id: slow, wait_seconds: 10 }, { timeout: 300 }), /timed out/i);
    await delay(200);
    assert.ok(!mock.state.cancelled.includes(slow), "left running, to be collected later");
  });

  test("stopping a call while it waits to try a restarting server again sends nothing", async () => {
    const back = mock.restart(300);
    const { client } = await connect({}, { refusedRetryMs: 600 });
    const controller = new AbortController();
    const pending = call(client, "generate_image", { prompt: "stopped before sending" }, { signal: controller.signal });
    await delay(100);
    controller.abort();
    await assert.rejects(pending);
    await back;
    await delay(700); // past the moment the second try would have gone out
    assert.equal(mock.state.submitted.filter((job) => job.prompt === "stopped before sending").length, 0);
  });

  test("an upload for a call already stopped sends nothing, and the extension keeps running", async () => {
    const api = createClient({ serverUrl: mock.url, problem: null, token: TOKEN });
    const form = new FormData();
    form.append("file", new Blob([PNG], { type: "image/png" }), "photo.png");
    const stopped = AbortSignal.abort(new Error("stopped by the person"));
    // Node 24's fetch ended the whole process here a moment later (FormData + a stopped signal).
    await assert.rejects(api.uploadImage(form, { signal: stopped, call: { deadline: now() + 10_000, signal: stopped } }),
      /stopped by the person/);
    await delay(200);
    assert.deepEqual(mock.state.requests, []);
  });

  test("stopping edit_image or save_reference while the picture is read leaves other calls running", async () => {
    mock.state.jobMs = 600;
    const { client } = await connect();
    const photo = path.join(tmp, "big photo.png");
    await fs.writeFile(photo, PNG);
    const realReadFile = fs.readFile;
    fs.readFile = async (...args) => {
      await delay(400); // a large photo on a slow disk
      return realReadFile(...args);
    };
    try {
      const other = call(client, "generate_image", { prompt: "carry on" });
      for (const [tool, args] of [["edit_image", { instruction: "add a hat", source_path: photo }],
        ["save_reference", { name: "Hat", source_path: photo }]]) {
        const controller = new AbortController();
        const stopped = call(client, tool, args, { signal: controller.signal });
        await delay(100);
        controller.abort();
        await assert.rejects(stopped);
      }
      await delay(500); // past the end of both reads
      const done = await other;
      assert.ok(!done.isError, textOf(done));
      assert.match(textOf(done), /Made 1 picture/);
    } finally {
      fs.readFile = realReadFile;
    }
    assert.equal(mock.state.requests.filter((r) => /^POST \/(images|references)/.test(r)).length, 0, "nothing uploaded");
  });

  test("only a reason that says timed out counts as a timeout", () => {
    assert.equal(looksLikeTimeout("McpError: MCP error -32001: Request timed out"), true);
    assert.equal(looksLikeTimeout(new Error("Request timed out")), true);
    assert.equal(looksLikeTimeout(new DOMException("The operation was aborted due to timeout", "TimeoutError")), true);
    assert.equal(looksLikeTimeout("request time out"), true);
    assert.equal(looksLikeTimeout("AbortError: This operation was aborted"), false);
    assert.equal(looksLikeTimeout(undefined), false);
  });
});

test("cancelling a job the server doesn't know says so plainly", async () => {
  const { client } = await connect();
  const result = await call(client, "cancel_image_job", { job_id: "job_0000000000000000" });
  assert.equal(result.isError, true);
  assert.match(textOf(result), /has no job job_0000000000000000, so there is nothing to cancel/);
});

test("times name their zone, since Claude may be on another clock", async () => {
  const { client } = await connect();
  await call(client, "generate_image", { prompt: "a clock" });
  const listed = textOf(await call(client, "list_recent_images", { limit: 1 }));
  assert.match(listed, /\d{4}-\d{2}-\d{2} \d{2}:\d{2} (?:[A-Z]{2,5}|GMT[+-]\d{1,2}(?::\d{2})?|UTC)\b/);
});

describe("connection problems", () => {
  test("a wrong token is explained", async () => {
    const { client } = await connect({ token: "wrong" });
    const result = await call(client, "generate_image", { prompt: "x" });
    assert.equal(result.isError, true);
    assert.match(textOf(result), /didn't accept this computer's token/);
    const status = await call(client, "image_server_status");
    assert.match(textOf(status), /Reachable: yes/);
    assert.match(textOf(status), /Token: The image server didn't accept/);
  });

  test("a missing token is explained before any request", async () => {
    const { client } = await connect({ token: "" });
    const result = await call(client, "list_references");
    assert.match(textOf(result), /No API token is set/);
  });

  test("an unreachable server mentions the Local Network permission", async () => {
    const { client } = await connect({ serverUrl: "http://image-server.invalid:7865" });
    const result = await call(client, "list_references");
    assert.equal(result.isError, true);
    assert.match(textOf(result), /Can't reach the image server at http:\/\/image-server\.invalid:7865/);
    assert.match(textOf(result), /Privacy & Security > Local Network/);
  });

  test("a refused loopback connection is tried once more, then suggests starting the server", async () => {
    const serverUrl = `http://127.0.0.1:${await closedPort()}`;
    let attempts = 0;
    const counting = (...args) => {
      attempts++;
      return fetch(...args);
    };
    const { client } = await connect({ serverUrl }, { fetchImpl: counting, refusedRetryMs: 100 });
    const result = await call(client, "list_references");
    const here = process.platform === "darwin" ? "this Mac" : "this computer";
    assert.match(textOf(result), new RegExp("isn't accepting connections \\(ECONNREFUSED\\)\\. It may be restarting .* or " +
      `not running: start it on ${here} with scripts/install_image_server\\.sh\\. If the image server runs on another ` +
      "computer, put that Mac's address in the extension settings \\(scripts/enable_lan\\.sh on it prints it\\)\\.$"));
    assert.equal(attempts, 2);
    attempts = 0;
    const { client: hurried } = await connect({ serverUrl }, { fetchImpl: counting, refusedRetryMs: 100, callBudgetMs: 1050 });
    await call(hurried, "list_references");
    assert.equal(attempts, 1, "no time left to wait and try again");
  });

  test("a server that is restarting gets the first request again", async () => {
    const back = mock.restart(150);
    const { client, config } = await connect({}, { refusedRetryMs: 400 });
    const result = await call(client, "generate_image", { prompt: "welcome back" });
    await back;
    assert.ok(!result.isError, textOf(result));
    assert.equal(mock.state.submitted.filter((job) => job.prompt === "welcome back").length, 1);
    assert.deepEqual(await files(config.outputDir), ["welcome-back.png"]);
  });

  test("a timeout reads as one, never as error 23, and points at the firewall rather than a busy server", async () => {
    const config = { serverUrl: "http://image-server.invalid:7865", problem: null, token: TOKEN };
    const timeout = new DOMException("The operation was aborted due to timeout", "TimeoutError");
    for (const err of [timeout, new TypeError("fetch failed", { cause: timeout })]) {
      const api = createClient(config, { fetchImpl: async () => { throw err; } });
      await assert.rejects(api.references(), (e) => {
        assert.match(e.message, /didn't answer within 15 s\. If this keeps happening, the server Mac's firewall may be holding connections from this computer: run scripts\/check_lan\.sh on that Mac\./);
        assert.doesNotMatch(e.message, /restarting|busy/);
        assert.equal(e.transient, true);
        return true;
      });
    }
    const local = createClient({ ...config, serverUrl: "http://127.0.0.1:7865" }, { fetchImpl: async () => { throw timeout; } });
    const here = process.platform === "darwin" ? "this Mac" : "this computer";
    await assert.rejects(local.references(), new RegExp(`didn't answer within 15 s\\. If this keeps happening, run ` +
      `scripts/doctor\\.sh on ${here}\\. If the image server runs on another computer, put that Mac's address`));
    const odd = Object.assign(new Error("socket hang up"), { code: 23 });
    const api = createClient(config, { fetchImpl: async () => { throw odd; } });
    await assert.rejects(api.references(), /Can't reach the image server at \S+ \(socket hang up\)/);
  });

  test("refused, dropped and unreachable connections each get advice that fits", async () => {
    const config = { serverUrl: "http://teachers-mac.local:7865", problem: null, token: TOKEN };
    const failing = (code) => createClient(config, { fetchImpl: async () => { throw networkError(code); } });
    const messageFor = (code) => failing(code).references().then(() => assert.fail("should fail"), (err) => err.message);
    const refused = await messageFor("ECONNREFUSED");
    assert.match(refused, /isn't accepting connections \(ECONNREFUSED\)\. It may be restarting .*LAN access may be off.*scripts\/doctor\.sh.*scripts\/enable_lan\.sh/);
    assert.doesNotMatch(refused, /awake|Local Network/, "something at the address answered");
    for (const code of ["ECONNRESET", "UND_ERR_SOCKET", "EPIPE"]) {
      const dropped = await messageFor(code);
      assert.match(dropped, new RegExp(`closed the connection \\(${code}\\); it may be restarting`));
      assert.doesNotMatch(dropped, /awake|Local Network/);
    }
    for (const code of ["EHOSTUNREACH", "ENOTFOUND", "UND_ERR_CONNECT_TIMEOUT"]) {
      const away = await messageFor(code);
      assert.match(away, /awake and on the same network/);
      assert.match(away, /Privacy & Security > Local Network > Claude/);
    }
    // Once the server answered in this call, the address and the permission are fine.
    let calls = 0;
    const api = createClient(config, {
      fetchImpl: async () => {
        if (calls++ === 0) return new Response(JSON.stringify({ references: [] }), { status: 200 });
        throw networkError("EHOSTUNREACH");
      },
    });
    const sameCall = { deadline: now() + 10_000 };
    await api.references(undefined, { call: sameCall });
    await assert.rejects(api.references(undefined, { call: sameCall }), (err) => {
      assert.match(err.message, /Lost the connection to the image server at \S+ \(EHOSTUNREACH\)\. Check that the Mac running it is still awake/);
      assert.doesNotMatch(err.message, /Local Network|server address/);
      return true;
    });
  });

  test("a loopback address points at the server on this computer, or on another one, and says Mac only on a Mac", async () => {
    const local = { serverUrl: "http://127.0.0.1:7865", problem: null, token: TOKEN };
    const messageFor = (err) => createClient(local, { fetchImpl: async () => { throw err; } }).references()
      .then(() => assert.fail("should fail"), (e) => e.message);
    const elsewhere = " If the image server runs on another computer, put that Mac's address in the extension " +
      "settings (scripts/enable_lan.sh on it prints it).";
    const timeout = new DOMException("The operation was aborted due to timeout", "TimeoutError");
    const realPlatform = Object.getOwnPropertyDescriptor(process, "platform");
    try {
      for (const [platform, here] of [["darwin", "this Mac"], ["win32", "this computer"], ["linux", "this computer"]]) {
        Object.defineProperty(process, "platform", { ...realPlatform, value: platform });
        const refused = await messageFor(networkError("ECONNREFUSED"));
        assert.ok(refused.endsWith(`start it on ${here} with scripts/install_image_server.sh.${elsewhere}`), refused);
        const away = await messageFor(networkError("EHOSTUNREACH"));
        assert.ok(away.endsWith(`Is it running on ${here}? Start it with scripts/install_image_server.sh.${elsewhere}`), away);
        const silent = await messageFor(timeout);
        assert.ok(silent.endsWith(`run scripts/doctor.sh on ${here}.${elsewhere}`), silent);
        if (platform !== "darwin") for (const message of [refused, away, silent]) assert.doesNotMatch(message, /this Mac/);
      }
    } finally {
      Object.defineProperty(process, "platform", realPlatform);
    }
    // An address on the network needs none of that.
    const lan = createClient({ ...local, serverUrl: "http://teachers-mac.local:7865" },
      { fetchImpl: async () => { throw networkError("ECONNREFUSED"); } });
    await assert.rejects(lan.references(), (err) => !err.message.includes("another computer"));
  });

  test("the status check waits longer than a connection attempt, so a Mac that is asleep reads as unreachable", async () => {
    const seen = [];
    const real = AbortSignal.timeout;
    AbortSignal.timeout = (ms) => {
      seen.push(ms);
      return real.call(AbortSignal, ms);
    };
    try {
      await createClient({ serverUrl: mock.url, problem: null, token: TOKEN }).health();
    } finally {
      AbortSignal.timeout = real;
    }
    assert.ok(seen[0] > 10_000, `health gives up after ${seen[0]} ms; undici stops connecting after 10 s`);
  });

  test("image_server_status says what failed: the token, the models, or the address", async () => {
    const { client } = await connect();
    mock.trouble("GET /models", { status: 500, body: { detail: "internal error" } });
    let status = textOf(await call(client, "image_server_status"));
    assert.match(status, /Reachable: yes \(version 2\.0\.0, up 2 min\)/);
    assert.match(status, /Models: couldn't be listed \(Image server error 500: internal error\)/);
    assert.doesNotMatch(status, /Token:/);
    mock.trouble("GET /health", { status: 200, body: { status: true } }); // e.g. Open WebUI's /health
    const elsewhere = await call(client, "image_server_status");
    assert.equal(elsewhere.isError, true);
    assert.match(textOf(elsewhere), /something answers at this address, but it isn't the image server/);
    assert.doesNotMatch(textOf(elsewhere), /Reachable: yes|undefined|NaN/);
    mock.trouble("GET /health", { status: 200, body: { status: "ok" } }); // a 1.0 image server
    const older = await call(client, "image_server_status");
    assert.equal(older.isError, true);
    assert.match(textOf(older), /older image server \(update it on the server Mac with scripts\/install_image_server\.sh\)/);
    const { client: tokenless } = await connect({ token: "" });
    status = textOf(await call(tokenless, "image_server_status"));
    assert.match(status, /Token: No API token is set/);
  });

  test("image_server_status shows the whole server's queue and how many pictures an edit can use", async () => {
    const { client } = await connect();
    // Open WebUI's pictures count in queue_depth; jobs_active counts only the job API's.
    mock.trouble("GET /health", { status: 200, body: { status: "ok", version: "2.0.0", loaded_model: "flux2-klein-4b",
      queue_depth: 3, jobs_active: 1, memory: {}, uptime_s: 600 } });
    const status = textOf(await call(client, "image_server_status"));
    assert.match(status, /\nLoaded model: flux2-klein-4b\nPictures waiting or being drawn \(all apps, Open WebUI included\): 3\n/);
    assert.match(status, /\n- flux2-klein-4b \(default, default for edits\): edits with up to 4 pictures \(source \+ references\), apache-2\.0\n/);
    assert.match(status, /\n- z-image-turbo: new pictures only, apache-2\.0\n/);
    assert.doesNotMatch(status, /: 1\n|4 references/);
    mock.trouble("GET /health", { status: 200, body: { status: "ok", version: "2.0.0", uptime_s: 1 } });
    assert.match(textOf(await call(client, "image_server_status")), /Open WebUI included\): unknown\n/);
  });

  test("too many open connections is shown as the server says it, asked once, with no warning about a picture made twice", async () => {
    const detail = "too many open connections from this computer; try again in a moment";
    mock.trouble("POST /jobs", { status: 503, body: { detail }, headers: { "retry-after": "5" } });
    const { client } = await connect();
    const before = mock.state.jobs.size;
    const refused = await call(client, "generate_image", { prompt: "crowded" });
    assert.equal(refused.isError, true);
    assert.equal(textOf(refused), "Too many open connections from this computer; try again in a moment.");
    assert.equal(mock.state.requests.filter((r) => r.startsWith("POST /jobs")).length, 1);
    assert.equal(mock.state.jobs.size, before, "no job was made");
    // While a picture is awaited, it passes like a restart does: the picture is still collected.
    mock.state.jobMs = 300;
    mock.trouble(/^GET \/jobs\//, { status: 503, body: { detail }, headers: { "retry-after": "5" }, times: 2 });
    const later = await call(client, "generate_image", { prompt: "less crowded" });
    assert.ok(!later.isError, textOf(later));
    assert.match(textOf(later), /Saved to: \S+less-crowded\.png/);
  });

  test("a 503 gives the server's reason, and a 413 claims no size", async () => {
    const { client } = await connect();
    mock.trouble("GET /references", { status: 503, body: { detail: RESTARTING }, headers: { "retry-after": "5" } });
    let result = await call(client, "list_references");
    assert.equal(result.isError, true);
    assert.equal(textOf(result), "The image server is restarting; try again in a few seconds.");
    mock.trouble("GET /references", { status: 503, body: "<html>Service Unavailable</html>" });
    result = await call(client, "list_references");
    assert.equal(textOf(result), "The image server is busy with other pictures. Try again in a minute.");
    const photo = path.join(tmp, "big.png");
    await fs.writeFile(photo, PNG);
    mock.trouble("POST /images", { status: 413, body: { detail: "the picture is too large" } });
    result = await call(client, "edit_image", { instruction: "x", source_path: photo });
    assert.match(textOf(result), /larger than the image server accepts/);
    assert.doesNotMatch(textOf(result), /MB/);
  });

  test("answers from something that isn't the image server are never echoed or blamed on it", async () => {
    const { client } = await connect();
    const elsewhere = /^Something at \S+ answered \(HTTP \d+\), but it doesn't look like the image server\. Check the server address in the extension settings \(on the server Mac, scripts\/enable_lan\.sh prints it\)\.$/;
    // Ollama's 404, FastAPI's answers for a route another app doesn't have, AirPlay's 403, a page.
    for (const [status, body] of [[404, "404 page not found"], [404, { detail: "Not Found" }],
      [405, { detail: "Method Not Allowed" }], [403, "Forbidden"], [400, "<html><body>Bad request</body></html>"]]) {
      mock.trouble("GET /references", { status, body });
      const result = await call(client, "list_references");
      assert.equal(result.isError, true);
      assert.match(textOf(result), elsewhere, `${status} ${JSON.stringify(body)}`);
    }
    // The server's own plain-text refusal and its own details still come through.
    mock.trouble("GET /references", { status: 403, body: "forbidden: private network clients only" });
    assert.match(textOf(await call(client, "list_references")), /^The image server only answers computers on the local network/);
    mock.trouble("GET /references", { status: 500, body: "Internal Server Error" });
    assert.equal(textOf(await call(client, "list_references")),
      "Image server error 500. Try again in a minute; if it keeps happening, run scripts/doctor.sh on the server Mac.");
    // However long the answer, the result stays small (Claude Desktop rejects ~150k characters).
    mock.trouble("POST /jobs", { status: 500, body: `<html>${"x".repeat(200_000)}</html>` });
    let result = await call(client, "generate_image", { prompt: "huge error page" });
    assert.equal(result.isError, true);
    assert.match(textOf(result), /^Image server error 500\. Try again in a minute/);
    assert.doesNotMatch(textOf(result), /xxx/);
    mock.trouble("GET /references", { status: 500, body: { detail: "y".repeat(5000) } });
    result = await call(client, "list_references");
    assert.match(textOf(result), /^Image server error 500: y+…$/);
    assert.ok(textOf(result).length < 350, `${textOf(result).length} characters`);
    // The status check keeps what it knows, and says what answers at the address.
    mock.trouble("GET /health", { status: 404, body: "404 page not found" });
    const status = await call(client, "image_server_status");
    assert.equal(status.isError, true);
    assert.match(textOf(status), /^Image server: http:\/\/127\.0\.0\.1:\d+\nPictures are saved in: \S+/);
    assert.match(textOf(status), /\nReachable: something answers at this address, but it isn't the image server\.\nCheck the server address/);
    assert.doesNotMatch(textOf(status), /page not found/);
    // A job the server doesn't know (its own 404) is still told apart from a wrong address.
    mock.trouble(/^GET \/jobs\//, { status: 404, body: "404 page not found" });
    const lost = await call(client, "get_image_job", { job_id: "job_0123456789abcdef" });
    assert.match(textOf(lost), /^Something at \S+ answered \(HTTP 404\)/);
    assert.match(textOf(lost), /job_id job_0123456789abcdef: call get_image_job with it later/);
  });

  test("an https address and a redirect are explained, with no warning about a picture made twice", async () => {
    const { client: secure } = await connect({ serverUrl: mock.url.replace("http:", "https:") });
    const result = await call(secure, "generate_image", { prompt: "secure owl" });
    assert.equal(result.isError, true);
    assert.match(textOf(result),
      /^Couldn't make a secure \(https\) connection to https:\/\/127\.0\.0\.1:\d+ \(\w+\)\. The image server uses plain http: change the address to http:\/\/127\.0\.0\.1:\d+ in the extension settings\.$/);
    const status = await call(secure, "image_server_status");
    assert.match(textOf(status), /^Image server: https:.*\nPictures are saved in: .*\nCouldn't make a secure \(https\) connection/);
    mock.trouble("POST /jobs", { status: 307, headers: { location: "/login" } });
    const { client } = await connect();
    const redirected = await call(client, "generate_image", { prompt: "redirected owl" });
    assert.equal(redirected.isError, true);
    assert.match(textOf(redirected), /^Something at \S+ redirected the request elsewhere, but it doesn't look like the image server\. Check the server address/);
    for (const text of [textOf(result), textOf(redirected)]) {
      assert.doesNotMatch(text, /may have started|Is it running|install_image_server/);
    }
  });

  test("every request asks for a connection of its own", async () => {
    // Node 26's fetch stalls a request on a kept-alive connection idle for half a second or more.
    const asked = [];
    const recording = (url, init) => {
      asked.push(init.headers.connection);
      return fetch(url, init);
    };
    const { client } = await connect({}, { fetchImpl: recording });
    const result = await call(client, "generate_image", { prompt: "fresh line" });
    assert.ok(!result.isError, textOf(result));
    assert.ok(asked.length >= 4, `${asked.length} requests`);
    assert.deepEqual([...new Set(asked)], ["close"]);
  });

  test("a rejected Host header points at enable_lan.sh", async () => {
    const strict = await startMock({ hostCheck: () => false });
    try {
      const { client } = await connect({ serverUrl: strict.url });
      const result = await call(client, "list_references");
      assert.match(textOf(result), /scripts\/enable_lan\.sh/);
    } finally {
      await strict.close();
    }
  });

  test("a bad server address in settings is reported on use", async () => {
    const config = loadConfig({ IMAGE_SERVER_URL: "ftp://example", IMAGE_SERVER_TOKEN: TOKEN });
    assert.match(config.problem, /http:\/\//);
    const { client } = await connect({ problem: config.problem });
    const result = await call(client, "list_references");
    assert.match(textOf(result), /must start with http/);
  });
});

describe("settings and file names", () => {
  test("empty or unexpanded settings fall back to defaults", () => {
    const config = loadConfig({
      IMAGE_SERVER_URL: "http://192.168.1.20:7865/v1/", IMAGE_SERVER_TOKEN: "Bearer abc",
      IMAGE_OUTPUT_DIR: "${user_config.output_dir}", IMAGE_WAIT_SECONDS: "500",
    });
    assert.equal(config.serverUrl, "http://192.168.1.20:7865");
    assert.equal(config.token, "abc");
    assert.ok(config.outputDir.endsWith("Generated Images") || config.outputDir.endsWith("Teach Image"));
    assert.equal(config.waitSeconds, 40, "waiting longer would leave no time to save the pictures");
    assert.equal(loadConfig({ IMAGE_WAIT_SECONDS: "1" }).waitSeconds, 5);
    assert.equal(loadConfig({ IMAGE_WAIT_SECONDS: "${user_config.wait}" }).waitSeconds, 40);
    assert.equal(loadConfig({ IMAGE_SERVER_URL: "mac.local:7865" }).serverUrl, "http://mac.local:7865");
    const custom = loadConfig({ IMAGE_OUTPUT_DIR: "/Volumes/School/Pictures" });
    assert.ok(custom.inputRoots.includes(path.resolve("/Volumes/School/Pictures")), "saved pictures can be edited");
    assert.match(loadConfig({ IMAGE_SERVER_URL: "http://user:pw@mac.local:7865" }).problem, /token/);
  });

  test("localhost becomes 127.0.0.1, and an http:// address without a port gets the server's 7865", () => {
    const url = (value) => loadConfig({ IMAGE_SERVER_URL: value }).serverUrl;
    // The server listens on IPv4, and Node 18 tries localhost's IPv6 address first.
    assert.equal(url("http://localhost:7865"), "http://127.0.0.1:7865");
    assert.equal(url("LOCALHOST:7865/v1"), "http://127.0.0.1:7865");
    assert.equal(url("https://localhost:7865"), "https://127.0.0.1:7865");
    // Nothing serves the image server on port 80: a missing port was just left out.
    assert.equal(url("http://localhost"), "http://127.0.0.1:7865");
    assert.equal(url("http://teachers-mac.local"), "http://teachers-mac.local:7865");
    assert.equal(url("teachers-mac.local"), "http://teachers-mac.local:7865");
    assert.equal(url("http://192.168.1.20/v1/"), "http://192.168.1.20:7865");
    assert.equal(url("http://192.168.1.20:"), "http://192.168.1.20:7865");
    assert.equal(url("http://[fd00::20]"), "http://[fd00::20]:7865");
    // A port that is given stays, even 80; an https:// address is taken as given.
    assert.equal(url("http://192.168.1.20:80"), "http://192.168.1.20");
    assert.equal(url("http://[fd00::20]:8080"), "http://[fd00::20]:8080");
    assert.equal(url("https://teachers-mac.local"), "https://teachers-mac.local");
    assert.equal(url(""), "http://127.0.0.1:7865");
  });

  test("a pasted token is cleaned, and one that can't be sent is reported", async () => {
    const clean = loadConfig({ IMAGE_SERVER_TOKEN: "​ Bearer  abc 123⁠﻿\r\n" });
    assert.equal(clean.token, "abc123");
    assert.equal(clean.problem, null);
    assert.equal(loadConfig({ IMAGE_SERVER_TOKEN: "abc\n123‍" }).token, "abc123");
    const odd = loadConfig({ IMAGE_SERVER_TOKEN: "abc–123" });
    assert.equal(odd.problem, "The API token contains characters that can't be sent. Copy it again from the " +
      "server Mac and paste it into the extension settings.");
    const { client } = await connect({ token: odd.token, problem: odd.problem });
    const result = await call(client, "list_references");
    assert.match(textOf(result), /characters that can't be sent/);
    assert.match(loadConfig({ IMAGE_SERVER_URL: "ftp://x", IMAGE_SERVER_TOKEN: "é" }).problem, /http/,
      "the address problem is reported first");
  });

  test("names are safe on every platform", () => {
    assert.equal(safeName("../../etc/passwd"), "etc passwd");
    assert.equal(safeName("..."), "image");
    assert.equal(safeName("CON"), "CON_");
    assert.equal(safeName("con.txt"), "con_.txt");
    assert.equal(safeName("COM1.v2"), "COM1_.v2");
    assert.equal(safeName("lpt3.x"), "lpt3_.x");
    assert.equal(safeName("com¹"), "com¹_");
    assert.equal(safeName("lpt².notes"), "lpt²_.notes");
    assert.equal(safeName("console.txt"), "console.txt");
    assert.equal(safeName("com10"), "com10");
    assert.equal(safeName("Owl: final?.png"), "Owl final");
    assert.equal(nameFromPrompt("A cute OWL, wearing glasses!"), "a-cute-owl-wearing-glasses");
    assert.equal(nameFromPrompt("¡¿?!"), "image");
    const root = path.join(os.tmpdir(), "root");
    assert.equal(outputFolder(root, "../../x"), path.join(root, "x"));
    assert.equal(outputFolder(root, ""), root);
    assert.equal(outputFolder(root, "a/b/c/d/e"), path.join(root, "a", "b", "c"));
    assert.equal(outputFolder(root, path.join(root, "Bear book")), path.join(root, "Bear book"));
    assert.equal(outputFolder(root, `${root}${path.sep}`), root);
    assert.throws(() => outputFolder(root, path.join(path.dirname(root), "Bear book")), /not a full path/);
  });

  test("names keep the vowel signs and other combining marks of their script", () => {
    // Hindi, Thai, Tamil and Arabic write vowels as combining marks that NFC leaves separate.
    const nfc = (text) => text.normalize("NFC");
    for (const word of ["बिल्ली", "นกฮูก", "பூனை", "قِطَّة", "café"].map(nfc)) {
      assert.equal(safeName(word), word);
      assert.equal(safeName(word.normalize("NFD")), word, "decomposed, as macOS file names can be");
    }
    assert.equal(nameFromPrompt("बिल्ली किताब पढ़ रही है"), nfc("बिल्ली-किताब-पढ़-रही-है"));
    const root = path.join(os.tmpdir(), "root");
    assert.equal(outputFolder(root, "कहानी"), path.join(root, nfc("कहानी")), "the teacher's own folder, not a new one");
  });

  test("a prompt that starts with a very long word is named after its beginning, never half a character", () => {
    const chinese = "一只友好的猫头鹰在图书馆里安静地读一本关于星星和月亮的大书旁边还有一只小老鼠在认真地听它讲故事给大家听然后一起回家";
    assert.ok(chinese.length > 48);
    assert.equal(nameFromPrompt(chinese), chinese.slice(0, 48));
    const compound = "Donaudampfschifffahrtsgesellschaftskapitänsmütze".normalize("NFC");
    assert.equal(nameFromPrompt(`${compound}${compound} owl`), `${compound}${compound}`.toLowerCase().slice(0, 48));
    // A character outside the Basic Multilingual Plane takes two UTF-16 units; a cut never splits one.
    const rare = "𠀀"; // U+20000, a CJK ideograph
    assert.equal(nameFromPrompt(`a${rare.repeat(30)}`), `a${rare.repeat(23)}`);
    assert.equal(safeName(`b${rare.repeat(50)}`), `b${rare.repeat(39)}`);
    assert.doesNotMatch(safeName(`b${rare.repeat(50)}`), /[\ud800-\udbff]$/);
  });

  test("a file in the way of the pictures folder is explained, and nothing is downloaded for nothing", async () => {
    const { client, config } = await connect();
    await fs.mkdir(config.outputDir, { recursive: true });
    await fs.writeFile(path.join(config.outputDir, "Bear book"), "a file, not a folder");
    const downloads = mock.state.fullDownloads;
    const pair = await call(client, "generate_image", { prompt: "bear cub", n: 2, folder: "Bear book" });
    assert.ok(!pair.isError, textOf(pair));
    const text = textOf(pair);
    assert.match(text, /Pictures can't be saved in \S+out.Bear book: a file is in the way \(it has the name of that folder, or of a folder above it\)\. Use another folder name, or choose another folder for "Save pictures in"/);
    assert.match(text, /once that is fixed, call get_image_job with job_id job_\w+ to save them\./);
    assert.doesNotMatch(text, /EEXIST|try saving again/);
    assert.equal(mock.state.fullDownloads - downloads, 1, "the second picture isn't downloaded for nothing");
    const imageId = /img_[0-9a-f]{16}/.exec(text)[0];
    const copy = await call(client, "get_image", { image_id: imageId, save_copy: true, folder: "Bear book/Unit 2" });
    assert.equal(copy.isError, true);
    assert.match(textOf(copy), /^Pictures can't be saved in .*Unit 2: a file is in the way/);
  });

  test("a pictures folder on a drive that isn't connected says so", async () => {
    const realMkdir = fs.mkdir;
    const failing = (code) => async (dir) => {
      throw Object.assign(new Error(`${code}: mkdir '${dir}'`), { code, syscall: "mkdir" });
    };
    try {
      fs.mkdir = failing("ENOENT"); // Windows, for a drive letter that isn't there
      await assert.rejects(savePng(path.join(tmp, "E", "Teach Image"), "owl", PNG),
        /The folder for pictures, \S+Teach Image, can't be created: is the drive it is on connected\? Connect it, or choose another folder for "Save pictures in"/);
      for (const code of ["EACCES", "EROFS"]) {
        fs.mkdir = failing(code); // a Mac: only the system may create /Volumes/<disk>
        if (process.platform === "darwin") {
          await assert.rejects(savePng("/Volumes/Teach Image Test 7c1f/Pictures", "owl", PNG), (err) => {
            assert.match(err.message, /is the drive it is on connected\?/, code);
            assert.doesNotMatch(err.message, /Privacy & Security|read-only/, code);
            return true;
          });
        }
      }
      fs.mkdir = failing("EACCES");
      // On a disk that is there, the same refusal is about permission.
      await assert.rejects(savePng(path.join(tmp, "pictures"), "owl", PNG), /Claude isn't allowed to save pictures in/);
    } finally {
      fs.mkdir = realMkdir;
    }
  });

  test("a read-only disk is explained as that, not as a missing permission", async () => {
    const realMkdir = fs.mkdir;
    fs.mkdir = async (dir) => {
      throw Object.assign(new Error(`EROFS: read-only file system, mkdir '${dir}'`), { code: "EROFS", syscall: "mkdir" });
    };
    try {
      const card = path.join(tmp, "card");
      await assert.rejects(savePng(card, "owl", PNG), (err) => {
        assert.equal(err.message, `Pictures can't be saved in ${card}: the disk it is on is read-only; choose another ` +
          "folder for \"Save pictures in\" in the extension settings.");
        return true;
      });
      // A folder problem: the picture waits on the server until another folder is chosen.
      const { client } = await connect({ outputDir: card });
      const result = await call(client, "generate_image", { prompt: "locked card" });
      assert.ok(!result.isError, textOf(result));
      assert.match(textOf(result), /Not saved on this computer: Pictures can't be saved in \S+card: the disk it is on is read-only/);
      assert.match(textOf(result), /once that is fixed, call get_image_job with job_id job_\w+ to save it\./);
      assert.doesNotMatch(textOf(result), /Privacy & Security|EROFS/);
    } finally {
      fs.mkdir = realMkdir;
    }
  });

  test("pictures are allowed by their path first, and each folder's real path is asked for only once", {
    skip: process.platform === "win32" && "needs symlinks",
    timeout: 10_000, // waiting for the stalled drive would never end
  }, async () => {
    const real = path.join(tmp, "real");
    await fs.mkdir(real);
    const link = path.join(tmp, "link");
    await fs.symlink(real, link);
    const photo = path.join(real, "photo.png");
    await fs.writeFile(photo, PNG);
    const stalled = path.join(tmp, "stalled drive"); // a folder on a drive that stopped answering
    const realRealpath = fs.realpath;
    const asked = [];
    fs.realpath = async (p, ...rest) => {
      asked.push(p);
      return p === stalled ? new Promise(() => {}) : realRealpath(p, ...rest);
    };
    try {
      // Inside a folder as written: no folder's real path is needed.
      assert.equal((await readInputImage(photo, [stalled, real])).name, "photo.png");
      assert.deepEqual(asked, [photo]);
      // Inside a folder only through a link: allowed by real paths, without waiting for the stalled drive.
      asked.length = 0;
      let { ms } = await timed(readInputImage(photo, [stalled, link]));
      assert.ok(ms < 500, `took ${ms} ms`);
      assert.deepEqual(asked.sort(), [link, photo, stalled].sort());
      // Each folder is asked once: the link's real path is remembered, and the stalled drive isn't asked again.
      asked.length = 0;
      await readInputImage(photo, [stalled, link]);
      assert.deepEqual(asked, [photo]);
      // A picture in neither folder is refused: the stalled drive's answer is waited for a little
      // while (it might be the one), then no more.
      const outside = path.join(tmp, "outside.png");
      await fs.writeFile(outside, PNG);
      for (const most of [2500, 300]) {
        asked.length = 0;
        ({ ms } = await timed(assert.rejects(readInputImage(outside, [stalled, link]), /can only be sent from your home folder/)));
        assert.ok(ms < most, `took ${ms} ms`);
        assert.deepEqual(asked, [outside]);
      }
    } finally {
      fs.realpath = realRealpath;
    }
  });

  test("file errors without their own words are blamed on the file, not the extension", async () => {
    const { client } = await connect();
    const made = await call(client, "generate_image", { prompt: "flaky disk" });
    const imageId = /img_[0-9a-f]{16}/.exec(textOf(made))[0];
    const realOpen = fs.open;
    fs.open = async (...args) => {
      const handle = await realOpen(...args);
      return {
        writeFile: async () => {
          throw Object.assign(new Error("EIO: i/o error, write"), { code: "EIO", syscall: "write" });
        },
        close: () => handle.close(),
      };
    };
    try {
      const copy = await call(client, "get_image", { image_id: imageId, save_copy: true });
      assert.equal(copy.isError, true);
      assert.equal(textOf(copy), "A file or folder on this computer couldn't be used: EIO: i/o error, write");
    } finally {
      fs.open = realOpen;
    }
  });

  test("paths that can't lead to a picture are explained", { skip: process.platform === "win32" && "needs symlinks" },
    async () => {
      const unusable = /Claude can't open .+: that isn't a usable path to a picture file\. Give the full path of the picture itself/;
      await assert.rejects(readInputImage(path.join(tmp, `${"x".repeat(300)}.png`), [tmp]), unusable, "ENAMETOOLONG");
      const loop = path.join(tmp, "loop.png");
      await fs.symlink(path.join(tmp, "loop2.png"), loop);
      await fs.symlink(loop, path.join(tmp, "loop2.png"));
      await assert.rejects(readInputImage(loop, [tmp]), unusable, "ELOOP");
    });

  test("pictures that can't be sent are explained", async () => {
    await assert.rejects(readInputImage(path.join(tmp, "missing.png"), [tmp]), /There's no file at/);
    const plain = path.join(tmp, "plain.png");
    await fs.writeFile(plain, PNG);
    await assert.rejects(readInputImage(path.join(plain, "inside.png"), [tmp]), /There's no file at/, "ENOTDIR");
    await assert.rejects(readInputImage(plain, [path.join(tmp, "elsewhere")]), (err) => {
      assert.match(err.message, /Copy plain\.png into your home folder first, for example into Pictures/);
      assert.match(err.message, /IMAGE_INPUT_ROOTS/);
      return true;
    });
  });

  test("a folder Claude may not read is explained", {
    skip: (process.platform === "win32" || process.getuid?.() === 0) && "needs folder permissions",
  }, async () => {
    const locked = path.join(tmp, "locked");
    await fs.mkdir(locked);
    await fs.writeFile(path.join(locked, "photo.png"), PNG);
    await fs.chmod(locked, 0o000);
    try {
      await assert.rejects(readInputImage(path.join(locked, "photo.png"), [tmp]), (err) => {
        assert.match(err.message, /Claude isn't allowed to read the folder/);
        assert.match(err.message, /Privacy & Security > Files and Folders \(or Full Disk Access\) > Claude/);
        assert.match(err.message, /copy the picture into your home folder/);
        return true;
      });
    } finally {
      await fs.chmod(locked, 0o700);
    }
  });

  test("a picture this account can't open is blamed on its own permissions, not the folder", {
    skip: noFolderPermissions,
  }, async () => {
    const photo = path.join(tmp, "private.png");
    await fs.writeFile(photo, PNG);
    await fs.chmod(photo, 0o000);
    try {
      await assert.rejects(readInputImage(photo, [tmp]), (err) => {
        assert.match(err.message, /isn't allowed to open \S+private\.png: the picture's own permissions don't allow it/);
        // The same account can't open it to save a copy: giving it access is the way out.
        assert.match(err.message, /Give your account Read access to it \(on a Mac: select it in Finder, choose File > Get Info, and under Sharing & Permissions/);
        assert.match(err.message, /or ask whoever owns the file for a copy\. Then try again\./);
        assert.doesNotMatch(err.message, /save a copy|Privacy & Security|the folder|into your home folder/);
        return true;
      });
    } finally {
      await fs.chmod(photo, 0o600);
    }
  });

  test("a locked folder outside the allowed ones gets the copy advice, never System Settings", {
    skip: noFolderPermissions,
  }, async () => {
    const usb = path.join(tmp, "USB");
    await fs.mkdir(usb);
    await fs.writeFile(path.join(usb, "photo.png"), PNG);
    await fs.mkdir(path.join(tmp, "home"));
    await fs.chmod(usb, 0o000);
    try {
      await assert.rejects(readInputImage(path.join(usb, "photo.png"), [path.join(tmp, "home")]), (err) => {
        assert.match(err.message, /Pictures can only be sent from your home folder.*Copy photo\.png into your home folder first/);
        assert.doesNotMatch(err.message, /Privacy & Security/);
        return true;
      });
    } finally {
      await fs.chmod(usb, 0o700);
    }
  });

  test("quoted paths and file:// URLs are full paths too", async () => {
    const photo = path.join(tmp, "class photo.png");
    await fs.writeFile(photo, PNG);
    for (const given of [`"${photo}"`, `“${photo}”`, ` '${photo}' `, pathToFileURL(photo).href]) {
      const { name, type } = await readInputImage(given, [tmp]);
      assert.deepEqual([name, type], ["class photo.png", "image/png"], given);
    }
    await assert.rejects(readInputImage("\"relative/photo.png\"", [tmp]), /isn't a full path/);
  });

  test("a save cut off by a full disk leaves no broken file behind", async () => {
    const dir = path.join(tmp, "pictures");
    const realOpen = fs.open;
    fs.open = async (...args) => {
      const handle = await realOpen(...args);
      return {
        writeFile: async (bytes) => {
          await handle.write(bytes.subarray(0, 8));
          throw Object.assign(new Error("ENOSPC: no space left on device, write"), { code: "ENOSPC" });
        },
        close: () => handle.close(),
      };
    };
    try {
      await assert.rejects(savePng(dir, "owl", PNG),
        /There's no room to save pictures in \S+pictures: the disk is full\. Free up some space/);
    } finally {
      fs.open = realOpen;
    }
    assert.deepEqual(await files(dir), [], "the cut-off owl.png was removed");
    assert.equal(path.basename(await savePng(dir, "owl", PNG)), "owl.png", "so the next save gets the real name");
  });

  test("pictures are recognized by their bytes", () => {
    assert.equal(sniffImage(PNG), "image/png");
    assert.equal(sniffImage(Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 0])), "image/jpeg");
    assert.equal(sniffImage(Buffer.from("RIFF\0\0\0\0WEBPVP8 ", "latin1")), "image/webp");
    assert.equal(sniffImage(Buffer.from("GIF89a....", "latin1")), "image/gif");
    assert.equal(sniffImage(Buffer.from("%PDF-1.7")), null);
  });
});
