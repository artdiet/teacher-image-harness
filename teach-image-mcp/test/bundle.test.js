// Runs the built extension (dist/, from `npm run build`) the way Claude Desktop does:
// a child process speaking MCP over stdio, configured through environment variables.

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

import { TOKEN, startMock } from "./mock-server.js";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const bundle = path.join(root, "dist", "teach-image", "server", "index.mjs");
const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"));
const example = JSON.parse(fs.readFileSync(path.join(root, "claude_desktop_config.example.json"), "utf8"));
const SOURCES = ["src", "manifest.json", "package.json"];

// The time of the newest change among these files, looking inside folders.
function newestChange(paths) {
  let newest = 0;
  for (const p of paths) {
    const stat = fs.statSync(p);
    const changed = stat.isDirectory() ? newestChange(fs.readdirSync(p).map((name) => path.join(p, name))) : stat.mtimeMs;
    newest = Math.max(newest, changed);
  }
  return newest;
}

// Why the bundle can't be tested, if it can't: a bundle older than its sources would test
// yesterday's code. (`npm run build` runs this file right after building.)
function cannotTest(bundleFile, base = root) {
  if (!fs.existsSync(bundleFile)) return "run npm run build first";
  if (newestChange(SOURCES.map((p) => path.join(base, p))) > fs.statSync(bundleFile).mtimeMs) {
    return "the bundle is older than src: run npm run build";
  }
  return false;
}

test("a bundle older than its sources is not tested", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "teach-image-stale-"));
  try {
    const built = path.join(dir, "dist", "index.mjs");
    assert.equal(cannotTest(built, dir), "run npm run build first");
    fs.mkdirSync(path.join(dir, "src", "deeper"), { recursive: true });
    fs.mkdirSync(path.dirname(built));
    const files = ["src/index.js", "src/deeper/more.js", "manifest.json", "package.json", "dist/index.mjs"];
    const at = (seconds) => new Date(1_790_000_000_000 + seconds * 1000);
    for (const file of files) {
      fs.writeFileSync(path.join(dir, file), "");
      fs.utimesSync(path.join(dir, file), at(0), at(0));
    }
    fs.utimesSync(built, at(10), at(10));
    assert.equal(cannotTest(built, dir), false);
    for (const file of ["src/deeper/more.js", "manifest.json", "package.json"]) {
      fs.utimesSync(path.join(dir, file), at(20), at(20));
      assert.equal(cannotTest(built, dir), "the bundle is older than src: run npm run build", file);
      fs.utimesSync(path.join(dir, file), at(0), at(0));
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("Node gets more file threads, set before the server starts", () => {
  // A file operation that hangs (a network folder gone, a question about folder access left
  // unanswered) holds one of them; Node's default is 4, and it reads the number only at startup.
  assert.equal(manifest.server.mcp_config.env.UV_THREADPOOL_SIZE, "16");
  assert.equal(example.mcpServers["teach-image"].env.UV_THREADPOOL_SIZE, "16");
});

test("the bundled server answers over stdio", { skip: cannotTest(bundle) }, async () => {
  const mock = await startMock();
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "teach-image-bundle-"));
  // The manifest's fixed settings, as Claude Desktop passes them (the user's come from the test).
  const fixed = Object.entries(manifest.server.mcp_config.env).filter(([, value]) => !value.includes("${"));
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [bundle],
    env: { PATH: process.env.PATH, HOME: os.homedir(), ...Object.fromEntries(fixed), IMAGE_SERVER_URL: mock.url,
      IMAGE_SERVER_TOKEN: TOKEN, IMAGE_OUTPUT_DIR: out },
    stderr: "pipe",
  });
  let stderr = "";
  transport.stderr?.on("data", (chunk) => { stderr += chunk; });
  const client = new Client({ name: "bundle-test", version: "1.0.0" });
  try {
    await client.connect(transport);
    const { tools } = await client.listTools();
    assert.deepEqual(tools.map((t) => t.name).sort(), manifest.tools.map((t) => t.name).sort(),
      "manifest.json lists the same tools the server registers");
    const status = await client.callTool({ name: "image_server_status", arguments: {} });
    assert.match(status.content[0].text, /Reachable: yes/);
    assert.match(status.content[0].text, /Token: accepted/);
    const made = await client.callTool({ name: "generate_image", arguments: { prompt: "bundle check" } });
    assert.ok(!made.isError, made.content[0].text);
    assert.ok(fs.existsSync(path.join(out, "bundle-check.png")));
    assert.match(stderr, /ready; image server/);
    assert.doesNotMatch(stderr, new RegExp(TOKEN), "the token never appears in logs");
  } finally {
    await client.close();
    await mock.close();
    fs.rmSync(out, { recursive: true, force: true });
  }
});
