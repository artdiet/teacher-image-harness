// Builds dist/teach-image.mcpb: one bundled server file (dependencies included, so the
// extension runs on Claude Desktop's built-in Node with nothing to install), the manifest,
// icon and license.
//
//   npm run build

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"));
if (manifest.version !== pkg.version) {
  throw new Error(`manifest.json version ${manifest.version} != package.json ${pkg.version}`);
}

const stage = path.join(root, "dist", "teach-image");
fs.rmSync(stage, { recursive: true, force: true });
fs.mkdirSync(path.join(stage, "server"), { recursive: true });

await build({
  entryPoints: [path.join(root, "src", "index.js")],
  outfile: path.join(stage, "server", "index.mjs"),
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node18",
  legalComments: "eof",
  // Some dependencies still call require(); give the ESM bundle one.
  banner: { js: "import { createRequire as __createRequire } from 'node:module';\nconst require = __createRequire(import.meta.url);" },
  logLevel: "warning",
});

fs.copyFileSync(path.join(root, "manifest.json"), path.join(stage, "manifest.json"));
fs.copyFileSync(path.join(root, "icon.png"), path.join(stage, "icon.png"));
fs.copyFileSync(path.join(path.dirname(root), "LICENSE.md"), path.join(stage, "LICENSE.md"));

// The mcpb CLI's JavaScript entry, run with this Node: node_modules/.bin/mcpb is a .cmd shim on
// Windows, which execFileSync can't start.
function mcpbCli() {
  const dir = path.join(root, "node_modules", "@anthropic-ai", "mcpb");
  const { bin } = JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf8"));
  return path.join(dir, typeof bin === "string" ? bin : bin.mcpb);
}

const mcpb = (...args) => execFileSync(process.execPath, [mcpbCli(), ...args], { stdio: "inherit" });
const out = path.join(root, "dist", "teach-image.mcpb");
mcpb("validate", path.join(stage, "manifest.json"));
mcpb("pack", stage, out);
execFileSync(process.execPath, ["--test", path.join(root, "test", "bundle.test.js")], { stdio: "inherit" });
console.log(`\nbuilt ${path.relative(process.cwd(), out)}`);
