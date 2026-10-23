# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## What this is

A working proof of concept (Oct 2026): local image generation for a teacher's
classroom materials on an Apple Silicon Mac (M5 Pro, 24 GB), used from Open
WebUI (1.0) and from Claude Desktop / Cowork (2.0). Three parts:

- `image-server/`: FastAPI + [mflux](https://github.com/mflux-community/mflux)
  (Apple MLX), runs **natively** under launchd on port 7865 (`127.0.0.1`, or
  all interfaces in opt-in LAN mode), bearer token. OpenAI-compatible
  `/v1/images/*` for Open WebUI, plus an async job API (`/jobs`, `/images`,
  `/references`, `/models`, `/health`; contract in `docs/API.md` and
  `docs/openapi.json`) with per-device tokens (`client_tokens.json`, hashed).
- `openwebui/teach_image_pipe.py`: an Open WebUI 0.9.2 "pipe" function (runs
  inside the Open WebUI Docker container). A director LLM (`gemma4:12b` on the
  Mac's Ollama) classifies each message (draw/reply, kind, edit previous); fixed
  templates build the image prompt; results are stored as the user's Open WebUI
  files; the reply streams a `<think>` trace.
- `teach-image-mcp/`: the **Teach Image** Claude Desktop extension (.mcpb), a
  Node stdio MCP server that each Claude computer runs; it calls the job API
  over the LAN, saves full PNGs locally and returns previews + facts as text.
  User setup in `docs/MCP.md`.

The next phase (a whole-book harness with its own server and UI) is planned
for a separate repo that builds on this one. Read `docs/LEARNINGS.md` before
designing anything new: it records what was tried, measured and decided.

## Commands

```bash
cd image-server && uv run pytest -q              # 130 tests: server + pipe (fake backend, no GPU; a few run a real uvicorn)
cd teach-image-mcp && npm test                   # 104 tests: MCP tools vs a mock server, plus the bundle once built
cd teach-image-mcp && npm run build              # dist/teach-image.mcpb (esbuild bundle, mcpb validate/pack, smoke test)
scripts/doctor.sh                                # health of the whole chain (prints no secrets)
scripts/check_lan.sh                             # in LAN mode: reachable from the network? spots a firewall block
scripts/client_tokens.sh add|list|revoke <name>  # per-device tokens (prints a new token once)
scripts/smoke_test_api.sh http://<host>:7865     # curl every endpoint (makes 3 real pictures, unloads the model)
uv run --project image-server python scripts/export_openapi.py   # after changing routes
scripts/install_image_server.sh                  # (re)start the launchd agent; validates .env first
scripts/install_openwebui.sh                     # install/update pipe + model entry (backs up webui.db)
scripts/download_models.sh                       # pinned MLX weights into the HF cache (~29 GB)
cd image-server && uv run python ../scripts/bakeoff.py --quick   # model comparison on real prompts
```

Logs: `~/Library/Logs/teacher-image-harness/image-server.log`. Data (outputs,
backups, bake-offs): `~/Library/Application Support/teacher-image-harness/`.

After changing server code or `.env`: re-run `install_image_server.sh`. After
changing the pipe: re-run `install_openwebui.sh` (idempotent; keeps admin edits).
After changing the extension: bump `version` in both `package.json` and
`manifest.json`, rebuild, reinstall the .mcpb, fully restart Claude Desktop.

## Rules

- **Public repo.** Never commit secrets, `.env`, model weights, generated
  images, database backups, personal names, emails, LAN addresses or other
  machine details. The teacher-facing model name lives only in `.env`
  (`OWUI_MODEL_NAME`). A gitleaks pre-commit hook (`core.hooksPath=.githooks`)
  and a CI scan back this up; the hook needs `brew install gitleaks`.
- `user_notes.md` (gitignored) is the user's async channel: answer each note
  inline beneath it as `> **Claude:** …` and put your own questions there.
- Device tokens and the server URL go in Claude Desktop's extension settings
  or a local config, never in `manifest.json`, docs examples or tests (tests
  use obviously fake tokens).
- Commit or push only when asked.
- Don't load models or run generation while another MLX process is using the
  GPU: only one model fits (see memory below). `bakeoff.py` asks the server to
  unload first.
- Licenses matter: default models are Apache-2.0. Qwen-Image-2.1 is
  research/evaluation only and stays off unless the user decides otherwise.

## Constraints and gotchas (verified)

- **GPU memory:** the Metal working set on 24 GB is ~17.8 GiB. Only one image
  model is resident at a time; the server asks Ollama to unload its models
  right before each job (FIFO lock) and unloads after 10 idle minutes, then
  restarts itself lean (~50 MB) unless `.env`/code changed since startup.
- **Docker on macOS has no GPU access**, so the image server must stay native.
  Docker Desktop's `host.docker.internal` reaches services bound to the Mac's
  `127.0.0.1`, so nothing needs to listen on the network.
- **Ollama has no image generation** (removed in v0.32.6); it is only the LLM.
- **mflux:** pin `pillow<13` on Python 3.12 (Pillow 13 breaks mflux imports).
  Always pass `model_path` (the pinned snapshot dir) and an explicit
  `ModelConfig`, `quantize=None` for pre-quantized repos. All MLX calls stay on
  one thread. klein text-to-image and edit are separate classes; the edit view
  shares weights via `dict.update` + `__dict__.update`. Qwen-2.1 needs
  `ModelConfig.qwen_image_21_turbo()`, sizes in multiples of 32, guidance 1.
  Z-Image-Turbo needs 9 steps (API default is 4).
- **Open WebUI 0.9.2:** a pipe model is invisible to non-admins until a model
  row with the pipe's id exists *and* has a read grant; image attachments need
  `vision` and `file_upload` capabilities. Background tasks (titles, tags,
  follow-ups) call the pipe with `__task__`, so answer them without drawing.
  `<think>` renders as a collapsed block only when `body["stream"]` is true.
  Return images via `upload_file_handler` + `Chats.insert_chat_files` and
  markdown links to `/api/v1/files/{id}/content` (never data: URLs). API keys
  are off on the target instance, so the installer runs inside the container
  through Open WebUI's data layer (needs `WEBUI_SECRET_KEY` from the key file
  and `ENABLE_DB_MIGRATIONS=False`).

- **Claude Desktop / Cowork (verified Oct 2026, Claude Desktop 2.42):**
  - Custom connectors and plugin HTTP servers are called from Anthropic's cloud
    and can't reach a LAN.
  - Local stdio servers (.mcpb extensions) run on the user's machine. Cowork
    tasks run in the cloud and reach them through the open Desktop app
    (`[localMcpBridge]` in `~/Library/Logs/Claude/main.log`).
  - Local tool calls time out at ~60 s, so a tool call does everything within
    a 50 s budget (waiting at most 40 s) and hands back a job id. Results over ~150k characters are rejected, so previews
    share a ~75 KB budget. Avoid `outputSchema`/`structuredContent`.
  - Desktop bundles Node (v24 in 2.42).
- **macOS firewall:** the uv-managed Python is ad-hoc signed. Until it is
  allowed (`socketfilterfw --add` + `--unblockapp`, redone after Python
  upgrades) the firewall accepts the TCP connection and then holds it, so
  requests time out. `--getappblocked` still says "permitted" for apps it has
  never asked about. Test with a request to the Mac's own LAN IP
  (`scripts/check_lan.sh`); on the Mac itself `<name>.local` resolves to
  127.0.0.1 and proves nothing.
- **macOS `/bin/bash` is 3.2:** JSON with commas inside `"$( … -d "{…}" )"`
  gets brace-expanded into separate words. Build bodies in a variable first.
- **LAN server details (2.0):**
  - Host names are compared in lowercase (macOS LocalHostNames are mixed case;
    Node's fetch lowercases).
  - `connections.py` subclasses uvicorn's h11 protocol (per-client cap, 10 s
    header deadline) because uvicorn has neither. `tests/test_connections.py`
    guards those internals: re-run it after upgrading uvicorn.
  - Open WebUI's primary token is accepted only from loopback; Docker's
    `host.docker.internal` arrives as 127.0.0.1.
  - Job records are written on accept, with a marker in `jobs/active/` until
    they finish.
  - A stop starts one 25 s grace clock at once (`__main__.serve` hooks
    uvicorn's `handle_exit`; `timeout_graceful_shutdown` uses the same 25 s)
    for open requests and background jobs, then stops what's left and records
    it as "interrupted". Keep grace + 5 s + unload under launchd's
    `ExitTimeOut` (60 s), which `install_image_server.sh` waits out.

## Code conventions

Python 3.12, uv, type hints, small modules, comments explain *why*. Server
routes never declare body parameters (FastAPI would read the body before the
auth dependency). Tests use a fake backend and in-memory Open WebUI stand-ins;
add a test with every behavior change. The extension is plain ESM JavaScript
(no TypeScript build), zod input schemas, `node:test`; tool errors are
returned as `isError` results written for the teacher, never thrown, and
logs go to stderr only (stdout carries the protocol).
