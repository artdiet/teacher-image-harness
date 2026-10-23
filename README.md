# Teacher Image Harness

Private, local image generation for classroom materials (worksheet icons,
coloring pages, storybook art) running on an Apple Silicon MacBook and used
from [Open WebUI](https://github.com/open-webui/open-webui) and from Claude
Desktop and Cowork. No cloud image APIs and no per-image cost: the models run
on the Mac, and teachers use them from any computer on the home network.

**Version 1.0** is a chat model in Open WebUI:

- Describe a picture and it appears inline with a **Download PNG** link.
- Attach a picture to change it ("turn this into a coloring page").
- Follow-ups change the last picture ("now give it a red hat"); ask for something new to start fresh.
- Ask questions ("what prompt did you use for that?") and get a text answer.
- Every reply has a collapsible **Thought for N seconds** trace: the decision,
  the exact prompt sent to the image model, model, seed, size and timings.
- Options anywhere in the message: `--portrait` `--landscape` `--wide`
  `--tall`, `--n 3`, `--seed 42`, `--transparent`, `--new`, `--raw`. Type `help` for the list.

**Version 2.0** adds **Teach Image**, a Claude Desktop extension (local MCP
server), so Claude chat and Cowork tasks can make pictures on the Mac from
this computer or any other on the network ([docs/MCP.md](docs/MCP.md)):

- Claude picks the style (icon, coloring page, storybook), size and number of
  variations; full-size PNGs are saved on the computer running Claude, and
  Claude sees a preview plus the exact prompt, seed and `image_id`.
- Edit a picture made earlier or a file on the computer; save characters as
  reference pictures to keep them the same across pages.
- Behind it, the image server gained an asynchronous job API with stored
  images, JSON sidecars and references ([docs/API.md](docs/API.md),
  [docs/openapi.json](docs/openapi.json)), an opt-in LAN mode and per-device
  tokens. The 1.0 endpoints are unchanged.

**Status:** working proof of concept. What was tried, measured and decided
(models, runtimes, Open WebUI internals, benchmarks) is in
[docs/LEARNINGS.md](docs/LEARNINGS.md). The next phase, a harness that plans
and illustrates a whole picture book with consistent characters, builds on
this image server in its own project.

## How it works

```
 Teacher's browser (any computer on the LAN)
        │  Open WebUI login, port 3000
        ▼
 ┌─ Docker: Open WebUI ───────────────────────────────────────────┐
 │  "Teach Image" pipe function (openwebui/teach_image_pipe.py)   │
 │   1. director: a local LLM decides draw / reply, picture kind, │
 │      whether to change the previous picture                    │
 │   2. a fixed style template turns that into the image prompt   │
 │   3. results are stored as the teacher's own files             │
 └──────┬─────────────────────────────────────────┬───────────────┘
        │ host.docker.internal:11434              │ host.docker.internal:7865 (bearer token)
        ▼                                         ▼
 Ollama (native)                    ┌─ macOS (launchd): image-server ──────┐
 director: gemma4:12b               │  FastAPI + mflux on Apple MLX (GPU)  │
                                    │  loopback or LAN · one job at a time │
                                    └──────────────────┬───────────────────┘
                                                       ▼
                             FLUX.2 [klein] 4B (8-bit) · Z-Image-Turbo (8-bit)
```

The image model runs natively, not in Docker, because Docker containers on
macOS can't use the Apple GPU. MLX also uses the M5's GPU neural accelerators.
Open WebUI reaches the server through Docker Desktop's `host.docker.internal`,
so for 1.0 alone the server listens only on `127.0.0.1`.

Version 2.0 adds a second way in:

```
 Claude Desktop / Cowork (this Mac or any computer on the LAN)
   └─ Teach Image extension (teach-image-mcp/, local MCP server over stdio)
        │  HTTP on the LAN, port 7865, per-device bearer token
        ▼
 image-server job API: POST /jobs → poll → GET /images/{id} (one GPU queue, shared with Open WebUI)
```

Claude's cloud-side connectors can't reach a home network, so the extension
runs on each computer inside Claude Desktop, which also bridges it into Cowork
([why](docs/MCP.md)). LAN mode is opt-in (`scripts/enable_lan.sh`).

The director only classifies the request. A small model writing image prompts
freely drifted away from what the teacher asked; classification plus fixed
templates keeps prompts faithful. In tests `gemma4:12b` classified all cases
correctly. Smaller models were faster but missed some, so the pipe falls back
to the teacher's own words if the director is unavailable, and `--raw` skips it.

The image server speaks a subset of the OpenAI Images API
(`/v1/images/generations` and multipart `/v1/images/edits`) for Open WebUI,
and an asynchronous job API (`/jobs`, `/images`, `/references`) for the
extension and the future book harness.

## Models

| Model | License | Good for | Measured on an M5 Pro (24 GB), 1024² |
|---|---|---|---|
| **FLUX.2 [klein] 4B**, 8-bit (default) | Apache-2.0 | icons, coloring pages, storybook art; edits with up to 4 reference images | ~10 s per picture warm (up to ~25 s under sustained load); edits ~19 s |
| Z-Image-Turbo, 8-bit | Apache-2.0 | text-to-image only | ~30–45 s |
| Qwen-Image-2.1-Turbo, 4-bit (off by default) | Qwen Research License: research/evaluation only | evaluation | ~37–60 s |
| `gemma4:12b` via Ollama (director) | Apache-2.0 | draw/reply decisions | ~2–6 s per message |

## Memory and standby

A 24 GB Mac gives the GPU ~17.8 GB, so only one big model fits at a time:

- Before each picture, the server asks Ollama to unload its models (the
  director reloads in a few seconds on the next message).
- After 10 idle minutes the server unloads its model, then restarts itself as
  a fresh process (launchd brings it back). Standby is ~80 MB including the uv
  launcher, at ~0% CPU; the first picture afterwards takes a second or two
  longer. If a request arrives during that brief restart, the pipe retries once.
  The restart is skipped (with a warning in the log) if `.env` or the server
  code changed since startup, so an unfinished edit can't take the server down.

## Requirements

The image server runs only on an **Apple Silicon Mac (M1 or later) with macOS 14
or later**: it uses Apple's MLX. Measured on an M5 Pro MacBook Pro with 24 GB:

| Piece | Disk | Memory while working | Notes |
|---|---|---|---|
| Image server + FLUX.2 klein 4B, 8-bit (needed) | 8 GB model | 9–11 GB GPU memory at peak; ~80 MB when idle | ~10 s per 1024² picture, ~19 s per edit, a few s more when cold |
| Z-Image-Turbo, 8-bit (optional) | 10 GB | 12 GB at peak | 30–45 s per picture |
| Qwen-Image-2.1-Turbo, 4-bit (off: research-only license) | 9 GB | 15 GB at peak | 37–61 s per picture |
| Open WebUI in Docker Desktop (optional) | 6 GB image | a few GB with Docker's VM | the chat front end (1.0) |
| Ollama with `gemma4:12b` (optional, for Open WebUI) | 7.6 GB | ~8 GB; unloaded before each picture | the "director" |
| Teach Image extension (each Claude computer) | < 2 MB | ~50 MB | macOS or Windows, inside Claude Desktop |
| Pictures you keep | 0.3–0.6 MB per 1024² PNG | | kept until you delete them |

- **Memory:** 24 GB is what this runs on: the image model, Claude Desktop and
  the Open WebUI pieces fit because the server unloads Ollama's model before
  each picture. 16 GB is untested and tight (klein's peak is close to the
  ~11 GB of GPU memory a 16 GB Mac allows), so plan on the image server alone
  there. 32 GB or more keeps the director and the image model loaded together
  and leaves room for bigger models.
- **Disk:** about 9 GB for the image server with klein; about 45 GB for
  everything above. 256 GB of storage is enough; more leaves room for
  pictures and future models.
- **Speed** has only been measured on the M5 Pro (16-core GPU). The chip
  generation matters more than its size: M5- and M6-family GPUs have neural
  accelerators that make image generation roughly 3–4× faster than M4 and
  older chips, and MLX uses them only on macOS 26.2 or later. Expect an M4
  Mac to be several times slower than the numbers above.
- **Network:** for other computers, a home network the Mac stays on (a
  fixed address or its `.local` name). No internet connection is needed at
  run time; the models download once (`scripts/download_models.sh`).

Other hardware, untested: mflux has an early Linux/NVIDIA path through MLX's
CUDA backend, which might run the server unchanged. Otherwise
`image-server/src/image_server/backends.py` can be ported to diffusers'
`Flux2KleinPipeline` (PyTorch, Apache-2.0; text-to-image and edits in one
class, about 8 GB of GPU memory with FP8 weights). Either way the job API, the
extension and the Open WebUI pipe stay as they are. Hardware and hosting
options are in [docs/LEARNINGS.md](docs/LEARNINGS.md#13-hardware-and-hosting-as-of-oct-2026).

## Setup

Needs the hardware above, [uv](https://docs.astral.sh/uv/), and for the Open
WebUI front end, Ollama with `gemma4:12b` pulled and Open WebUI running in
Docker Desktop.

```bash
scripts/setup_env.sh            # create .env (gitignored) with a random token
scripts/download_models.sh      # ~29 GB into the Hugging Face cache, pinned revisions
scripts/install_image_server.sh # launchd agent: starts at login, restarts on crash
scripts/install_openwebui.sh    # add the model to Open WebUI (backs up its database first)
scripts/doctor.sh               # check every link in the chain
```

Then reload Open WebUI. Set `OWUI_MODEL_NAME` in `.env` to choose the name
shown in the model list.

For Claude Desktop and Cowork (2.0), see [docs/MCP.md](docs/MCP.md):

```bash
scripts/enable_lan.sh               # opt in to LAN access (writes .env only)
scripts/install_image_server.sh     # restart; checks the LAN address and the macOS firewall
scripts/client_tokens.sh add <pc>   # one token per computer, revocable
cd teach-image-mcp && npm ci && npm run build   # → dist/teach-image.mcpb, double-click to install
```

- **After editing `.env`**, re-run `scripts/install_image_server.sh`: it checks the file before restarting the server, which reads it only at startup. If you changed the token, also re-run `scripts/install_openwebui.sh`.
- **After changing the pipe**, re-run `scripts/install_openwebui.sh`. It keeps what an admin changed in Open WebUI (settings, picture, description, access).
- **Access:** the first install lets every existing non-admin account use the model. New accounts don't see it until you re-run the installer with `OWUI_GRANT_ALL=1`, or set `OWUI_PUBLIC=1` in `.env` so every account, including future ones, can use it. Leave `OWUI_PUBLIC` blank to keep whatever is set in Open WebUI.
- The director model and other options are in Admin Panel > Functions > Teach Image (gear icon).

Logs: `~/Library/Logs/teacher-image-harness/image-server.log`.

## Security and privacy

- The image server binds to `127.0.0.1` unless LAN mode is turned on,
  requires a bearer token on every route but `/health` (checked before any
  request body is read), rejects unexpected `Host` headers (DNS rebinding),
  caps upload sizes and image dimensions, and sends no CORS headers.
- In LAN mode it answers only private-network clients (RFC 1918, link-local,
  CGNAT, loopback) and accepts per-device tokens from
  `scripts/client_tokens.sh`, stored only as SHA-256 hashes and revocable
  without a restart; Open WebUI's token works only from the Mac itself. Each
  computer may hold at most 16 connections, idle or stalled connections are
  closed, and a stopping server waits at most 25 s. Details in
  [docs/MCP.md](docs/MCP.md#security).
- The image server makes no internet requests at runtime (model downloads are
  pinned and it runs with `HF_HUB_OFFLINE=1`). Prompts and pictures travel
  between the Mac and the teachers' browsers on your network (plain HTTP, port
  3000) and, in LAN mode, the computers running the Teach Image extension
  (port 7865); don't forward either port to the internet. With Claude, the
  prompts Claude writes and the previews it looks at are part of the Claude
  conversation (Anthropic's cloud); the drawing and the full-size pictures stay
  on your computers.
- The token lives in `.env` (gitignored, mode 600), in Open WebUI's database
  (the function's settings) and in its database backups under
  `~/Library/Application Support/teacher-image-harness/backups` (the newest 3
  are kept; they also hold password hashes and chats).
- A gitleaks pre-commit hook blocks committed secrets once enabled in each
  clone (`git config core.hooksPath .githooks`). The GitHub Action only
  detects secrets after a push; rotate anything it finds.
- PNG metadata records the model and seed, never the prompt.
- Images are stored as the teacher's own Open WebUI files. The image server
  also archives Open WebUI's pictures with their prompts under
  `~/Library/Application Support/teacher-image-harness/outputs/<date>/`
  (`IMAGE_SERVER_SAVE_OUTPUTS=false` turns that off), and keeps every picture
  made through the job API, with a JSON sidecar, in `images/` (clients fetch
  them by id). Folders are mode 700; nothing is deleted automatically, and
  deleting a chat doesn't delete them. The job API's pictures form one library
  shared by every device token; Open WebUI's archive isn't served to them.
- The Teach Image extension saves pictures only inside its pictures folder,
  never overwrites a file, and uploads a local file only when asked to edit it
  or keep it as a reference (pictures only, from the home or pictures folder).
- **Run Open WebUI 0.11.0 or later.** This was built against 0.9.2, which has
  published cross-user file-access advisories: a signed-in user who knows
  another user's file id can read (and in some cases change or delete) that
  file. See [GHSA-vrhc-3fr6-pc3c](https://github.com/open-webui/open-webui/security/advisories/GHSA-vrhc-3fr6-pc3c)
  (CVE-2026-54010, fixed in 0.9.6), [GHSA-r472-mw7m-967f](https://github.com/open-webui/open-webui/security/advisories/GHSA-r472-mw7m-967f)
  (CVE-2026-45402, fixed in 0.9.5) and [GHSA-6xhv-rxhv-pwm4](https://github.com/open-webui/open-webui/security/advisories/GHSA-6xhv-rxhv-pwm4)
  (CVE-2026-70487, fixed in 0.11.0). The installer handles 0.12's API changes;
  re-run `scripts/install_openwebui.sh` and test after upgrading.

## Development

```bash
cd image-server
uv run pytest                                    # server + Open WebUI pipe tests (130)
uv run python ../scripts/bakeoff.py --quick      # compare models on real prompts
cd ../teach-image-mcp
npm test                                         # extension tools against a mock server (104, incl. the bundle)
npm run build                                    # dist/teach-image.mcpb
uv run --project ../image-server python ../scripts/export_openapi.py   # refresh docs/openapi.json
```

The bake-off asks the running server and Ollama to unload first, then writes
PNGs and an HTML contact sheet to `~/Library/Application Support/teacher-image-harness/bakeoff/`.

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md): free for personal, educational
and other noncommercial use. Model weights are downloaded separately and keep
their own licenses (see the table above).
