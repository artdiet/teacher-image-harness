# Teacher Image Harness

Private, local image generation for classroom materials (worksheet icons,
coloring pages, storybook art) running on an Apple Silicon MacBook and used
from [Open WebUI](https://github.com/open-webui/open-webui). No cloud image
APIs and no per-image cost: the models run on the Mac, and teachers use them
from any browser on the home network.

**Version 1.0** is a chat model in Open WebUI:

- Describe a picture and it appears inline with a **Download PNG** link.
- Attach a picture to change it ("turn this into a coloring page").
- Follow-ups change the last picture ("now give it a red hat"); ask for something new to start fresh.
- Ask questions ("what prompt did you use for that?") and get a text answer.
- Every reply has a collapsible **Thought for N seconds** trace: the decision,
  the exact prompt sent to the image model, model, seed, size and timings.
- Options anywhere in the message: `--portrait` `--landscape` `--wide`
  `--tall`, `--n 3`, `--seed 42`, `--transparent`, `--new`, `--raw`. Type `help` for the list.

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
                                    │  127.0.0.1 only · one job at a time  │
                                    └──────────────────┬───────────────────┘
                                                       ▼
                             FLUX.2 [klein] 4B (8-bit) · Z-Image-Turbo (8-bit)
```

The image model runs natively, not in Docker, because Docker containers on
macOS can't use the Apple GPU. MLX also uses the M5's GPU neural accelerators.
Open WebUI reaches the server through Docker Desktop's `host.docker.internal`,
so the server never listens on the network.

The director only classifies the request. A small model writing image prompts
freely drifted away from what the teacher asked; classification plus fixed
templates keeps prompts faithful. In tests `gemma4:12b` classified all cases
correctly. Smaller models were faster but missed some, so the pipe falls back
to the teacher's own words if the director is unavailable, and `--raw` skips it.

The image server speaks a subset of the OpenAI Images API
(`/v1/images/generations` and multipart `/v1/images/edits`), so other tools
and the future 2.0 harness can use it too.

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

## Setup

Needs an Apple Silicon Mac (24 GB or more), [uv](https://docs.astral.sh/uv/),
Ollama with `gemma4:12b` pulled, and Open WebUI running in Docker Desktop.

```bash
scripts/setup_env.sh            # create .env (gitignored) with a random token
scripts/download_models.sh      # ~29 GB into the Hugging Face cache, pinned revisions
scripts/install_image_server.sh # launchd agent: starts at login, restarts on crash
scripts/install_openwebui.sh    # add the model to Open WebUI (backs up its database first)
scripts/doctor.sh               # check every link in the chain
```

Then reload Open WebUI. Set `OWUI_MODEL_NAME` in `.env` to choose the name
shown in the model list.

- **After editing `.env`**, re-run `scripts/install_image_server.sh`: it checks the file before restarting the server, which reads it only at startup. If you changed the token, also re-run `scripts/install_openwebui.sh`.
- **After changing the pipe**, re-run `scripts/install_openwebui.sh`. It keeps what an admin changed in Open WebUI (settings, picture, description, access).
- **Access:** the first install lets every existing non-admin account use the model. New accounts don't see it until you re-run the installer with `OWUI_GRANT_ALL=1`, or set `OWUI_PUBLIC=1` in `.env` so every account, including future ones, can use it. Leave `OWUI_PUBLIC` blank to keep whatever is set in Open WebUI.
- The director model and other options are in Admin Panel > Functions > Teach Image (gear icon).

Logs: `~/Library/Logs/teacher-image-harness/image-server.log`.

## Security and privacy

- The image server binds to `127.0.0.1`, requires a bearer token (checked
  before any request body is read), rejects unexpected `Host` headers (DNS
  rebinding), caps upload sizes and image dimensions, and sends no CORS headers.
- The image server makes no internet requests at runtime (model downloads are
  pinned and it runs with `HF_HUB_OFFLINE=1`). Prompts and pictures travel only
  between the Mac and the teachers' browsers on your network, over plain HTTP
  on port 3000; don't forward that port to the internet.
- The token lives in `.env` (gitignored, mode 600), in Open WebUI's database
  (the function's settings) and in its database backups under
  `~/Library/Application Support/teacher-image-harness/backups` (the newest 3
  are kept; they also hold password hashes and chats).
- A gitleaks pre-commit hook blocks committed secrets once enabled in each
  clone (`git config core.hooksPath .githooks`). The GitHub Action only
  detects secrets after a push; rotate anything it finds.
- PNG metadata records the model and seed, never the prompt.
- Images are stored as the teacher's own Open WebUI files. The image server
  also archives every picture with its prompt under
  `~/Library/Application Support/teacher-image-harness/outputs/` (folder mode
  700) until you delete them; set `IMAGE_SERVER_SAVE_OUTPUTS=false` to turn
  that off. Deleting a chat doesn't delete the archive.
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
uv run pytest                                    # server + Open WebUI pipe tests (79)
uv run python ../scripts/bakeoff.py --quick      # compare models on real prompts
```

The bake-off asks the running server and Ollama to unload first, then writes
PNGs and an HTML contact sheet to `~/Library/Application Support/teacher-image-harness/bakeoff/`.

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md): free for personal, educational
and other noncommercial use. Model weights are downloaded separately and keep
their own licenses (see the table above).
