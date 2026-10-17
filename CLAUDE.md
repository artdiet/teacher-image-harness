# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## What this is

A working proof of concept (Oct 2026): local image generation for a teacher's
classroom materials on an Apple Silicon Mac (M5 Pro, 24 GB), used from Open
WebUI. Two parts:

- `image-server/`: FastAPI + [mflux](https://github.com/mflux-community/mflux)
  (Apple MLX), runs **natively** under launchd on `127.0.0.1:7865`, bearer
  token, OpenAI-compatible `/v1/images/generations` and `/v1/images/edits`.
- `openwebui/teach_image_pipe.py`: an Open WebUI 0.9.2 "pipe" function (runs
  inside the Open WebUI Docker container). A director LLM (`gemma4:12b` on the
  Mac's Ollama) classifies each message (draw/reply, kind, edit previous); fixed
  templates build the image prompt; results are stored as the user's Open WebUI
  files; the reply streams a `<think>` trace.

The next phase (a whole-book harness with its own server and UI) is planned
for a separate repo that builds on this one. Read `docs/LEARNINGS.md` before
designing anything new: it records what was tried, measured and decided.

## Commands

```bash
cd image-server && uv run pytest -q              # 79 tests: server + pipe (fake backend, no GPU)
scripts/doctor.sh                                # health of the whole chain (prints no secrets)
scripts/install_image_server.sh                  # (re)start the launchd agent; validates .env first
scripts/install_openwebui.sh                     # install/update pipe + model entry (backs up webui.db)
scripts/download_models.sh                       # pinned MLX weights into the HF cache (~29 GB)
cd image-server && uv run python ../scripts/bakeoff.py --quick   # model comparison on real prompts
```

Logs: `~/Library/Logs/teacher-image-harness/image-server.log`. Data (outputs,
backups, bake-offs): `~/Library/Application Support/teacher-image-harness/`.

After changing server code or `.env`: re-run `install_image_server.sh`. After
changing the pipe: re-run `install_openwebui.sh` (idempotent; keeps admin edits).

## Rules

- **Public repo.** Never commit secrets, `.env`, model weights, generated
  images, database backups, personal names, emails, LAN addresses or other
  machine details. The teacher-facing model name lives only in `.env`
  (`OWUI_MODEL_NAME`). A gitleaks pre-commit hook (`core.hooksPath=.githooks`)
  and a CI scan back this up; the hook needs `brew install gitleaks`.
- `user_notes.md` (gitignored) is the user's async channel: answer each note
  inline beneath it as `> **Claude:** …` and put your own questions there.
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

## Code conventions

Python 3.12, uv, type hints, small modules, comments explain *why*. Server
routes never declare body parameters (FastAPI would read the body before the
auth dependency). Tests use a fake backend and in-memory Open WebUI stand-ins;
add a test with every behavior change.
