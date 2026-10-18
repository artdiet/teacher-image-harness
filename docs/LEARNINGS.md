# Learnings: local image generation for classroom materials

What this proof of concept tried, measured and decided (October 2026), written
so the next project, a whole-book illustration harness, can start from facts
instead of assumptions. Versions and dates matter: this ecosystem changes
monthly, so re-check anything marked *as of* before relying on it.

**Verdict:** a 24 GB Apple Silicon MacBook can host a genuinely useful local
image model. The winner was **FLUX.2 [klein] 4B (8-bit) via mflux**: ~10 s per
1024² picture warm, ~19 s for an edit with a reference picture, about 11 GB at
peak, Apache-2.0. It produces exactly the flat, bold-outline worksheet style
teachers need, and good storybook art. A local LLM director (`gemma4:12b`)
turns it into a chat model that can also answer questions.

---

## 1. Platform constraints (the facts that shaped everything)

| Fact (as of Oct 2026) | Consequence |
|---|---|
| **Docker containers on macOS cannot use the Apple GPU** (Docker Desktop GPU passthrough is Windows/NVIDIA only; Apple's `container` CLI 1.5 has no GPU; Podman/krunkit Vulkan works only for llama.cpp, at roughly 50–80% of native speed). CPU-only diffusion would take minutes per image. | The image model runs **natively** (launchd). Only the UI lives in Docker. |
| Docker Desktop's `host.docker.internal` reaches services bound to the Mac's **127.0.0.1** (verified). | Native services never need to listen on the network. |
| **Ollama removed image generation** in v0.32.6 (2026-08-04) after an experimental run from v0.14 (Jan 2026, z-image-turbo and flux2-klein only). Maintainers say it will return on the new MLX runner, no date. 0.40.x returns 404 on `/v1/images/*`. | Ollama is only the LLM (director). The image server has an adapter seam in case Ollama gains images again. |
| LM Studio: no image generation. Docker Model Runner: a diffusers engine exists on macOS in code, but it is pinned to an old diffusers (no Qwen-Image-2.1), does text-to-image only, and a contributor says DMR is being deprecated in favour of `llmman`. | Not usable as the foundation. |
| **24 GB Mac → ~17.76 GiB Metal working set** (`recommendedMaxWorkingSetSize`, `iogpu.wired_limit_mb=0`). | **One big model at a time.** No pair of the candidate image models fits together (weights alone: klein 8.0 + Z-Image 10.2 GiB = 18.2 GiB), nor an image model next to a 12B LLM. |
| The M5 GPU has per-core **Neural Accelerators**; MLX uses them on macOS 26.2+ (Apple: FLUX ≥3.8× faster on M5 than M4). PyTorch MPS only partially (2.14 prefill attention); stable-diffusion.cpp's Metal tensor path had an open M5 blank-image bug. | **MLX-based runtimes win on M5.** |

## 2. Models: landscape, licenses, measurements

### Candidates that fit 24 GB (as of Oct 2026)

| Model | Params (incl. text encoder) | License | Notes |
|---|---|---|---|
| **FLUX.2 [klein] 4B** | 3.9B + Qwen3-4B | Apache-2.0 | 4 distilled steps; text-to-image **and** editing with up to 4 references; LoRA-trainable. mflux q8 = 8.6 GB. **Chosen.** |
| Z-Image-Turbo | 6.2B + Qwen3-4B | Apache-2.0 | 9 steps; strong text-to-image; **no editing** (Z-Image-Edit unreleased). q8 = 11 GB. |
| Qwen-Image-2.1 / -Turbo | 7.1B DiT + **8.8B** Qwen3-VL encoder (≈33 GB bf16, so "8B" is misleading) | **Qwen Research License: research or evaluation only** | Top open-weights quality, native **RGBA**, **up to 10 references**, text rendering. Needs ~4-bit on 24 GB (q4 = 10 GB). Earlier Qwen-Image releases were Apache-2.0; 2.1 is not. |
| Ming-Image-0.1-Design | 6.2B + ~16B MoE encoder | MIT | Native RGBA, strong design/typography; slow (12 steps); not benchmarked here. |
| ERNIE-Image-Turbo | 8B | Apache-2.0 | Text-to-image only; not benchmarked here. |
| FLUX 3 Image Klein (2026-10-15) | 5.6B | FLUX Kommunity (commercial outputs only under US$5M revenue; terminable on notice; gated) | 10 references, strong. mflux peaks 18.5–22.8 GB, **over** the 24 GB budget today; also wants a 9.4B prompt upsampler. Watch for 2.0. |

Avoid for this use: FLUX.2 klein 9B, FLUX.2 dev, FLUX.1 dev/Kontext. Their
non-commercial licenses exclude use "connected to … employment
responsibilities", which arguably covers a teacher's own job. Chroma has no
safety alignment, HunyuanImage is territory-restricted, and the 20B Qwen-Image
line is too heavy.

Other licensing traps:

- **rembg's default background model is RMBG-2.0, which is non-commercial.**
  Select `birefnet-general` (MIT), or flood-fill a white background, as this
  repo does.
- Third-party quantized checkpoints can mislabel their license.
- Teachers Pay Teachers announced (Oct 2026) a coming ban on AI-generated
  resources. That matters for anything sold there.

### Bake-off (this Mac, 1024², same prompts and seed)

Prompts: three worksheet icons (apple, bus, sun), a coloring page (turtle), two
storybook scenes, plus an edit ("same style, but a banana") from the apple.

| Model | s / image (sustained run) | Warm, fresh process | Peak GPU memory | Edit |
|---|---|---|---|---|
| **FLUX.2 klein 4B q8** | 10–25 s | **~9–11 s** | 9–11 GB | 34 s sustained / **~19 s** warm (1 ref) |
| Z-Image-Turbo q8 | 31–48 s | — | 11.8 GB | n/a |
| Qwen-Image-2.1-Turbo q4 | 37–61 s | — | 14.2 GB | 61 s / 15.0 GB |

Times rose during long runs (sustained load). Quality on the classroom styles:

- **klein:** crisp flat icons with bold outlines and charm, clean thick-line
  coloring pages, lovely watercolor storybook art. The edit kept the style
  exactly: an apple icon became a matching banana icon, and an owl icon became
  the same owl as a coloring page.
- **Z-Image:** chunkier, bolder icons that also work well. Its coloring page
  picked up gray shading despite "no gray". It's 3–4× slower and can't edit.
- **Qwen-2.1:** clean and comparable. Its coloring lines are thinner, which
  makes them less kid-friendly. Excellent edits, but the slowest, the most
  memory, and a license problem.

**For the full harness, compare per task.** klein is the default for speed,
editing and consistent icon sets. Qwen-2.1 has the strongest multi-reference
and RGBA features if licensing allows. Z-Image is an alternative look for
single illustrations. FLUX 3 klein is the upgrade to watch once its memory use
fits.

## 3. Runtimes compared (model server role)

| Runtime | Verdict | Why |
|---|---|---|
| **mflux** (MLX, MIT) | **Chosen** | Fastest scriptable path on M5; supports klein, Z-Image, Qwen-2.1 (10 refs, RGBA), FLUX 3 klein; clean Python API. No HTTP server, so this repo wrote a ~300-line FastAPI wrapper. Releases weekly: pin versions. |
| stable-diffusion.cpp `sd-server` | Good no-code fallback | OpenAI-style `/v1/images/*` out of the box, GGUF (smallest memory). But no auth, echoes any CORS origin, one model per process, slower than MLX, and had an M5 blank-image bug. |
| ComfyUI | Good for complex multi-pass graphs | Full API and day-0 model support, but PyTorch MPS is slower (enable `--use-pytorch-cross-attention`), int8 has no speedup on MPS, and the API is workflow-JSON. Worth revisiting for 2.0 pipelines. |
| Draw Things | Great manual tool | Probably the fastest engine, but its HTTP API lives only inside the GUI app: no auth, a single init image. Its headless server is gRPC. |
| LocalAI, Rapid-MLX, koboldcpp, InvokeAI, diffusers+MPS | Not chosen | Large surfaces or insecure defaults; Qwen-2.1 edits single-image only or unreleased; diffusers bf16 doesn't fit 24 GB. |

## 4. mflux practical notes (0.23.0, mlx 0.32.4)

- **Pin `pillow>=12.3,<13`** on Python 3.12: Pillow 13 uses PEP 810 lazy
  imports, which break `import mflux`.
- **Load from the pinned snapshot directory** (`~/.cache/huggingface/hub/models--org--name/snapshots/<rev>`)
  with an **explicit `ModelConfig`** and `quantize=None` (pre-quantized weights
  carry their level). Without `model_path`, mflux silently falls back to the
  original bf16 repo, a 15–33 GB download. Set `HF_HUB_OFFLINE=1` before
  anything imports `huggingface_hub`.
- **FLUX.2 klein:** `Flux2Klein` does text-to-image; `Flux2KleinEdit` handles
  references and crashes without them. Loading both would double memory, so
  build the edit object as a view over the loaded model:
  `Flux2KleinEdit.__new__`, `nn.Module.__init__`, then `dict.update(edit, t2i)`
  and `edit.__dict__.update(t2i.__dict__)`. Don't use `Module.update()`, which
  expects a parameter tree. Guidance stays 1.0 (distilled), 4 steps.
  References are PIL images or paths, each scaled to ≤1 MP; output size comes
  only from width/height.
- **Z-Image-Turbo:** pass `num_inference_steps=9` (the API default of 4 is too
  few). Guidance and negative prompts are ignored.
- **Qwen-Image-2.1-Turbo:** use `QwenImage21Edit` for everything; it's the only
  class that returns RGBA. Pass `ModelConfig.qwen_image_21_turbo()` explicitly
  (the class default is the base model, and the repo name infers 2512).
  Width and height must be multiples of 32 (it raises, it doesn't round),
  guidance exactly 1, and `output_resolution=512` for references to stay under
  24 GB. Clear `prompt_cache` after each call, since it grows without bound.
  Transparency is requested in the prompt ("This is an RGBA image with transparency…").
- **Threading:** run every MLX call on **one long-lived thread**. MLX ≥0.31.2
  binds streams to threads.
- **Memory:** to free it, drop every reference, then `gc.collect()`, then
  `mx.clear_cache()` (`mx.metal.clear_cache` is deprecated). Cap the cache with
  `mx.set_cache_limit(2 GB)`. On a failure, `traceback.clear_frames()` first,
  because failed frames still hold the model. Weights load lazily, and
  `mx.compile` recompiles for each new size, so the first image at a size is
  slower.
- **Cancel / progress:** `model.callbacks.register(obj)`. mflux calls
  `obj.call_in_loop(t=…, seed=…, latents=…, config=…, …)` with keywords at each
  step, and any exception except KeyboardInterrupt stops generation cleanly.
  There is no unregister, so register one hook per loaded model. This repo uses
  it to stop jobs whose client left; the same hook can stream step progress.
- `TQDM_DISABLE=1` keeps logs clean; mflux still reads the elapsed time.

## 5. Image server design (what worked)

- **OpenAI Images API subset:** `/v1/images/generations` (JSON) and
  `/v1/images/edits` (multipart, `image` or `image[]`). Unknown model names
  (e.g. `dall-e-2` from Open WebUI's built-in engine) fall back to the
  defaults. Responses include seed, seconds and the real width/height.
- **One MLX worker thread, one resident model, FIFO turns.** Ollama is freed
  right before each job (not at enqueue). Jobs check a cancel flag between
  images and steps; a disconnect watcher sets it.
- **Standby:** unload after 10 idle minutes, then SIGTERM self; launchd
  KeepAlive restarts a fresh process (~50 MB, measured live) instead of ~400 MB
  of imported libraries. The restart is skipped if a fingerprint of `.env`,
  `uv.lock` and the sources changed since startup, so a half-done edit can't
  crash-loop it. The first picture after standby costs ~1–2 s extra.
- **Security for a loopback service:**
  - token checked **before any body is read**: routes declare no body
    parameters, because FastAPI reads bodies before dependencies run
  - `TrustedHostMiddleware` against DNS rebinding, and no CORS
  - capped JSON and multipart bodies (including chunked)
  - image decoding with a format allowlist, a 50 MP decompression-bomb cap,
    tRNS / 16-bit handling, and best-effort EXIF
  - PNG metadata records model and seed, never the prompt
  - outputs archived to a 0700 folder
- **Edits keep the reference's aspect ratio** unless a size is given (as in
  OpenAI's API). Phone photos are rarely square.

## 6. Open WebUI 0.9.2 integration (internals that weren't obvious)

**How a pipe model appears**

- A single pipe's model id is the function id, and its name is the function's
  DB name. The model is **invisible to `user`-role accounts** until a `model`
  row with that id exists and has a read **access grant**. In 0.9.2,
  `access_control` is silently ignored: use `access_grants`. A row with
  `base_model_id=None` overrides name, capabilities and avatar, the same way
  admins override Ollama models.
- Image attachments need both `capabilities.vision` and `file_upload`. Set
  `file_context: false` (no RAG on attachments) and `builtin_tools: false`.

**What a pipe receives**

- Attached images arrive in `body["messages"][-1]["content"]` as
  `image_url` parts with `data:` URLs, and earlier user uploads are re-sent
  every turn.
- Assistant history comes back as text, including earlier reasoning blocks in
  their original `<think>…</think>` tags (`raw=True`). That's how the director
  can answer "what prompt did you use?". `__files__` never contains images.
- **Background tasks call the pipe itself** (title, tags, follow-ups) with
  `__task__` set, when no external task model is configured. Answer them with
  tiny JSON, never by drawing.

**Returning results**

- Store PNGs with `open_webui.routers.files.upload_file_handler(..., process=False, process_in_background=False, user=UserModel)`,
  then `Chats.insert_chat_files(...)`, and return markdown
  `![alt](/api/v1/files/{id}/content)` plus
  `[Download PNG](…/content?attachment=true)`.
  - These work for the owner over the session cookie, from other machines.
  - Never inline base64: it's stored about 4 times per image and re-sent every
    turn.
  - Raw HTML such as `<a download>` is printed as text.
- A pipe can return an **async generator**; Open WebUI streams it.
  `<think>` tags become a collapsed "Thought for N seconds" block, but **only
  in streaming mode**. Non-streaming shows them raw, so check `body["stream"]`.
  Defuse model-written text: a `<thinking>` inside a reply opens a reasoning
  block, and a chunk starting with `data:` is treated as a raw SSE line.
- Status events (`{"type":"status","data":{description,done,hidden}}`) are
  persisted. Close them on every exit, including Stop (`CancelledError`).
- `ConfigVar` settings (formerly PersistentConfig) ignore environment
  variables after the first launch; configure through the admin UI or DB.

**Installing without API keys**

API keys were off on the target instance, so `scripts/owui_install.py` runs
**inside the container** with Open WebUI's own async data layer:

- `WEBUI_SECRET_KEY` read from the container's key file, and
  `ENABLE_DB_MIGRATIONS=False`
- a sqlite-backup of `webui.db` before every write
- validation through `load_function_module_by_id` under a probe id
- idempotent re-runs that keep admin edits (fingerprints of the values it
  wrote), store valves only as overrides, and add grants only on first install

**Moving to 0.12.x**

- `load_function_module_by_id` returns 4 values (handled).
- `upload_image` returns a dict.
- Native function calling adds an `<attached_files>` block to user messages
  (stripped).
- A new `context_compaction` task exists.
- **Security:** 0.9.2 is affected by three published cross-user file-access
  advisories: [GHSA-vrhc-3fr6-pc3c](https://github.com/open-webui/open-webui/security/advisories/GHSA-vrhc-3fr6-pc3c)
  (CVE-2026-54010, CVSS 8.3, fixed in 0.9.6), [GHSA-r472-mw7m-967f](https://github.com/open-webui/open-webui/security/advisories/GHSA-r472-mw7m-967f)
  (CVE-2026-45402, CVSS 8.1, fixed in 0.9.5) and [GHSA-6xhv-rxhv-pwm4](https://github.com/open-webui/open-webui/security/advisories/GHSA-6xhv-rxhv-pwm4)
  (CVE-2026-70487, CVSS 5.3 GitHub / 6.5 NVD, fixed in 0.11.0). Build the next
  project against 0.11.0 or later; 0.12.3 is current (as of 2026-10-17).
- The generated secret key lives in the container layer, so recreating the
  container signs everyone out.

## 7. The director (LLM in the loop)

- **Free-form prompt rewriting by small models drifted badly.**
  `ministral-3:3b` turned "a cute owl for my reading worksheet" into a
  watercolor branch scene, gave a coloring page pastel fills, and answered an
  attached "turn this into a coloring page" with a text reply.
- **Classification plus templates fixed it.** The director returns
  `{action: draw|reply, kind: icon|coloring|storybook|other, edit_previous, subject, reply}`
  through Ollama structured output (`format` = JSON schema, `think: false`,
  `temperature: 0`). Code builds the prompt from fixed style templates. Hard
  rules override the model: an attached picture always means draw; options
  mean draw unless the teacher is asking about them.
- **Scores on 7 test cases:** `gemma4:12b` 7/7 (≈2–3 s warm, ~6 s cold),
  `qwen2.5:3b` 5/7, `ministral-3:3b` 4/7. Default: `gemma4:12b`
  (Apache-2.0, vision-capable).
- **Memory interplay:** the director (~8 GB) is unloaded before each picture
  and reloads in a few seconds on the next message. Acceptable; a 3B director
  would stay resident but decides worse.
- Show the trace by default during testing. It made every decision auditable
  and was the user's own request.

## 8. Prompting that works for teachers (FLUX.2 klein)

| Kind | Template |
|---|---|
| Worksheet icon | `{subject}, flat vector icon, bold clean black outlines, simple shapes, solid bright colors, centered, plain white background, no text` |
| Coloring page | `{subject}, coloring book page for kids, black and white line art, thick clean outlines, no shading, no gray, plain white background, no text` |
| Storybook | `children's picture book illustration: {subject}, warm gentle colors, soft watercolor style, no text` (style only if none named; without "no text", tall pages came back with made-up poster titles) |

Keep words out of pictures and add them in the worksheet tool. An edit with
the previous picture as reference ("same style, but a banana") keeps an icon
set consistent. Transparent icons come from generating on white and then
flood-filling the white background from the border; interior whites survive.

## 9. Operations

- launchd agent: `KeepAlive`, `ProcessType=Interactive` (no background
  throttling), `ThrottleInterval=15`, logs under `~/Library/Logs/…`. The
  installer validates `.env` with the server's own rules **before** stopping
  the running agent, waits for bootout, then health-polls.
- `scripts/doctor.sh` checks:
  - the agent, the loopback-only bind (in LAN mode: that the LAN address
    answers through the firewall), and that Ollama is running but not exposed
  - the Open WebUI container
  - the pipe's stored token against the server (from inside the container)
  - the director model being pulled
- Teachers reach Open WebUI at `http://<mac-ip>:3000` on the LAN. The Mac must
  be awake and Docker Desktop running; Docker doesn't start at login by
  default.

## 10. Lessons from two review rounds (56 confirmed issues fixed)

Multi-agent reviews, with every finding adversarially verified, caught things
tests and manual runs didn't:

- **FastAPI reads a body parameter before the auth dependency runs.** That
  allowed unauthenticated memory exhaustion until the routes read bodies
  themselves.
- **Disconnected clients don't cancel FastAPI handlers.** Stop in Open WebUI
  left orphan GPU jobs until a disconnect watcher, a cancel flag and an mflux
  step hook were added.
- Freeing Ollama "before a job" must mean right before it runs, not at
  enqueue time.
- Self-restart re-reads `.env`, so guard it or a bad edit becomes a crash loop.
- Installer "preserve admin edits" logic was correct for one re-run and wrong
  on the second; test idempotency with two consecutive runs.
- Small things break real use:
  - square output for a portrait photo
  - a garbage EXIF block causing a 500
  - a trailing backslash breaking markdown
  - the status line left spinning after early exits
  - documentation that overstated security

## 11. Claude Desktop and Cowork via MCP (2.0)

Goal: let Claude chat and Cowork tasks use the Mac's image model, from this
Mac or another computer on the home network.

### Facts that decided the design (as of Oct 2026, Claude Desktop 2.42)

| Fact | Consequence |
|---|---|
| Custom connectors (Settings > Connectors) and plugin `http`/`sse` MCP servers are connected from Anthropic's cloud, over HTTPS. | They can't reach a home LAN without a public tunnel, which would expose the server. Rejected. |
| Cowork tasks run in Anthropic's cloud, but local stdio servers (desktop extensions, `claude_desktop_config.json`) run on the user's computer, and Cowork reaches them through the open Desktop app (`[localMcpBridge]` in `main.log`). | A small local MCP server on each computer, packaged as a `.mcpb` extension. Claude Desktop must stay open while a Cowork task uses it. |
| Plugins can't carry secrets: `${VAR}` in a plugin's MCP config stays literal, and Cowork doesn't prompt for plugin settings. | The token goes in the extension's settings page (`sensitive`), not in a plugin. |
| Local tool calls time out at about 60 s, and results over about 150k characters are rejected. `outputSchema`, `structuredContent` and embedded resources have open client bugs. | Calls wait up to 40 s, then hand back a job id. Previews share a ~75 KB budget. Results are plain text plus image blocks. |
| The model may not see an image result until its next turn. | The text carries the saved path, `image_id`, seed and the exact prompt. |
| Claude Desktop runs `.mcpb` Node servers on its bundled Node (v24 in 2.42). | One esbuild bundle (1.3 MB), nothing to install on clients, Windows included. |
| macOS clients need Privacy & Security > Local Network permission for Claude. | It's the first troubleshooting step. |

The design is a thin proxy and a fat server:

- The GPU server stays the single queue (shared with Open WebUI) and owns the
  previews; the extension has no native image libraries.
- The extension is stateless apart from a short memory of where each job's
  files go. It saves full PNGs on the client, never overwriting, and uploads
  local files only when asked.

### Measured through the extension (live server, FLUX.2 klein)

| Call | Time | Result size |
|---|---|---|
| icon 1024², transparent, model cold | 12.1 s (9.3 s drawing) | 52k characters (38 KB preview) |
| 4 coloring pages, 1024² | 41.2 s | 93k characters (4 previews, 15–18 KB each) |
| edit of a local file (uploaded) | 23.2 s | 39k characters |
| storybook 1152×896 with one reference | 19.0 s | 69k characters |
| 2 storybook pages, 768×1344 | 20.3 s | 83k characters |

Four storybook pages would pass 40 s, which is why the job-id path exists.
The reference kept the owl's look (colors, scarf) in a new watercolor scene.

### Lessons

- **The macOS firewall misreports.** The uv-managed Python is ad-hoc signed.
  - Until it is allowed, the firewall completes the TCP handshake and then
    holds the connection, so clients time out without an error.
  - `socketfilterfw --getappblocked` says "permitted" for programs it has never
    prompted about.
  - A request from the Mac to its own LAN IP goes through the firewall like a
    remote one, so it's a reliable self-test (`scripts/check_lan.sh`). Apple's
    `/usr/bin/python3`, which is on the allow list, answered on the same IP.
  - The Mac resolves its own `<name>.local` to 127.0.0.1, so testing that name
    on the Mac proves nothing.
  - The approval is tied to the Python build: redo it after uv upgrades
    Python.
- **macOS `/bin/bash` 3.2** brace-expands JSON with commas inside
  `"$( … -d "{…}" )"`, splitting it into separate words: the curl test's 422s
  were the test's fault. Build JSON bodies in a variable first.
- **"no text" belongs in the storybook template too.** Without it, a tall
  storybook page came back with made-up poster lettering.
- **LAN hardening in layers:**
  - a `Host` allowlist (DNS rebinding)
  - a private-client filter (RFC 1918, link-local, CGNAT, loopback)
  - per-device token hashes, reloaded when the file changes and compared in
    constant time across all of them
  - opt-in binding to all interfaces, so the 1.0 setup stays loopback-only

### First real use (Claude cloud session through the Desktop app)

A Claude session in the cloud, linked to this Mac through the Desktop app,
made a kindergarten counting set: 8 transparent icons in one folder, no
errors. What it showed:

- **Getting the tools to appear takes three things:**
  - Settings filled in: an extension installed before its required token was
    entered is not switched on ("missing required configuration, not enabling
    automatically" in `main.log`).
  - A full restart of Claude Desktop.
  - The person choosing when Claude may use the tools, through a link Claude
    shows. Until then the device bridge announced "an empty local-MCP group"
    and the session saw no tools (`localMcpServers = []`). The extension only
    started when that choice was made; the next request found its 10 tools, as
    `mcp__remote-devices__Teach_Image__<tool>`.
- **Parallel tool calls run one at a time.** Seven calls issued together
  reached the server 12.5 s apart with nothing queued, so the bridge
  serializes them; one GPU would anyway.
- **Where the time goes, per icon:**
  - 9.4 s of drawing (warm) on the server.
  - 0.4–0.7 s from "finished on the server" to "saved on the Mac" (polling,
    download, write).
  - About 2.9 s from one call saving its file to the next job arriving: the
    result going back to the cloud session and the next call coming in.

  Faster polling would gain little; fewer, bigger calls would save the
  bridge hops. The first, cold call took 17 s (model load).

### What the 2.0 review found

Four parallel reviewers (security, server concurrency, extension, docs)
reproduced about 25 issues that tests and live runs had missed:

- **Host names are case-insensitive, Starlette's check isn't.** macOS
  LocalHostNames are mixed case, Node's fetch lowercases them, so every
  `<name>.local` client got "Invalid host header". The check now lowercases
  both sides.
- **uvicorn has no header timeout.** Its only guard, the global
  `limit_concurrency`, let any LAN device take the server offline by holding
  16 idle sockets. Now a subclass of its h11 protocol caps connections per
  client and closes connections that don't send headers within 10 s; it is
  tested under a real uvicorn.
- **Accepted jobs vanished on restart.** uvicorn waits for open connections,
  not background tasks, so a reinstall, logout or even the idle restart (a job
  arriving during the graceful shutdown) left job ids that answered 404. Now:
  - records are written on accept
  - shutdown gives running jobs 25 s to finish, then records the rest as
    interrupted (launchd's `ExitTimeOut` is raised to 60 s)
  - startup marks leftovers from a crash
  - new jobs get a "restarting" 503 while draining
- **Capacity checks must reserve.** Checking "queue full?" in the request
  and counting the job later let a burst get `202` and fail afterwards.
  Cancelling a queued job also waited behind the jobs ahead of it; a task
  waiting on an `asyncio.Lock` can be cancelled safely, a running executor job
  can't.
- **Fail closed:** a damaged token file used to keep a revoked token valid.
  Open WebUI's primary token is now accepted only from loopback (Docker's
  `host.docker.internal` arrives as 127.0.0.1).
- **Don't serve one front end's archive to another's token.** Archiving Open
  WebUI's pictures in the job API's store let any device token read other
  users' prompts; they went back to `outputs/<date>/`.
- **Decode big uploads small.** JPEG `draft()` plus an early integer
  `reduce()` (before colour conversion and EXIF rotation, which copy the whole
  image) cut a 50 MP RGBA upload from about 1 GB to the size of one decoded
  copy; two decodes at most run at once.
- **Be exact about privacy.** "Nothing leaves your network" was wrong for the
  Claude path: prompts and previews are part of the Claude conversation.
- **The extension must never lose a job id:**
  - one dropped poll used to fail the call after the job was accepted
  - the ~60 s limit has to cover uploads and downloads, not just waiting
  - a client timeout must not cancel a picture that is about to finish

A second find-then-refute round on the fixes (one finder and one skeptic per
area) confirmed 17 more, mostly about shutdown:

- **uvicorn's graceful shutdown has no default limit.** On SIGTERM it closes
  the listener first, then waits for every open request before the app's
  shutdown runs. One client stalled mid-upload (a laptop going to sleep) kept a
  stopping server alive forever, refusing everyone, and launchd's `KeepAlive`
  never restarted it. Now the wait is bounded, and a regression test sends
  SIGTERM to a real server process with a stalled client attached. The bound
  must not be shorter than real work: a first 10 s limit cut off Open WebUI
  pictures mid-drawing while background jobs got 25 s, so a stop now starts
  one shared 25 s clock in uvicorn's `handle_exit`.
- **A header deadline isn't enough.** Pipelining without reading, or a body
  that stops, also pins a connection, so a watchdog now covers stalled bodies
  (30 s) and stalled readers (60 s), and TCP keepalive notices vanished peers.
  `transport.close()` waits to flush data a stalled client never reads; use
  `abort()`. Judge a reader by whether the send buffer drains, not by
  `pause_writing`: uvicorn hands an ordinary response to the socket in one
  write and calls it complete at once.
- **Cancelling a request mid-picture** used to release the GPU queue while the
  MLX thread kept drawing. Now the job is stopped at its next step and keeps
  its turn until then.
- **Interrupted-job recovery needs an index:** a time window missed jobs after
  a long power-off. Empty marker files per unfinished job make it exact and
  cheap.
- **Scripts:**
  - The installer must wait out launchd's `ExitTimeOut`, or `bootstrap` fails
    and the server stays down.
  - A LAN self-test must tell "TCP connects but is held" (firewall) from
    "nothing listening" (`curl -w %{time_connect}`).
  - Take the interface from the default route, not just en0/en1 (USB and
    Thunderbolt Ethernet).

## 12. Recommendations for the book-building harness

1. **Reuse the image server instead of rewriting it.** Extract `image-server/`
   as a package (pip-installable from git) or vendor it with attribution. Its
   OpenAI-style API, memory discipline, cancellation and security are already
   paid for. The 2.0 job API already has step progress (mflux
   `call_in_loop`), cancellation, stored images with sidecars and references.
   Still to add:
   - batch jobs
   - a model-per-pass scheduler
2. **Plan memory by pass, not by request.** Run all LLM planning first (story,
   pages, character sheet, prompts), unload, then all image passes grouped by
   model. Swaps cost seconds; ping-ponging between an LLM and an image model
   costs minutes per book.
3. **Character consistency comes from references:**
   - klein: up to 4 references, Apache-2.0
   - Qwen-2.1: up to 10 references plus RGBA, if the license is acceptable
     for the use
   - FLUX 3 klein: 10 references, once it fits 24 GB
   Keep a per-book character library (front, side and expression sheets) and
   pass the relevant sheets with every page prompt. Consider training a klein
   LoRA per recurring character; check the memory needed on 24 GB first.
4. **Build model comparison into the harness.** Generalize
   `scripts/bakeoff.py`: same page prompts across models, timings and peak
   memory recorded, a human picks in a contact-sheet UI, and choices are stored
   per task (icons, scenes, covers).
5. **Provenance per page:** model, revision, seed, prompt, references, steps
   and size, so any page can be regenerated or tweaked exactly.
6. **Keep the director pattern:** an LLM classifies or plans with structured
   output; code templates write the image prompts. Story planning may want a
   bigger model (`gpt-oss:20b`), which is fine as long as it runs in its own
   pass before images.
7. **Licensing for a commercial future:** stay on Apache-2.0/MIT image models
   (klein 4B, Z-Image, Ming, ERNIE) or get licenses. Note the FLUX
   non-commercial "employment responsibilities" clause, Qwen-2.1's
   research-only terms, and marketplace AI policies.
8. **A custom UI earns its keep there:** a book/page/character model, gallery,
   template library, page layout and PDF export. Chat stops fitting once state
   spans many pages. The Open WebUI pipe can stay as a quick chat front end on
   the same server.
9. **Start on a current Open WebUI** (≥0.11.0, see the advisories in section 6)
   if it stays in the picture as a chat front end.
10. **Claude can already drive it:** the Teach Image extension (section 11)
    gives Claude chat and Cowork the job API, so a Cowork task can plan a book
    and call the tools page by page before any custom UI exists.
11. **Open questions to re-check:**
   - Ollama re-adding image generation (MLX runner)
   - FLUX 3 klein memory on 24 GB
   - Z-Image-Edit release
   - running Open WebUI natively to drop Docker's VM (a few GB) from standby

## 13. Hardware and hosting (as of Oct 2026)

Researched 2026-10-18 for "what would I buy to run or share this?". Prices
are US list prices and move fast (a 2026 memory shortage raised Mac, GPU, RAM
and SSD prices and stretched delivery times); re-check before buying.

**What decides speed on a Mac:** the GPU generation. M5/M6 GPUs have neural
accelerators: Apple measured about 3.8× faster FLUX on an M5 than an M4, and
an $899 M6 Mac mini drew a FLUX.1 image in 36 s against 94 s on an M4 (third
party). MLX uses them only on macOS 26.2+. The pinned MLX needs macOS 14+.

| Machine (Oct 2026) | US price | klein 4B per 1024² | Notes |
|---|---|---|---|
| Mac mini M5 Pro, 24 GB / 512 GB | $1,699 | ~9.5 s (same chip as this MacBook) | the known-good setup; 6 W idle |
| Mac mini M5 Pro, 48 GB / 512 GB | ~$2,299 | ~9.5 s (~7.5–8 s with the 20-core GPU, +$200) | room for the director and image model together (estimate) |
| Mac mini M6, 24 GB / 512 GB | $1,299 | ~13–22 s (estimate) | budget; 16 GB ($899) only for the image server alone |
| Mac Studio M5 Max, 36 GB | $2,499 | ~5–6.5 s (estimate) | for several teachers on one machine |
| PC + RTX 5060 Ti 16 GB | ~$1,750–2,500 (card ~$620–880) | ~5–6 s (estimate) | needs FP8 weights to keep everything on the GPU; ~40–80 W idle (estimate) |
| PC + RTX 5070 Ti 16 GB | ~$2,100–2,800 | ~3 s (estimate) | |
| DGX Spark / Strix Halo 128 GB | ~$3,400–7,000 | ~3–12 s (estimates) | only worth it for large LLMs; not for picture speed |

Not worth buying for this: M4/M4 Pro Macs (no GPU neural accelerators), an
RTX 5090 (about 2.5× its $1,999 list price, supply uncertain). Faster hardware
has diminishing returns through Claude: each tool call spends about 2.9 s on
the cloud-to-Mac link (section 11).

**Licenses for sharing or a service** (from the model cards):

- FLUX.2 klein 4B and Z-Image-Turbo: Apache-2.0, commercial use allowed.
- FLUX.2 klein 9B and FLUX.2 dev: FLUX Non-Commercial. A salaried teacher's
  classroom use is arguably "connected to ... employment responsibilities",
  so keep them off.
- Qwen-Image-2.1: research or evaluation only (a 2026 rollback from Apache).
- FLUX 3 Image Klein (2026-10-15, mflux 0.23 runs it): free for people and
  companies under $5M revenue generating their own outputs; hosting it for
  others is unclear, so opt-in only.
- Open WebUI keeps its branding clause above 50 users. Purely AI-generated
  images are generally not copyrightable (US Copyright Office, 2025).

**Hosted alternatives** (no GPU needed): klein 4B costs about $0.0012 per
1024² picture on Cloudflare Workers AI (about 96 free per day), $0.005 on fal,
$0.014 on BFL. A heavy classroom year of 10,000 pictures is roughly $12–140.
An always-on cloud L4 ($320–580 a month) only beats those above roughly
80,000–370,000 pictures a month. For a future paid service: start with a
hosted API or a per-second serverless GPU running a diffusers port, not
owned hardware.

**Schools:** local generation keeps prompts and any student photos on the
teacher's machine, which keeps FERPA/COPPA questions simple. With a cloud
backend, keep student names and photos out of prompts. Districts usually
approve tools rather than host AI servers (CoSN 2026: 86% vet tools, 7%
prepare AI storage). Not legal advice.
