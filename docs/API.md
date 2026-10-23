# Image server API

The image server runs natively on the Mac (launchd) and does all image work on
its GPU, one job at a time. Two sets of routes:

- **Job API** (this document): asynchronous jobs, stored images with metadata,
  reference pictures. Used by the Teach Image MCP extension
  ([MCP.md](MCP.md)) and meant for the future book harness.
- **OpenAI-compatible** routes (`/v1/images/generations`, `/v1/images/edits`,
  `/v1/models`, `/v1/status`, `/v1/unload`): synchronous, used by the Open WebUI
  pipe. Same requests and responses as in 1.0 (a busy `503` now also carries
  `Retry-After`).

The machine-readable contract is [openapi.json](openapi.json) (OpenAPI 3.1,
regenerate with `uv run --project image-server python scripts/export_openapi.py`).

## Basics

| | |
|---|---|
| Base URL | `http://127.0.0.1:7865` on the Mac; `http://<mac-name>.local:7865` or `http://<mac-ip>:7865` from the network once LAN access is on ([MCP.md](MCP.md#1-on-the-image-server-mac-once)) |
| Auth | `Authorization: Bearer <token>` on every route except `GET /health`. Other computers use their own device token from `scripts/client_tokens.sh add <device>`. `IMAGE_SERVER_TOKEN` from `.env` (Open WebUI's) is accepted only from the Mac itself. |
| Bodies | JSON (64 KB cap) or multipart for uploads (20 MB per picture by default). Unknown fields in a job are rejected, so typos don't pass silently. |
| Errors | JSON `{"detail": "..."}` (validation errors: a list of `{loc, msg}`), except the two refusals made before the app runs: `403` and the `Invalid host header` `400` are plain text |
| Connections | at most 16 open at once per computer (the Mac itself is exempt); one more gets `503` `{"detail": "too many open connections from this computer; try again in a moment"}`. Request headers must arrive within 10 s of connecting or of the previous response; a request body that stops arriving for 30 s, or a response the client stops reading for 60 s, ends the connection. Slow but moving transfers and long-running requests are fine. |
| CORS | none: browsers on other sites can't call it |

Status codes:

| Code | Meaning |
|---|---|
| `400` | the request can't be run as asked: unknown model, missing reference or image, a `size` that doesn't parse or is out of range, `n` above the server's limit, an upload that isn't a picture. Plain-text `Invalid host header`: the address used isn't in `IMAGE_SERVER_ALLOWED_HOSTS` (names match in any letter case). |
| `401` | missing or wrong token (or Open WebUI's token used from another computer) |
| `403` | client outside the private network |
| `404` | unknown id |
| `413` | too large |
| `422` | malformed body: unknown fields, wrong types, values out of range (e.g. `width` 3000, `n` 9) |
| `503` | busy: too many jobs waiting (`Retry-After: 30`), or restarting (`Retry-After: 5`, detail says so) |

Ids are random and strictly formatted: `job_<16 hex>`, `img_<16 hex>`,
`ref_<16 hex>`.

## Jobs

### `POST /jobs` → `202`

Queue a picture. With `source_image_id` or `reference_ids` it is an edit and
uses the edit model (FLUX.2 klein by default).

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string, 1–4000 | required | what to draw or change |
| `style` | `icon` `coloring` `storybook` `none` | `none` | wraps the prompt in a classroom template (below) |
| `width`, `height` | 256–2048 | 1024×1024; for edits the shape of the source (or of the first reference) at about 1 MP | give both, or use `size` |
| `size` | `"WIDTHxHEIGHT"` | | alternative to width/height |
| `seed` | 0–4294967295 | random | picture *i* of the job uses `seed + i` |
| `n` | 1–8 | 1 | more than `IMAGE_SERVER_MAX_N` (4) is refused with `400` |
| `source_image_id` | `img_…` | | picture to edit (generated or uploaded) |
| `reference_ids` | list of `ref_…` | | up to the model's limit (klein: 4, counting the source) |
| `transparent` | bool | false | flood-fills the white background from the edges |
| `model` | string | server default | see `GET /models` |
| `steps` | 1–100 | model default (klein 4, Z-Image 9) | advanced |
| `guidance` | 0–20 | | accepted and recorded, but the current models ignore it (klein and Qwen use a fixed 1.0) |
| `negative_prompt` | string | | recorded but not used by the current models (`result.ignored` says so) |

Sizes snap to multiples of 16 and are scaled down to about 1.5 megapixels.

Style templates (`{subject}` is the prompt):

| Style | Prompt sent to the model |
|---|---|
| `icon` | `{subject}, flat vector icon, bold clean black outlines, simple shapes, solid bright colors, centered, plain white background, no text` |
| `coloring` | `{subject}, coloring book page for kids, black and white line art, thick clean outlines, no shading, no gray, plain white background, no text` |
| `storybook` | `children's picture book illustration: {subject}, warm gentle colors, soft watercolor style, no text` (the watercolor part only when the prompt names no medium) |

```json
{"job_id": "job_8f2c…", "status": "queued", "queue_depth": 1}
```

`queue_depth` counts the jobs ahead of this one (0 on an idle server). The job
is recorded on disk as soon as it is accepted.

### `GET /jobs/{job_id}`

```json
{
  "job_id": "job_8f2c…", "status": "done", "progress": 1.0,
  "params": {"prompt": "a friendly owl", "style": "icon", "full_prompt": "a friendly owl, flat vector icon, …",
             "model": "flux2-klein-4b", "width": 1024, "height": 1024, "seed": 21, "n": 1, "…": "…"},
  "created_at": 1792270000.1, "started_at": 1792270000.2, "finished_at": 1792270009.6,
  "result": {
    "image_id": "img_1c41…", "seed": 21, "model": "flux2-klein-4b",
    "images": [{"image_id": "img_1c41…", "seed": 21, "width": 1024, "height": 1024, "seconds": 9.3}],
    "params": {"…": "…"}
  }
}
```

`status` is `queued`, `running`, `done`, `failed` or `cancelled`. `error` appears
only on failed jobs and `result` only on finished ones. `progress` runs from 0
to 1 across all requested pictures. Poll about once a second.

A job id always resolves, even across restarts (`jobs/<id>.json`). When the
server stops (idle restart, reinstall, logout), it accepts no new jobs (`503`
"restarting"), loads no other model, and gives open requests and queued or
running jobs one shared grace of 25 s to finish. Anything still unfinished,
including after a crash or a long power-off, then reads as `failed` with
`error: "interrupted by a server restart; please try again"`. A job that can't
be recorded on disk (a full disk) is refused with `503` rather than accepted.

### `DELETE /jobs/{job_id}`

Cancels a job. A job still waiting for its turn is dropped at once (the
response already says `cancelled`) and frees its place in the queue. One that
is starting or running stops at its next denoising step, so its status turns
`cancelled` a moment later (the response may still say `queued` or `running`).

### `GET /jobs?limit=20`

Recent jobs, newest first (`limit` 1–200), including ones from before the last
restart.

## Images

Every picture made through the job API is stored with a JSON sidecar, whatever
`IMAGE_SERVER_SAVE_OUTPUTS` says (clients fetch them by id). These pictures,
jobs and references form one library shared by every device token: any of them
can list, view and edit all of it and delete any reference. Open WebUI's
pictures (the OpenAI-compatible routes) are archived separately, by date, in
`outputs/`, and are never served by these routes.

### `POST /images` (multipart) → `201`

Upload a picture to edit: field `file` (PNG, JPEG, WebP or GIF; it is
normalized to RGB, about 1 megapixel at most).

```json
{"image_id": "img_c7c9…", "width": 1024, "height": 1024}
```

### `GET /images/{image_id}?size=full|preview&budget=70000`

- `full` (default): the original PNG. PNG text chunks hold the model and seed,
  never the prompt.
- `preview`: WebP, 768 px on the long edge, stepped down in size and quality
  until it fits `budget` bytes (8000–70000, default 70000). The MCP extension
  asks for smaller previews when a result carries several pictures.

### `GET /images/{image_id}/meta`

The sidecar: `prompt`, `style`, `full_prompt`, `model`, `seed`, `width`,
`height`, `steps`, `guidance`, `reference_ids`, `source_image_id`, `job_id`,
`seconds`, `license`, `created_at`. Uploads have `source: "upload"` and
`filename` instead.

### `GET /images?limit=20&before=<created_at>`

Sidecars, newest first. Pass the last `created_at` as `before` to page.

## References

Reference pictures (character sheets, style samples) for `reference_ids`.

### `POST /references` → `201`

Either multipart (`file`, optional `name`, optional `tags` as a comma-separated
string) or JSON that promotes a stored image:

```json
{"from_image_id": "img_7e08…", "name": "Benny the bear, front view", "tags": ["bear book"]}
```

```json
{"id": "ref_5a98…", "name": "Benny the bear, front view", "tags": ["bear book"],
 "width": 1024, "height": 1024, "source": "image:img_7e08…", "created_at": 1792270400.0}
```

### `GET /references?tag=bear%20book`, `GET /references/{id}`, `GET /references/{id}/image`, `DELETE /references/{id}` → `204`

Deleting a reference keeps the pictures made with it.

## Service

- `GET /models`: per model `id`, `license`, `supports_edit`,
  `max_reference_images`, `default_steps`, `research_only`, `default`,
  `edit_default`, `supports_negative_prompt`, `supports_transparent`.
- `GET /health` (no token): `status`, `version`, `loaded_model`,
  `queue_depth`, `jobs_active`, `memory` (GPU GB after the last job),
  `uptime_s`. No prompts, pictures or secrets, but anyone on the network can
  see whether the server is busy.

## Storage

Under `IMAGE_SERVER_DATA_DIR` (default
`~/Library/Application Support/teacher-image-harness`), folders mode 700 and
files mode 600:

```
images/img_<id>.png            the picture
images/img_<id>.json           sidecar (prompt, seed, model, params, references)
images/img_<id>.preview.webp   cached preview
references/ref_<id>.png/.json  reference picture and its name, tags, size, source
jobs/job_<id>.json             job records (written when accepted and when finished)
jobs/active/job_<id>           empty marker while a job is unfinished (finds interrupted jobs at startup)
outputs/<date>/…png/.json      Open WebUI's pictures with their prompts (IMAGE_SERVER_SAVE_OUTPUTS)
client_tokens.json             device token hashes (scripts/client_tokens.sh)
```

Nothing is deleted automatically.

## Testing every endpoint

`scripts/smoke_test_api.sh http://<mac-name>.local:7865` runs about 35 checks
against a running server with curl (it asks for a device token). Run it from
another computer, or from the Mac against its LAN address, so the Host check and
the firewall are part of the test. It makes three real pictures on the shared
GPU and unloads the model at the end.

## curl walkthrough

```bash
S=http://<mac-name>.local:7865
TOKEN=<device token>
auth=(-H "Authorization: Bearer $TOKEN")

curl -s $S/health
curl -s "${auth[@]}" $S/models

job=$(curl -s "${auth[@]}" -H 'content-type: application/json' \
  -d '{"prompt":"a friendly owl holding a pencil","style":"icon","transparent":true}' \
  $S/jobs | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

curl -s "${auth[@]}" $S/jobs/$job          # repeat until "status": "done"
img=<result.image_id from that response>
curl -s "${auth[@]}" "$S/images/$img?size=full" -o owl.png
curl -s "${auth[@]}" "$S/images/$img?size=preview&budget=20000" -o owl-preview.webp

# edit a picture from disk
up=$(curl -s "${auth[@]}" -F "file=@photo.jpg" $S/images | python3 -c 'import json,sys; print(json.load(sys.stdin)["image_id"])')
curl -s "${auth[@]}" -H 'content-type: application/json' \
  -d "{\"prompt\":\"turn this into a coloring page\",\"style\":\"coloring\",\"source_image_id\":\"$up\"}" $S/jobs

# keep a character
curl -s "${auth[@]}" -H 'content-type: application/json' \
  -d "{\"from_image_id\":\"$img\",\"name\":\"Owl\",\"tags\":[\"owl\"]}" $S/references
```
