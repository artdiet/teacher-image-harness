"""Compare the local image models on the teacher's real use cases.

Runs each model on the same prompts and seed, records seconds per image and
peak GPU memory, and writes PNGs plus an HTML contact sheet outside the repo.

    uv run --project image-server python scripts/bakeoff.py
    uv run --project image-server python scripts/bakeoff.py --models flux2-klein-4b --quick
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import os
import resource
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # same HF_HOME, port and token as the server
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx  # noqa: E402

from image_server.backends import create_backend  # noqa: E402
from image_server.config import MODELS  # noqa: E402
from image_server.engine import release_mlx_memory  # noqa: E402

PROMPTS = {
    "icon-apple": "a friendly cartoon apple, flat vector icon, bold black outlines, simple "
    "shapes, solid bright colors, centered, plain white background, no text",
    "icon-bus": "a yellow school bus, flat vector clipart icon for a children's worksheet, "
    "thick clean outlines, simple shapes, plain white background, no text",
    "icon-sun": "a smiling sun, simple flat icon for a kindergarten worksheet, bold outline, "
    "two colors, plain white background, no text",
    "coloring-turtle": "coloring book page for kids: a cute turtle sitting on a rock, black "
    "and white line art, thick clean outlines, no shading, no gray, plain white background",
    "story-fox": "children's picture book illustration: a small fox and a rabbit sharing an "
    "umbrella in the rain, soft watercolor style, warm gentle colors",
    "story-tree": "children's storybook illustration: a child reading a book under a big oak "
    "tree on a sunny afternoon, cozy, soft colored pencil style",
}
EDIT_PROMPT = "same flat icon style and outlines, but a banana instead of an apple"
SEED = 42
SIZE = 1024


def peak_gb() -> float:
    getter = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return getter() / 1e9


def reset_peak() -> None:
    resetter = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    resetter()


def run_model(model_id: str, prompts: dict[str, str], out: Path) -> list[dict]:
    spec = MODELS[model_id]
    rows = []
    reset_peak()
    started = time.perf_counter()
    backend = create_backend(spec)
    load_s = time.perf_counter() - started
    print(f"[{model_id}] loaded in {load_s:.1f}s")
    try:
        for name, prompt in prompts.items():
            reset_peak()
            started = time.perf_counter()
            image = backend.generate(prompt=prompt, seed=SEED, width=SIZE, height=SIZE,
                                     steps=spec.default_steps, guidance=spec.default_guidance,
                                     references=[])
            seconds = time.perf_counter() - started
            path = out / f"{model_id}--{name}.png"
            image.save(path)
            rows.append({"model": model_id, "case": name, "seconds": round(seconds, 1),
                         "peak_gb": round(peak_gb(), 2), "file": path.name, "load_s": round(load_s, 1)})
            print(f"[{model_id}] {name}: {seconds:.1f}s, peak {peak_gb():.1f} GB")
        apple = out / f"{model_id}--icon-apple.png"
        if spec.supports_edit and apple.exists():
            from PIL import Image

            reset_peak()
            started = time.perf_counter()
            image = backend.generate(prompt=EDIT_PROMPT, seed=SEED, width=SIZE, height=SIZE,
                                     steps=spec.default_steps, guidance=spec.default_guidance,
                                     references=[Image.open(apple).convert("RGB")])
            seconds = time.perf_counter() - started
            path = out / f"{model_id}--edit-banana.png"
            image.save(path)
            rows.append({"model": model_id, "case": "edit-banana", "seconds": round(seconds, 1),
                         "peak_gb": round(peak_gb(), 2), "file": path.name, "load_s": round(load_s, 1)})
            print(f"[{model_id}] edit-banana: {seconds:.1f}s, peak {peak_gb():.1f} GB")
    finally:
        backend.close()
        del backend
        gc.collect()
        release_mlx_memory()
    return rows


def contact_sheet(rows: list[dict], out: Path) -> Path:
    models = list(dict.fromkeys(r["model"] for r in rows))
    cases = list(dict.fromkeys(r["case"] for r in rows))
    cell = {(r["model"], r["case"]): r for r in rows}
    head = "".join(f"<th>{html.escape(m)}</th>" for m in models)
    body = []
    for case in cases:
        tds = []
        for m in models:
            r = cell.get((m, case))
            tds.append(
                f'<td><a href="{r["file"]}"><img src="{r["file"]}"></a>'
                f'<div>{r["seconds"]} s · peak {r["peak_gb"]} GB</div></td>' if r else "<td>–</td>"
            )
        prompt = PROMPTS.get(case, EDIT_PROMPT)
        body.append(f"<tr><th>{html.escape(case)}<p>{html.escape(prompt)}</p></th>{''.join(tds)}</tr>")
    page = out / "index.html"
    page.write_text(
        "<!doctype html><meta charset=utf-8><title>Image model bake-off</title><style>"
        "body{font:14px system-ui;margin:16px}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}"
        "img{width:256px;height:256px;object-fit:contain;background:#eee}th p{font-weight:400;max-width:220px}"
        f"</style><h1>Bake-off {datetime.now():%Y-%m-%d %H:%M}</h1><table><tr><th></th>{head}</tr>"
        + "".join(body) + "</table>"
    )
    return page


def free_gpu() -> None:
    """Ask the running image server and Ollama to release their models first."""
    import asyncio
    import urllib.request

    from image_server.config import Settings
    from image_server.ollama import free_ollama_memory

    settings = Settings.from_env()
    asyncio.run(free_ollama_memory(settings.ollama_url))
    if settings.token:
        request = urllib.request.Request(f"http://127.0.0.1:{settings.port}/v1/unload", method="POST",
                                         headers={"Authorization": f"Bearer {settings.token}"})
        try:
            urllib.request.urlopen(request, timeout=30).read()
            print("asked the image server to unload its model")
        except OSError as exc:
            print(f"note: could not ask the image server to unload ({exc}); continuing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["flux2-klein-4b", "z-image-turbo",
                                                         "qwen-image-2.1-turbo"])
    parser.add_argument("--quick", action="store_true", help="one icon + one story prompt")
    parser.add_argument("--out", type=Path, default=Path(
        "~/Library/Application Support/teacher-image-harness/bakeoff").expanduser())
    args = parser.parse_args()
    prompts = {k: PROMPTS[k] for k in ("icon-apple", "story-fox")} if args.quick else PROMPTS
    free_gpu()
    out = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_id in args.models:
        rows += run_model(model_id, prompts, out)
        (out / "results.json").write_text(json.dumps(rows, indent=2))
    page = contact_sheet(rows, out)
    max_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"\ncontact sheet: {page}\nprocess max RSS: {max_rss_gb:.1f} GB")


if __name__ == "__main__":
    main()
