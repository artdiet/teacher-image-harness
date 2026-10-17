"""
title: Teach Image
author: Art Dietrich
version: 0.2.0
license: PolyForm-Noncommercial-1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0)
Required Notice: Copyright 2026 Art Dietrich (https://github.com/artdiet)
description: Classroom pictures (worksheet icons, coloring pages, storybook art) from the local teacher-image-harness image server, with a local LLM that decides when to draw and when to answer.
required_open_webui_version: 0.9.2
"""

# Open WebUI "pipe" function. It runs inside the Open WebUI container.
#
# Each chat message goes through:
#   1. a small "director" LLM on the Mac's Ollama, which decides whether the
#      teacher wants a picture or a text answer, what kind of picture, and
#      whether to change the previous picture (it classifies; it does not
#      write the image prompt);
#   2. a fixed style template that turns the subject into the image prompt;
#   3. the image server on the Mac (http://host.docker.internal:7865).
# Results are stored as Open WebUI files owned by the user and shown inline
# with download links. A collapsible trace ("Thought for N seconds") shows
# every decision, the exact prompt, seed and timings.
#
# Install/update with scripts/install_openwebui.sh. The display name shown in
# the model list is set at install time, not here.

import asyncio
import base64
import contextlib
import io
import json
import logging
import re
import time
from typing import AsyncIterator, Optional

import httpx
from fastapi import UploadFile
from pydantic import BaseModel, Field

from open_webui.models.chats import Chats
from open_webui.models.files import Files
from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler
from open_webui.storage.provider import Storage

log = logging.getLogger(__name__)

SIZES = {
    "square": "1024x1024",
    "portrait": "896x1152",
    "landscape": "1152x896",
    "wide": "1344x768",
    "tall": "768x1344",
}
VALUE_FLAGS = {"n", "seed", "size", "model"}
BOOL_FLAGS = {"new", "transparent", "raw", *SIZES}
DRAW_FLAGS = VALUE_FLAGS | BOOL_FLAGS  # any option means "draw"

STYLE = {
    "icon": "{subject}, flat vector icon, bold clean black outlines, simple shapes, solid bright "
    "colors, centered, plain white background, no text",
    "coloring": "{subject}, coloring book page for kids, black and white line art, thick clean "
    "outlines, no shading, no gray, plain white background, no text",
    "storybook": "children's picture book illustration: {subject}, warm gentle colors{style}",
    "other": "{subject}",
}
NAMED_STYLE_RE = re.compile(
    r"watercolor|water colou?r|pencil|crayon|pastel|cartoon|painting|painted|digital|3d|pixel|"
    r"photo|realistic|ink|collage|clay|gouache|oil",
    re.I,
)

DIRECTOR_SYSTEM = """You help a teacher make classroom pictures. Read the newest teacher message and classify it. Do not write image prompts; just fill in the fields.

action:
- "draw" if the teacher wants a picture made, changed, redone or continued (including "make it ...", "now a ...", "turn this into ...").
- "reply" only if the teacher asks a question or is chatting and does not want a new picture (for example "what prompt did you use?").
kind: "icon" (worksheet icon or clipart), "coloring" (coloring page or line art), "storybook" (picture book or story scene), or "other".
edit_previous: true if the teacher is changing or continuing the most recent picture; false for a new subject or a fresh picture.
subject: what to draw, in the teacher's own words, including any changes they asked for (for example "an owl wearing a red graduation cap"). Do not add details the teacher did not ask for. Empty when action is "reply".
Words starting with -- are options for the picture tool (for example --portrait or --n 3). They mean the teacher wants a picture, unless the teacher is asking what an option does.
reply: for "draw", one short friendly sentence presenting the finished picture (for example "Here's a cute owl for your reading worksheet!"). For "reply", the full answer, using the conversation: earlier assistant turns contain a trace with the exact prompt and settings that were used."""

DIRECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["draw", "reply"]},
        "kind": {"type": "string", "enum": ["icon", "coloring", "storybook", "other"]},
        "edit_previous": {"type": "boolean"},
        "subject": {"type": "string"},
        "reply": {"type": "string"},
    },
    "required": ["action", "kind", "edit_previous", "subject", "reply"],
}

HELP = """**How to use this model**

- Describe the picture: *"a friendly owl for my reading worksheet"*, *"a coloring page of a dinosaur"*, *"a storybook page: a bear and a bunny having a picnic"*.
- **Change a picture:** attach it (📎) and say what to change, e.g. *"turn this into a coloring page"*.
- **Follow-ups change the last picture:** *"now give it a red hat"*. Ask for something new (or start a new chat) to get a fresh one.
- **Ask questions:** *"what prompt did you use for that?"* gets a text answer.
- Open **Thought for … seconds** above a reply to see exactly what was sent to the image model.

Options you can add anywhere in the message:

| Option | What it does |
|---|---|
| `--portrait` `--landscape` `--wide` `--tall` | picture shape (default square) |
| `--n 3` | make several versions at once (up to 4) |
| `--raw --seed 42` | repeat a picture exactly: the same words with `--raw`, the same seed, and `--new` in a chat that already has pictures |
| `--transparent` | remove the white background (for icons) |
| `--new` | don't change earlier pictures, start fresh |
| `--raw` | send your exact words to the image model (no rewording) |

Tips: add words in your worksheet tool rather than in the picture."""

ATTACHED_FILES_RE = re.compile(r"<attached_files>.*?</attached_files>\s*", re.S)  # Open WebUI 0.12+
OUR_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(/api/v1/files/([0-9a-fA-F-]{36})/content\)")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
DOWNLOAD_LINE_RE = re.compile(r"^\[Download PNG\].*$", re.M)
FILE_URL_RE = re.compile(r"^/api/v1/files/([0-9a-fA-F-]{36})/content")
FILE_ID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")

TASK_REPLIES = {
    "tags_generation": '{"tags": []}',
    "follow_up_generation": '{"follow_ups": []}',
    "query_generation": '{"queries": []}',
    "autocomplete_generation": '{"text": ""}',
    "emoji_generation": "\U0001f3a8",
}


class ServerError(Exception):
    """The image server answered with an error the teacher should see."""


def task_reply(task: str, task_body: Optional[dict]) -> str:
    """Answer Open WebUI's background tasks (titles, tags, ...) without drawing."""
    if task == "title_generation":
        for message in (task_body or {}).get("messages") or []:
            if message.get("role") == "user":
                text, _ = split_content(message.get("content"))
                text, _ = parse_options(text)
                title = " ".join(text.split()[:6])[:60]
                return json.dumps({"title": title or "Classroom picture"})
        return json.dumps({"title": "Classroom picture"})
    return TASK_REPLIES.get(task, "")


def split_content(content) -> tuple[str, list[str]]:
    """Text and image URLs (data: URLs in Open WebUI 0.9) from one message's content."""
    if isinstance(content, str):
        return ATTACHED_FILES_RE.sub("", content).strip(), []
    texts, images = [], []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(part.get("text") or "")
        elif part.get("type") in ("image_url", "input_image"):
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if isinstance(url, str) and url:
                images.append(url)
    return ATTACHED_FILES_RE.sub("", "\n".join(texts)).strip(), images


def parse_options(text: str) -> tuple[str, dict]:
    """Pull --flags out of the prompt. Unknown --words stay in the prompt."""
    if "--" not in text:
        return text.strip(), {}
    words, options = [], {}
    tokens = text.split()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        name = token[2:].lower() if token.startswith("--") else ""
        if name in BOOL_FLAGS:
            options[name] = True
        elif name in VALUE_FLAGS and i + 1 < len(tokens):
            options[name] = tokens[i + 1]
            i += 1
        else:
            words.append(token)
        i += 1
    return " ".join(words), options


def previous_images(messages: list) -> tuple[list[str], str]:
    """Newest earlier picture in the chat: our own output, else the last user upload."""
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, str):
            ids = OUR_IMAGE_RE.findall(content)
            if ids:
                return [f"/api/v1/files/{ids[-1]}/content"], "previous picture"
        elif message.get("role") == "user":
            images = split_content(content)[1]
            if images:
                return images, "picture attached earlier"
    return [], ""


def history_for_director(messages: list, last_index: int, prompt: str, attached: int) -> list:
    """The conversation as plain text for the director (pictures become markers)."""
    out = []
    for message in messages[max(0, last_index - 12) : last_index]:
        role = message.get("role")
        if role == "user":
            text, images = split_content(message.get("content"))
            text, _ = parse_options(text)
            if images:
                text += f"\n[The teacher attached {len(images)} picture(s).]"
            out.append({"role": "user", "content": text})
        elif role == "assistant" and isinstance(message.get("content"), str):
            text = re.sub(r"</?think>", "\n", message["content"])  # keep the trace as plain text
            text = MD_IMAGE_RE.sub("[picture]", text)
            out.append({"role": "assistant", "content": DOWNLOAD_LINE_RE.sub("", text).strip()})
    current = prompt + (f"\n[The teacher attached {attached} picture(s).]" if attached else "")
    out.append({"role": "user", "content": current})
    return out


def build_prompt(kind: str, subject: str) -> str:
    template = STYLE.get(kind, STYLE["other"])
    style = "" if NAMED_STYLE_RE.search(subject) else ", soft watercolor style"
    return template.format(subject=subject.strip().rstrip("."), style=style)


def slug(text: str, limit: int = 40) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = ""
    for word in words:
        if len(out) + len(word) + 1 > limit:
            break
        out = f"{out}-{word}" if out else word
    return out or "picture"


def alt_text(prompt: str) -> str:
    return " ".join(re.sub(r"[\[\]()*_`<>\\]", " ", prompt).split())[:80] or "picture"


def safe_reply(text: str) -> str:
    """Director text, defused: a zero-width space after '<' stops it from opening Open WebUI's
    reasoning tags, and a leading 'data:' would be taken for a raw stream line."""
    text = re.sub(r"([<\u25c1])", "\\1\u200b", text)
    return "\u200b" + text if text.startswith("data:") else text


def quote(text: str) -> str:
    """Neutralize markdown/HTML in text echoed into the trace."""
    return " ".join(re.sub(r"[<>`*_\[\]|]", " ", text).split())


async def read_owned_file(file_id: str, user) -> Optional[tuple[bytes, str]]:
    """Bytes of an Open WebUI file, only if it belongs to this user (or user is admin)."""
    file = await Files.get_file_by_id(file_id)
    if not file or (file.user_id != user.id and user.role != "admin"):
        return None
    def read() -> bytes:
        with open(Storage.get_file(file.path), "rb") as fh:
            return fh.read()

    try:
        data = await asyncio.to_thread(read)
    except Exception:
        log.warning("could not read file %s", file_id, exc_info=True)
        return None
    return data, (file.meta or {}).get("content_type") or "image/png"


async def load_reference(url: str, user) -> Optional[tuple[bytes, str]]:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        mime = header[5:].split(";")[0] or "image/png"
        try:
            return base64.b64decode(data), mime
        except ValueError:
            return None
    match = FILE_URL_RE.match(url)
    if match:
        return await read_owned_file(match.group(1), user)
    if FILE_ID_RE.match(url):  # Open WebUI could not inline it; it's a bare file id
        return await read_owned_file(url, user)
    return None


async def store_png(request, user, png: bytes, filename: str, chat_id, message_id) -> str:
    """Save as an Open WebUI file owned by the user and linked to the chat; return its URL."""
    linkable = bool(chat_id and message_id and not str(chat_id).startswith("local:"))
    upload = UploadFile(file=io.BytesIO(png), filename=filename, headers={"content-type": "image/png"})
    item = await upload_file_handler(
        request,
        file=upload,
        metadata={"chat_id": chat_id, "message_id": message_id} if linkable else {},
        process=False,
        process_in_background=False,
        user=user,
    )
    file_id = item["id"] if isinstance(item, dict) else item.id
    if linkable:
        await Chats.insert_chat_files(
            chat_id=chat_id, message_id=message_id, file_ids=[file_id], user_id=user.id
        )
    return f"/api/v1/files/{file_id}/content"


async def emit_status(emitter, description: str, done: bool = False, hidden: bool = False) -> None:
    if emitter:
        with contextlib.suppress(Exception):  # a failing emitter must not hide the real error
            await emitter(
                {"type": "status", "data": {"description": description, "done": done, "hidden": hidden}}
            )


async def with_heartbeat(coro, emitter, label: str, every: float = 10.0):
    """Await coro, refreshing the status line every few seconds."""
    task = asyncio.ensure_future(coro)
    started = time.monotonic()
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=every)
            if not task.done():
                await emit_status(emitter, f"{label}… {int(time.monotonic() - started)} s")
        return task.result()
    finally:
        if not task.done():
            task.cancel()  # Stop button: CancelledError arrives here


def server_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail if isinstance(d, dict))
    return str(detail or response.text or response.reason_phrase)[:300]


class _ReplyState:
    """What a reply has opened (trace block, status line), so every exit can close it."""

    def __init__(self, emitter, trace: bool) -> None:
        self.emitter = emitter
        self.trace = trace
        self.trace_open = False
        self.status_open = False

    def line(self, text: str) -> str:
        return f"{text}\n\n" if self.trace_open else ""

    def end_trace(self) -> str:
        if not self.trace_open:
            return ""
        self.trace_open = False
        return "</think>\n\n"

    async def status(self, description: str) -> None:
        await emit_status(self.emitter, description)
        self.status_open = True

    async def close_status(self, description: str, hidden: bool) -> None:
        if self.status_open:
            self.status_open = False
            await emit_status(self.emitter, description, done=True, hidden=hidden)


class Pipe:
    class Valves(BaseModel):
        IMAGE_SERVER_URL: str = Field(
            default="http://host.docker.internal:7865",
            description="Image server address as seen from inside the Open WebUI container",
        )
        IMAGE_SERVER_TOKEN: str = Field(
            default="",
            description="IMAGE_SERVER_TOKEN from the repo's .env",
            json_schema_extra={"input": {"type": "password"}},
        )
        MODEL: str = Field(default="", description="Image model id; empty = the server's default")
        DEFAULT_SIZE: str = Field(default="1024x1024")
        MAX_IMAGES: int = Field(default=4, ge=1, le=8, description="Upper limit for --n")
        MAX_REFERENCES: int = Field(default=4, ge=1, le=10)
        EDIT_PREVIOUS_IMAGE: bool = Field(
            default=True, description="Follow-up messages may change the most recent picture"
        )
        USE_DIRECTOR: bool = Field(
            default=True, description="Let a local LLM decide between drawing and answering"
        )
        DIRECTOR_MODEL: str = Field(default="gemma4:12b", description="Ollama model for the director")
        OLLAMA_URL: str = Field(default="http://host.docker.internal:11434")
        DIRECTOR_TIMEOUT_SECONDS: int = Field(default=90, ge=5, le=600)
        SHOW_TRACE: bool = Field(
            default=True, description="Show the collapsible trace (prompt, model, seed, timings)"
        )
        TIMEOUT_SECONDS: int = Field(default=600, ge=30, le=3600)

    def __init__(self):
        self.valves = self.Valves()  # Open WebUI swaps in the saved valves before each call
        self._transport = None  # tests inject httpx transports here
        self._ollama_transport = None
        self._retry_delay = 3.0

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__=None,
        __event_emitter__=None,
        __task__: Optional[str] = None,
        __task_body__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __message_id__: Optional[str] = None,
    ):
        if __task__:  # titles, tags, follow-ups: answer cheaply, never draw
            return task_reply(__task__, __task_body__)
        messages = body.get("messages") or []
        last_index = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        if last_index < 0:
            return HELP
        text, images = split_content(messages[last_index].get("content"))
        prompt, options = parse_options(text)
        if prompt.lower().strip(" ?!./") in {"help", ""} and not images:
            return HELP
        try:
            self._build_request("x", options)  # bad options: explain before any work
        except ValueError as exc:
            return f"{exc}\n\nType **help** to see the options."
        # Open WebUI turns <think> into a collapsed "Thought for N seconds" block only
        # when streaming; otherwise it would show the trace as raw text.
        state = _ReplyState(__event_emitter__, trace=self.valves.SHOW_TRACE and bool(body.get("stream")))
        return self._respond(
            state, messages, last_index, text, prompt, options, images,
            __user__, __request__, __chat_id__, __message_id__,
        )

    async def _respond(self, state, *args) -> AsyncIterator[str]:
        """Stream the reply; whatever happens, close the trace and the status line."""
        try:
            async for chunk in self._steps(state, *args):
                yield chunk
        except asyncio.CancelledError:  # Stop button
            await asyncio.shield(state.close_status("Stopped", hidden=False))
            raise
        except Exception as exc:
            log.exception("teach image pipe failed")
            await state.close_status("Something went wrong", hidden=False)
            yield state.end_trace()
            yield f"Something went wrong while making the picture ({type(exc).__name__}). Please try again."
            return
        await state.close_status("Done", hidden=True)

    async def _steps(
        self, state, messages, last_index, text, prompt, options, images,
        user_info, request, chat_id, message_id,
    ) -> AsyncIterator[str]:
        started = time.monotonic()
        if state.trace:
            state.trace_open = True
            yield f"<think>\n**Request:** {quote(text) or '(picture only)'}\n\n"

        # 1. Decide: draw or reply, what kind, change the previous picture or not.
        decision = {"action": "draw", "kind": "other", "edit_previous": True, "subject": prompt,
                    "reply": ""}
        # An attached picture always means "draw". So do options, unless the teacher is
        # asking about one ("what does --transparent do?").
        forced = bool(images) or (bool(DRAW_FLAGS & set(options)) and not prompt.rstrip().endswith("?"))
        if options.get("raw") or not self.valves.USE_DIRECTOR:
            note = "skipped (--raw)" if options.get("raw") else "off"
            yield state.line(f"**Director:** {note}, using your words as the prompt")
        else:
            await state.status("Reading your message…")
            t0 = time.monotonic()
            try:
                decision = await self._direct(
                    history_for_director(messages, last_index, text, len(images))
                )
                decision["subject"] = parse_options(decision["subject"])[0]
                if forced and decision["action"] != "draw":
                    decision.update(action="draw", reply="")
                yield state.line(
                    f"**Director** ({quote(self.valves.DIRECTOR_MODEL)}, {time.monotonic() - t0:.1f} s): "
                    f"{decision['action']} · {decision['kind']} · "
                    f"{'change the previous picture' if decision['edit_previous'] else 'new picture'}"
                )
            except Exception as exc:  # the director is optional: fall back to the raw words
                decision["subject"] = prompt
                yield state.line(f"**Director:** unavailable ({quote(str(exc))[:120]}), using your words")

        if decision["action"] == "reply":
            yield state.line(f"**Total:** {time.monotonic() - started:.1f} s")
            yield state.end_trace()
            yield safe_reply(decision["reply"]) or "I'm here to make pictures. Type **help** to see how."
            return

        # 2. Build the image request.
        subject = decision.get("subject") or prompt or "a clean, simple version of this picture"
        if options.get("raw") or not self.valves.USE_DIRECTOR:
            image_prompt = prompt or "a clean, simple version of this picture"
        else:
            image_prompt = build_prompt(decision.get("kind", "other"), subject)
        request_body = self._build_request(image_prompt, options)  # options already checked

        user = await Users.get_user_by_id(user_info["id"])
        source = "attached" if images else ""
        wants_previous = (
            not images
            and self.valves.EDIT_PREVIOUS_IMAGE
            and not options.get("new")
            and decision.get("edit_previous", True)
        )
        if wants_previous:
            images, source = previous_images(messages[:last_index])
        references = []
        for url in images[: self.valves.MAX_REFERENCES]:
            ref = await load_reference(url, user)
            if ref:
                references.append(ref)
        if images and not references:
            yield state.end_trace()
            if source == "attached":
                yield "I couldn't open the attached picture. Please attach it again as a PNG or JPEG."
            else:
                yield ("I can't open the earlier picture any more (it was deleted, or it belongs to "
                       "another account). Attach it again, or add `--new` for a fresh picture.")
            return
        if not references:
            request_body.setdefault("size", self.valves.DEFAULT_SIZE)
        yield state.line(f"**Prompt sent to the image model:** {quote(image_prompt)}")
        if references:
            yield state.line(f"**Reference pictures:** {len(references)} ({source})")

        # 3. Draw.
        noun = "pictures" if request_body["n"] > 1 else "picture"
        label = f"Changing your {noun}" if references else f"Drawing your {noun}"
        await state.status(f"{label}…")
        try:
            result = await with_heartbeat(
                self._call_server(request_body, references), state.emitter, label
            )
        except ServerError as exc:
            await state.close_status("Something went wrong", hidden=False)
            yield state.end_trace()
            yield str(exc)
            return

        # 4. Save as the user's files and present.
        await state.status("Saving…")
        stem = slug(subject)
        parts = []
        for item in result.get("data", []):
            png = base64.b64decode(item["b64_json"])
            seed = item.get("seed", "x")
            url = await store_png(request, user, png, f"{stem}-{seed}.png", chat_id, message_id)
            parts += [f"![{alt_text(subject)}]({url})", f"[Download PNG]({url}?attachment=true)"]
            yield state.line(
                f"**Image model:** {quote(str(result.get('model', '?')))} · seed {seed} · "
                f"{item.get('width', '?')}×{item.get('height', '?')} · {item.get('seconds', 0):.1f} s"
            )
        yield state.line(f"**Total:** {time.monotonic() - started:.1f} s")
        yield state.end_trace()
        if decision.get("reply"):
            yield safe_reply(decision["reply"]) + "\n\n"
        yield "\n\n".join(parts) or "The image server returned no pictures."
        if source == "previous picture":
            yield "\n\n_Changed the previous picture. Add `--new` for a fresh one._"

    async def _direct(self, conversation: list) -> dict:
        """Ask the director LLM (Ollama, structured output) to classify the request."""
        valves = self.valves
        async with httpx.AsyncClient(
            base_url=valves.OLLAMA_URL.rstrip("/"),
            timeout=httpx.Timeout(valves.DIRECTOR_TIMEOUT_SECONDS, connect=5),
            transport=self._ollama_transport,
        ) as client:
            response = await client.post("/api/chat", json={
                "model": valves.DIRECTOR_MODEL,
                "stream": False,
                "think": False,
                "format": DIRECTOR_SCHEMA,
                "keep_alive": "5m",
                "options": {"temperature": 0, "num_ctx": 8192},
                "messages": [{"role": "system", "content": DIRECTOR_SYSTEM}, *conversation],
            })
            response.raise_for_status()
        decision = json.loads(response.json()["message"]["content"])
        if decision.get("action") not in ("draw", "reply"):
            raise ValueError("director gave no action")
        decision["kind"] = decision.get("kind") if decision.get("kind") in STYLE else "other"
        decision["edit_previous"] = bool(decision.get("edit_previous"))
        decision["subject"] = str(decision.get("subject") or "")[:600]
        decision["reply"] = str(decision.get("reply") or "")[:2000]
        return decision

    def _build_request(self, prompt: str, options: dict) -> dict:
        """The image request. Size is set only if the teacher chose one: edits then keep the
        reference picture's shape, and text-to-image gets DEFAULT_SIZE later."""
        request = {"prompt": prompt[:4000], "n": 1}
        for shape in SIZES:
            if options.get(shape):
                request["size"] = SIZES[shape]
        if "size" in options:
            if not re.fullmatch(r"\d{3,4}x\d{3,4}", options["size"].lower()):
                raise ValueError("`--size` should look like `1024x768`.")
            request["size"] = options["size"].lower()
        if "n" in options:
            if not options["n"].isdigit() or not 1 <= int(options["n"]) <= self.valves.MAX_IMAGES:
                raise ValueError(f"`--n` should be a number from 1 to {self.valves.MAX_IMAGES}.")
            request["n"] = int(options["n"])
        if "seed" in options:
            if not options["seed"].isdigit():
                raise ValueError("`--seed` should be a whole number, like `--seed 42`.")
            request["seed"] = int(options["seed"])
        model = options.get("model") or self.valves.MODEL
        if model:
            request["model"] = model
        if options.get("transparent"):
            request["background"] = "transparent"
        return request

    async def _call_server(self, request: dict, references: list) -> dict:
        try:
            response = await self._post(request, references)
        except httpx.ConnectError:
            # The server restarts itself after long idle periods; give it a moment.
            await asyncio.sleep(self._retry_delay)
            try:
                response = await self._post(request, references)
            except httpx.ConnectError:
                raise ServerError(
                    f"Couldn't reach the image server at {self.valves.IMAGE_SERVER_URL}. "
                    "Is it running on the Mac?"
                ) from None
        if response.status_code == 401:
            raise ServerError(
                "The image server rejected the token. An admin should check this function's "
                "IMAGE_SERVER_TOKEN setting."
            )
        if response.status_code == 503:
            raise ServerError("The image server is busy with other pictures. Try again in a minute.")
        if response.status_code >= 400:
            raise ServerError(f"The image server couldn't do that: {server_detail(response)}")
        return response.json()

    async def _post(self, request: dict, references: list) -> httpx.Response:
        valves = self.valves
        base_url = valves.IMAGE_SERVER_URL.rstrip("/")
        try:
            async with httpx.AsyncClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {valves.IMAGE_SERVER_TOKEN}"},
                timeout=httpx.Timeout(valves.TIMEOUT_SECONDS, connect=10),
                transport=self._transport,
            ) as client:
                if references:
                    fields = {k: str(v) for k, v in request.items()}
                    files = [
                        ("image[]", (f"reference-{i}.png", data, mime))
                        for i, (data, mime) in enumerate(references)
                    ]
                    return await client.post("/v1/images/edits", data=fields, files=files)
                return await client.post("/v1/images/generations", json=request)
        except httpx.ConnectError:
            raise
        except httpx.TimeoutException:
            raise ServerError(
                "The image server took too long. Try a smaller size or fewer pictures."
            ) from None
        except httpx.HTTPError as exc:
            raise ServerError(
                f"Couldn't reach the image server at {base_url} ({type(exc).__name__}). "
                "Is it running on the Mac?"
            ) from None
