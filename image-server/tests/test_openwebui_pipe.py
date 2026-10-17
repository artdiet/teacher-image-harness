"""The Open WebUI pipe, run against the real image-server app with a fake backend.

Open WebUI's internals (files, chats, users, storage) are replaced by small
in-memory stand-ins, and the director LLM by a fake Ollama endpoint, so this
runs without the container.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from conftest import TOKEN, FakeBackend, png_bytes
from image_server.app import create_app
from image_server.config import Settings
from image_server.engine import Engine

PIPE_PATH = Path(__file__).resolve().parents[2] / "openwebui" / "teach_image_pipe.py"


class FakeOpenWebUI:
    """In-memory stand-ins for the open_webui modules the pipe imports."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.files: dict[str, SimpleNamespace] = {}
        self.uploads: list[dict] = []
        self.chat_links: list[dict] = []
        self.fail_uploads = False
        self.users = {
            "teacher": SimpleNamespace(id="teacher", role="user"),
            "someone-else": SimpleNamespace(id="someone-else", role="user"),
        }

    def add_file(self, owner: str, data: bytes) -> str:
        file_id = str(uuid.uuid4())
        path = self.tmp / file_id
        path.write_bytes(data)
        self.files[file_id] = SimpleNamespace(
            id=file_id, user_id=owner, path=str(path), meta={"content_type": "image/png"}
        )
        return file_id

    def install(self) -> None:
        fake = self

        async def get_user_by_id(user_id):
            return fake.users.get(user_id)

        async def get_file_by_id(file_id):
            return fake.files.get(file_id)

        async def upload_file_handler(request, *, file, metadata, process, process_in_background, user):
            assert process is False and process_in_background is False
            if fake.fail_uploads:
                raise RuntimeError("disk full")
            file_id = fake.add_file(user.id, file.file.read())
            fake.uploads.append({"id": file_id, "filename": file.filename, "metadata": metadata,
                                 "content_type": file.content_type})
            return fake.files[file_id]

        async def insert_chat_files(*, chat_id, message_id, file_ids, user_id):
            fake.chat_links.append({"chat_id": chat_id, "message_id": message_id,
                                    "file_ids": file_ids, "user_id": user_id})

        modules = {
            "open_webui": {},
            "open_webui.models": {},
            "open_webui.models.users": {"Users": SimpleNamespace(get_user_by_id=get_user_by_id)},
            "open_webui.models.files": {"Files": SimpleNamespace(get_file_by_id=get_file_by_id)},
            "open_webui.models.chats": {"Chats": SimpleNamespace(insert_chat_files=insert_chat_files)},
            "open_webui.routers": {},
            "open_webui.routers.files": {"upload_file_handler": upload_file_handler},
            "open_webui.storage": {},
            "open_webui.storage.provider": {"Storage": SimpleNamespace(get_file=lambda p: p)},
        }
        for name, attrs in modules.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            sys.modules[name] = module


class FakeDirector:
    """Stands in for Ollama's /api/chat with structured output."""

    def __init__(self) -> None:
        self.decision = {"action": "draw", "kind": "other", "edit_previous": False,
                         "subject": "", "reply": ""}
        self.requests: list[dict] = []
        self.fail = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail:
            raise httpx.ConnectError("ollama down", request=request)
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(self.decision)}})


@pytest.fixture
def owui(tmp_path):
    fake = FakeOpenWebUI(tmp_path)
    saved = {k: v for k, v in sys.modules.items() if k == "open_webui" or k.startswith("open_webui.")}
    fake.install()
    yield fake
    for name in [k for k in sys.modules if k == "open_webui" or k.startswith("open_webui.")]:
        del sys.modules[name]
    sys.modules.update(saved)


@pytest.fixture
def pipe_module(owui):
    spec = importlib.util.spec_from_file_location("teach_image_pipe", PIPE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def director():
    return FakeDirector()


@pytest.fixture
def pipe(pipe_module, tmp_path, director):
    engine = Engine(FakeBackend, release_memory=lambda: None)
    app = create_app(Settings(token=TOKEN, data_dir=tmp_path / "data", free_ollama=False), engine)
    p = pipe_module.Pipe()
    p.valves = p.Valves(IMAGE_SERVER_TOKEN=TOKEN, USE_DIRECTOR=False)
    p._transport = httpx.ASGITransport(app=app)
    p._ollama_transport = httpx.MockTransport(director.handler)
    p._retry_delay = 0
    yield p
    engine.shutdown()


def run(pipe, messages, *, chat_id="chat-1", events=None, stream=True, **kw) -> str:
    async def emitter(event):
        if events is not None:
            events.append(event)

    async def go():
        result = await pipe.pipe(
            {"messages": messages, "stream": stream}, {"id": "teacher"}, __request__=object(),
            __event_emitter__=emitter, __chat_id__=chat_id, __message_id__="msg-1", **kw,
        )
        if isinstance(result, str):
            return result
        return "".join([chunk async for chunk in result])

    return asyncio.run(go())


def split(reply: str) -> tuple[str, str]:
    """(trace, visible text) of a streamed reply."""
    if "</think>" in reply:
        trace, visible = reply.split("</think>", 1)
        return trace.replace("<think>", "").strip(), visible.strip()
    return "", reply.strip()


def user(text=None, images=()):
    parts = [{"type": "text", "text": text}] if text else []
    parts += [{"type": "image_url", "image_url": {"url": url}} for url in images]
    return {"role": "user", "content": parts}


def data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def last_call():
    return FakeBackend.instances[-1].calls[-1]


def with_director(pipe, **decision):
    pipe.valves = pipe.Valves(IMAGE_SERVER_TOKEN=TOKEN, USE_DIRECTOR=True)
    return decision


# --- drawing, without the director -------------------------------------------------------


def test_text_to_image_returns_inline_image_download_link_and_trace(pipe, owui):
    events = []
    trace, visible = split(run(pipe, [user("a friendly cartoon apple, flat icon")], events=events))
    upload = owui.uploads[0]
    url = f"/api/v1/files/{upload['id']}/content"
    assert f"![a friendly cartoon apple, flat icon]({url})" in visible
    assert f"[Download PNG]({url}?attachment=true)" in visible
    assert upload["filename"].startswith("a-friendly-cartoon-apple-flat-icon-")
    assert upload["metadata"] == {"chat_id": "chat-1", "message_id": "msg-1"}
    assert owui.chat_links[0]["file_ids"] == [upload["id"]]
    assert owui.files[upload["id"]].user_id == "teacher"  # owned by the chatting user
    assert "**Prompt sent to the image model:** a friendly cartoon apple, flat icon" in trace
    assert "**Image model:** flux2-klein-4b · seed" in trace and "1024×1024" in trace
    assert last_call()["references"] == []
    assert events[0]["data"]["description"].startswith("Drawing your picture")
    assert events[-1]["data"] == {"description": "Done", "done": True, "hidden": True}


def test_trace_can_be_turned_off(pipe, owui):
    pipe.valves = pipe.Valves(IMAGE_SERVER_TOKEN=TOKEN, USE_DIRECTOR=False, SHOW_TRACE=False)
    reply = run(pipe, [user("a sun")])
    assert "<think>" not in reply and "[Download PNG]" in reply


def test_attached_picture_is_edited_and_keeps_its_shape(pipe, owui):
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (200, 30, 30)).save(buf, format="PNG")
    trace, visible = split(run(pipe, [user("turn this into a coloring page", [data_url(buf.getvalue())])]))
    call = last_call()
    assert len(call["references"]) == 1
    assert call["width"] > call["height"]  # landscape in, landscape out
    assert "**Reference pictures:** 1 (attached)" in trace
    assert "Changed the" not in visible


def test_follow_up_edits_previous_output_unless_new(pipe, owui):
    file_id = owui.add_file("teacher", png_bytes())
    history = [
        user("an owl icon"),
        {"role": "assistant", "content": f"<think>trace</think>\n![owl](/api/v1/files/{file_id}/content)\n\n[Download PNG](x)"},
    ]
    _, visible = split(run(pipe, history + [user("give it a red hat")]))
    assert len(last_call()["references"]) == 1
    assert "Changed the previous picture" in visible
    run(pipe, history + [user("a fox icon --new")])
    assert last_call()["references"] == [] and last_call()["prompt"] == "a fox icon"


def test_unopenable_previous_picture_is_reported_not_redrawn(pipe, owui):
    file_id = owui.add_file("someone-else", png_bytes())
    history = [user("x"), {"role": "assistant", "content": f"![x](/api/v1/files/{file_id}/content)"}]
    _, visible = split(run(pipe, history + [user("make it blue")]))
    assert "can't open the earlier picture" in visible
    assert FakeBackend.instances == []


def test_options(pipe, owui):
    reply = run(pipe, [user("a castle --portrait --n 2 --seed 7")])
    assert len(owui.uploads) == 2 and reply.count("[Download PNG]") == 2
    call = last_call()
    assert (call["width"], call["height"]) == (896, 1152)
    assert call["prompt"] == "a castle" and call["seed"] == 8  # second of two: 7, 8


def test_bad_options_explain_instead_of_calling_server(pipe, owui):
    assert "`--n`" in run(pipe, [user("a cat --n 99")])
    assert "`--seed`" in run(pipe, [user("a cat --seed abc")])
    assert "`--size`" in run(pipe, [user("a cat --size huge")])
    assert FakeBackend.instances == []


def test_help_and_empty_message(pipe, owui):
    assert "How to use this model" in run(pipe, [user("help")])
    assert "How to use this model" in run(pipe, [user("--new")])


def test_background_tasks_never_generate(pipe, owui, director):
    with_director(pipe)
    title = run(pipe, [], __task__="title_generation",
                __task_body__={"messages": [user("a friendly cartoon apple flat icon for worksheets --n 2")]})
    assert json.loads(title) == {"title": "a friendly cartoon apple flat icon"}
    assert json.loads(run(pipe, [], __task__="tags_generation")) == {"tags": []}
    assert json.loads(run(pipe, [], __task__="follow_up_generation")) == {"follow_ups": []}
    assert run(pipe, [], __task__="context_compaction") == ""
    assert FakeBackend.instances == [] and director.requests == []


def test_temporary_chat_is_not_linked(pipe, owui):
    run(pipe, [user("a sun")], chat_id="local:abc")
    assert owui.uploads[0]["metadata"] == {} and owui.chat_links == []


def test_friendly_errors(pipe, owui):
    pipe.valves = pipe.Valves(IMAGE_SERVER_TOKEN="wrong-token", USE_DIRECTOR=False)
    assert "rejected the token" in run(pipe, [user("a sun")])

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    pipe.valves = pipe.Valves(IMAGE_SERVER_TOKEN=TOKEN, USE_DIRECTOR=False)
    pipe._transport = httpx.MockTransport(refuse)
    assert "Couldn't reach the image server" in run(pipe, [user("a sun")])


def test_retry_once_after_connection_refused(pipe, owui):
    real = pipe._transport
    attempts = []

    class FlakyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.ConnectError("restarting", request=request)
            return await real.handle_async_request(request)

    pipe._transport = FlakyTransport()
    assert "[Download PNG]" in run(pipe, [user("a sun")])
    assert len(attempts) == 2


def test_unexpected_errors_close_trace_and_status(pipe, owui):
    owui.fail_uploads = True
    events = []
    trace, visible = split(run(pipe, [user("a sun")], events=events))
    assert "Something went wrong while making the picture (RuntimeError)" in visible
    assert events[-1]["data"]["done"] is True and "<think>" not in visible


def test_no_trace_without_streaming(pipe, owui):
    reply = run(pipe, [user("a sun")], stream=False)
    assert "<think>" not in reply and "[Download PNG]" in reply


def test_stop_closes_the_status_line(pipe, owui, pipe_module):
    async def slow_director(request):
        await asyncio.sleep(30)

    pipe.valves = pipe.Valves(IMAGE_SERVER_TOKEN=TOKEN, USE_DIRECTOR=True)
    pipe._ollama_transport = httpx.MockTransport(slow_director)
    events = []

    async def emitter(event):
        events.append(event)

    async def go():
        stream = await pipe.pipe({"messages": [user("a sun")], "stream": True}, {"id": "teacher"},
                                 __request__=object(), __event_emitter__=emitter,
                                 __chat_id__="c", __message_id__="m")

        async def consume():
            async for _ in stream:
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert events[0]["data"]["description"] == "Reading your message…"
    assert events[-1]["data"] == {"description": "Stopped", "done": True, "hidden": False}


def test_early_exits_leave_no_open_status(pipe, owui, director):
    with_director(pipe)
    director.decision = {"action": "draw", "kind": "icon", "edit_previous": True,
                         "subject": "make it blue", "reply": ""}
    file_id = owui.add_file("someone-else", png_bytes())
    history = [user("x"), {"role": "assistant", "content": f"![x](/api/v1/files/{file_id}/content)"}]
    events = []
    run(pipe, history + [user("make it blue")], events=events)
    assert events[-1]["data"]["done"] is True
    events.clear()
    asked = len(director.requests)
    run(pipe, [user("a cat --n 99")], events=events)
    assert events == [] and len(director.requests) == asked  # rejected before the director runs


# --- the director ------------------------------------------------------------------------


def test_director_reply_answers_in_text_without_drawing(pipe, owui, director):
    with_director(pipe)
    director.decision = {"action": "reply", "kind": "icon", "edit_previous": False, "subject": "",
                         "reply": "I used: a cute owl, flat vector icon."}
    history = [user("an owl"), {"role": "assistant",
                                "content": "<think>**Prompt sent to the image model:** a cute owl, flat vector icon</think>\n![owl](/api/v1/files/abc/content)\n\n[Download PNG](/api/v1/files/abc/content?attachment=true)"}]
    trace, visible = split(run(pipe, history + [user("what prompt did you use?")]))
    assert visible == "I used: a cute owl, flat vector icon."
    assert "**Director** (gemma4:12b" in trace and "reply" in trace
    assert FakeBackend.instances == []
    sent = director.requests[0]["messages"]
    assert sent[0]["role"] == "system"
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert "[picture]" in assistant["content"] and "Download PNG" not in assistant["content"]
    assert "a cute owl, flat vector icon" in assistant["content"]  # the trace reaches the director
    assert sent[-1] == {"role": "user", "content": "what prompt did you use?"}


def test_director_draw_uses_style_template_and_reply(pipe, owui, director):
    with_director(pipe)
    director.decision = {"action": "draw", "kind": "icon", "edit_previous": False,
                         "subject": "a cute owl", "reply": "Here's a cute owl!"}
    trace, visible = split(run(pipe, [user("a cute owl for my reading worksheet")]))
    assert last_call()["prompt"].startswith("a cute owl, flat vector icon, bold clean black outlines")
    assert visible.startswith("Here's a cute owl!")
    assert "draw · icon · new picture" in trace


def test_director_cannot_refuse_an_attached_picture(pipe, owui, director):
    with_director(pipe)
    director.decision = {"action": "reply", "kind": "coloring", "edit_previous": True,
                         "subject": "", "reply": "Done!"}
    run(pipe, [user("turn this into a coloring page", [data_url(png_bytes())])])
    assert len(last_call()["references"]) == 1
    assert "coloring book page" in last_call()["prompt"]
    attached = director.requests[0]["messages"][-1]["content"]
    assert "[The teacher attached 1 picture(s).]" in attached


def test_director_new_subject_does_not_edit_previous(pipe, owui, director):
    with_director(pipe)
    director.decision = {"action": "draw", "kind": "coloring", "edit_previous": False,
                         "subject": "a dinosaur", "reply": "Here's your dinosaur!"}
    file_id = owui.add_file("teacher", png_bytes())
    history = [user("an owl"), {"role": "assistant", "content": f"![owl](/api/v1/files/{file_id}/content)"}]
    _, visible = split(run(pipe, history + [user("now a coloring page of a dinosaur")]))
    assert last_call()["references"] == []
    assert "Changed the previous picture" not in visible


def test_director_failure_falls_back_to_raw_words(pipe, owui, director):
    with_director(pipe)
    director.fail = True
    trace, _ = split(run(pipe, [user("a red barn")]))
    assert "**Director:** unavailable" in trace
    assert last_call()["prompt"] == "a red barn"


def test_raw_flag_skips_the_director(pipe, owui, director):
    with_director(pipe)
    trace, _ = split(run(pipe, [user("exactly this: a blue triangle --raw")]))
    assert director.requests == [] and "skipped (--raw)" in trace
    assert last_call()["prompt"] == "exactly this: a blue triangle"


def test_storybook_style_is_added_only_when_none_named(pipe_module):
    assert pipe_module.build_prompt("storybook", "a bear picnic").endswith("soft watercolor style")
    assert "watercolor" not in pipe_module.build_prompt("storybook", "a bear picnic in colored pencil")


def test_open_webui_012_attached_files_block_is_ignored(pipe_module):
    text, images = pipe_module.split_content([
        {"type": "text", "text": "<attached_files>\n<file id='1'/>\n</attached_files>\nmake it blue"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ])
    assert text == "make it blue" and images == ["data:image/png;base64,AAAA"]


def test_asking_about_an_option_gets_an_answer(pipe, owui, director):
    with_director(pipe)
    director.decision = {"action": "reply", "kind": "other", "edit_previous": False, "subject": "",
                         "reply": "It removes the white background."}
    _, visible = split(run(pipe, [user("what does --transparent do?")]))
    assert visible == "It removes the white background." and FakeBackend.instances == []
    assert director.requests[0]["messages"][-1]["content"] == "what does --transparent do?"


def test_options_still_force_a_picture(pipe, owui, director):
    with_director(pipe)
    director.decision = {"action": "reply", "kind": "other", "edit_previous": False, "subject": "",
                         "reply": "Sure."}
    run(pipe, [user("a castle --portrait")])
    assert (last_call()["width"], last_call()["height"]) == (896, 1152)


def test_director_sees_the_trace_without_think_tags(pipe_module):
    history = [user("an owl"), {"role": "assistant",
               "content": "<think>**Prompt sent to the image model:** an owl icon</think>\n![owl](/x)"},
               user("what prompt?")]
    out = pipe_module.history_for_director(history, 2, "what prompt?", 0)
    assert "<think>" not in out[1]["content"] and "an owl icon" in out[1]["content"]


def test_director_reply_cannot_open_reasoning_tags(pipe, owui, director, pipe_module):
    with_director(pipe)
    director.decision = {"action": "reply", "kind": "other", "edit_previous": False, "subject": "",
                         "reply": "The <thinking> tag shows my notes."}
    _, visible = split(run(pipe, [user("what is the thinking tag?")]))
    assert "<thinking>" not in visible and "<\u200bthinking>" in visible
    assert pipe_module.safe_reply("data: hi").startswith("\u200bdata:")


def test_alt_text_drops_backslashes(pipe_module):
    assert "\\" not in pipe_module.alt_text("a path like C:\\ ends with \\")
