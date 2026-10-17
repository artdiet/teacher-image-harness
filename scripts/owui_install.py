"""Install or update the Teach Image pipe inside a running Open WebUI container.

Runs INSIDE the container with Open WebUI's own Python and data layer, so no
API key or admin token is needed. Don't run it directly; use
scripts/install_openwebui.sh, which backs up the database first and passes:

  argv[1]                 path to the pipe source inside the container
  argv[2]                 optional path to the model's avatar PNG
  IMAGE_SERVER_URL        image server address as seen from the container
  IMAGE_SERVER_TOKEN      token for the pipe's valve (never printed)
  OWUI_MODEL_NAME         name shown in the model list
  OWUI_PUBLIC             1/true = every account (also future ones) may use the model,
                          0/false = remove that, blank = leave access as set in Open WebUI
  OWUI_GRANT_ALL          1/true = re-grant read access to every current non-admin account

Re-running keeps what an admin changed in Open WebUI: valve edits, the model's
picture, description and suggestions (unless still at the installer's last
values), its on/off switch and its access list.

Written against Open WebUI 0.9.2; also handles 0.12's changed plugin loader.
Changes appear after a browser reload.
"""

import asyncio
import base64
import hashlib
import json
import os
import pathlib
import sys

from open_webui.env import DATA_DIR
from open_webui.internal.db import async_engine
from open_webui.models.access_grants import AccessGrants
from open_webui.models.functions import FunctionForm, FunctionMeta, Functions
from open_webui.models.models import ModelForm, ModelMeta, ModelParams, Models
from open_webui.models.users import Users
from open_webui.utils.plugin import load_function_module_by_id, replace_imports

FUNCTION_ID = "teach_image"
FUNCTION_NAME = "Teach Image"
DESCRIPTION = (
    "Makes classroom pictures on this computer: worksheet icons, coloring pages and "
    "storybook art. Attach a picture to change it, ask questions about what it made, "
    "or type help for options."
)
CAPABILITIES = {
    "vision": True,  # both needed for image attachments
    "file_upload": True,
    "status_updates": True,
    "file_context": False,  # no RAG on attachments
    "builtin_tools": False,  # keeps Open WebUI 0.12+ from rewriting the prompt
    "web_search": False,
    "image_generation": False,
    "code_interpreter": False,
    "terminal": False,
    "citations": False,
    "usage": False,
}
SUGGESTIONS = [
    {"title": ["Worksheet icon", "a cheerful apple"],
     "content": "a cheerful apple icon for a counting worksheet"},
    {"title": ["Coloring page", "a turtle on a rock"],
     "content": "a coloring page of a cute turtle sitting on a rock"},
    {"title": ["Storybook page", "fox and rabbit in the rain"],
     "content": "a storybook page: a small fox and a rabbit sharing an umbrella in the rain"},
    {"title": ["Help", "what can this model do?"], "content": "help"},
]
MANAGED = "teach_image_installer"  # meta key: fingerprints of what the installer last wrote


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def flag(name: str):
    """1/true/yes/on -> True, 0/false/no/off -> False, blank -> None."""
    value = (os.environ.get(name) or "").strip().lower()
    if value == "":
        return None
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    sys.exit(f"{name} must be 1, 0 or blank, got {value!r}")


async def main() -> None:
    content = replace_imports(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    avatar = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
    model_name = os.environ.get("OWUI_MODEL_NAME") or "Teach Image"
    public, grant_all = flag("OWUI_PUBLIC"), flag("OWUI_GRANT_ALL")
    admin = await Users.get_super_admin_user()
    if admin is None:
        sys.exit("no admin account found")

    # Validate the code the way the admin UI does, under a throwaway id.
    # 0.9.x returns (instance, type, frontmatter); 0.12+ adds the module.
    probe = f"{FUNCTION_ID}__probe"
    loaded = await load_function_module_by_id(probe, content=content)
    instance, kind, frontmatter = loaded[0], loaded[1], loaded[2]
    sys.modules.pop(f"function_{probe}", None)
    if kind != "pipe":
        sys.exit(f"expected a Pipe class, found {kind}")

    # Valves: keep admin edits, refresh address and token, and store only real
    # overrides so later default changes in the pipe take effect. Validate first.
    existing_function = await Functions.get_function_by_id(FUNCTION_ID)
    saved = (await Functions.get_function_valves_by_id(FUNCTION_ID) or {}) if existing_function else {}
    updates = {"IMAGE_SERVER_URL": os.environ.get("IMAGE_SERVER_URL") or "http://host.docker.internal:7865"}
    if os.environ.get("IMAGE_SERVER_TOKEN"):
        updates["IMAGE_SERVER_TOKEN"] = os.environ["IMAGE_SERVER_TOKEN"]
    defaults = instance.Valves().model_dump()
    merged = instance.Valves(**{**saved, **updates}).model_dump()
    valves = {k: v for k, v in merged.items() if k in updates or v != defaults.get(k)}

    meta = FunctionMeta(description=DESCRIPTION, manifest=frontmatter)
    if existing_function is None:
        function = await Functions.insert_new_function(
            admin.id, "pipe",
            FunctionForm(id=FUNCTION_ID, name=FUNCTION_NAME, content=content, meta=meta),
        )
        (DATA_DIR / "cache" / "functions" / FUNCTION_ID).mkdir(parents=True, exist_ok=True)
        action = "created"
    else:
        function = await Functions.update_function_by_id(
            FUNCTION_ID,
            {"name": FUNCTION_NAME, "content": content, "meta": meta.model_dump(), "type": "pipe"},
        )
        action = "updated"
    if function is None:
        sys.exit("writing the function failed; see: docker logs open-webui")
    if not function.is_active:
        function = await Functions.update_function_by_id(FUNCTION_ID, {"is_active": True})
    await Functions.update_function_valves_by_id(FUNCTION_ID, valves)

    # Model row with the pipe's id: overrides its name/capabilities and carries the
    # access grants (the same pattern as the existing Ollama model overrides).
    wanted = {"description": DESCRIPTION, "suggestion_prompts": SUGGESTIONS,
              "tags": [{"name": "images"}]}
    if avatar and avatar.is_file():
        wanted["profile_image_url"] = "data:image/png;base64," + base64.b64encode(avatar.read_bytes()).decode()
    current = await Models.get_model_by_id(FUNCTION_ID)
    if current is None:
        meta = {**wanted, "capabilities": CAPABILITIES,
                MANAGED: {key: fingerprint(value) for key, value in wanted.items()}}
        form = ModelForm(id=FUNCTION_ID, base_model_id=None, name=model_name, is_active=True,
                         params=ModelParams(), meta=ModelMeta(**meta))
        model = await Models.insert_new_model(form, admin.id)
        created = True
    else:
        old = current.meta.model_dump() if hasattr(current.meta, "model_dump") else dict(current.meta or {})
        last = old.get(MANAGED) or {}
        meta, managed = dict(old), dict(last)
        for key, value in wanted.items():
            # Replace only values the installer wrote itself; an admin's edits stay, and so
            # does the fingerprint of the installer's value, so they keep staying.
            if key not in old or not last or fingerprint(old.get(key)) == last.get(key):
                meta[key] = value
                managed[key] = fingerprint(value)
        meta["capabilities"] = {**(old.get("capabilities") or {}), **CAPABILITIES}
        meta[MANAGED] = managed
        form = ModelForm(id=FUNCTION_ID, base_model_id=None, name=model_name,
                         is_active=current.is_active, params=current.params, meta=ModelMeta(**meta))
        model = await Models.update_model_by_id(FUNCTION_ID, form)  # leaves grants alone
        created = False
    if model is None:
        sys.exit("writing the model failed; see: docker logs open-webui")

    # Access: on first install, read access for every non-admin account (admins see
    # everything). Later runs keep the admin's access list unless asked otherwise.
    existing = await AccessGrants.get_grants_by_resource("model", FUNCTION_ID)
    grants = {(g.principal_type, g.principal_id, g.permission) for g in existing}
    before = set(grants)
    if created or grant_all:
        listing = await Users.get_users()
        users = listing.get("users", []) if isinstance(listing, dict) else listing
        grants |= {("user", u.id, "read") for u in users if u.role == "user"}
    everyone = ("user", "*", "read")
    if public is True:
        grants.add(everyone)
    elif public is False:
        grants.discard(everyone)
    if grants != before:
        await AccessGrants.set_access_grants(
            "model", FUNCTION_ID,
            [{"principal_type": t, "principal_id": p, "permission": perm} for t, p, perm in sorted(grants)],
        )

    readers = sum(1 for t, p, perm in grants if perm == "read" and p != "*")
    print(f"function '{FUNCTION_ID}' {action}, active={function.is_active}")
    print(f"model '{model_name}' (id {FUNCTION_ID}) ready; read access for {readers} account(s)"
          + (" + every account" if everyone in grants else ""))
    print("valves: token " + ("set" if merged.get("IMAGE_SERVER_TOKEN") else "MISSING")
          + f", server {merged['IMAGE_SERVER_URL']}")
    await async_engine.dispose()


asyncio.run(main())
