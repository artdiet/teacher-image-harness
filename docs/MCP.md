# Teach Image in Claude Desktop and Cowork (MCP)

The **Teach Image** extension lets Claude make and edit classroom pictures with
the image server on your Mac, from Claude Desktop chat or from a Cowork task,
on that Mac or on any computer on the same network. Pictures are drawn on the
Mac's GPU, so there's no per-picture cost, and the full-size pictures stay on
your computers. Claude itself runs in Anthropic's cloud: the prompts it writes
and the small previews it looks at become part of your Claude conversation,
like anything else you share with it.

```
 Any computer on your network                     The image-server Mac
 ┌──────────────────────────────────┐             ┌───────────────────────────────┐
 │ Claude Desktop (chat or Cowork)  │             │ image server (launchd)        │
 │   └─ Teach Image extension       │  HTTP, LAN  │   job queue → FLUX.2 klein    │
 │      local MCP server (stdio)    │────────────▶│   images, references, jobs    │
 │      saves full PNGs here        │  device     │   port 7865, token required   │
 └──────────────────────────────────┘  token      └───────────────────────────────┘
```

**Why an extension and not a "custom connector":** connectors added under
Settings > Connectors are called from Anthropic's cloud, which can't reach a
server inside your home network. A desktop extension is a small local MCP
server that Claude Desktop starts on your computer, and Cowork reaches it
through the Desktop app. So it works for chat and Cowork alike, as long as
Claude Desktop is open on that computer.

## 1. On the image-server Mac (once)

```bash
scripts/enable_lan.sh              # listen on the network (LAN only), allow this Mac's name and IP
scripts/install_image_server.sh    # restart; then it checks the network address
```

The restart ends with one of these lines from `scripts/check_lan.sh`:

- `reachable on the network at http://<mac-name>.local:7865 and http://192.168.x.y:7865`:
  done.
- `blocked: … the macOS firewall is holding connections to the server's
  Python`: click **Allow** if macOS asks about `python3.12`, or run the two
  `sudo socketfilterfw` commands it prints, then `scripts/check_lan.sh` again.
  Redo this after uv upgrades Python (the approval is per Python build).

Then make a token for each computer that will use the extension (this Mac
included). Each can be revoked on its own, and the server's own token from
`.env` (Open WebUI's) is refused from other computers:

```bash
scripts/client_tokens.sh add classroom-pc     # prints the token once: copy it
scripts/client_tokens.sh list
scripts/client_tokens.sh revoke classroom-pc  # takes effect within seconds, no restart
```

Only SHA-256 hashes are stored (`client_tokens.json` in the data folder, mode
600).

Tips: give the Mac a fixed address in your router (or use its
`<name>.local` address), and keep it from sleeping while people use it.

## 2. Build the extension (once per version)

On a computer with Node 20.10 or later (or 18.20+) and this repository:

```bash
cd teach-image-mcp
npm ci
npm run build          # → dist/teach-image.mcpb (also runs a smoke test)
```

Copy `dist/teach-image.mcpb` to each computer (USB stick, shared folder, …).
It contains no settings or tokens.

## 3. Install on each computer

1. Double-click `teach-image.mcpb` (or drag it onto Claude Desktop's
   Settings > Extensions page) and choose **Install**. The extension isn't
   signed, so expect Claude Desktop to ask you to confirm.
2. Fill in the settings:
   - **Image server address**: `http://<mac-name>.local:7865` or
     `http://<mac-ip>:7865` (`scripts/enable_lan.sh` prints both). On the
     image-server Mac itself, keep `http://127.0.0.1:7865`.
   - **API token**: that computer's token from `client_tokens.sh add`.
   - **Save pictures in** (optional): empty means `Generated Images` inside
     your Claude folder (`Documents/Claude`, or `~/Claude`), where Cowork can
     use them; otherwise `Pictures/Teach Image`.
3. In Settings > Extensions, make sure **Teach Image is switched on**. Claude
   Desktop doesn't switch on an extension that was installed before its
   required settings (the token) were filled in.
4. **Quit Claude Desktop completely and reopen it** (on a Mac, Claude >
   Quit, not just closing the window).
5. **Allow Claude to use the tools.** In the first chat or task that wants
   them, Claude shows a link to choose when it may use Teach Image's tools.
   Until you choose, chats and tasks don't see the tools at all (Claude may
   say it has no image tools), even though the extension is installed. The
   extension starts at that moment; asking again then finds the tools.
6. Mac only: when macOS asks whether Claude may find devices on your local
   network, choose **Allow**. If you missed it: System Settings > Privacy &
   Security > Local Network > Claude.
7. Check it: ask Claude *"check the Teach Image server status"*.

The extension runs on Claude Desktop's built-in Node, so nothing else needs
installing on Windows or macOS.

## 4. Using it

Ask in plain words; Claude picks the tool and the style:

- *"Make a worksheet icon of a friendly owl holding a pencil, transparent background."*
- *"Three coloring pages of farm animals for my kindergarten class."*
- *"A storybook page: a bear and a bunny having a picnic, landscape."*
- *"Give the owl a red scarf."* (edits the last picture)
- *"Turn ~/Pictures/class-pet.jpg into a coloring page."*
- *"Save the bear as a reference called Benny, then draw Benny planting a tree."*

Each finished picture is saved as a full-size PNG on that computer (Claude
says where) and shown as a smaller preview. The first picture after the
server has been idle takes a few seconds longer while the model loads.
Requests for several pictures can take longer than one tool call allows;
Claude then gets a job id and collects the pictures with a second call. A
finished picture is never lost on the way: after a dropped connection, Claude
gets the job id and can collect it later. If the server restarts while a
picture is still being drawn, the job reports that it was interrupted, and
Claude can ask again. Pressing Stop on a picture cancels it on the server.

**Cowork:** keep Claude Desktop open on the computer while a task runs; Cowork
reaches the extension through it.

**Several pictures at once** (a worksheet's icon set) are made one after
another: the Mac's GPU draws one picture at a time, and Claude's link to the
computer runs one tool call at a time. Expect about 10 seconds of drawing per
1024×1024 icon plus about 3 seconds per call for the trip to Claude and back:
eight icons took about two minutes in a test. Pictures land in the `Generated Images`
folder inside your Claude folder, so a task working in that folder can put
them into documents and slides.

### Tools

| Tool | What it does |
|---|---|
| `generate_image` | new pictures: prompt, style (icon, coloring, storybook, none), aspect or size, 1–4 variations, seed, references, transparent background, file name, subfolder |
| `edit_image` | change a picture made earlier (`image_id`) or a file on this computer (`source_path`) |
| `get_image_job` | collect a picture that took longer than one call |
| `cancel_image_job` | stop a waiting or running picture |
| `get_image` | show a picture made earlier with its prompt and seed; optionally save a copy |
| `list_recent_images` | recent pictures with ids, prompts and seeds |
| `save_reference`, `list_references`, `delete_reference` | reference pictures (characters, style samples) for consistent pictures |
| `image_server_status` | connection, token, models, queue, where pictures are saved |

## Manual setup (without the .mcpb)

For other MCP clients, or to run from a checkout, start the bundled server with
Node 18 or later and the same settings as environment variables. Unzip the `.mcpb`
(it's a zip) or use `teach-image-mcp/dist/teach-image/server/index.mjs` after
a build.

Claude Desktop, `claude_desktop_config.json` (Settings > Developer > Edit
Config): see [teach-image-mcp/claude_desktop_config.example.json](../teach-image-mcp/claude_desktop_config.example.json).
This route needs Node installed and the token sits in that file in plain text.

Claude Code:

```bash
claude mcp add teach-image \
  -e IMAGE_SERVER_URL=http://<mac-name>.local:7865 \
  -e IMAGE_SERVER_TOKEN=<device token> \
  -e UV_THREADPOOL_SIZE=16 \
  -- node /path/to/teach-image/server/index.mjs
```

| Variable | Meaning |
|---|---|
| `IMAGE_SERVER_URL` | server address (default `http://127.0.0.1:7865`) |
| `IMAGE_SERVER_TOKEN` | this computer's token |
| `IMAGE_OUTPUT_DIR` | where full-size PNGs go (default above) |
| `IMAGE_WAIT_SECONDS` | how long a call waits for a picture, 5–40 (default 40). Claude Desktop stops local tool calls at about 60 s, so a call does everything (upload, wait, saving) within 50 s and hands back a job id for anything longer. |
| `IMAGE_INPUT_ROOTS` | extra folders pictures may be sent from, separated by `:` (`;` on Windows); the home and output folders are always allowed |
| `UV_THREADPOOL_SIZE` | set it to `16` (the extension does): Node's file and name lookups share a few threads, and a pictures folder on a network drive that stops answering could otherwise tie them all up |

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Can't reach the image server" | Is the Mac awake, on the same network, and does `scripts/check_lan.sh` on it say reachable? On a Mac client, allow Claude under Privacy & Security > Local Network. Check the address in the extension settings. |
| "doesn't accept the address" | The Mac's IP changed or you used a name it doesn't know: run `scripts/enable_lan.sh` and `scripts/install_image_server.sh` on it, or use `<name>.local`. |
| `<name>.local` doesn't work on a Windows PC | Use the Mac's IP address instead (and reserve it in the router). |
| "didn't accept this computer's token" | Wrong or revoked token: `scripts/client_tokens.sh add <name>` and paste the new one. The token in the server's `.env` only works on the server Mac itself. |
| "The image server is busy" | Several pictures are waiting already; try again in a minute. |
| "too many open connections from this computer" | Many requests at once from one computer; try again in a moment. |
| Tools don't appear | Check that Settings > Extensions shows Teach Image switched on with its token filled in, quit and reopen Claude Desktop, then use the link Claude shows to allow its tools and ask again. `~/Library/Logs/Claude/main.log` says `[localMcpBridge] announcing Teach Image: 10 tool(s)` once chats and tasks can see them (`announcing an empty local-MCP group` means it isn't running yet). |
| Cowork can't use the tools | Claude Desktop must be open on the computer that has the extension. |

Logs: the extension writes to Claude Desktop's MCP logs
(`~/Library/Logs/Claude/mcp-server-*.log` on a Mac,
`%APPDATA%\Claude\logs` on Windows), starting with a `[teach-image] … ready`
line that shows the address and whether a token is set (never the token).
Cowork's connection to local extensions shows up as `[localMcpBridge]` in
`~/Library/Logs/Claude/main.log`. Server side:
`~/Library/Logs/teacher-image-harness/image-server.log`, `scripts/doctor.sh`,
and `scripts/smoke_test_api.sh http://<mac-name>.local:7865`, which tests every
endpoint from any computer with curl (see [API.md](API.md#testing-every-endpoint)).

## Security

- **LAN only.** The server answers only private-network addresses
  (`IMAGE_SERVER_PRIVATE_CLIENTS_ONLY`), only under the names listed in
  `IMAGE_SERVER_ALLOWED_HOSTS`, and needs a token on every route except
  `/health`. Never forward port 7865 on your router.
- **Plain HTTP on your network.** Someone on the same network who can watch
  traffic could read a token and pictures; use it on a network you trust, and
  revoke a token if a computer leaves.
- **One token per computer**, stored on the server only as a hash; revoking
  one doesn't affect the others or Open WebUI. Open WebUI's token works only
  from the server Mac itself. If the token file gets damaged, device tokens are
  refused until it's fixed, rather than a revoked one staying valid.
- **What Claude sees:** the prompt, the saved file's path on that computer,
  `image_id`, seed and a downscaled preview of each picture. Uploaded photos
  are stored on the image server; Claude sees previews of results (and of an
  upload, if asked to show it).
- **One shared library:** pictures, jobs and references made through the
  extension are shared by every computer with a token, so one computer's
  Claude can list and show pictures another made. Open WebUI's pictures are
  kept separately and aren't visible to the extension.
- **Connection limits:** at most 16 open connections per computer, and a
  connection that sends nothing for 10 seconds, or stalls in the middle of a
  transfer, is closed; a stopping server waits at most 25 seconds for open
  requests and pictures. One misbehaving or sleeping device can't tie up the
  server.
- The extension keeps its token in Claude Desktop's extension settings
  (marked sensitive), not in any file in this repository.
- It sends a file to the server only when asked to edit it or save it as a
  reference, and only pictures (checked by their bytes, 20 MB at most) from
  your home folder, the pictures folder or `IMAGE_INPUT_ROOTS`.
- It saves new files only inside the pictures folder and never overwrites an
  existing file.
- The firewall approval in step 1 applies to that Python build, so any
  program run with the same uv-managed Python may accept connections too.
