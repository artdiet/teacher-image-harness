# teach-image-mcp

The **Teach Image** desktop extension: a local MCP server (stdio) that Claude
Desktop starts on each computer and that forwards tool calls to the image
server's job API ([docs/API.md](../docs/API.md)). Setup for users is in
[docs/MCP.md](../docs/MCP.md).

```
src/index.js     entry: stdio transport, settings from the environment
src/server.js    the MCP tools; saves finished PNGs, returns previews + facts as text
src/client.js    REST client; turns failures into messages for the person using Claude
src/files.js     safe file names, never-overwrite saving, which pictures may be uploaded
src/config.js    environment variables and defaults
manifest.json    MCPB manifest (settings page: server address, token, pictures folder)
scripts/build.mjs  esbuild bundle → dist/teach-image/ → mcpb validate + pack → smoke test
```

```bash
npm ci
npm test         # tools against a mock image server (no GPU), plus the bundle once built
npm run build    # dist/teach-image.mcpb
```

Running the tests from source needs Node 18.20+ or 20.10+ (the source imports
package.json with an import attribute). The built bundle has no such
requirement: esbuild inlines the JSON and targets Node 18.

Design notes (why it looks like this):

- **Local stdio, not a remote connector.** Cowork and claude.ai connectors call
  MCP servers from Anthropic's cloud, which can't reach a home LAN. Claude
  Desktop runs extensions on the user's machine and bridges them into Cowork.
- **One time budget per call.** Claude Desktop stops local tool calls at about
  60 s, so everything a call does (reading a picture to send, uploading,
  waiting, downloading, writing the PNGs) fits in 50 s, measured on a clock
  that system time changes don't move: it waits up to 40 s
  (`IMAGE_WAIT_SECONDS`, 5 to 40; the last look at the job is at the end of
  the wait), saves the full PNGs first and adds previews only with time to
  spare (5 s at most). Writing a PNG always gets at least 1.5 s, even just
  before the deadline, so a download that finished in the nick of time
  doesn't make a healthy folder look slow. A picture that isn't
  finished or saved in time (also when the pictures folder is on a slow or
  sleeping drive) comes back as a `job_id` for `get_image_job`, which saves
  only what is still missing, so no file is saved twice. Two calls collecting
  the same job take turns saving, and the one waiting gives up within its own
  budget. A picture on this computer that is too slow to read (an iCloud file
  that must download first) is reported as that before anything is sent.
- **Stuck file operations.** Node shares a few threads (4 by default) between
  all file work and looking up the server's name, and a file operation that
  never returns (a network folder that went away, a macOS question about
  folder access left unanswered) holds one for as long as it hangs. So while
  a write into a folder, or a read of a picture, is still going, no other is
  started there: another call waits for it only until it counts as slow
  (1.5 s), then says so at once, with the `job_id`. The real paths of the
  folders pictures may be sent from are asked for once, and only when a
  picture isn't inside one as written. As a backstop, the manifest starts
  Node with 16 threads (`UV_THREADPOOL_SIZE`, read only at startup, so it
  is set in the environment rather than in code).
- **A job is never lost.** Once the server has made a job, every result names
  its `job_id`. Brief trouble while waiting (a dropped connection, a server
  restart, a 5xx) is retried until the wait is over; a connection refused on a
  call's first request (the server restarting) is tried once more after 3 s.
  If the answer to the request that makes the job is lost (a timeout, a
  dropped connection), the server may have made it anyway, so the result
  says to wait a minute and check `list_recent_images` (which lists only
  finished pictures) before asking again. A `503` is an answer, not a lost
  one: its reason (busy, restarting, too many open connections from this
  computer) is shown as the server gives it, and the request isn't repeated.
- **Stopping.** Stopping a call that waits for a picture (`generate_image`,
  `edit_image` or `get_image_job`) cancels that picture on the server, even
  when the call is stopped while the job is being submitted; stopped during
  the pause before retrying a refused connection, nothing is sent at all. A
  call that Claude Desktop stops for taking too long ("timed out") leaves its
  job running, so the picture can still be collected.
- **Results stay small.** Claude Desktop rejects tool results over ~150k
  characters, so previews share a ~75 KB budget (the server's `budget`
  parameter) and the full PNGs are saved to disk. An answer that isn't the
  image server's (another program at the address, a login page) is never
  echoed, and the server's own details are cut at 300 characters. Pictures
  are checked by their bytes, so such a page is never saved as a PNG or shown
  as a preview. No `outputSchema` or `structuredContent`: Claude clients have
  mishandled them.
- **Node quirks.** Uploads are sent as finished bytes rather than `FormData`:
  Node 24's fetch ends the whole process when a `FormData` body meets a call
  that was already stopped. Every request asks for a connection of its own:
  Node 26's fetch holds back a request on a kept-alive connection that was
  idle for half a second or more, which would slow every poll (Node 18 to 24
  don't). Both matter for the manual setup too, which runs on the computer's
  own `node` rather than Claude Desktop's.
- **Addresses.** `localhost` becomes `127.0.0.1` (the server listens on IPv4,
  and Node 18 tries localhost's IPv6 address first), and an `http://` address
  without a port gets the server's 7865 rather than 80 (an explicit port, and
  an `https://` address, are kept as given).
- **Facts go in text.** The model may only see an image on its next turn, so
  the text carries the saved path, `image_id`, seed and the exact prompt.
- **One bundled file.** Dependencies are bundled with esbuild so the extension
  runs on Claude Desktop's built-in Node (v24 in Claude 2.42) with no install
  step; the build targets Node 18.
