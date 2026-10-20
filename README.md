# Self-hosted MCP and API stack

Python services that expose a local Bible database, Flux image generation, and ACE-Step music generation as [MCP](https://modelcontextprotocol.io) servers, plus a small bearer-token gateway in front of the two servers that are meant to be reachable from the internet. A read-only stack status server and a patched ACE-Step web UI sit beside them.

This repository is the application code. It does not include model weights, the ComfyUI or ACE-Step images, a Bible SQLite file, or a reverse-proxy config. Those stay outside the repo and are named below where the code expects them.

## Services

| Directory | Process | Container port | MCP name | Reached from the internet? |
|---|---|---|---|---|
| `bible-api/` | FastMCP | 8000 | Bible API | No. Local only. |
| `image-api/` | FastMCP | 8010 | `imagegen` | Yes, through `mcp-gateway` |
| `music-api/` | FastMCP | 8030 | `musicgen` | Yes, through `mcp-gateway` |
| `stack-api/` | FastMCP | 8020 | Stack Status | No. Local only. |
| `mcp-gateway/` | Starlette proxy | 8040 | — | Yes. This is the public entry point. |
| `studio-ui/` | nginx + the upstream Node UI | 3000 | — | Optional. A web UI, not an MCP server. |

`tools/mcpcall.py` is a stdlib test client. `mcp-gateway/mcp-token.py` mints and revokes gateway tokens. Neither is a service.

## Architecture

Remote clients talk only to the gateway. The MCP servers stay unauthenticated and are reached on the Docker network (or on localhost for a client on the same machine).

```
MCP client
  │  Authorization: Bearer <token>
  ▼
reverse proxy  ──►  mcp-gateway:8040/{musicgen|imagegen}/mcp
                      │  checks the token hash, rate limit, and body size
                      │  sets X-MCP-Gateway: <token name>
                      ▼
                 music-api:8030/mcp    image-api:8010/mcp
                      │                      │
                      │                      └── comfyui:8188  (Flux, on demand)
                      ├── acestep:8001                 (ACE-Step REST, on demand)
                      ├── acestep-gradio:7860          (studio / XL)
                      ├── studio-ui:3000
                      └── stack-api:8020/gpu           (read-only GPU snapshot)

browser  ──►  reverse proxy  ──►  image-api /images/{file}.png
                             ──►  music-api /audio/{file}.mp3
                             ──►  studio-ui (optional)

bible-api:8000/mcp and stack-api:8020/mcp are not in the gateway's upstream map.
```

`image-api`, `music-api`, and `stack-api` talk to Docker through `DOCKER_HOST`. Point that at a proxy that allows only the calls you intend. The gateway is not given a Docker client.

Image and music generation share one GPU. Each server refuses or warns when the other side's model container is running. `stack-api` is the only service that reads the GPU directly (NVML).

## What each part does

### bible-api

Local Bible lookup over SQLite (`aiosqlite`). Tools, registered in `routers/`:

| Tool | Purpose |
|---|---|
| `translations` | List loaded translations |
| `books`, `book` | List books, or look one up by number, name, or abbreviation |
| `verse`, `passage`, `chapter` | Fetch one verse, a verse range, or a whole chapter |
| `compare` | The same verse in several translations |
| `search` | Full-text search (`verses_fts`) |
| `strongs`, `strongs_verses` | Strong's lexicon entry, and verses that use that number |
| `interlinear`, `xref` | Word-level original-language data for one verse |

The database is not built here. On startup the server requires `DB_PATH` (default `/data/bibles.db`) and reads `books`, `translations`, `verses`, `verses_fts`, `lexicon`, and `word_occurrences`. `GET /health` is the health check. The MCP endpoint is `POST/GET /mcp` (streamable HTTP).

### image-api (`imagegen`)

Wraps a ComfyUI Flux workflow (`image-api/flux_workflow_api.json`). The workflow loads `flux1-dev-Q5_K_S.gguf`, `t5xxl_fp8_e4m3fn.safetensors`, `clip_l.safetensors`, and `ae.safetensors` from ComfyUI's own model directories.

Tools: `generate` (background job), `job_status`, `status`, `gpu_status`, `stop_model`.

`generate` starts the `comfyui` container if it is stopped, runs the workflow, copies the PNG into `OUTPUT_DIR`, and asks ComfyUI to free VRAM. An idle loop stops ComfyUI after `COMFYUI_IDLE_TIMEOUT` seconds (default 600). PNGs older than `IMAGE_RETENTION_DAYS` (default 30) are deleted.

Routes: `GET /images/{filename}` (only a 32-hex-character name plus `.png`) and `GET /health`. Returned URLs use `IMAGE_URL_BASE`, which should be the public prefix that proxies this route.

`gpu_status` calls `STACK_API_URL` (default `http://stack-api:8020/gpu`). It does not read NVML itself.

### music-api (`musicgen`)

Drives ACE-Step 1.5. Three pathways, one GPU, so only one ACE-Step container runs at a time:

- **REST 2B** (`acestep` at `http://acestep:8001`): `generate_song`, `generate_quick`, `modify_song`, `describe_track`. Started on demand and stopped after `MUSIC_IDLE_TIMEOUT` (default 600 seconds). `describe_track` loads the LM named by `DESCRIBE_LM`, then stops the container.
- **XL** (`acestep-xl` or `acestep-xl-sft`, plus `studio-ui`): `generate_song_xl`. Started on demand and stopped after `XL_IDLE_TIMEOUT` (default 900 seconds). The XL container is expected to carry the Docker network alias `acestep-gradio`, because the UI calls `http://acestep-gradio:7860`.
- **Studio** (`acestep-gradio` or XL, left running): `studio_on`, `studio_off`, `studio_status`, `studio_generate`, `studio_job`. Also exposed as `GET /studio/status` and `POST /studio/on` and `POST /studio/off`.

Also: `list_tracks`, `job_status`, `status`, `gpu_status`, `stop_model`.

`list_tracks` and `modify_song` read files from `UPLOAD_DIR` (default `/uploads`). `describe_track` stages a copy under `INBOX_DIR` (default `/inbox`). Generated MP3s go to `OUTPUT_DIR` (default `/data`) and are removed after `MUSIC_RETENTION_DAYS` (default 30). Public URLs use `AUDIO_URL_BASE`.

`GET /audio/{filename}` serves only a 32-hex-character name plus `.mp3`.

Studio and XL generation talk to the studio UI's HTTP API. The UI's own login is a passwordless setup call; `STUDIO_UI_USERNAME` is the username sent to `POST /api/auth/setup`. Real access control for browsers is whatever you put at the reverse proxy. `STUDIO_PUBLIC_URL` is prefixed onto audio paths the UI returns.

For the non-distilled XL model, music-api writes `/flags/dcw_off`. The `studio-ui` image is patched to turn DCW off while that file exists. Mount the same directory at `/flags` in both containers.

`XL_THINKING_DEFAULT=1` and `XL_COT_CAPTION=1` turn on the LM planner and caption rewrite. The defaults are off.

### stack-api

Read-only view of the Docker containers and the host GPU. Tools: `stack_status`, `stack_logs`, `system_resources`, `gpu_status`, `gpu_processes`. `GET /gpu` returns the same GPU reading plus per-process VRAM, which is what image-api and music-api call. `GET /health` is the health check.

`gpu_processes` and `GET /gpu` open `/proc/<pid>/comm` and `/proc/<pid>/cgroup` to tell a container from a host process. That only works if this container can see the host process table. `pynvml.nvmlInit()` runs at import, so the process needs an NVIDIA device. There is no code path that starts, stops, or execs a container; keep `DOCKER_HOST` pointed at a read-only Docker API.

### mcp-gateway

Proxies `GET`, `POST`, and `DELETE` on `/{server}/mcp` to:

- `musicgen` → `http://music-api:8030/mcp`
- `imagegen` → `http://image-api:8010/mcp`

Anything else is 404. Adding a key to `UPSTREAMS` in `mcp-gateway/main.py` publishes that server.

Auth is a bearer token. The gateway stores only SHA-256 hashes in `TOKENS_FILE` (default `/secrets/tokens.json`) and re-reads the file when it changes. Each entry has a name, a hash, and the list of servers that token may call. Comparison covers every entry so the response time does not reveal which one matched. Limits: `RATE_LIMIT_PER_MIN` (default 60) and `MAX_BODY_BYTES` (default 1 MiB). Every upstream request gets `X-MCP-Gateway` set to the token's name; a client-supplied value is removed first.

The image runs as uid 1000. The hash file has to be readable by that user. `GET /health` does not require a token.

### studio-ui

Build of [fspecii/ace-step-ui](https://github.com/fspecii/ace-step-ui) at commit `a1fdf91829ec6f7b98844f80e323529cd155dbf2`, with two patches applied at build time:

- `patch-gradio-args.py` realigns the UI's Gradio argument list and makes DCW depend on `/flags/dcw_off`.
- `patch-auth-header.py` sends the UI's JWT as `X-Auth-Token` so it does not replace a browser `Authorization` header. `nginx.conf` maps that header back to `Bearer` for the Node API.

nginx listens on 3000 and proxies `/api`, `/audio`, and a few other prefixes to the Node server on 127.0.0.1:3001. The build clones the upstream repo and runs `npm ci`, so it needs network access at build time. The patches assert their anchors and fail the build if the pinned UI no longer matches.

### Background jobs

`image-api` and `music-api` treat a request as remote when the `X-MCP-Gateway` header is present (the gateway always sets it). Remote calls wait at most `REMOTE_WAIT_SECONDS` (default 40) and otherwise return `job_id` with `status: running`. Poll `job_status`. Callers without that header wait until the tool finishes. Jobs are kept in memory and disappear on restart; finished files stay on disk. `job_status` waits up to 45 seconds per call.

## Prerequisites

- Docker, for every service.
- An NVIDIA GPU for the ComfyUI and ACE-Step containers. `stack-api` also needs the NVIDIA driver visible inside its container. `image-api` and `music-api` do not load the models themselves; they start and stop those containers.
- A Docker HTTP API address for `DOCKER_HOST` on `image-api`, `music-api`, and `stack-api`. Those three processes exit at import if `DOCKER_HOST` is unset.
- A reverse proxy with TLS if clients outside the machine should connect. Publish the gateway and, separately, the image, audio, and studio routes. Do not publish the MCP servers' own `/mcp` ports on a public address; they do not check tokens.
- A SQLite Bible database for `bible-api`, if you run that service.
- Python 3.12 only for the two host scripts. They use the standard library.

## Configure

Every variable the code reads is listed in [`.env.example`](.env.example), with the same defaults the code uses. The services do not load a `.env` file themselves. Pass the variables into each container, and export the host-script variables (`MCP_GATEWAY_SECRETS`, `MCP_PUBLIC_URL`, `MCP_PUBLIC_IP`, and the `*_PORT` variables) in the shell where you run `mcp-token.py` or `tools/mcpcall.py`.

Values you will replace:

| Variable | Used by | Replace with |
|---|---|---|
| `IMAGE_URL_BASE` | image-api | Public URL prefix for generated PNGs |
| `AUDIO_URL_BASE` | music-api | Public URL prefix for generated MP3s |
| `STUDIO_PUBLIC_URL` | music-api | Public origin of the studio UI |
| `STUDIO_UI_USERNAME` | music-api | Username the studio UI was set up with |
| `MCP_PUBLIC_URL` | `tools/mcpcall.py` | Public origin of the gateway |
| `MCP_GATEWAY_SECRETS` | `mcp-token.py`, `tools/mcpcall.py` | Host directory for token files |
| `DOCKER_HOST` | image-api, music-api, stack-api | Docker API address |
| `DB_PATH` | bible-api | Path of your Bible SQLite file inside the container |

`MCP_PUBLIC_IP` is optional. Set it only when `tools/mcpcall.py --remote` should resolve the gateway hostname to a specific address. Leave it empty to use DNS.

Container paths (`/data`, `/uploads`, `/inbox`, `/flags`, `/secrets/tokens.json`, `/run/secrets/mcp-gateway`) are the paths inside the container or the default host path for the token directory. Bind-mount your data there, or set the variables.

Docker DNS names such as `music-api:8030` and `comfyui:8188` are fixed in the code. Containers that should talk to each other need to share a network and use those names (or the matching `*_CONTAINER` / `STACK_API_URL` overrides).

## Run

Build each service from its directory. Example for the two services that do not need a GPU model behind them:

```bash
docker build -t bible-api ./bible-api
docker run --rm --name bible-api -p 127.0.0.1:8001:8000 \
  -v /path/to/bibles.db:/data/bibles.db:ro \
  bible-api

docker build -t stack-api ./stack-api
docker run --rm --name stack-api -p 127.0.0.1:8020:8020 \
  --gpus all \
  -e DOCKER_HOST=tcp://docker-socket-proxy-ro:2375 \
  stack-api
```

`bible-api` listens on 8000 inside the container. Publishing it as host port 8001 matches the default `BIBLE_API_PORT` in `tools/mcpcall.py`. The other host ports that client expects are 8010 (image), 8020 (stack), and 8030 (music), which match those Dockerfiles' exposed ports.

```bash
docker build -t image-api ./image-api
docker build -t music-api ./music-api
docker build -t mcp-gateway ./mcp-gateway
docker build -t studio-ui ./studio-ui
```

`image-api` and `music-api` need `DOCKER_HOST`, the output volume, and a network where `comfyui`, `acestep`, `studio-ui`, and `stack-api` resolve. `mcp-gateway` needs a read-only mount of the hash file at `TOKENS_FILE` and a network where `music-api` and `image-api` resolve. Do not mount the plaintext `*.token` files into the gateway; it only reads hashes.

Mint a token on the host (the plaintext is written to a mode-600 file and is not printed):

```bash
export MCP_GATEWAY_SECRETS=/run/secrets/mcp-gateway
python3 mcp-gateway/mcp-token.py add my-client --servers musicgen,imagegen
python3 mcp-gateway/mcp-token.py list
python3 mcp-gateway/mcp-token.py revoke my-client
```

Hashes are written to `$MCP_GATEWAY_SECRETS/store/tokens.json`. Mount that file into the gateway as `/secrets/tokens.json`. Copy the token out of `$MCP_GATEWAY_SECRETS/my-client.token` into the client, then delete the file if you do not want it kept on disk. A lost token is revoked and minted again; it cannot be recovered from the hash.

## Connect an MCP client

Local client on the same machine, no token, for any of the four MCP servers:

```text
http://127.0.0.1:8030/mcp    music-api
http://127.0.0.1:8010/mcp    image-api
http://127.0.0.1:8020/mcp    stack-api
http://127.0.0.1:8001/mcp    bible-api (if published as 8001:8000)
```

Transport is streamable HTTP. A client sends `initialize`, then `notifications/initialized`, then `tools/list` or `tools/call`.

Remote client, gateway only, music and image:

```text
POST https://mcp.example.com/musicgen/mcp
POST https://mcp.example.com/imagegen/mcp
Authorization: Bearer <token>
Accept: application/json, text/event-stream
```

Use your real gateway origin in place of `mcp.example.com`. The path after the host is the upstream name (`musicgen` or `imagegen`) plus `/mcp`.

`tools/mcpcall.py` does that handshake for a quick test:

```bash
python3 tools/mcpcall.py musicgen LIST
python3 tools/mcpcall.py musicgen status
export MCP_PUBLIC_URL=https://mcp.example.com
python3 tools/mcpcall.py --remote my-client musicgen status
```

`--remote` reads `$MCP_GATEWAY_SECRETS/my-client.token` and never prints it. Local calls block until the tool returns. Remote calls get the gateway's shorter wait, then a `job_id` when the job is still running.

Generated files are not fetched over MCP. The tool result contains an `image_url` or `audio_url` under the public prefix you set. The browser, or the client, downloads that URL from the image or music server's file route.

## Notes

- Gateway logs one line per request. A denied token is `DENY`, a limit is `LIMIT`, a success is `OK`.
- `stop_model` on image-api stops ComfyUI. `stop_model` on music-api stops whichever ACE-Step container is running, and the studio UI. Neither stops the other's model.
- The ComfyUI, ACE-Step, and Bible-database build steps are not in this repository. `music-api` will raise if an XL container it was asked to start does not exist.
- License: [MIT](LICENSE), copyright Richard Di Cosmo, 2026.
