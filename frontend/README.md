# PixelHolo Avatar Studio

The frontend is the React 19 + Vite web app for PixelHolo. It keeps the main
flow simple: create a profile from one talking video, type or speak a prompt,
and watch the answer appear in the portrait preview.

![PixelHolo Alvin avatar studio preview](../docs/assets/avatar-studio-preview.png)

This is a representative development capture, not a benchmark or a promise
that the layout will never change. The canvas and controls are still being
refined for different screen sizes.

## User flow

1. The landing page introduces PixelHolo and links to **Create a profile**.
2. The user chooses a unique profile name. If the name is already in use, the
   form explains the problem instead of opening that profile.
3. The user uploads a video or records a guided **25-second** camera sample.
   The camera view includes a head-position guide, lighting tips, and a short
   script to read.
4. That same video supplies the face and voice. The backend extracts the audio,
   creates **24 kHz mono** voice segments, and bakes a **25 FPS** avatar cache.
   New users do not need to train StyleTTS2.
5. Once preparation finishes, the new profile is selected and appears in the
   left sidebar. Switching profiles clears the old preview and warms the new
   one before generation starts.
6. The user types a prompt or clicks the microphone. Browser speech recognition
   shows the live transcript in the composer and sends the final text
   automatically when speech ends.
7. Chatterbox generates the audio and MuseTalk generates the frames. The client
   lines the frames up with the audio; the default baked loop is **20 seconds**
   with a **0.15-second** cross-fade.

## Local development

The frontend commands below were verified with a clean dependency install:

```bash
cd frontend
npm ci
npm run typecheck
npm run build
```

For the live application, start a pre-provisioned FastAPI worker in one
terminal. Full Chatterbox + MuseTalk inference needs a Linux NVIDIA GPU host,
the MuseTalk weights, and a worker environment where both model stacks have
already been resolved; the repository's legacy `setup_env.sh` is not a
complete one-command model installer. See the root
[`README`](../README.md#full-local-gpu-worker) before setting up a new GPU host.

```bash
cd voice_cloning
source .venv/bin/activate
python -m uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

Then start the Vite app in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

The port is set explicitly in `vite.config.ts`:

```text
http://127.0.0.1:5174
```

Available scripts:

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start Vite with host `0.0.0.0` on port `5174` |
| `npm run typecheck` | Run TypeScript without emitting files |
| `npm run build` | Create the deployable `frontend/dist` build |
| `npm run preview` | Serve the production build locally for a smoke test |

## API connection

The development client talks to `http://127.0.0.1:8000` by default. Point it
at another worker with:

```bash
VITE_PIXELHOLO_API_BASE=http://127.0.0.1:8000 npm run dev
```

Useful client-side tuning variables:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `VITE_PIXELHOLO_API_BASE` | development: `http://127.0.0.1:8000`; production: `/api` | FastAPI origin override |
| `VITE_PIXELHOLO_BINARY_STREAM` | enabled in the current build | Prefer binary stream packets over NDJSON |
| `VITE_PIXELHOLO_BINARY_PCM_AUDIO` | enabled in the current build | Prefer PCM audio packets when available |
| `VITE_PIXELHOLO_AVATAR_MAX_FRAME_EDGE` | `1280` px | Downscale very large incoming avatar frames while preserving more mouth detail |
| `VITE_PIXELHOLO_AUDIO_START_DELAY_SEC` | `0.04` s | Audio-only startup offset |
| `VITE_PIXELHOLO_AVATAR_AUDIO_START_DELAY_SEC` | `0.34` s | Avatar startup offset while frames buffer |
| `VITE_PIXELHOLO_AVATAR_AUDIO_CHUNK_LEAD_SEC` | `0.08` s | Lead time for following audio chunks |
| `VITE_PIXELHOLO_VIDEO_PREDECODE_PREWAIT_MS` | `45` ms | Predecode wait before drawing a frame |

In production, the browser uses the same-origin `/api` path exposed by
`tools/web_proxy.py`, so public visitors never need direct access to port 8000.
Older saved settings that point to `127.0.0.1` are discarded automatically in
production because that address would refer to the visitor's own device.

## Anonymous device workspaces

Accounts are not implemented yet. On the first visit, the client creates a UUID
and stores it in `localStorage` as `pixelholo_workspace_id`. Every `fetch` and
`XMLHttpRequest` includes:

```http
X-PixelHolo-Workspace: 11111111-2222-4333-8444-555555555555
```

That header keeps profiles, uploads, preprocessing output, generated audio, and
avatar caches in the current browser/device workspace. Opening the site in
another browser or clearing site data starts a new empty workspace. This is
workspace isolation, not authentication; a future account layer can replace
the UUID with a server-issued tenant id.

For local CLI requests, omit the header to use the legacy developer workspace,
or send a UUID to reproduce a browser workspace.

## Streaming contract

These are the endpoints used by the main avatar flow:

```text
GET  /profiles                  list profile metadata
POST /upload                    store a talking video
POST /preprocess                extract voice + bake avatar cache
POST /warmup                    prepare the selected profile
POST /speak                     stream one fixed prompt
POST /chat                      stream an optional LLM response
POST /interrupt                 stop the active stream
```

`/speak` and `/chat` send JSON with `profile_type: "avatar"`, `speaker`,
`avatar_profile`, `tts_backend: "chatterbox"`, and
`lipsync_backend: "musetalk"`. With binary mode, the response media type is
`application/vnd.pixelholo.stream-v1` and packets are framed with the `PHS1`
magic. Without it, the response is `application/x-ndjson`, one JSON event per
line. The parser accepts audio chunks, JPEG frame batches, `done`, and `error`
events.

The session details panel shows the last prompt's time-to-first-audio and total
latency. These are per-prompt observations, not benchmark results; use the
repository benchmark suite for repeatable comparisons across seeds and
backends.

## Production build

```bash
cd frontend
npm run typecheck
npm run build
```

The generated `dist/` directory is served by `tools/web_proxy.py`. The checked-
in systemd unit binds the proxy to `100.120.224.119:8080` and forwards `/api/*`
to `127.0.0.1:8000`. Change those addresses when deploying to another host.

## Common UI debugging

- **No microphone button:** use a browser that supports SpeechRecognition and
  grant microphone permission. Text prompts still work without it.
- **Camera permission does not appear:** click **Enable camera**, then allow
  camera and microphone access in the browser. Camera recording can begin
  before naming the profile; a unique name is required only when creating it.
  Use HTTPS or `localhost` so `getUserMedia()` is allowed.
- **The old face appears after switching:** wait for “Preparing this profile…”;
  the preview stays blank until the selected profile is warm.
- **No frames:** check `/lipsync_backend` and confirm it reports `musetalk`, then
  make sure the profile contains `avatar_cache/frames.npy` and `coords.npy`.
- **Video is large or cropped:** use a portrait source (`9:16` or `3:4`) and
  let the client cap frames at **1,280 px** (**768 px** in Firefox). The canvas
  preserves the portrait instead of stretching it horizontally.
