# Lip-sync engines

This directory holds the Wav2Lip runner and the model files used by the MuseTalk
bridge. The FastAPI worker uses **MuseTalk by default** for live avatars. Use
Wav2Lip for standalone/offline jobs or when you explicitly need the fallback.

## Which path should I use?

### MuseTalk through the PixelHolo bridge (current default)

When an avatar is prepared, PixelHolo bakes a portrait cache once:

```text
voice_cloning/data/avatar_profiles/<name>/avatar_cache/
├── frames.npy
├── coords.npy
└── runtime metadata / MuseTalk latents / face masks
```

At inference time, the bridge loads that cache, prepares MuseTalk, and streams
JPEG frames alongside Chatterbox audio. The current defaults are:

| Setting | Default |
| --- | ---: |
| Source/baked playback rate | 25 FPS |
| Text window | 120 characters |
| First text window | 72 characters |
| Stream window | 1.2 s |
| Look-ahead | 0.16 s |
| JPEG quality | 92 |
| Face scale | 1.0 (clamped to 0.75–1.15) |
| Browser frame edge cap | 1,080 px in the current frontend |

These are tuning defaults, not benchmark results. Change them after measuring a
representative prompt set on the deployment GPU.

Install the MuseTalk repository and weights like this:

```bash
cd lip_syncing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p lib
cd lib
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
bash download_weights.sh
```

If the environment has been changed, select MuseTalk explicitly when starting
the worker:

```bash
cd ../../voice_cloning
source .venv/bin/activate
LIPSYNC_BACKEND=musetalk uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

Call `POST /warmup` for a profile before its first prompt. The first warmup may
create `musetalk_latents.pt`, `musetalk_masks.pkl`, and runtime metadata in the
profile's avatar cache.

### Wav2Lip standalone or fallback path

Wav2Lip can turn a complete source video and audio track into an output file. It
can also be selected by the FastAPI worker with `LIPSYNC_BACKEND=wav2lip`.

Install the legacy repository and checkpoints:

```bash
cd lip_syncing
mkdir -p lib models
git clone https://github.com/Rudrabha/Wav2Lip.git lib/Wav2Lip
```

Place these files in `lip_syncing/models/`:

- `s3fd-619a316812.pth` — face detector;
- `wav2lip_gan.pth` — Wav2Lip GAN checkpoint.

Then run a standalone output:

```bash
source .venv/bin/activate
python src/run_lipsync.py \
  --video input.mp4 \
  --audio speech.wav \
  --output result.mp4 \
  --resize_factor 1
```

`--resize_factor 1` keeps the source resolution. Larger values reduce the
working resolution and memory use, but also reduce detail. If the runner runs
out of memory, lower `--wav2lip_batch_size` or `--face_det_batch_size`.

Select Wav2Lip for the API:

```bash
cd voice_cloning
source .venv/bin/activate
LIPSYNC_BACKEND=wav2lip uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

To let MuseTalk fall back automatically when initialization fails:

```bash
LIPSYNC_BACKEND=musetalk LIPSYNC_BACKEND_FALLBACK=1 \
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

## Why bake the avatar?

`voice_cloning/src/preprocess_video.py` extracts the audio, processes the
speech, and invokes the avatar baker with defaults of **25 FPS**, a
**20-second** loop, and a **0.15-second** loop fade. At runtime, the worker
reads `frames.npy` and `coords.npy` instead of running full face detection for
every prompt. Re-bake a profile whenever its source video or portrait crop
changes. After preprocessing, the backend drops the warmup markers so it does
not reuse stale cache data.

## Useful commands

Create a cache from a source video:

```bash
cd voice_cloning
python src/preprocess_video.py \
  --video /absolute/path/to/talking_video.mp4 \
  --name alvin \
  --avatar_fps 25 \
  --avatar_loop_sec 20 \
  --avatar_loop_fade_sec 0.15
```

Generate an offline video from the cached avatar path:

```bash
python src/speak_video.py \
  --profile alvin \
  --text "This is a generated video using the baked avatar cache."
```

## Environment variables

| Variable | Default | Effect |
| --- | ---: | --- |
| `LIPSYNC_BACKEND` | `musetalk` | Select `musetalk` or `wav2lip` |
| `LIPSYNC_BACKEND_FALLBACK` | disabled | Permit MuseTalk → Wav2Lip fallback |
| `MUSE_TALK_MODELS_DIR` | `lib/MuseTalk/models` | MuseTalk model root |
| `MUSE_TALK_FACE_SCALE` | `1.0` | Face scale (runtime clamps it) |
| `MUSE_TALK_MAX_CHUNK_CHARS` | `120` | Maximum MuseTalk text window |
| `MUSE_TALK_FIRST_CHUNK_CHARS` | `72` | First window for startup |
| `MUSE_TALK_STREAM_WINDOW_SEC` | `1.2` | Stream work window |
| `MUSE_TALK_LOOKAHEAD_SEC` | `0.16` | Frame look-ahead |
| `MUSE_TALK_JPEG_QUALITY` | `92` | Streamed JPEG quality |
| `MUSE_TALK_FRAME_CROSSFADE` | `2` | Runtime frame crossfade parameter |

## Troubleshooting

- **`MuseTalk repo not found`:** clone it at `lip_syncing/lib/MuseTalk` or set
  `MUSE_TALK_MODELS_DIR`, then verify that the bridge can see the repo.
- **Audio but no frames:** check that `avatar_cache/frames.npy` and
  `avatar_cache/coords.npy` exist for the selected profile. Re-run preprocessing
  if the source changed.
- **Wrong crop or blurry mouth:** use a steady, front-lit portrait source at
  **720p or higher**, keep the eyes and mouth visible, and avoid aggressive face
  scaling. Re-bake after changing the source.
- **Wav2Lip checkpoint errors:** make sure both the S3FD detector and
  `wav2lip_gan.pth` are under `lip_syncing/models/`.
- **Out of memory:** lower batch sizes, reduce the browser frame edge cap from
  **1,080**, or use a smaller source for offline tests.
- **Stale profile after switching:** wait for the selected profile to warm up;
  the frontend intentionally leaves the preview blank until it is ready.
