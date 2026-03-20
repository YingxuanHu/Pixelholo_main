# Voice Cloning and Orchestration (StyleTTS2)

This module handles voice cloning (StyleTTS2) and orchestrates the full pipeline. It calls `lip_syncing/` to generate video frames during streaming.

## Key Concepts

### Avatar Baking
Real-time face detection is too slow. Baking happens once during preprocessing:
1) Extract a short loop from the video.
2) Run face detection on each frame offline.
3) Save `frames.npy` and `coords.npy`.
4) Runtime loads the cache instantly.

### Staircase Chunking
Streaming uses a text pattern to prevent stalls:
- Chunk 1: 4 words (warmup).
- Chunk 2: 10 words (bridge).
- Chunk 3+: 25 words (cruise).

## Workflow

Choose a mode:
- Voice-only: audio output only.
- Voice + Avatar: audio + lip-synced video.

### 1a) Preprocess (voice-only)
```bash
python src/preprocess.py \
  --video /path/to/me.mp4 \
  --name alvin
```

### 1b) Preprocess (video -> dataset + avatar cache)
```bash
python src/preprocess_video.py \
  --video /path/to/me.mp4 \
  --name alvin
```

### 2) Train (recommended sprint)
```bash
python src/train.py \
  --dataset_path data/avatar_profiles/alvin \
  --profile_type avatar \
  --epochs 25
```

### 3) Streaming API
```bash
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```
Default lip-sync backend is `wav2lip`.

Backend selection is controlled by env flag at backend startup:
```bash
# Wav2Lip (default path, unchanged)
LIPSYNC_BACKEND=wav2lip uvicorn src.inference:app --host 0.0.0.0 --port 8000

# MuseTalk (low-latency defaults)
LIPSYNC_BACKEND=musetalk \
MUSE_TALK_BATCH_SIZE=24 \
MUSE_TALK_INFER_FPS=12 \
MUSE_TALK_STREAM_WINDOW_SEC=1.0 \
MUSE_TALK_BLEND_EXPAND=1.2 \
MUSE_TALK_FACE_SCALE=1.0 \
MUSE_TALK_ALPHA_BLUR_RATIO=0.05 \
uvicorn src.inference:app --host 0.0.0.0 --port 8000

# Optional fallback: if MuseTalk init fails, fallback to Wav2Lip
LIPSYNC_BACKEND=musetalk LIPSYNC_BACKEND_FALLBACK=1 uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

MuseTalk notes:
- First request for a profile can rebuild `musetalk_latents.pt` and `musetalk_masks.pkl`.
- Runtime now auto-converts avatar `coords.npy` from bake format (`[y1, y2, x1, x2]`) before MuseTalk processing.
- For lower startup latency, call `POST /warmup` for the target avatar profile once after backend start.
- Default tuning is balanced for latency + quality. Override only if you need profile-specific changes.

Endpoints:
- `POST /stream_avatar` - NDJSON stream (audio + JPEG frames)
- `POST /stream` - audio-only stream

### 4) CLI video output
```bash
python src/speak_video.py \
  --profile alvin \
  --text "This is a generated video using the baked cache."
```

### Voice-only inference (audio)
```bash
python src/speak.py \
  --profile alvin \
  --profile_type voice \
  --text "Hello from voice-only."
```

## Configuration
Files of interest:
- `outputs/training/avatar/<name>/profile.json` - inference defaults (model path, ref wav, alpha/beta, f0_scale).
- `outputs/training/avatar/<name>/best_epoch.txt` - selected checkpoint.
- `outputs/training/avatar/<name>/epoch_scores.json` - scoring history.
- `data/<type>_profiles/<name>/lexicon.json` - pronunciation overrides. Supports both single words and full phrases (for example `\"1300 b c\"` or a proper noun phrase) when you need exact spoken output.

Pronunciation pipeline:
- text is normalized first (numbers, dotted abbreviations such as `B.C.`, contractions, punctuation cleanup)
- runtime then prefers a CMUdict-backed base pronunciation dictionary when available
- profile `lexicon.json` overrides still win over the base dictionary
- espeak phonemizer remains the fallback for out-of-vocabulary words

Suggested training settings:
- `epochs: 25`
- `save_every: 1` to capture early sweet-spot checkpoints.

## Setup
```bash
cd voice_cloning
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
If `python` is not available, use `python3 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`.
If you see an "externally-managed-environment" error, use `.venv/bin/python -m pip install -r requirements.txt`.

System deps: `ffmpeg`, `espeak-ng`.

LLM chat defaults:
- `GROQ_MODEL_DEFAULT=llama-3.1-8b-instant`
- `OPENAI_MODEL_LIVE=gpt-4o-mini-search-preview`
- `LLM_ENABLE_LIVE_ROUTING=1`

With those defaults, ordinary chat stays on the fast Groq base model. Prompts that clearly ask for current information such as weather, latest news, scores, prices, or post-2024 facts are routed to the OpenAI live-search model instead. If OpenAI is not configured or the live request fails before it produces any text, the backend falls back to the default Groq chat model instead of crashing the stream.

Clone StyleTTS2 and download LibriTTS weights:
```bash
mkdir -p lib
cd lib
git clone https://github.com/yl4579/StyleTTS2.git
mkdir -p StyleTTS2/Models/LibriTTS
wget -O StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth   https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth
wget -O StyleTTS2/Models/LibriTTS/config.yml   https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml
```

Optional: install MuseTalk repo + weights (needed only when `LIPSYNC_BACKEND=musetalk`):
```bash
cd lip_syncing/lib
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
bash download_weights.sh
```

## Troubleshooting
- "Command not found": activate `.venv` first.
- "Lip sync bridge failed": confirm `lip_syncing/` exists at the repo root and the backend path helpers still point to it.
- "Stream stalls": check GPU VRAM usage and network stability.
- Training crashes on tiny datasets:
  - `ValueError: high <= 0` means too few segments (use a longer clip or `--legacy_split`).
  - `IndexError: Dimension out of range` can happen with `batch_size=1` on tiny datasets; use `--batch_size 2` for smoke tests.
