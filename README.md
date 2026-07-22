# PixelHolo

PixelHolo is a low-latency AI avatar system that clones voices (StyleTTS2) and syncs lips (Wav2Lip) in real-time. It is designed for interactive chatbots with instant streaming using Staircase Chunking and Avatar Baking.

## System Architecture
- `frontend/` - React/Vite UI for chatting with the avatar.
- `ios/` - Native iOS thin client sources.
- `pixelholo_2_ios.xcodeproj/` - Xcode project for the iOS app.
- `voice_cloning/` - The brain: voice training, text-to-speech, and orchestration.
- `lip_syncing/` - The engine: Wav2Lip inference runners (standalone or bridged).
- `setup_env.sh` - Bootstrap script for the shared Python environment.
- `reference/` - legacy UI reference (not used).

## Concepts
- Avatar Baking: Precompute face boxes and frames once so streaming avoids live face detection.
- Staircase Chunking: 4 words -> 10 words -> 25 words for fast first frame and steady buffering.

## Quickstart

Prereqs: Linux (Ubuntu 22.04+), NVIDIA GPU (8GB+), Python 3.12, Node 18+, `ffmpeg`, `espeak-ng`.
Model downloads are documented in `voice_cloning/README.md` and `lip_syncing/README.md`.

1) Voice engine
```bash
cd voice_cloning
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2) Lip sync engine
```bash
cd lip_syncing
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

3) Frontend
```bash
cd frontend
npm install
```

4) Create a profile (video -> voice + avatar cache)
```bash
cd voice_cloning
source .venv/bin/activate
python src/preprocess_video.py --video /path/to/my_video.mp4 --name alvin
```

5) Train (recommended sprint)
```bash
python src/train.py --dataset_path data/avatar_profiles/alvin --profile_type avatar --epochs 25
```

6) Run streaming
```bash
# Terminal 1 (backend)
cd voice_cloning
uvicorn src.inference:app --host 0.0.0.0 --port 8000

# Terminal 2 (frontend)
cd frontend
npm run dev
```
Open http://localhost:5173

Optional (LLM chat): create `.env` at the repo root with:
```
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# Optional live-search tuning
OPENAI_LIVE_SEARCH_MODEL=gpt-4o-mini
OPENAI_REALTIME_MAX_OUTPUT_TOKENS=512
OPENAI_REASONING_EFFORT=none
```

## Key Features
- Voice cloning with StyleTTS2 fine-tuning.
- Real-time lip sync with Wav2Lip.
- Baked avatars for zero face-detect latency.
- Staircase chunking for stable streaming.

## Troubleshooting
- Permission errors: activate the correct `.venv` in each repo.
- Audio but no video: confirm `lip_syncing/models/wav2lip_gan.pth` exists.
- Stream stalling: ensure GPU VRAM headroom and stable network.
