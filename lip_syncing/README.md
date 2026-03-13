# Lip Sync Engine (Wav2Lip)

This is the high-performance inference engine. It can run standalone, but is designed to be controlled by `voice_cloning` via the bridge.

## How It Works

### Legacy Mode (Slow)
Reads a full video -> detects faces -> runs inference -> writes video.
Time: tens of seconds per sentence.

### Cached Mode (Fast / Real-Time)
Loads pre-baked `frames.npy` and `coords.npy` -> runs inference -> streams frames.
Time: sub-second per sentence.

## Setup
```bash
cd lip_syncing
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
If `python` is not available, use `python3 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`.

## Download Models
Place these in `lip_syncing/models/`:
- `s3fd-619a316812.pth`
  - https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
- `wav2lip_gan.pth`
  - https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k

Clone Wav2Lip:
```bash
git clone https://github.com/Rudrabha/Wav2Lip.git lib/Wav2Lip
```

## Standalone Usage (Slow / Quality)
```bash
python src/run_lipsync.py \
  --video input.mp4 \
  --audio speech.wav \
  --output result.mp4 \
  --resize_factor 1
```

## Cached Mode (Fast / Real-Time)
Cached mode is driven by `voice_cloning` via the bridge and uses the baked cache:
```
voice_cloning/data/avatar_profiles/<name>/avatar_cache/
```
Use `voice_cloning/src/speak_video.py` or the streaming API for this path.

## Run from voice_cloning
`voice_cloning/src/speak_video.py` calls this repo automatically.
```bash
python voice_cloning/src/speak_video.py \
  --profile alvin \
  --text "Hello from video"
```

## Performance Notes
- `--resize_factor 1` keeps full resolution (slow but sharp).
- If you hit OOM: lower `--wav2lip_batch_size` or `--face_det_batch_size`.
- Wav2Lip is optimized for 25 FPS; changing FPS requires re-baking.
