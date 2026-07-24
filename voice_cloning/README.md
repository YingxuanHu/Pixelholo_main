# Voice cloning and inference worker

`voice_cloning/` is PixelHolo's Python backend. It handles profile data,
preprocessing, TTS model loading, optional LLM responses, and the streaming API
used by `frontend/`.

The active avatar path is:

```text
video → audio extraction + Whisper/VAD → processed voice reference
      → avatar bake (frames + face coordinates)
      → Chatterbox (24 kHz audio) → MuseTalk (portrait frames)
```

StyleTTS2 is still available for trained voices and comparisons, but a new
avatar does not need that training step.

## What each model does

| Component | Input | Output | Default/public role |
| --- | --- | --- | --- |
| Chatterbox TTS | Text + processed reference WAV | 24 kHz voice audio | Default one-shot voice cloning |
| StyleTTS2 | Text + trained profile/checkpoint | Voice audio | Legacy trained-speaker path and benchmark comparison |
| MuseTalk | Baked face cache + audio timing | JPEG avatar frames | Default real-time lip-sync backend |
| Wav2Lip | Video/cache + audio | Lip-synced frames/video | Standalone legacy runner and optional fallback |
| Whisper + VAD | Source video/audio | Speech segments and metadata | Preprocessing quality control, not the chat model |

## Profile data layout

For a normal `avatar` profile, the backend writes a directory like this:

```text
data/avatar_profiles/<profile>/
├── raw_videos/        # original uploaded video(s)
├── raw_audio/         # optional separate audio override(s)
├── processed_wavs/    # cleaned 24 kHz speech segments
├── metadata.csv       # segment timing/transcription metadata
└── avatar_cache/
    ├── frames.npy     # baked portrait frames
    ├── coords.npy     # face boxes aligned to frames
    └── *runtime*      # MuseTalk latents/masks/metadata as generated
```

Trained legacy profiles also use:

```text
outputs/training/<profile_type>/<profile>/
├── profile.json            # selected inference defaults
├── best_epoch.txt          # selected StyleTTS2 checkpoint
├── epoch_scores.json       # checkpoint scoring history
└── *.pt / *.pth            # model artifacts, when present
```

When a request includes `X-PixelHolo-Workspace: <uuid>`, both trees live below
`data/workspaces/<uuid>/` and `outputs/workspaces/<uuid>/`. Requests without
that header use the legacy developer tree.

## Runtime defaults

These are the values currently used by the code. They are defaults, not
measurements from a particular GPU:

| Setting | Default |
| --- | ---: |
| Processed/reference sample rate | 24,000 Hz mono |
| Whisper model | `large-v3` |
| Whisper device/compute | CUDA when configured / `float16` |
| Minimum speech segment | 2.0 s |
| Maximum speech segment | 10.0 s |
| Minimum words per segment | 4 |
| VAD merge gap | 0.20 s |
| Minimum speech ratio | 0.60 |
| Avatar bake rate | 25 FPS |
| Avatar face padding | `0 10 0 0` |
| TTS chunk size | up to 180 characters or 45 words |
| Inter-chunk pause | 40 ms |
| Chatterbox alpha / beta | 0.20 / 0.70 |
| Chatterbox diffusion steps | 10 |
| Chatterbox embedding scale | 1.70 |
| F0 scale / pace / volume | 1.0 / 1.0 / 1.0 |
| MuseTalk text window | 120 characters |
| MuseTalk first window | 72 characters |
| MuseTalk stream window / look-ahead | 1.2 s / 0.16 s |
| MuseTalk JPEG quality | 92 |

## Install

Install the system tools first:

```bash
# package-manager commands vary by Linux distribution
ffmpeg -version
espeak-ng --version
```

Then create the worker environment:

```bash
cd voice_cloning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository-level [`setup_env.sh`](../setup_env.sh) is an alternative conda
bootstrap. It creates a Python **3.10** environment and installs a nightly CUDA
**13.0** PyTorch stack. Read the script first if the host uses a different
driver or toolkit.

## Install model assets

### MuseTalk (default avatar path)

```bash
cd ../lip_syncing/lib
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
bash download_weights.sh
```

The bridge expects the repository at `lip_syncing/lib/MuseTalk` and its model
files under `models/`, unless `MUSE_TALK_MODELS_DIR` points somewhere else.

### Wav2Lip (legacy/fallback)

See [`lip_syncing/README.md`](../lip_syncing/README.md) for the Wav2Lip repo,
S3FD detector, and checkpoint setup. Use `LIPSYNC_BACKEND=wav2lip` when you
need this path.

### StyleTTS2 (legacy trained voice)

```bash
mkdir -p lib
cd lib
git clone https://github.com/yl4579/StyleTTS2.git
mkdir -p StyleTTS2/Models/LibriTTS
wget -O StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth \
  https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth
wget -O StyleTTS2/Models/LibriTTS/config.yml \
  https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml
```

## Prepare an avatar profile

`preprocess_video.py` is the command-line equivalent of the web onboarding
flow. By default, one video supplies both the face and the voice:

```bash
cd voice_cloning
source .venv/bin/activate
python src/preprocess_video.py \
  --video /absolute/path/to/talking_video.mp4 \
  --name alvin \
  --avatar_fps 25 \
  --avatar_loop_sec 20 \
  --avatar_loop_fade_sec 0.15 \
  --avatar_pads '0 10 0 0'
```

For the best source clip:

- Prefer a portrait clip (`9:16` or `3:4`) at **720p or higher**.
- Record **5–20 seconds** of clear speech; the guided camera flow records
  **20 seconds**.
- Face the camera with bright, even light from the front. Keep eyes and mouth
  visible and avoid a bright window behind you.
- Keep music, echo, multiple speakers, and large head turns out of the sample.

If you already have a separate voice recording, pass it with
`--audio /path/to/reference.wav`. Otherwise, the script extracts the video's
audio with `ffmpeg`.

## Start the API

```bash
cd voice_cloning
source .venv/bin/activate
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

Check which runtimes the worker resolved:

```bash
curl http://127.0.0.1:8000/lipsync_backend
```

Example response shape:

```json
{
  "backend": "musetalk",
  "tts_backend": "chatterbox",
  "tts_backends": ["chatterbox", "styletts2"],
  "runtime_instance_id": "..."
}
```

### Backend selection

Chatterbox and MuseTalk are the defaults:

```bash
LIPSYNC_BACKEND=musetalk \
PIXELHOLO_TTS_BACKEND=chatterbox \
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

To run the legacy Wav2Lip path:

```bash
LIPSYNC_BACKEND=wav2lip uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

To let MuseTalk fall back to Wav2Lip if initialization fails:

```bash
LIPSYNC_BACKEND=musetalk LIPSYNC_BACKEND_FALLBACK=1 \
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

## Warmup and streaming

Warm the selected profile before its first prompt:

```bash
curl -X POST http://127.0.0.1:8000/warmup \
  -H 'Content-Type: application/json' \
  -d '{
    "profile":"alvin",
    "profile_type":"avatar",
    "tts_backend":"chatterbox",
    "lipsync_backend":"musetalk"
  }'
```

Then stream a sentence as an avatar:

```bash
curl -N -X POST http://127.0.0.1:8000/speak \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/vnd.pixelholo.stream-v1' \
  -H 'X-PixelHolo-Transport: binary' \
  -d '{
    "text":"Welcome to PixelHolo. Your avatar is ready.",
    "speaker":"alvin",
    "avatar_profile":"alvin",
    "profile_type":"avatar",
    "tts_backend":"chatterbox",
    "lipsync_backend":"musetalk",
    "avatar_emotion":"neutral",
    "avatar_fps":25,
    "avatar_max_frame_edge":1080
  }' > avatar.stream
```

Omit the binary headers to use `application/x-ndjson` instead. Audio chunks
contain base64 WAV data and sample-rate metadata; avatar events contain frame
payloads and timing metadata. Binary packets use the media type
`application/vnd.pixelholo.stream-v1` and `PHS1` framing. The parser in
`App.tsx` is the best reference for clients that consume this stream.

## HTTP API

| Route | Description |
| --- | --- |
| `GET /lipsync_backend` | Resolve current TTS/lip-sync backends |
| `POST /warmup` | Prepare one profile's TTS and lip-sync resources |
| `POST /interrupt` | Interrupt active generation in the request workspace |
| `GET /profiles` | List profiles (`?profile_type=voice|avatar`) |
| `GET /profiles/{name}/voice-controls` | Read profile-specific voice controls |
| `PATCH /profiles/{name}` | Rename profile data and references |
| `DELETE /profiles/{name}` | Remove a profile and its warmup markers |
| `POST /upload` | Store a video file (`multipart/form-data`) |
| `POST /upload_audio` | Store a separate audio override |
| `POST /preprocess` | Run voice extraction and optional avatar baking |
| `POST /train` | Run the legacy StyleTTS2 training pipeline |
| `POST /stream` | Stream audio-only NDJSON |
| `POST /stream_avatar` | Stream one fixed avatar prompt |
| `POST /generate` | Return one non-streaming WAV or base64 WAV |
| `POST /chat` | LLM response followed by voice/avatar streaming |
| `POST /speak` | Fixed text voice/avatar streaming |

Browser workspace requests should include:

```http
X-PixelHolo-Workspace: <uuid>
```

For a complete upload and preprocessing example, see the root
[`README.md`](../README.md#api-examples).

## Voice controls and stability

The Chatterbox adapter normalizes the reference before conditioning and
serializes preparation and generation per runtime engine. The defaults are
deliberately conservative:

| Control | Default |
| --- | ---: |
| `alpha` | `0.20` |
| `beta` | `0.70` |
| `diffusion_steps` | `10` |
| `embedding_scale` | `1.70` |
| `f0_scale` | `1.0` |
| `pace_scale` | `1.0` |
| `volume_gain` | `1.0` |
| `max_chunk_chars` | `180` |
| `max_chunk_words` | `45` |
| `pause_ms` | `40` |

For longer prompts, the splitter starts with a **4-word** chunk, moves to a
**10-word** bridge, and then uses **25-word** cruise chunks where punctuation
allows. Supply a fixed seed when you need repeatable output while debugging
voice drift; the benchmark suite deliberately varies the seed to measure it.

## Legacy StyleTTS2 training

Keep this path for trained-speaker comparisons and existing profiles:

```bash
python src/preprocess.py \
  --video /absolute/path/to/talking_video.mp4 \
  --name alvin

python src/train.py \
  --dataset_path data/avatar_profiles/alvin \
  --profile_type avatar \
  --epochs 25
```

The `config.py` defaults are batch size **2**, maximum sequence length **400**,
and **15** epochs. The example uses **25** epochs for a fuller run. Chatterbox
avatars do not need this path.

Pronunciation overrides live in `lexicon.json`. The text pipeline normalizes
numbers, dotted abbreviations, contractions, and punctuation. It then prefers a
CMUdict pronunciation and falls back to espeak for words it does not know.

## Run the benchmark

From the repository root, run:

```bash
python benchmarks/tts_benchmark.py \
  --project-root /home/alvin/PixelHolo_trial \
  --reference voice_cloning/data/avatar_profiles/alvin2_video/processed_wavs/alvin2_video_0065.wav \
  --backends chatterbox styletts2 \
  --output-dir outputs/benchmarks/tts_suite
```

The default suite covers **4 prompts × 3 seeds = 12 generations per backend**.
Each JSON row records load time, reference preparation, warm generation time,
audio duration, real-time factor, peak GPU memory, and speaker-embedding
similarity. The summary also reports mean and maximum cross-seed drift. Audio
samples are written for listening, and missing optional backends are marked as
skipped. The script does not change profile or model files.

## Useful environment variables

| Variable | Default | Use |
| --- | ---: | --- |
| `PIXELHOLO_WORKSPACE_ID` | `legacy` | Scope CLI/subprocess work to a UUID workspace |
| `PIXELHOLO_TTS_BACKEND` | `chatterbox` | Default TTS adapter |
| `CHATTERBOX_DEVICE` | CUDA if available | Chatterbox device |
| `LIPSYNC_BACKEND` | `musetalk` | `musetalk` or `wav2lip` |
| `LIPSYNC_BACKEND_FALLBACK` | `0` | Permit MuseTalk → Wav2Lip fallback |
| `MUSE_TALK_MODELS_DIR` | `lip_syncing/lib/MuseTalk/models` | MuseTalk weights root |
| `MUSE_TALK_MAX_CHUNK_CHARS` | `120` | MuseTalk text window |
| `MUSE_TALK_FIRST_CHUNK_CHARS` | `72` | First window size |
| `MUSE_TALK_STREAM_WINDOW_SEC` | `1.2` | Streaming work window |
| `MUSE_TALK_LOOKAHEAD_SEC` | `0.16` | Frame look-ahead |
| `MUSE_TALK_JPEG_QUALITY` | `92` | Frame JPEG quality |
| `STYLE_TTS2_MODEL` | unset | Optional default legacy checkpoint |
| `STYLE_TTS2_CONFIG` | inferred | Optional StyleTTS2 config path |

## Troubleshooting

- **Command not found:** activate `voice_cloning/.venv` before starting Uvicorn
  or running preprocessing.
- **No avatar frames:** make sure MuseTalk is at `lip_syncing/lib/MuseTalk`, its
  weights are present, and the selected profile has
  `avatar_cache/frames.npy` and `coords.npy`.
- **The first request is slow:** call `/warmup`. Cold Chatterbox/MuseTalk loads
  are expected and are separate from warm generation time.
- **The wrong profile is speaking:** send `speaker` and `avatar_profile`
  together, keep the workspace header stable, and wait for warmup before
  starting a new stream.
- **Training fails on a tiny clip:** use a longer, cleaner source or reduce the
  split requirements. `ValueError: high <= 0` means there are not enough
  segments.
- **GPU memory errors:** reduce MuseTalk batch/window settings, cap the browser
  frame edge at **1,080 px**, or use the standalone Wav2Lip runner for offline
  output.
