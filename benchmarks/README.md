# PixelHolo benchmark harnesses

These scripts are optional comparison tools. They never modify model weights or
profile files. Instead, they write generated samples and JSON summaries to the
output directory you choose.

## TTS benchmark

Run the main suite from the repository root on the GPU host:

```bash
python benchmarks/tts_benchmark.py \
  --project-root /home/alvin/PixelHolo_trial \
  --reference voice_cloning/data/avatar_profiles/alvin2_video/processed_wavs/alvin2_video_0065.wav \
  --backends chatterbox styletts2 \
  --output-dir outputs/benchmarks/tts_suite
```

The default test matrix is:

```text
4 prompts × 3 seeds (101, 202, 303) = 12 generations/backend
```

The prompts cover a neutral greeting, identity/camera language, happy emotion,
and concerned emotion. Each generated WAV is saved so you can listen for accent
drift, pronunciation, and prosody alongside the numerical scores.

For each backend, prompt, and seed, the JSON summary records:

| Field | Meaning |
| --- | --- |
| `load_sec` | Cold model load time |
| `prepare_sec` | Reference conditioning time |
| `generation_sec` | Time spent generating audio |
| `audio_sec` | Duration of generated audio |
| `rtf` | `generation_sec / audio_sec`; lower is faster |
| `peak_gpu_mem_mb` | Peak allocated CUDA memory, when available |
| `speaker_cosine_to_ref` | Resemblyzer cosine similarity to reference, when available |
| `cross_seed_drift_mean` | Mean pairwise embedding distance across seeds |
| `cross_seed_drift_max` | Maximum pairwise embedding distance across seeds |

The summary has this general shape:

```json
{
  "backend": "chatterbox",
  "prompt_id": "neutral_short",
  "seed": 101,
  "load_sec": 0.0,
  "prepare_sec": 0.0,
  "generation_sec": 0.0,
  "audio_sec": 0.0,
  "rtf": 0.0,
  "peak_gpu_mem_mb": null,
  "speaker_cosine_to_ref": null
}
```

The zeros and nulls above are placeholders, not observed results. Run the suite
on the target machine to get real values. Missing optional backends are listed
as `skipped` in the final JSON instead of being treated as failures.

## Optional comparison scripts

- `f5_batch.py` — batch prompts through an F5-TTS-compatible setup;
- `cosyvoice_batch.py` — batch prompts through CosyVoice;
- `seedvc_batch.py` — compare voice-conversion outputs.

Read each script's command-line help before running it. The repository bootstrap
does not install the optional model APIs or weights.
