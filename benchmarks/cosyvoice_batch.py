#!/usr/bin/env python3
"""Benchmark Fun-CosyVoice 3 zero-shot English inference."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


PROMPTS = {
    "neutral_short": "Welcome to PixelHolo. Your voice profile is ready.",
    "identity": "Today we are testing the camera, the microphone, and the live avatar.",
    "emotion_happy": "I cannot believe we made it! This is wonderful news.",
    "emotion_concerned": "Wait a moment. Something seems wrong with the connection.",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--prompt-audio", type=Path, required=True)
    p.add_argument("--prompt-text", required=True)
    p.add_argument("--target-ref", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    args = p.parse_args()
    sys.path[:0] = [str(args.repo_root / "third_party/Matcha-TTS"), str(args.repo_root)]

    import torchaudio

    def load(path, *unused, **kwargs):
        wave, sr = sf.read(path, dtype="float32")
        wave = torch.from_numpy(wave)
        wave = wave.unsqueeze(0) if wave.ndim == 1 else wave.T
        return wave, sr

    # CosyVoice requests the soundfile backend, but TorchAudio 2.9 delegates
    # that call to TorchCodec.  SoundFile is sufficient for this harness.
    torchaudio.load = load
    from cosyvoice.cli.cosyvoice import AutoModel

    t0 = time.perf_counter()
    model = AutoModel(model_dir=str(args.model_dir), load_trt=False, load_vllm=False, fp16=True)
    load_sec = time.perf_counter() - t0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target, target_sr = sf.read(args.target_ref, dtype="float32")
    from resemblyzer import VoiceEncoder

    encoder = VoiceEncoder(device="cpu")
    target16 = librosa.resample(target, orig_sr=target_sr, target_sr=16000)
    target_emb = encoder.embed_utterance(target16)
    rows = []
    prompt_text = "You are a helpful assistant.<|endofprompt|>" + args.prompt_text
    for name, text in PROMPTS.items():
        for seed in args.seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            t1 = time.perf_counter()
            result = next(model.inference_zero_shot(text, prompt_text, str(args.prompt_audio), stream=False))
            generation_sec = time.perf_counter() - t1
            audio = result["tts_speech"].squeeze().detach().cpu().numpy().astype(np.float32)
            path = args.output_dir / f"cosyvoice3_{name}_seed{seed}.wav"
            sf.write(path, audio, model.sample_rate)
            out16 = librosa.resample(audio, orig_sr=model.sample_rate, target_sr=16000)
            out_emb = encoder.embed_utterance(out16)
            cosine = float(np.dot(target_emb, out_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(out_emb)))
            rows.append({"prompt": name, "seed": seed, "generation_sec": generation_sec,
                         "audio_sec": len(audio) / model.sample_rate,
                         "rtf": generation_sec / (len(audio) / model.sample_rate),
                         "speaker_cosine_to_ref": cosine, "path": str(path)})
            print(name, seed, f"{generation_sec:.3f}s", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"model": "fun-cosyvoice3-0.5b", "load_sec": load_sec, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
