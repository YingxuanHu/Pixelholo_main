#!/usr/bin/env python3
"""Small F5-TTS v1 inference benchmark for the isolated F5 package."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import random
import sys
import time
import types
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-audio", type=Path, required=True)
    p.add_argument("--ref-text", required=True)
    p.add_argument("--target-ref", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--nfe-step", type=int, default=16)
    p.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    args = p.parse_args()

    # The inference package imports training-only modules.  They are not used
    # here, so tiny module stubs keep this benchmark isolated from training
    # dependencies such as datasets and wandb.
    wandb = types.ModuleType("wandb")
    wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    wandb.api = types.SimpleNamespace(api_key=None)
    sys.modules["wandb"] = wandb
    datasets = types.ModuleType("datasets")
    datasets.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
    datasets.Dataset = object
    datasets.load_from_disk = lambda *a, **k: None
    sys.modules["datasets"] = datasets

    import soundfile as soundfile_module
    import torchaudio

    # TorchAudio 2.9 delegates file I/O to TorchCodec; soundfile is enough for
    # this read-only benchmark and avoids adding a codec dependency.
    torchaudio.load = lambda path: (
        torch.from_numpy(soundfile_module.read(path, dtype="float32")[0]).unsqueeze(0),
        soundfile_module.info(path).samplerate,
    )

    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
    from f5_tts.model.backbones.dit import DiT
    from omegaconf import OmegaConf
    from importlib.resources import files
    from cached_path import cached_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml"))
    ckpt = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
    t0 = time.perf_counter()
    vocoder = load_vocoder(vocoder_name="vocos", device=device)
    model = load_model(DiT, cfg.model.arch, ckpt, mel_spec_type="vocos", device=device)
    load_sec = time.perf_counter() - t0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target, target_sr = sf.read(args.target_ref, dtype="float32")
    import resemblyzer

    encoder = resemblyzer.VoiceEncoder(device="cpu")
    rows = []
    for prompt_name, text in PROMPTS.items():
        for seed in args.seeds:
            seed_all(seed)
            t1 = time.perf_counter()
            audio, sr, _ = infer_process(
                str(args.ref_audio), args.ref_text, text, model, vocoder,
                mel_spec_type="vocos", nfe_step=args.nfe_step,
                cfg_strength=2.0, device=device,
            )
            generation_sec = time.perf_counter() - t1
            audio = np.asarray(audio, dtype=np.float32)
            path = args.output_dir / f"f5_{prompt_name}_seed{seed}.wav"
            sf.write(path, audio, sr)
            ref16 = librosa.resample(target, orig_sr=target_sr, target_sr=16000)
            out16 = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            a, b = encoder.embed_utterance(ref16), encoder.embed_utterance(out16)
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            rows.append({"prompt": prompt_name, "seed": seed, "generation_sec": generation_sec,
                         "audio_sec": len(audio) / sr, "rtf": generation_sec / (len(audio) / sr),
                         "speaker_cosine_to_ref": cosine, "path": str(path)})
            print(prompt_name, seed, f"{generation_sec:.3f}s", flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"model": "f5-tts-v1-base", "load_sec": load_sec, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
