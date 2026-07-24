#!/usr/bin/env python3
"""Benchmark Seed-VC v1 as an optional Chatterbox speaker-locking stage.

This is kept separate from the PixelHolo runtime because Seed-VC currently
pins a different dependency stack.  It assumes the Seed-VC repository is on
``PYTHONPATH`` and writes one JSON summary plus converted WAVs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio


def cosine(encoder, a: np.ndarray, b: np.ndarray, sr: int) -> float:
    if sr != 16000:
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
        b = librosa.resample(b, orig_sr=sr, target_sr=16000)
    ea, eb = encoder.embed_utterance(a), encoder.embed_utterance(b)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))


def convert_one(models, source_path: Path, target_path: Path, args) -> tuple[np.ndarray, int, float]:
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = models
    device = args.device
    sr = mel_fn_args["sampling_rate"]
    source_audio = librosa.load(source_path, sr=sr)[0]
    ref_audio = librosa.load(target_path, sr=sr)[0]
    source = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)
    t0 = time.perf_counter()
    source_16k = torchaudio.functional.resample(source, sr, 16000)
    ref_16k = torchaudio.functional.resample(ref, sr, 16000)
    s_alt = semantic_fn(source_16k)
    s_ori = semantic_fn(ref_16k)
    mel = mel_fn(source.float())
    mel2 = mel_fn(ref.float())
    target_lengths = torch.LongTensor([int(mel.size(2) * args.length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
    feat2 = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    style2 = campplus_model((feat2 - feat2.mean(dim=0, keepdim=True)).unsqueeze(0))
    cond, *_ = model.length_regulator(s_alt, ylens=target_lengths, n_quantizers=3, f0=None)
    prompt_condition, *_ = model.length_regulator(s_ori, ylens=target2_lengths, n_quantizers=3, f0=None)
    max_source_window = max(args.max_context_window - mel2.size(2), 1)
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * args.hop_length
    processed_frames = 0
    chunks: list[np.ndarray] = []
    previous = None
    with torch.inference_mode():
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last = processed_frames + max_source_window >= cond.size(1)
            cat = torch.cat([prompt_condition, chunk_cond], dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                target = model.cfm.inference(
                    cat,
                    torch.LongTensor([cat.size(1)]).to(mel2.device),
                    mel2,
                    style2,
                    None,
                    args.diffusion_steps,
                    inference_cfg_rate=args.inference_cfg_rate,
                )
                target = target[:, :, mel2.size(-1):]
            wave = vocoder_fn(target.float()).squeeze()[None, :]
            if processed_frames == 0:
                if is_last:
                    chunks.append(wave[0].cpu().numpy())
                    break
                chunks.append(wave[0, :-overlap_wave_len].cpu().numpy())
                previous = wave[0, -overlap_wave_len:]
            elif is_last:
                current = wave[0].cpu().numpy()
                overlap = min(overlap_wave_len, len(current), len(previous))
                fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
                fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
                chunks.append(np.concatenate([previous[-overlap:].cpu().numpy() * fade_out + current[:overlap] * fade_in, current[overlap:]]))
                break
            else:
                current = wave[0, :-overlap_wave_len].cpu().numpy()
                overlap = min(overlap_wave_len, len(current), len(previous))
                fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
                fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
                chunks.append(previous[-overlap:].cpu().numpy() * fade_out + current[:overlap] * fade_in)
                previous = wave[0, -overlap_wave_len:]
            processed_frames += target.size(2) - overlap_frame_len
    return np.concatenate(chunks).astype(np.float32), sr, time.perf_counter() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seedvc-root", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--diffusion-steps", type=int, default=10)
    args = p.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(args.seedvc_root))
    import inference as seed_inference

    args.f0_condition = False
    args.auto_f0_adjust = False
    args.semi_tone_shift = 0
    args.checkpoint = None
    args.config = None
    args.fp16 = True
    args.length_adjust = 1.0
    args.inference_cfg_rate = 0.7
    args.max_context_window = 30 * 22050 // 256
    args.hop_length = 256
    load_args = SimpleNamespace(
        f0_condition=False,
        checkpoint=None,
        config=None,
        fp16=True,
    )
    t0 = time.perf_counter()
    models = seed_inference.load_models(load_args)
    load_sec = time.perf_counter() - t0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_audio, target_sr = sf.read(args.target, dtype="float32")
    from resemblyzer import VoiceEncoder

    encoder = VoiceEncoder(device="cpu")
    rows = []
    for source in sorted(args.source_dir.glob("*.wav")):
        tts_audio, tts_sr = sf.read(source, dtype="float32")
        audio, sr, generation_sec = convert_one(models, source, args.target, args)
        out = args.output_dir / f"seedvc_{source.stem}.wav"
        sf.write(out, audio, sr)
        rows.append({
            "source": str(source),
            "path": str(out),
            "generation_sec": generation_sec,
            "audio_sec": len(audio) / sr,
            "rtf": generation_sec / max(len(audio) / sr, 1e-9),
            "speaker_cosine_to_ref": cosine(encoder, target_audio, audio, sr),
            "source_speaker_cosine_to_ref": cosine(encoder, target_audio, tts_audio, tts_sr),
        })
        print(source.name, f"{generation_sec:.3f}s", flush=True)
    summary = {"model": "seed-vc-v1-whisper-small-wavenet", "load_sec": load_sec, "rows": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(args.output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
