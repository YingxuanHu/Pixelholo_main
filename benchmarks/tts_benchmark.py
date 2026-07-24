#!/usr/bin/env python3
"""Repeatable local TTS benchmark for PixelHolo voice backends.

The benchmark deliberately keeps the model adapters small and records the
measurements that matter for a live avatar: cold load time, reference
conditioning time, warm generation latency, real-time factor, GPU peak memory,
and speaker-embedding similarity.  It writes audio samples next to a JSON
summary so accent/voice drift can be reviewed by listening as well as by the
embedding metric.

Run this from ``voice_cloning``'s virtualenv on the GPU host, for example:

    python benchmarks/tts_benchmark.py \
      --project-root /home/alvin/PixelHolo_trial \
      --reference voice_cloning/data/avatar_profiles/alvin2_video/processed_wavs/alvin2_video_0065.wav \
      --backends chatterbox styletts2 \
      --output-dir outputs/benchmarks/tts_suite

The script does not install packages or change model/profile files.  Missing
optional backends are reported as skipped in the JSON result.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import random
import statistics
import sys
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch


PROMPTS: dict[str, str] = {
    "neutral_short": "Welcome to PixelHolo. Your voice profile is ready.",
    "identity": "Today we are testing the camera, the microphone, and the live avatar.",
    "emotion_happy": "I cannot believe we made it! This is wonderful news.",
    "emotion_concerned": "Wait a moment. Something seems wrong with the connection.",
}
DEFAULT_SEEDS = (101, 202, 303)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1024**2, 2)


def audio_array(value: Any) -> np.ndarray:
    if isinstance(value, tuple) and value:
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    out = np.asarray(value, dtype=np.float32).squeeze()
    return out.reshape(-1)


def safe_cosine(encoder: Any, reference: np.ndarray, sample: np.ndarray, sr: int) -> float | None:
    try:
        import librosa

        ref_16k = librosa.resample(reference, orig_sr=sr, target_sr=16000) if sr != 16000 else reference
        sample_16k = librosa.resample(sample, orig_sr=sr, target_sr=16000) if sr != 16000 else sample
        ref_emb = encoder.embed_utterance(ref_16k)
        sample_emb = encoder.embed_utterance(sample_16k)
        denom = float(np.linalg.norm(ref_emb) * np.linalg.norm(sample_emb))
        if denom <= 0:
            return None
        return float(np.dot(ref_emb, sample_emb) / denom)
    except Exception as exc:  # metric should never make generation fail
        warnings.warn(f"speaker embedding skipped: {exc}")
        return None


@dataclass
class Backend:
    name: str
    model: Any
    sample_rate: int
    prepare: Callable[[Path], None] | None
    generate: Callable[[str, Path, int], np.ndarray]
    load_sec: float
    close: Callable[[], None] | None = None


def load_chatterbox(device: str, kind: str, cfg_weight: float = 0.5, exaggeration: float = 0.5) -> Backend:
    if kind == "chatterbox":
        module = importlib.import_module("chatterbox.tts")
        cls = module.ChatterboxTTS
        name = "chatterbox"
    elif kind == "chatterbox_mtl":
        module = importlib.import_module("chatterbox.mtl_tts")
        cls = module.ChatterboxMultilingualTTS
        name = "chatterbox_mtl"
    elif kind == "chatterbox_turbo":
        module = importlib.import_module("chatterbox.tts_turbo")
        cls = module.ChatterboxTurboTTS
        name = "chatterbox_turbo"
    else:
        raise ValueError(kind)

    t0 = time.perf_counter()
    model = cls.from_pretrained(device=device)
    load_sec = time.perf_counter() - t0
    sample_rate = int(getattr(model, "sr", getattr(model, "sample_rate", 24000)))
    prepared: dict[str, tuple[str, float]] = {}

    def prepare(ref: Path) -> None:
        # Turbo's prepare_conditionals has an optional loudness argument but
        # all versions accept at least the path and exaggeration.
        key = str(ref.resolve())
        if prepared.get("key") == (key, exaggeration):
            return
        model.prepare_conditionals(str(ref), exaggeration=exaggeration)
        prepared["key"] = (key, exaggeration)

    def generate(text: str, ref: Path, seed: int) -> np.ndarray:
        seed_everything(seed)
        # Passing the prompt path is intentional: it makes every row a true
        # one-shot call and exercises the same code path as a new user profile.
        key = str(ref.resolve())
        kwargs: dict[str, Any] = {
            "text": text,
            # The main loop prepares a profile once and then uses cached
            # conditionals, matching the long-lived live-avatar process.
            "audio_prompt_path": None if prepared.get("key", (None,))[0] == key else str(ref),
            "temperature": 0.8,
            "repetition_penalty": 1.2,
        }
        if kind == "chatterbox_mtl":
            kwargs["language_id"] = "en"
        elif kind == "chatterbox":
            kwargs.update({"exaggeration": exaggeration, "cfg_weight": cfg_weight})
        else:  # Turbo does not support cfg/exaggeration.
            kwargs.update({"exaggeration": 0.0, "cfg_weight": 0.0, "top_p": 0.95})
        return audio_array(model.generate(**kwargs))

    def close() -> None:
        nonlocal model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return Backend(name, model, sample_rate, prepare, generate, load_sec, close)


def load_styletts2(project_root: Path, device: str, model_path: Path, config_path: Path) -> Backend:
    src = project_root / "voice_cloning"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    module = importlib.import_module("src.inference")
    t0 = time.perf_counter()
    model = module.StyleTTS2RepoEngine(model_path=model_path, config_path=config_path, device=device)
    load_sec = time.perf_counter() - t0
    sample_rate = int(model.sample_rate)

    def generate(text: str, ref: Path, seed: int) -> np.ndarray:
        seed_everything(seed)
        return audio_array(
            model.generate(
                text=text,
                ref_wav_path=ref,
                alpha=0.2,
                beta=0.7,
                diffusion_steps=15,
                embedding_scale=1.7,
                f0_scale=0.7729140751263205,
                seed=seed,
            )
        )

    def close() -> None:
        nonlocal model
        model.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return Backend("styletts2", model, sample_rate, None, generate, load_sec, close)


def load_chatterbox_vc(device: str) -> Backend:
    t0 = time.perf_counter()
    tts = importlib.import_module("chatterbox.tts").ChatterboxTTS.from_pretrained(device=device)
    vc = importlib.import_module("chatterbox.vc").ChatterboxVC.from_pretrained(device=device)
    load_sec = time.perf_counter() - t0
    sample_rate = 24000
    prepared_ref: str | None = None

    def prepare(ref: Path) -> None:
        nonlocal prepared_ref
        key = str(ref.resolve())
        if prepared_ref == key:
            return
        tts.prepare_conditionals(str(ref), exaggeration=0.5)
        prepared_ref = key

    def generate(text: str, ref: Path, seed: int) -> np.ndarray:
        seed_everything(seed)
        source = audio_array(
            tts.generate(
                text=text,
                audio_prompt_path=None if prepared_ref == str(ref.resolve()) else str(ref),
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.8,
                repetition_penalty=1.2,
            )
        )
        source_path = Path(os.environ.get("PIXELHOLO_BENCH_TMP", "/tmp")) / f"pixelholo_tts_{uuid.uuid4().hex}.wav"
        sf.write(source_path, source, 24000)
        try:
            return audio_array(vc.generate(audio=str(source_path), target_voice_path=str(ref)))
        finally:
            source_path.unlink(missing_ok=True)

    def close() -> None:
        nonlocal tts, vc
        del tts, vc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return Backend("chatterbox_vc", (tts, vc), sample_rate, prepare, generate, load_sec, close)


def build_backend(kind: str, args: argparse.Namespace) -> Backend:
    if kind.startswith("chatterbox") and kind != "chatterbox_vc":
        return load_chatterbox(
            args.device,
            kind,
            cfg_weight=args.chatterbox_cfg_weight,
            exaggeration=args.chatterbox_exaggeration,
        )
    if kind == "chatterbox_vc":
        return load_chatterbox_vc(args.device)
    if kind == "styletts2":
        return load_styletts2(args.project_root, args.device, args.style_model, args.style_config)
    raise ValueError(f"Unknown backend: {kind}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    result: dict[str, Any] = {"row_count": len(rows)}
    for key in ("prepare_sec", "generation_sec", "audio_sec", "rtf", "speaker_cosine_to_ref", "peak_gpu_mem_mb"):
        values = vals(key)
        if values:
            result[f"{key}_mean"] = statistics.mean(values)
            result[f"{key}_median"] = statistics.median(values)
            result[f"{key}_min"] = min(values)
            result[f"{key}_max"] = max(values)

    # Cross-seed speaker drift is the pairwise distance between generated
    # embeddings for the same prompt.  Lower distance means more stable.
    try:
        from resemblyzer import VoiceEncoder

        encoder = VoiceEncoder(device="cpu")
        by_prompt: dict[str, list[np.ndarray]] = {}
        import librosa

        for row in rows:
            wav, sr = sf.read(row["path"], dtype="float32")
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            by_prompt.setdefault(row["prompt"], []).append(encoder.embed_utterance(wav))
        drifts: list[float] = []
        for embeddings in by_prompt.values():
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    a, b = embeddings[i], embeddings[j]
                    drifts.append(1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        if drifts:
            result["cross_seed_drift_mean"] = statistics.mean(drifts)
            result["cross_seed_drift_max"] = max(drifts)
    except Exception as exc:
        result["cross_seed_drift_error"] = str(exc)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--style-model", type=Path, default=None)
    parser.add_argument("--style-config", type=Path, default=None)
    parser.add_argument("--backends", nargs="+", default=["chatterbox", "styletts2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks/tts_suite"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chatterbox-cfg-weight", type=float, default=0.5)
    parser.add_argument("--chatterbox-exaggeration", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.reference = args.reference if args.reference.is_absolute() else (args.project_root / args.reference)
    args.reference = args.reference.resolve()
    if args.style_model is None:
        args.style_model = args.project_root / "voice_cloning/outputs/training/avatar/alvin2_video/epoch_2nd_00004.pth"
    if args.style_config is None:
        args.style_config = args.project_root / "voice_cloning/outputs/training/avatar/alvin2_video/config_ft.yml"
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else (args.project_root / args.output_dir)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        from resemblyzer import VoiceEncoder

        reference, ref_sr = sf.read(args.reference, dtype="float32")
        encoder = VoiceEncoder(device="cpu")
    except Exception as exc:
        print(f"speaker metric unavailable: {exc}", file=sys.stderr)
        reference = None
        ref_sr = None
        encoder = None

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "reference": str(args.reference),
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "backends": {},
        "skipped": {},
    }
    for kind in args.backends:
        try:
            backend = build_backend(kind, args)
        except Exception as exc:
            manifest["skipped"][kind] = f"{type(exc).__name__}: {exc}"
            continue
        rows: list[dict[str, Any]] = []
        try:
            for prompt_name, text in PROMPTS.items():
                for seed in args.seeds:
                    reset_peak_memory()
                    t0 = time.perf_counter()
                    if backend.prepare is not None:
                        backend.prepare(args.reference)
                    prepare_sec = time.perf_counter() - t0
                    t1 = time.perf_counter()
                    audio = backend.generate(text, args.reference, seed)
                    generation_sec = time.perf_counter() - t1
                    audio_sec = len(audio) / backend.sample_rate
                    path = run_dir / f"{backend.name}_{prompt_name}_seed{seed}.wav"
                    sf.write(path, audio, backend.sample_rate)
                    row = {
                        "backend": backend.name,
                        "prompt": prompt_name,
                        "seed": seed,
                        "prepare_sec": prepare_sec,
                        "generation_sec": generation_sec,
                        "audio_sec": audio_sec,
                        "rtf": generation_sec / max(audio_sec, 1e-9),
                        "peak_gpu_mem_mb": peak_memory_mb(),
                        "speaker_cosine_to_ref": safe_cosine(encoder, reference, audio, backend.sample_rate)
                        if encoder is not None and ref_sr == backend.sample_rate
                        else None,
                        "path": str(path),
                    }
                    rows.append(row)
                    print(f"{backend.name:18} {prompt_name:18} seed={seed}: {generation_sec:.3f}s ({audio_sec:.2f}s audio)", flush=True)
        except Exception as exc:
            manifest["skipped"][kind] = f"runtime {type(exc).__name__}: {exc}"
        finally:
            manifest["backends"][backend.name] = {
                "load_sec": backend.load_sec,
                "sample_rate": backend.sample_rate,
                "rows": rows,
                "summary": summarize(rows) if rows else {},
            }
            if backend.close:
                backend.close()

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2))
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
