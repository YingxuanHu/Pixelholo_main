from __future__ import annotations

import base64
import queue
import threading
import contextlib
import gc
import json
import logging
import traceback
import io
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Iterator, Protocol
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import cv2
import librosa
import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.text_normalize import clean_text_for_tts, warmup_text_normalizer
from src.utils.smart_buffer import SmartStreamBuffer
from src.utils.prosody_chunker import prosody_split
from src.llm.llm_service import LLMService
from src.utils.audio_stitcher import AudioStitcher

SILENCE_CULLING_ENABLED = True
SILENCE_RMS_THRESHOLD = 0.003

_AI_EXECUTOR = ThreadPoolExecutor(max_workers=1)

_llm_service: LLMService | None = None


def _get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(
            system_prompt="You are a helpful, witty AI assistant. Keep answers concise."
        )
    return _llm_service


warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pixelholo")
STYLE_TTS2_DIR = PROJECT_ROOT / "lib" / "StyleTTS2"

sys.path.append(str(PROJECT_ROOT))
from config import (  # noqa: E402
    DEFAULT_AVATAR_FPS,
    DEFAULT_AVATAR_PADS,
    DEFAULT_AVATAR_NOSMOOTH,
    PROFILE_TYPE_AVATAR,
    PROFILE_TYPE_VOICE,
    OUTPUTS_DIR,
    profile_data_root,
    processed_wavs_dir,
    raw_audio_dir,
    raw_videos_dir,
    resolve_dataset_root,
    resolve_training_dir,
    TRAINING_DIRNAME,
    training_root,
    inference_audio_dir,
    inference_video_dir,
)

if STYLE_TTS2_DIR.exists():
    sys.path.insert(0, str(STYLE_TTS2_DIR))

MEAN = -4
STD = 4
DEFAULT_ALPHA = 0.2
DEFAULT_BETA = 0.7
DEFAULT_DIFFUSION_STEPS = 10
DEFAULT_EMBEDDING_SCALE = 1.7
DEFAULT_F0_SCALE = 1.0
DEFAULT_LANG = "en-ca"
DEFAULT_MAX_CHUNK_CHARS = 180
DEFAULT_MAX_CHUNK_WORDS = 45
DEFAULT_PAUSE_MS = 40

_WORD_RE = re.compile(r"[A-Za-z']+|[^A-Za-z']+")
_WORD_ONLY_RE = re.compile(r"[A-Za-z']+$")

app = FastAPI()


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_engine_lock = threading.Lock()
_engines: dict[tuple[str, str], "StyleTTS2RepoEngine"] = {}
_lipsync_lock = threading.Lock()
_lipsync_engines: dict[str, "_LipSyncEngineProtocol"] = {}
_active_stream_lock = threading.Lock()
_active_stream_stop_events: set[threading.Event] = set()
_warmup_profile_lock = threading.Lock()


def _register_stream_stop_event(stop_event: threading.Event) -> None:
    with _active_stream_lock:
        _active_stream_stop_events.add(stop_event)


def _unregister_stream_stop_event(stop_event: threading.Event) -> None:
    with _active_stream_lock:
        _active_stream_stop_events.discard(stop_event)


def _interrupt_active_streams() -> int:
    with _active_stream_lock:
        events = list(_active_stream_stop_events)
    for event in events:
        event.set()
    return len(events)
_SIGMA_WARNED_PATHS: set[str] = set()


class _LipSyncEngineProtocol(Protocol):
    fps: float

    def load_profile(self, profile: str, profile_type: str = PROFILE_TYPE_AVATAR) -> None: ...
    def sync_chunk(self, audio_16k: np.ndarray, fps: float | None = None) -> list[np.ndarray]: ...


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    return buffer.getvalue()


PIXELHOLO_BINARY_STREAM_MEDIA_TYPE = "application/vnd.pixelholo.stream-v1"
_PIXELHOLO_BINARY_MAGIC = b"PHS1"


def _binary_stream_requested(request: Request) -> bool:
    transport = (request.headers.get("x-pixelholo-transport") or "").strip().lower()
    if transport == "binary":
        return True
    accept = (request.headers.get("accept") or "").lower()
    return PIXELHOLO_BINARY_STREAM_MEDIA_TYPE in accept


def _mobile_stream_requested(request: Request) -> bool:
    client = (request.headers.get("x-pixelholo-client") or "").strip().lower()
    if client in {"ios", "iphone", "mobile"}:
        return True
    user_agent = (request.headers.get("user-agent") or "").lower()
    return "iphone" in user_agent or "ipad" in user_agent


def _pack_binary_stream_packet(metadata: dict[str, object], payload: bytes = b"") -> bytes:
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    header = _PIXELHOLO_BINARY_MAGIC + struct.pack(">II", len(meta_bytes), len(payload))
    return header + meta_bytes + payload


def _split_text(text: str, max_chars: int, max_words: int) -> list[str]:
    if not text:
        return []
    sentences = [s.strip() for s in re.findall(r"[^.!?]+[.!?]?", text) if s.strip()]
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars and len(sentence.split()) <= max_words:
            chunks.append(sentence)
            continue
        words = sentence.split()
        current: list[str] = []
        for word in words:
            current.append(word)
            if len(" ".join(current)) >= max_chars or len(current) >= max_words:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
    return chunks


def _split_text_warmup(
    text: str, max_chars: int, max_words: int, warmup_words: int = 4, warmup_scan: int = 8
) -> list[str]:
    words = text.split()
    if len(words) <= warmup_scan:
        return _split_text(text, max_chars, max_words)
    split_idx = None
    for i, word in enumerate(words[:warmup_scan]):
        if any(p in word for p in [",", ".", "!", "?"]):
            split_idx = i + 1
            break
    if split_idx is None:
        split_idx = min(warmup_words, len(words))
    warmup = " ".join(words[:split_idx]).strip()
    rest = " ".join(words[split_idx:]).strip()
    if not warmup:
        return _split_text(text, max_chars, max_words)
    chunks = [warmup]
    if rest:
        chunks.extend(_split_text(rest, max_chars, max_words))
    return chunks


def _split_text_staircase(
    text: str,
    max_chars: int,
    max_words: int,
    limits: list[int] | None = None,
    lookback: int = 5,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    limits = limits or [4, 10, 25]
    chunks: list[str] = []
    cursor = 0
    step = 0
    total = len(words)
    while cursor < total:
        limit = limits[step] if step < len(limits) else limits[-1]
        limit = min(limit, max_words)
        end = min(cursor + limit, total)
        if end < total:
            for i in range(end - 1, max(cursor, end - lookback), -1):
                if words[i][-1] in ",.!?;":
                    end = i + 1
                    break
        candidate = " ".join(words[cursor:end])
        while candidate and len(candidate) > max_chars and end > cursor + 1:
            end -= 1
            candidate = " ".join(words[cursor:end])
        if not candidate:
            break
        chunks.append(candidate)
        cursor = end
        step += 1
    return chunks


def _split_text_prosody(text: str, max_chars: int, first_chunk_max_chars: int | None = None) -> list[str]:
    limit = first_chunk_max_chars or max(60, int(max_chars * 0.5))
    return prosody_split(text, max_chars=max_chars, first_chunk_max_chars=limit)


def _plan_stream_tts_chunks(
    sentence: str,
    max_chars: int,
    max_words: int,
    *,
    first_chunk_max_chars: int | None = None,
    is_first_global_chunk: bool = False,
) -> list[str]:
    text = sentence.strip()
    if not text:
        return []

    # Favor full-sentence synthesis to keep timbre/prosody stable across output.
    single_max_chars = max(
        80,
        min(
            int(os.getenv("STREAM_SENTENCE_SINGLE_MAX_CHARS", "220")),
            600,
        ),
    )
    single_max_words = max(
        16,
        min(
            int(os.getenv("STREAM_SENTENCE_SINGLE_MAX_WORDS", "48")),
            120,
        ),
    )

    # Optional fast-first-byte split only for the very first global chunk.
    if (
        is_first_global_chunk
        and first_chunk_max_chars is not None
        and len(text) > int(first_chunk_max_chars)
    ):
        return _split_text_prosody(
            text,
            max_chars=max_chars,
            first_chunk_max_chars=int(first_chunk_max_chars),
        )

    if len(text) <= single_max_chars and len(text.split()) <= single_max_words:
        return [text]

    return _split_text(text, max_chars=max_chars, max_words=max_words)


def _smart_chunks(text: str, min_chars: int = 40, max_chars: int = 150) -> list[str]:
    buffer = SmartStreamBuffer(min_chunk_size=min_chars, max_chunk_size=max_chars)
    chunks: list[str] = []
    for word in text.split():
        chunk = buffer.add_token(word + " ")
        if chunk:
            chunks.append(chunk)
    tail = buffer.flush()
    if tail:
        chunks.append(tail)
    return chunks


def _apply_pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float,
) -> np.ndarray:
    if semitones == 0:
        return audio
    if shutil.which("rubberband"):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = Path(tmpdir) / "in.wav"
                out_path = Path(tmpdir) / "out.wav"
                import soundfile as sf

                sf.write(in_path, audio, sample_rate)
                result = subprocess.run(
                    ["rubberband", "-p", str(semitones), str(in_path), str(out_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or "rubberband failed")
                shifted, out_sr = sf.read(out_path)
                if shifted.ndim > 1:
                    shifted = shifted.mean(axis=1)
                if out_sr != sample_rate:
                    try:
                        import librosa
                    except Exception as exc:
                        raise RuntimeError("Missing librosa for pitch resampling.") from exc
                    shifted = librosa.resample(shifted, orig_sr=out_sr, target_sr=sample_rate)
                return np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
    try:
        import librosa
    except Exception as exc:
        raise RuntimeError("Missing librosa for pitch shifting.") from exc
    shifted = librosa.effects.pitch_shift(audio.astype(np.float32), sr=sample_rate, n_steps=semitones)
    return np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0)


def _apply_de_esser(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    if cutoff_hz <= 0:
        return audio
    nyquist = sample_rate * 0.5
    if cutoff_hz >= nyquist:
        return audio
    try:
        from scipy.signal import butter, sosfilt
    except Exception as exc:
        raise RuntimeError("Missing scipy for de-esser filtering.") from exc
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    filtered = sosfilt(sos, audio.astype(np.float32))
    return np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)


def _soft_clip(audio: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    if audio.size == 0:
        return audio
    audio = audio.astype(np.float32, copy=False)
    max_val = float(np.max(np.abs(audio)))
    if max_val <= threshold:
        return audio
    return np.tanh(audio / threshold) * threshold


def _remove_dc(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    audio = audio.astype(np.float32, copy=False)
    return audio - float(np.mean(audio))


def _apply_crossfade(chunks: list[np.ndarray], sample_rate: int, crossfade_ms: float) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    if crossfade_ms <= 0:
        return np.concatenate(chunks)
    cross_len = int(sample_rate * (crossfade_ms / 1000.0))
    if cross_len < 2:
        return np.concatenate(chunks)
    faded = chunks[0].astype(np.float32, copy=False)
    t = np.linspace(0.0, 1.0, cross_len, dtype=np.float32)
    fade_in = np.sin(t * (np.pi / 2.0))
    fade_out = np.cos(t * (np.pi / 2.0))
    for nxt in chunks[1:]:
        nxt = nxt.astype(np.float32, copy=False)
        if faded.size < cross_len or nxt.size < cross_len:
            faded = np.concatenate([faded, nxt])
            continue
        tail = faded[-cross_len:] * fade_out
        head = nxt[:cross_len] * fade_in
        blended = tail + head
        faded = np.concatenate([faded[:-cross_len], blended, nxt[cross_len:]])
    return faded


def _fade_edges(audio: np.ndarray, sample_rate: int, fade_ms: float = 5.0) -> np.ndarray:
    if audio.size == 0:
        return audio
    fade_len = int(sample_rate * (fade_ms / 1000.0))
    if fade_len <= 1:
        return audio
    if fade_len * 2 > audio.size:
        fade_len = max(1, audio.size // 2)
    audio = audio.astype(np.float32, copy=False)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out
    return audio


def _smart_vad_trim(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 30.0,
    frame_length: int = 1024,
    hop_length: int = 256,
    pad_ms: float = 50.0,
) -> np.ndarray:
    try:
        import librosa
    except Exception:
        return audio
    if audio.size == 0:
        return audio
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.power_to_db(rms * rms, ref=np.max)
    non_silent = np.flatnonzero(rms_db > -top_db)
    if non_silent.size == 0:
        return audio
    start = librosa.frames_to_samples(non_silent[0], hop_length=hop_length)
    end = librosa.frames_to_samples(non_silent[-1], hop_length=hop_length)
    pad = int(sample_rate * (pad_ms / 1000.0))
    start = max(0, start - pad)
    end = min(audio.size, end + pad)
    return audio[start:end]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _generate_with_seed(
    engine: "StyleTTS2RepoEngine",
    seed: int | None,
    text: str,
    ref_wav_path: Path,
    alpha: float,
    beta: float,
    diffusion_steps: int,
    embedding_scale: float,
    f0_scale: float,
    phonemizer_lang: str | None,
    lexicon: dict[str, str] | None,
    prepend_boundary: bool = False,
) -> np.ndarray:
    if seed is not None:
        _seed_everything(seed)
    # Optional sacrificial boundary to stabilize the very first chunk only.
    padded_text = f". {text}" if prepend_boundary else text
    audio = engine.generate(
        padded_text,
        ref_wav_path=ref_wav_path,
        alpha=alpha,
        beta=beta,
        diffusion_steps=diffusion_steps,
        embedding_scale=embedding_scale,
        f0_scale=f0_scale,
        phonemizer_lang=phonemizer_lang,
        lexicon=lexicon,
        seed=seed,
    )
    if prepend_boundary:
        # Trim a short lead-in to remove the sacrificial boundary.
        trim = int(engine.sample_rate * 0.04)  # ~40ms at 24k
        if audio is not None and audio.size > trim:
            audio = audio[trim:]
    return audio


def _stream_subprocess(command: list[str], cwd: Path) -> Iterator[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        yield f"{line.rstrip()}\n"
    exit_code = process.wait()
    if exit_code != 0:
        logger.error(
            "component=backend op=subprocess_stream status=error exit_code=%s cwd=%s cmd=%s",
            exit_code,
            cwd,
            command,
        )
        yield f"[process exited {exit_code}]\n"
        yield "ERROR: Subprocess failed. See logs above for details.\n"
    else:
        yield f"[process exited {exit_code}]\n"


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _first_wav(directory: Path | None) -> Path | None:
    if directory is None or not directory.exists():
        return None
    for wav_path in sorted(directory.glob("*.wav")):
        return wav_path
    return None


def _resolve_ref_wav(
    ref_wav_path: str | None,
    speaker: str | None,
    profile_type: str | None,
    model_path: Path | None = None,
) -> Path:
    if ref_wav_path:
        path = Path(ref_wav_path).expanduser()
        if path.exists():
            return path
        raise HTTPException(status_code=400, detail=f"ref_wav_path not found: {ref_wav_path}")

    if model_path is not None:
        profile = _load_profile_defaults_for_speaker(model_path, speaker, profile_type)
        ref_value = profile.get("ref_wav_path")
        if ref_value:
            base_dir = resolve_training_dir(speaker, profile_type) if speaker else model_path.parent
            candidate = _resolve_path(base_dir, str(ref_value))
            if candidate and candidate.exists():
                return candidate

    env_ref = os.getenv("STYLE_TTS2_REF_WAV")
    if env_ref:
        path = Path(env_ref).expanduser()
        if path.exists():
            return path
        raise HTTPException(status_code=400, detail=f"STYLE_TTS2_REF_WAV not found: {env_ref}")

    env_dir = os.getenv("STYLE_TTS2_REF_DIR")
    if env_dir:
        candidate = _first_wav(Path(env_dir).expanduser())
        if candidate:
            return candidate
        raise HTTPException(status_code=400, detail=f"No wav files in STYLE_TTS2_REF_DIR: {env_dir}")

    if speaker:
        candidate = _first_wav(processed_wavs_dir(speaker, profile_type))
        if candidate:
            return candidate

    data_root = PROJECT_ROOT / "data"
    processed_dirs = sorted(data_root.glob("*/*/processed_wavs")) + sorted(
        data_root.glob("*/processed_wavs")
    )
    if len(processed_dirs) == 1:
        candidate = _first_wav(processed_dirs[0])
        if candidate:
            return candidate

    raise HTTPException(
        status_code=400,
        detail=(
            "Provide ref_wav_path or set STYLE_TTS2_REF_WAV/STYLE_TTS2_REF_DIR. "
            "If using speaker, pass the speaker name."
        ),
    )


def _resolve_model_path(
    model_path: str | None,
    speaker: str | None,
    profile_type: str | None,
) -> Path:
    if model_path:
        candidate = Path(model_path).expanduser()
        if candidate.exists():
            return candidate
        raise HTTPException(status_code=400, detail=f"model_path not found: {candidate}")

    env_model = os.getenv("STYLE_TTS2_MODEL")
    if env_model:
        candidate = Path(env_model).expanduser()
        if candidate.exists():
            return candidate
        raise HTTPException(status_code=400, detail=f"STYLE_TTS2_MODEL not found: {candidate}")

    if speaker:
        training_dir = resolve_training_dir(speaker, profile_type)
        for filename in ("profile.json", "inference_defaults.json"):
            candidate = training_dir / filename
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text())
                except json.JSONDecodeError:
                    data = {}
                if isinstance(data, dict):
                    for key in ("model_path", "checkpoint_path", "checkpoint"):
                        value = data.get(key)
                        if value:
                            resolved = Path(str(value)).expanduser()
                            if not resolved.is_absolute():
                                resolved = training_dir / resolved
                            if resolved.exists():
                                return resolved
        best_path = training_dir / "best_epoch.txt"
        if best_path.exists():
            content = best_path.read_text().strip()
            if content:
                candidate = Path(content)
                if candidate.exists():
                    return candidate
        checkpoints = sorted(training_dir.glob("epoch_2nd_*.pth"))
        if checkpoints:
            return checkpoints[-1]

    raise HTTPException(status_code=400, detail="model_path is required")


def _list_profiles(profile_type: str | None) -> list[dict[str, object]]:
    types = [profile_type] if profile_type else [PROFILE_TYPE_VOICE, PROFILE_TYPE_AVATAR]
    profiles: list[dict[str, object]] = []
    legacy_training_root = OUTPUTS_DIR / TRAINING_DIRNAME

    def _has_training_artifacts(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        if (path / "profile.json").exists():
            return True
        if list(path.glob("epoch_2nd_*.pth")):
            return True
        return False

    for ptype in types:
        data_root = profile_data_root(ptype)
        training_root_dir = training_root(ptype)
        names: set[str] = set()
        if data_root.exists():
            names.update([p.name for p in data_root.iterdir() if p.is_dir()])
        if training_root_dir.exists():
            names.update([p.name for p in training_root_dir.iterdir() if p.is_dir()])
        if ptype == PROFILE_TYPE_VOICE and legacy_training_root.exists():
            legacy_names = []
            for p in legacy_training_root.iterdir():
                if p.is_dir() and _has_training_artifacts(p):
                    legacy_names.append(p.name)
            names.update(legacy_names)
        for name in sorted(names):
            data_dir = data_root / name
            training_dir = training_root_dir / name
            if not training_dir.exists():
                legacy_dir = legacy_training_root / name
                if legacy_dir.exists():
                    training_dir = legacy_dir
            has_training = _has_training_artifacts(training_dir)
            if not data_dir.exists() and not has_training:
                continue
            raw_count = len(list((data_dir / "raw_videos").glob("*"))) if data_dir.exists() else 0
            raw_audio_count = len(list((data_dir / "raw_audio").glob("*"))) if data_dir.exists() else 0
            processed_count = len(list((data_dir / "processed_wavs").glob("*.wav"))) if data_dir.exists() else 0
            profile_json = training_dir / "profile.json"
            best_epoch = None
            best_path = training_dir / "best_epoch.txt"
            if best_path.exists():
                content = best_path.read_text().strip()
                if content:
                    best_epoch = content
            checkpoints = sorted(training_dir.glob("epoch_2nd_*.pth")) if training_dir.exists() else []
            latest_ckpt = str(checkpoints[-1]) if checkpoints else None
            profiles.append(
                {
                    "name": name,
                    "profile_type": ptype,
                    "has_data": data_dir.exists(),
                    "raw_files": raw_count,
                    "raw_audio_files": raw_audio_count,
                    "processed_wavs": processed_count,
                    "has_profile": profile_json.exists() or bool(checkpoints),
                    "best_checkpoint": best_epoch,
                    "latest_checkpoint": latest_ckpt,
                }
            )
    return profiles


def _normalize_profile_type(profile_type: str | None) -> str:
    return PROFILE_TYPE_AVATAR if profile_type == PROFILE_TYPE_AVATAR else PROFILE_TYPE_VOICE


def _sanitize_profile_name(name: str, field_name: str = "profile") -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    if cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"{field_name} is invalid")
    if any(token in cleaned for token in ("/", "\\", "\x00")):
        raise HTTPException(status_code=400, detail=f"{field_name} contains invalid characters")
    return cleaned


def _profile_paths(profile_name: str, profile_type: str) -> list[Path]:
    ptype = _normalize_profile_type(profile_type)
    candidates: list[Path] = [
        profile_data_root(ptype) / profile_name,
        training_root(ptype) / profile_name,
        inference_audio_dir(profile_name, ptype),
    ]
    if ptype == PROFILE_TYPE_AVATAR:
        candidates.append(inference_video_dir(profile_name, ptype))
    else:
        legacy_training = OUTPUTS_DIR / TRAINING_DIRNAME / profile_name
        candidates.append(legacy_training)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _rewrite_profile_refs_in_dir(
    root: Path,
    *,
    old_name: str,
    new_name: str,
    path_replacements: dict[str, str],
) -> None:
    if not root.exists() or not root.is_dir():
        return
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".json", ".txt"}:
            continue
        try:
            if file_path.stat().st_size > 5_000_000:
                continue
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        updated = text
        for old_path, new_path in path_replacements.items():
            updated = updated.replace(old_path, new_path)
            updated = updated.replace(old_path.replace("\\", "/"), new_path.replace("\\", "/"))
        updated = updated.replace(f"/{old_name}/", f"/{new_name}/")
        updated = updated.replace(f"\\{old_name}\\", f"\\{new_name}\\")
        if updated != text:
            file_path.write_text(updated, encoding="utf-8")


def _drop_warmup_cache_for_profile(profile_name: str, profile_type: str) -> None:
    ptype = _normalize_profile_type(profile_type)
    global _warmed_tts_profiles, _warmed_lipsync_profiles
    _warmed_tts_profiles = {
        key for key in _warmed_tts_profiles if not (key[0] == profile_name and key[1] == ptype)
    }
    _warmed_lipsync_profiles = {
        key
        for key in _warmed_lipsync_profiles
        if not (key[0] == profile_name and key[1] == ptype)
    }


def _resolve_config_path(
    model_path: Path,
    config_path: str | None,
    speaker: str | None = None,
    profile_type: str | None = None,
) -> Path:
    if config_path:
        path = Path(config_path).expanduser()
        if path.exists():
            return path
        raise HTTPException(status_code=400, detail=f"config_path not found: {config_path}")

    profile = _load_profile_defaults_for_speaker(model_path, speaker, profile_type)
    if "config_path" in profile:
        base_dir = resolve_training_dir(speaker, profile_type) if speaker else model_path.parent
        candidate = _resolve_path(base_dir, str(profile["config_path"]))
        if candidate and candidate.exists():
            return candidate

    env_config = os.getenv("STYLE_TTS2_CONFIG")
    if env_config:
        path = Path(env_config).expanduser()
        if path.exists():
            return path
        raise HTTPException(status_code=400, detail=f"STYLE_TTS2_CONFIG not found: {env_config}")

    candidate = model_path.parent / "config_ft.yml"
    if candidate.exists():
        return candidate

    raise HTTPException(
        status_code=400,
        detail="config_path is required (or set STYLE_TTS2_CONFIG).",
    )


def _resolve_f0_scale(model_path: Path, f0_scale: float | None) -> float:
    if f0_scale is not None:
        return f0_scale

    candidate = model_path.parent / "f0_scale.txt"
    if candidate.exists():
        try:
            return float(candidate.read_text().strip())
        except ValueError:
            pass

    return DEFAULT_F0_SCALE


def _load_profile_defaults(model_path: Path) -> dict:
    for filename in ("profile.json", "inference_defaults.json"):
        candidate = model_path.parent / filename
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text())
            except json.JSONDecodeError as exc:
                logger.warning(
                    "component=backend op=load_profile_defaults fallback=empty_defaults reason=json_invalid path=%s error=%s",
                    candidate,
                    exc,
                )
                continue
            if isinstance(data, dict):
                return data
    return {}


def _load_inference_defaults(config_path: Path) -> dict:
    try:
        data = yaml.safe_load(config_path.read_text())
    except Exception as exc:
        logger.warning(
            "component=backend op=load_inference_defaults fallback=empty_defaults reason=yaml_invalid path=%s error=%s",
            config_path,
            exc,
        )
        return {}
    if not isinstance(data, dict):
        return {}
    for key in ("inference_params", "inference", "synthesis_params", "generate_params"):
        block = data.get(key)
        if isinstance(block, dict):
            return block
    return {}


def _load_profile_defaults_for_speaker(
    model_path: Path,
    speaker: str | None,
    profile_type: str | None,
) -> dict:
    if speaker:
        training_dir = resolve_training_dir(speaker, profile_type)
        for filename in ("profile.json", "inference_defaults.json"):
            candidate = training_dir / filename
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text())
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "component=backend op=load_profile_defaults fallback=empty_defaults reason=json_invalid path=%s profile=%s profile_type=%s error=%s",
                        candidate,
                        speaker,
                        profile_type,
                        exc,
                    )
                    data = {}
                if isinstance(data, dict):
                    return data
    return _load_profile_defaults(model_path)


def _resolve_phonemizer_lang(
    model_path: Path,
    req: "GenerateRequest",
    speaker: str | None,
    profile_type: str | None,
) -> str:
    if req.phonemizer_lang:
        return req.phonemizer_lang
    profile = _load_profile_defaults_for_speaker(model_path, speaker, profile_type)
    if "phonemizer_lang" in profile:
        return str(profile["phonemizer_lang"])
    return os.getenv("STYLE_TTS2_LANG", DEFAULT_LANG)


def _resolve_lexicon_path(
    model_path: Path,
    req: "GenerateRequest",
    speaker: str | None,
    profile_type: str | None,
) -> Path | None:
    if req.lexicon_path:
        return Path(req.lexicon_path).expanduser()
    profile = _load_profile_defaults_for_speaker(model_path, speaker, profile_type)
    if "lexicon_path" in profile:
        return Path(str(profile["lexicon_path"])).expanduser()
    env_path = os.getenv("STYLE_TTS2_LEXICON")
    if env_path:
        return Path(env_path).expanduser()
    if speaker:
        candidate = resolve_dataset_root(speaker, profile_type) / "lexicon.json"
        if candidate.exists():
            return candidate
    data_root = PROJECT_ROOT / "data"
    processed_dirs = sorted(data_root.glob("*/*/processed_wavs")) + sorted(
        data_root.glob("*/processed_wavs")
    )
    if len(processed_dirs) == 1:
        candidate = processed_dirs[0].parent / "lexicon.json"
        if candidate.exists():
            return candidate
    return None


def _load_lexicon(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"lexicon_path not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid lexicon JSON: {path}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Lexicon must be a JSON object.")
    cleaned = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            cleaned[key.lower()] = value.strip()
    return cleaned or None


def _resolve_inference_params(
    model_path: Path,
    config_path: Path,
    req: "GenerateRequest",
    speaker: str | None,
    profile_type: str | None,
) -> dict[str, float]:
    defaults = _load_inference_defaults(config_path)
    profile = _load_profile_defaults_for_speaker(model_path, speaker, profile_type)

    def _pick_value(name: str, req_value, profile_value, config_value):
        if req_value is not None:
            return req_value
        if profile_value is not None:
            return profile_value
        if config_value is not None:
            return config_value
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing inference param '{name}'. "
                "Set it in profile.json, request body, or config_ft.yml (inference_params)."
            ),
        )

    params = {
        "alpha": _pick_value("alpha", req.alpha, profile.get("alpha"), defaults.get("alpha")),
        "beta": _pick_value("beta", req.beta, profile.get("beta"), defaults.get("beta")),
        "diffusion_steps": _pick_value(
            "diffusion_steps",
            req.diffusion_steps,
            profile.get("diffusion_steps"),
            defaults.get("diffusion_steps"),
        ),
        "embedding_scale": _pick_value(
            "embedding_scale",
            req.embedding_scale,
            profile.get("embedding_scale"),
            defaults.get("embedding_scale"),
        ),
        "f0_scale": _pick_value(
            "f0_scale",
            req.f0_scale,
            profile.get("f0_scale"),
            defaults.get("f0_scale"),
        ),
    }

    if req.f0_scale is None and "f0_scale" not in profile and params["f0_scale"] is None:
        params["f0_scale"] = _resolve_f0_scale(model_path, None)

    return params


def _stream_avatar_from_text_iter(
    req: GenerateRequest,
    text_iter: Iterator[str],
    binary_transport: bool = False,
    mobile_profile: bool = False,
) -> StreamingResponse:
    start_time = time.perf_counter()
    profile = req.avatar_profile or req.speaker
    if not profile:
        raise HTTPException(status_code=400, detail="profile is required")

    profile_type = req.profile_type or PROFILE_TYPE_AVATAR
    model_path = _resolve_model_path(req.model_path, profile, profile_type)
    config_path = _resolve_config_path(model_path, req.config_path, profile, profile_type)
    ref_wav_path = _resolve_ref_wav(req.ref_wav_path, profile, profile_type, model_path)
    profile_dir = resolve_dataset_root(profile, profile_type)
    params = _resolve_inference_params(model_path, config_path, req, profile, profile_type)
    phonemizer_lang = _resolve_phonemizer_lang(model_path, req, profile, profile_type)
    lexicon_path = _resolve_lexicon_path(model_path, req, profile, profile_type)
    lexicon = _load_lexicon(lexicon_path)

    engine = _get_engine(model_path, config_path)
    try:
        lipsync = _get_lipsync_engine(req.lipsync_backend)
        with _lipsync_lock:
            lipsync.load_profile(profile, profile_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    is_musetalk = lipsync.__class__.__name__ == "MuseTalkBridge"
    fps = req.avatar_fps or lipsync.fps
    if mobile_profile and req.avatar_fps is None:
        try:
            fps = min(float(fps), float(os.getenv("IOS_STREAM_AVATAR_FPS", "15")))
        except (TypeError, ValueError):
            fps = min(float(fps), 15.0)

    profile_defaults = _load_profile_defaults(model_path)
    # Request-time MuseTalk knobs (ignored for Wav2Lip).
    # Reset to defaults when no override is provided so stale in-memory values do not linger.
    if is_musetalk:
        if req.musetalk_batch_size is not None:
            lipsync.batch_size = max(1, min(int(req.musetalk_batch_size), 128))
        if req.musetalk_infer_fps is not None:
            lipsync.infer_fps = max(6.0, min(float(req.musetalk_infer_fps), 30.0))
        elif mobile_profile:
            try:
                lipsync.infer_fps = max(
                    6.0,
                    min(float(lipsync.infer_fps), float(os.getenv("IOS_STREAM_MUSETALK_INFER_FPS", "15"))),
                )
            except (TypeError, ValueError):
                lipsync.infer_fps = max(6.0, min(float(lipsync.infer_fps), 15.0))

        default_face_scale = float(
            profile_defaults.get(
                "musetalk_face_scale",
                os.getenv("MUSE_TALK_FACE_SCALE", "1.0"),
            )
        )
        lipsync.face_scale = max(0.75, min(default_face_scale, 1.15))
        if req.musetalk_face_scale is not None:
            lipsync.face_scale = max(0.75, min(float(req.musetalk_face_scale), 1.15))

        if req.musetalk_audio_history_sec is not None:
            lipsync.audio_history_sec = max(
                0.5, min(float(req.musetalk_audio_history_sec), 6.0)
            )
    if is_musetalk and req.musetalk_max_chunk_chars is not None:
        max_chars = req.musetalk_max_chunk_chars
    elif req.max_chunk_chars is not None:
        max_chars = req.max_chunk_chars
    elif is_musetalk:
        max_chars = int(
            profile_defaults.get(
                "musetalk_max_chunk_chars",
                os.getenv("MUSE_TALK_MAX_CHUNK_CHARS", "120"),
            )
        )
    else:
        max_chars = profile_defaults.get("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS)
    max_words = (
        req.max_chunk_words
        if req.max_chunk_words is not None
        else profile_defaults.get("max_chunk_words", DEFAULT_MAX_CHUNK_WORDS)
    )
    pause_ms = (
        req.pause_ms
        if req.pause_ms is not None
        else profile_defaults.get("pause_ms", DEFAULT_PAUSE_MS)
    )
    pad_text = (
        req.pad_text
        if req.pad_text is not None
        else profile_defaults.get("pad_text", True)
    )
    pad_text_token = (
        req.pad_text_token
        if req.pad_text_token is not None
        else profile_defaults.get("pad_text_token", "...")
    )
    smart_trim_db = (
        req.smart_trim_db
        if req.smart_trim_db is not None
        else profile_defaults.get("stream_smart_trim_db", 0.0)
    )
    smart_trim_pad_ms = (
        req.smart_trim_pad_ms
        if req.smart_trim_pad_ms is not None
        else profile_defaults.get("stream_smart_trim_pad_ms", 30.0)
    )
    # Avoid per-TTS-chunk fade envelopes by default. AudioStitcher already handles
    # chunk boundary smoothing, and extra edge fades can introduce audible flutter/pops.
    stream_chunk_edge_fade_ms = float(
        profile_defaults.get(
            "stream_chunk_edge_fade_ms",
            os.getenv("STREAM_CHUNK_EDGE_FADE_MS", "0.0"),
        )
    )

    seed = req.seed if req.seed is not None else profile_defaults.get("seed")
    if seed is None:
        seed = 1234
    pause = np.zeros(int(engine.sample_rate * (pause_ms / 1000.0)), dtype=np.float32)
    comma_pause_ms = max(50.0, float(pause_ms) * 0.35)
    comma_pause = np.zeros(int(engine.sample_rate * (comma_pause_ms / 1000.0)), dtype=np.float32)
    first_chunk_max_chars: int | None = None
    if is_musetalk:
        if req.musetalk_first_chunk_chars is not None:
            first_chunk_max_chars = max(8, min(int(req.musetalk_first_chunk_chars), 200))
        else:
            first_chunk_max_chars = int(
                profile_defaults.get(
                    "musetalk_first_chunk_chars",
                    os.getenv("MUSE_TALK_FIRST_CHUNK_CHARS", "72"),
                )
            )
    musetalk_window_sec = 0.0
    if is_musetalk:
        if req.musetalk_stream_window_sec is not None:
            musetalk_window_sec = max(0.2, min(float(req.musetalk_stream_window_sec), 3.0))
        else:
            musetalk_window_sec = float(
                profile_defaults.get(
                    "musetalk_stream_window_sec",
                    os.getenv("MUSE_TALK_STREAM_WINDOW_SEC", "1.2"),
                )
            )
    musetalk_lookahead_sec = 0.0
    if is_musetalk:
        if req.musetalk_lookahead_sec is not None:
            musetalk_lookahead_sec = max(0.0, min(float(req.musetalk_lookahead_sec), 0.8))
        else:
            musetalk_lookahead_sec = float(
                profile_defaults.get(
                    "musetalk_lookahead_sec",
                    os.getenv("MUSE_TALK_LOOKAHEAD_SEC", "0.16"),
                )
            )
    jpeg_quality = 80
    if is_musetalk:
        if req.musetalk_jpeg_quality is not None:
            jpeg_quality = max(50, min(int(req.musetalk_jpeg_quality), 95))
        else:
            jpeg_quality = int(
                profile_defaults.get(
                    "musetalk_jpeg_quality",
                    os.getenv("MUSE_TALK_JPEG_QUALITY", "92"),
                )
            )
    if mobile_profile and req.musetalk_jpeg_quality is None:
        try:
            jpeg_quality = max(50, min(int(jpeg_quality), int(os.getenv("IOS_STREAM_JPEG_QUALITY", "72"))))
        except (TypeError, ValueError):
            jpeg_quality = max(50, min(int(jpeg_quality), 72))
    mobile_max_frame_edge = 0
    if mobile_profile:
        try:
            mobile_max_frame_edge = max(256, min(int(os.getenv("IOS_STREAM_MAX_FRAME_EDGE", "640")), 1280))
        except (TypeError, ValueError):
            mobile_max_frame_edge = 640

    logger.info(
        "component=stream op=avatar_stream_config profile=%s backend=%s mobile=%s fps=%.2f jpeg_quality=%s max_frame_edge=%s",
        profile,
        req.lipsync_backend or "default",
        mobile_profile,
        float(fps),
        jpeg_quality,
        mobile_max_frame_edge or "full",
    )

    def _generator() -> Iterator[str]:
        stop_event = threading.Event()
        _register_stream_stop_event(stop_event)
        result_queue: queue.Queue[tuple[str, int, np.ndarray | None, np.ndarray | None]] = queue.Queue(
            maxsize=3
        )
        sentence_queue: queue.Queue[str | None] = queue.Queue(maxsize=10)
        stitcher = AudioStitcher(sample_rate=engine.sample_rate, fade_len_ms=15.0)

        def _sentence_reader() -> None:
            try:
                for sentence in text_iter:
                    if stop_event.is_set():
                        break
                    while not stop_event.is_set():
                        try:
                            sentence_queue.put(sentence, timeout=0.2)
                            break
                        except queue.Full:
                            continue
            finally:
                while True:
                    try:
                        sentence_queue.put(None, timeout=0.2)
                        break
                    except queue.Full:
                        if stop_event.is_set():
                            break
                        continue

        def _put_result(
            kind: str,
            idx: int,
            audio: np.ndarray | None,
            audio_16k: np.ndarray | None,
        ) -> None:
            while not stop_event.is_set():
                try:
                    result_queue.put((kind, idx, audio, audio_16k), timeout=0.2)
                    return
                except queue.Full:
                    continue

        def _audio_worker() -> None:
            idx = 0
            tts_chunk_index = 0
            try:
                if seed is not None:
                    _seed_everything(seed)
                while not stop_event.is_set():
                    try:
                        sentence = sentence_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if sentence is None:
                        break
                    clean_sentence = clean_text_for_tts(sentence)
                    if not re.search(r"[A-Za-z0-9]", clean_sentence):
                        continue
                    sentence_ref = ref_wav_path
                    sentence_chunks = _plan_stream_tts_chunks(
                        clean_sentence,
                        max_chars=max_chars,
                        max_words=max_words,
                        first_chunk_max_chars=first_chunk_max_chars,
                        is_first_global_chunk=(tts_chunk_index == 0),
                    )
                    for chunk in sentence_chunks:
                        if stop_event.is_set():
                            return
                        if pad_text:
                            chunk = f"{pad_text_token} {chunk} {pad_text_token}"
                        is_first_tts_chunk = tts_chunk_index == 0
                        chunk_seed = seed
                        future = _AI_EXECUTOR.submit(
                            _generate_with_seed,
                            engine,
                            chunk_seed,
                            chunk,
                            sentence_ref,
                            params["alpha"],
                            params["beta"],
                            params["diffusion_steps"],
                            params["embedding_scale"],
                            params["f0_scale"],
                            phonemizer_lang,
                            lexicon,
                            is_first_tts_chunk,
                        )
                        while True:
                            if stop_event.is_set():
                                future.cancel()
                                return
                            try:
                                audio = future.result(timeout=0.2)
                                break
                            except FutureTimeoutError:
                                continue
                        tts_chunk_index += 1
                        if audio is None or audio.size == 0:
                            continue
                        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                        if smart_trim_db and smart_trim_db > 0:
                            audio = _smart_vad_trim(
                                audio,
                                engine.sample_rate,
                                top_db=float(smart_trim_db),
                                pad_ms=float(smart_trim_pad_ms),
                            )
                        audio = _remove_dc(audio.astype(np.float32, copy=False))
                        pitch_shift = (
                            req.pitch_shift
                            if req.pitch_shift is not None
                            else profile_defaults.get("pitch_shift", 0.0)
                        )
                        if pitch_shift:
                            audio = _apply_pitch_shift(audio, engine.sample_rate, pitch_shift)
                        de_esser_cutoff = (
                            req.de_esser_cutoff
                            if req.de_esser_cutoff is not None
                            else profile_defaults.get("de_esser_cutoff", 0.0)
                        )
                        de_esser_order = (
                            req.de_esser_order
                            if req.de_esser_order is not None
                            else profile_defaults.get("de_esser_order", 2)
                        )
                        if de_esser_cutoff:
                            audio = _apply_de_esser(
                                audio, engine.sample_rate, de_esser_cutoff, int(de_esser_order)
                            )
                        audio = _soft_clip(audio)
                        if stream_chunk_edge_fade_ms > 0:
                            audio = _fade_edges(
                                audio,
                                engine.sample_rate,
                                fade_ms=float(stream_chunk_edge_fade_ms),
                            )
                        is_sentence_end = chunk.rstrip().endswith((".", "!", "?"))
                        is_clause_end = chunk.rstrip().endswith((",", ";", ":"))
                        if is_sentence_end and pause.size:
                            audio = np.concatenate([audio, pause])
                        elif is_clause_end and comma_pause.size:
                            audio = np.concatenate([audio, comma_pause])
                        stitched = stitcher.process(audio)
                        if stitched.size:
                            audio_16k = librosa.resample(
                                stitched, orig_sr=engine.sample_rate, target_sr=16000
                            )
                            _put_result("data", idx, stitched, audio_16k.astype(np.float32))
                            idx += 1
                tail = stitcher.flush()
                if tail.size and not stop_event.is_set():
                    audio_16k = librosa.resample(
                        tail, orig_sr=engine.sample_rate, target_sr=16000
                    )
                    _put_result("data", idx, tail, audio_16k.astype(np.float32))
                    idx += 1
                _put_result("done", -1, None, None)
            except Exception as exc:
                _put_result("error", -1, np.array([str(exc)], dtype=object), None)

        sentence_reader_thread = threading.Thread(target=_sentence_reader, daemon=True)
        audio_worker_thread = threading.Thread(target=_audio_worker, daemon=True)
        sentence_reader_thread.start()
        audio_worker_thread.start()

        last_frame: np.ndarray | None = None
        musetalk_prev_tail_frames: list[np.ndarray] = []
        musetalk_frame_crossfade = 0
        if is_musetalk:
            try:
                musetalk_frame_crossfade = max(
                    0,
                    min(
                        6,
                        int(
                            profile_defaults.get(
                                "musetalk_frame_crossfade",
                                os.getenv("MUSE_TALK_FRAME_CROSSFADE", "2"),
                            )
                        ),
                    ),
                )
            except (TypeError, ValueError):
                musetalk_frame_crossfade = 2
        emit_index = 0
        window_src_samples = 0
        window_tgt_samples = 0
        lookahead_src_samples = 0
        lookahead_tgt_samples = 0
        pending_audio = np.zeros(0, dtype=np.float32)
        pending_audio_16k = np.zeros(0, dtype=np.float32)
        if is_musetalk and musetalk_window_sec > 0:
            window_src_samples = max(1, int(round(engine.sample_rate * musetalk_window_sec)))
            window_tgt_samples = max(1, int(round(16000.0 * musetalk_window_sec)))
        if is_musetalk and musetalk_lookahead_sec > 0:
            lookahead_src_samples = max(0, int(round(engine.sample_rate * musetalk_lookahead_sec)))
            lookahead_tgt_samples = max(0, int(round(16000.0 * musetalk_lookahead_sec)))

        def _emit_one(
            sub_audio: np.ndarray,
            sub_audio_16k: np.ndarray,
            lookahead_16k: np.ndarray | None = None,
        ) -> Iterator[str | bytes]:
            nonlocal last_frame, emit_index, musetalk_prev_tail_frames
            rms = (
                float(np.sqrt(np.mean(np.square(sub_audio))))
                if sub_audio is not None and sub_audio.size
                else 0.0
            )
            frames_needed = (
                max(1, int(round((len(sub_audio_16k) / 16000.0) * fps)))
                if sub_audio_16k is not None
                else 0
            )

            if SILENCE_CULLING_ENABLED and rms < SILENCE_RMS_THRESHOLD and last_frame is not None:
                frames = [last_frame.copy() for _ in range(frames_needed)]
            else:
                if is_musetalk and lookahead_16k is not None and lookahead_16k.size > 0:
                    frames = lipsync.sync_chunk(
                        sub_audio_16k,
                        fps=fps,
                        lookahead_16k=lookahead_16k,
                    )
                else:
                    frames = lipsync.sync_chunk(sub_audio_16k, fps=fps)
                if frames:
                    if is_musetalk and musetalk_frame_crossfade > 0 and musetalk_prev_tail_frames:
                        n = min(
                            musetalk_frame_crossfade,
                            len(musetalk_prev_tail_frames),
                            len(frames),
                        )
                        # Blend boundary frames to hide diffusion chunk-edge jitter.
                        for i in range(n):
                            alpha_new = float(i + 1) / float(n + 1)
                            alpha_old = 1.0 - alpha_new
                            frames[i] = cv2.addWeighted(
                                musetalk_prev_tail_frames[-n + i],
                                alpha_old,
                                frames[i],
                                alpha_new,
                                0.0,
                            )
                        musetalk_prev_tail_frames = [
                            f.copy() for f in frames[-musetalk_frame_crossfade:]
                        ]
                    elif is_musetalk and musetalk_frame_crossfade > 0:
                        musetalk_prev_tail_frames = [
                            f.copy() for f in frames[-musetalk_frame_crossfade:]
                        ]
                    last_frame = frames[-1].copy()
            frame_payloads: list[str] = []
            frame_blobs: list[bytes] = []
            for frame in frames:
                if mobile_max_frame_edge > 0:
                    h, w = frame.shape[:2]
                    max_dim = max(h, w)
                    if max_dim > mobile_max_frame_edge:
                        scale = float(mobile_max_frame_edge) / float(max_dim)
                        new_w = max(1, int(round(w * scale)))
                        new_h = max(1, int(round(h * scale)))
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                )
                if ok:
                    encoded = bytes(buf)
                    if binary_transport:
                        frame_blobs.append(encoded)
                    else:
                        frame_payloads.append(base64.b64encode(encoded).decode("ascii"))

            wav_bytes = _audio_to_wav_bytes(sub_audio, engine.sample_rate)
            duration_sec = float(len(sub_audio) / engine.sample_rate) if sub_audio.size else 0.0
            if binary_transport:
                metadata = {
                    "event": "chunk",
                    "chunk_index": emit_index,
                    "sample_rate": engine.sample_rate,
                    "fps": fps,
                    "duration_sec": round(duration_sec, 6),
                    "audio_bytes_len": len(wav_bytes),
                    "frame_lengths": [len(blob) for blob in frame_blobs],
                }
                payload = wav_bytes + b"".join(frame_blobs)
                yield _pack_binary_stream_packet(metadata, payload)
            else:
                payload = base64.b64encode(wav_bytes).decode("ascii")
                yield json.dumps(
                    {
                        "chunk_index": emit_index,
                        "audio_base64": payload,
                        "sample_rate": engine.sample_rate,
                        "fps": fps,
                        "frames_base64": frame_payloads,
                        "duration_sec": round(duration_sec, 6),
                    }
                ) + "\n"
            emit_index += 1

        try:
            while True:
                try:
                    kind, idx, audio, audio_16k = result_queue.get(timeout=0.2)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue
                if kind == "error":
                    detail = "Audio worker failed."
                    if audio is not None and audio.size > 0:
                        detail = str(audio[0])
                    if binary_transport:
                        yield _pack_binary_stream_packet({"event": "error", "detail": detail})
                    else:
                        yield json.dumps({"event": "error", "detail": detail}) + "\n"
                    break
                if kind == "done":
                    if is_musetalk and pending_audio.size and pending_audio_16k.size:
                        yield from _emit_one(pending_audio, pending_audio_16k)
                    break

                if audio is None or audio_16k is None:
                    continue

                if is_musetalk and window_src_samples > 0 and window_tgt_samples > 0:
                    pending_audio = np.concatenate([pending_audio, audio.astype(np.float32, copy=False)])
                    pending_audio_16k = np.concatenate(
                        [pending_audio_16k, audio_16k.astype(np.float32, copy=False)]
                    )
                    while (
                        pending_audio.size >= window_src_samples
                        and pending_audio_16k.size >= window_tgt_samples
                    ):
                        lookahead_16k = None
                        if lookahead_tgt_samples > 0:
                            if (
                                pending_audio.size < (window_src_samples + lookahead_src_samples)
                                or pending_audio_16k.size < (window_tgt_samples + lookahead_tgt_samples)
                            ):
                                break
                            lookahead_16k = pending_audio_16k[
                                window_tgt_samples : window_tgt_samples + lookahead_tgt_samples
                            ]
                        sub_audio = pending_audio[:window_src_samples]
                        sub_audio_16k = pending_audio_16k[:window_tgt_samples]
                        pending_audio = pending_audio[window_src_samples:]
                        pending_audio_16k = pending_audio_16k[window_tgt_samples:]
                        yield from _emit_one(sub_audio, sub_audio_16k, lookahead_16k=lookahead_16k)
                else:
                    yield from _emit_one(audio, audio_16k)
        except Exception:
            logger.exception("Avatar stream failed")
            if binary_transport:
                yield _pack_binary_stream_packet({"event": "error", "detail": traceback.format_exc()})
            else:
                yield json.dumps({"event": "error", "detail": traceback.format_exc()}) + "\n"
        finally:
            stop_event.set()
            with contextlib.suppress(Exception):
                close_text_iter = getattr(text_iter, "close", None)
                if callable(close_text_iter):
                    close_text_iter()
            with contextlib.suppress(Exception):
                sentence_reader_thread.join(timeout=0.5)
            with contextlib.suppress(Exception):
                audio_worker_thread.join(timeout=0.5)
            _unregister_stream_stop_event(stop_event)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if binary_transport:
            yield _pack_binary_stream_packet(
                {"event": "done", "inference_ms": round(elapsed_ms, 2)}
            )
        else:
            yield json.dumps({"event": "done", "inference_ms": round(elapsed_ms, 2)}) + "\n"

    return StreamingResponse(
        _generator(),
        media_type=(
            PIXELHOLO_BINARY_STREAM_MEDIA_TYPE
            if binary_transport
            else "application/x-ndjson"
        ),
    )


def _stream_voice_from_text_iter(
    req: GenerateRequest,
    text_iter: Iterator[str],
    binary_transport: bool = False,
    mobile_profile: bool = False,
) -> StreamingResponse:
    start_time = time.perf_counter()
    profile_type = req.profile_type or PROFILE_TYPE_VOICE
    model_path = _resolve_model_path(req.model_path, req.speaker, profile_type)
    config_path = _resolve_config_path(model_path, req.config_path, req.speaker, profile_type)
    ref_wav_path = _resolve_ref_wav(req.ref_wav_path, req.speaker, profile_type, model_path)
    profile_dir = resolve_dataset_root(req.speaker or "", profile_type) if req.speaker else None
    params = _resolve_inference_params(model_path, config_path, req, req.speaker, profile_type)
    phonemizer_lang = _resolve_phonemizer_lang(model_path, req, req.speaker, profile_type)
    lexicon_path = _resolve_lexicon_path(model_path, req, req.speaker, profile_type)
    lexicon = _load_lexicon(lexicon_path)

    engine = _get_engine(model_path, config_path)
    profile = _load_profile_defaults(model_path)
    max_chars = (
        req.max_chunk_chars
        if req.max_chunk_chars is not None
        else profile.get("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS)
    )
    max_words = (
        req.max_chunk_words
        if req.max_chunk_words is not None
        else profile.get("max_chunk_words", DEFAULT_MAX_CHUNK_WORDS)
    )
    pause_ms = (
        req.pause_ms
        if req.pause_ms is not None
        else profile.get("pause_ms", DEFAULT_PAUSE_MS)
    )
    pad_text = (
        req.pad_text
        if req.pad_text is not None
        else profile.get("pad_text", True)
    )
    pad_text_token = (
        req.pad_text_token
        if req.pad_text_token is not None
        else profile.get("pad_text_token", "...")
    )
    smart_trim_db = (
        req.smart_trim_db
        if req.smart_trim_db is not None
        else profile.get("stream_smart_trim_db", 0.0)
    )
    smart_trim_pad_ms = (
        req.smart_trim_pad_ms
        if req.smart_trim_pad_ms is not None
        else profile.get("stream_smart_trim_pad_ms", 30.0)
    )
    pause = np.zeros(int(engine.sample_rate * (pause_ms / 1000.0)), dtype=np.float32)
    comma_pause_ms = max(50.0, float(pause_ms) * 0.35)
    comma_pause = np.zeros(int(engine.sample_rate * (comma_pause_ms / 1000.0)), dtype=np.float32)
    seed = req.seed if req.seed is not None else profile.get("seed")
    if seed is None:
        seed = 1234

    def _generator() -> Iterator[str | bytes]:
        stop_event = threading.Event()
        _register_stream_stop_event(stop_event)
        idx = 0
        tts_chunk_index = 0
        stitcher = AudioStitcher(sample_rate=engine.sample_rate, fade_len_ms=15.0)
        try:
            if seed is not None:
                _seed_everything(seed)
            for sentence in text_iter:
                if stop_event.is_set():
                    break
                clean_sentence = clean_text_for_tts(sentence)
                sentence_ref = ref_wav_path
                sentence_chunks = _plan_stream_tts_chunks(
                    clean_sentence,
                    max_chars=max_chars,
                    max_words=max_words,
                    is_first_global_chunk=(tts_chunk_index == 0),
                )
                for chunk in sentence_chunks:
                    if stop_event.is_set():
                        break
                    if pad_text:
                        chunk = f"{pad_text_token} {chunk} {pad_text_token}"
                    chunk_seed = seed
                    audio = engine.generate(
                        chunk,
                        ref_wav_path=sentence_ref,
                        alpha=params["alpha"],
                        beta=params["beta"],
                        diffusion_steps=params["diffusion_steps"],
                        embedding_scale=params["embedding_scale"],
                        f0_scale=params["f0_scale"],
                        phonemizer_lang=phonemizer_lang,
                        lexicon=lexicon,
                        seed=chunk_seed,
                    )
                    tts_chunk_index += 1
                    if smart_trim_db and smart_trim_db > 0:
                        audio = _smart_vad_trim(
                            audio,
                            engine.sample_rate,
                            top_db=float(smart_trim_db),
                            pad_ms=float(smart_trim_pad_ms),
                        )
                    audio = _remove_dc(audio.astype(np.float32, copy=False))
                    pitch_shift = (
                        req.pitch_shift
                        if req.pitch_shift is not None
                        else profile.get("pitch_shift", 0.0)
                    )
                    if pitch_shift:
                        audio = _apply_pitch_shift(audio, engine.sample_rate, pitch_shift)
                    de_esser_cutoff = (
                        req.de_esser_cutoff
                        if req.de_esser_cutoff is not None
                        else profile.get("de_esser_cutoff", 0.0)
                    )
                    de_esser_order = (
                        req.de_esser_order
                        if req.de_esser_order is not None
                        else profile.get("de_esser_order", 2)
                    )
                    if de_esser_cutoff:
                        audio = _apply_de_esser(
                            audio, engine.sample_rate, de_esser_cutoff, int(de_esser_order)
                        )
                    audio = _soft_clip(audio)
                    audio = _fade_edges(audio, engine.sample_rate, fade_ms=5.0)
                    suffix = chunk.rstrip()
                    if suffix.endswith((".", "!", "?")) and pause.size:
                        audio = np.concatenate([audio, pause])
                    elif suffix.endswith((",", ";", ":")) and comma_pause.size:
                        audio = np.concatenate([audio, comma_pause])
                    stitched = stitcher.process(audio)
                    if stitched.size:
                        wav_bytes = _audio_to_wav_bytes(stitched, engine.sample_rate)
                        duration_sec = (
                            float(len(stitched) / engine.sample_rate) if stitched.size else 0.0
                        )
                        if binary_transport:
                            yield _pack_binary_stream_packet(
                                {
                                    "event": "chunk",
                                    "chunk_index": idx,
                                    "sample_rate": engine.sample_rate,
                                    "duration_sec": round(duration_sec, 6),
                                    "audio_bytes_len": len(wav_bytes),
                                    "frame_lengths": [],
                                },
                                wav_bytes,
                            )
                        else:
                            payload = base64.b64encode(wav_bytes).decode("ascii")
                            yield json.dumps(
                                {
                                    "chunk_index": idx,
                                    "audio_base64": payload,
                                    "sample_rate": engine.sample_rate,
                                    "duration_sec": round(duration_sec, 6),
                                }
                            ) + "\n"
                        idx += 1
            tail = stitcher.flush()
            if tail.size:
                wav_bytes = _audio_to_wav_bytes(tail, engine.sample_rate)
                duration_sec = float(len(tail) / engine.sample_rate) if tail.size else 0.0
                if binary_transport:
                    yield _pack_binary_stream_packet(
                        {
                            "event": "chunk",
                            "chunk_index": idx,
                            "sample_rate": engine.sample_rate,
                            "duration_sec": round(duration_sec, 6),
                            "audio_bytes_len": len(wav_bytes),
                            "frame_lengths": [],
                        },
                        wav_bytes,
                    )
                else:
                    payload = base64.b64encode(wav_bytes).decode("ascii")
                    yield json.dumps(
                        {
                            "chunk_index": idx,
                            "audio_base64": payload,
                            "sample_rate": engine.sample_rate,
                            "duration_sec": round(duration_sec, 6),
                        }
                    ) + "\n"
                idx += 1
        except Exception:
            logger.exception("Voice stream failed")
            if binary_transport:
                yield _pack_binary_stream_packet({"event": "error", "detail": traceback.format_exc()})
            else:
                yield json.dumps({"event": "error", "detail": traceback.format_exc()}) + "\n"
            return
        finally:
            stop_event.set()
            with contextlib.suppress(Exception):
                close_text_iter = getattr(text_iter, "close", None)
                if callable(close_text_iter):
                    close_text_iter()
            _unregister_stream_stop_event(stop_event)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if binary_transport:
            yield _pack_binary_stream_packet(
                {"event": "done", "inference_ms": round(elapsed_ms, 2)}
            )
        else:
            yield json.dumps({"event": "done", "inference_ms": round(elapsed_ms, 2)}) + "\n"

    return StreamingResponse(
        _generator(),
        media_type=(
            PIXELHOLO_BINARY_STREAM_MEDIA_TYPE
            if binary_transport
            else "application/x-ndjson"
        ),
    )


class GenerateRequest(BaseModel):
    text: str
    lipsync_backend: str | None = None
    model_path: str | None = None
    config_path: str | None = None
    ref_wav_path: str | None = None
    speaker: str | None = None
    profile_type: str | None = None
    phonemizer_lang: str | None = None
    lexicon_path: str | None = None
    alpha: float | None = None
    beta: float | None = None
    diffusion_steps: int | None = None
    embedding_scale: float | None = None
    f0_scale: float | None = None
    pitch_shift: float | None = None
    de_esser_cutoff: float | None = None
    de_esser_order: int | None = None
    pad_text: bool | None = None
    pad_text_token: str | None = None
    smart_trim_db: float | None = None
    smart_trim_pad_ms: float | None = None
    seed: int | None = None
    max_chunk_chars: int | None = None
    max_chunk_words: int | None = None
    pause_ms: int | None = None
    crossfade_ms: float | None = None
    avatar_profile: str | None = None
    avatar_fps: float | None = None
    musetalk_batch_size: int | None = None
    musetalk_infer_fps: float | None = None
    musetalk_stream_window_sec: float | None = None
    musetalk_lookahead_sec: float | None = None
    musetalk_jpeg_quality: int | None = None
    musetalk_face_scale: float | None = None
    musetalk_audio_history_sec: float | None = None
    musetalk_max_chunk_chars: int | None = None
    musetalk_first_chunk_chars: int | None = None
    return_base64: bool = False


class PreprocessRequest(BaseModel):
    profile: str
    filename: str | None = None
    audio_filename: str | None = None
    profile_type: str | None = None
    bake_avatar: bool = True
    avatar_fps: float | None = DEFAULT_AVATAR_FPS
    avatar_start_sec: float | None = None
    avatar_loop_sec: float | None = None
    avatar_loop_fade_sec: float | None = None
    avatar_resize_factor: int | None = None
    avatar_pads: str | None = DEFAULT_AVATAR_PADS
    avatar_batch_size: int | None = None
    avatar_nosmooth: bool = DEFAULT_AVATAR_NOSMOOTH
    avatar_blur_background: bool | None = None
    avatar_blur_kernel: int | None = None
    avatar_device: str | None = None


class TrainRequest(BaseModel):
    profile: str
    profile_type: str | None = None
    batch_size: int | None = None
    epochs: int | None = None
    max_len: int | None = None
    auto_select_epoch: bool = True
    auto_tune_profile: bool = True
    auto_build_lexicon: bool = True
    select_thorough: bool = True
    early_stop: bool = True


class WarmupRequest(BaseModel):
    profile: str
    profile_type: str | None = None
    lipsync_backend: str | None = None
    force: bool = False


class ProfileRenameRequest(BaseModel):
    new_name: str
    profile_type: str | None = None


class StyleTTS2RepoEngine:
    def __init__(self, model_path: Path, config_path: Path, device: str | None = None):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.style_cache: dict[str, torch.Tensor] = {}
        self._load_backend()

    def _load_backend(self) -> None:
        if not STYLE_TTS2_DIR.exists():
            raise RuntimeError("StyleTTS2 repo not found at lib/StyleTTS2.")

        try:
            import librosa
            import phonemizer
            import torchaudio
        except Exception as exc:
            raise RuntimeError(
                "Missing inference deps. Install `librosa`, `phonemizer`, and `torchaudio`."
            ) from exc

        try:
            from nltk.tokenize import word_tokenize
        except Exception:
            word_tokenize = None

        from models import build_model, load_ASR_models, load_F0_models, load_checkpoint
        from utils import length_to_mask, recursive_munch
        from text_utils import TextCleaner, symbols as tts_symbols
        from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
        from Utils.PLBERT.util import load_plbert

        self._librosa = librosa
        self._phonemizer_backend = phonemizer.backend.EspeakBackend
        self._phonemizers: dict[str, object] = {}
        self._word_tokenize = word_tokenize
        self._length_to_mask = length_to_mask
        self._text_symbols = set(tts_symbols)

        with self.config_path.open("r") as handle:
            self.config = yaml.safe_load(handle)

        dist_cfg = (
            self.config.get("model_params", {})
            .get("diffusion", {})
            .get("dist", {})
        )
        sigma_data = dist_cfg.get("sigma_data")
        try:
            sigma_value = float(sigma_data)
            sigma_ok = math.isfinite(sigma_value)
        except (TypeError, ValueError):
            sigma_ok = False

        if not sigma_ok:
            dist_cfg["sigma_data"] = 0.2
            dist_cfg["estimate_sigma_data"] = False
            updated = False
            try:
                with self.config_path.open("w") as handle:
                    yaml.dump(self.config, handle, default_flow_style=False)
                updated = True
            except Exception:
                updated = False
            key = str(self.config_path.resolve())
            if key not in _SIGMA_WARNED_PATHS:
                _SIGMA_WARNED_PATHS.add(key)
                suffix = " (config updated)" if updated else ""
                print(
                    f"Config sigma_data invalid ({sigma_data}); forcing to 0.2 for inference{suffix}."
                )

        preprocess_params = self.config.get("preprocess_params", {})
        self.sample_rate = int(preprocess_params.get("sr", 24000))
        spect_params = preprocess_params.get("spect_params", {})
        n_fft = spect_params.get("n_fft", 2048)
        win_length = spect_params.get("win_length", 1200)
        hop_length = spect_params.get("hop_length", 300)
        n_mels = int(self.config.get("model_params", {}).get("n_mels", 80))

        self._to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        model_params = recursive_munch(self.config["model_params"])
        self.model_params = model_params

        asr_config = _resolve_path(STYLE_TTS2_DIR, self.config.get("ASR_config", ""))
        asr_path = _resolve_path(STYLE_TTS2_DIR, self.config.get("ASR_path", ""))
        f0_path = _resolve_path(STYLE_TTS2_DIR, self.config.get("F0_path", ""))
        plbert_dir = _resolve_path(STYLE_TTS2_DIR, self.config.get("PLBERT_dir", ""))

        if not (asr_config and asr_path and f0_path and plbert_dir):
            raise RuntimeError("Missing ASR/F0/PLBERT paths in config.")

        quiet = os.getenv("STYLE_TTS2_QUIET", "1") == "1"
        suppress = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
        with suppress:
            self._text_cleaner = TextCleaner()
            text_aligner = load_ASR_models(str(asr_path), str(asr_config))
            pitch_extractor = load_F0_models(str(f0_path))
            plbert = load_plbert(str(plbert_dir))

            model = build_model(model_params, text_aligner, pitch_extractor, plbert)
            _ = [model[key].to(self.device) for key in model]
            model, _, _, _ = load_checkpoint(model, None, str(self.model_path), load_only_params=True)
            _ = [model[key].eval() for key in model]

        self.model = model
        self.sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

    def _tokenize(self, text: str) -> list[str]:
        if self._word_tokenize is None:
            return text.split()
        try:
            return self._word_tokenize(text)
        except LookupError:
            return text.split()

    def _get_phonemizer(self, lang: str | None):
        lang = lang or DEFAULT_LANG
        if lang not in self._phonemizers:
            try:
                self._phonemizers[lang] = self._phonemizer_backend(
                    language=lang,
                    preserve_punctuation=True,
                    with_stress=True,
                )
            except RuntimeError:
                fallback = "en-us" if lang != "en-us" else None
                if fallback:
                    self._phonemizers[lang] = self._phonemizer_backend(
                        language=fallback,
                        preserve_punctuation=True,
                        with_stress=True,
                    )
                else:
                    raise
        return self._phonemizers[lang]

    def _phonemize(self, text: str, lang: str | None, lexicon: dict[str, str] | None) -> str:
        phonemizer = self._get_phonemizer(lang)
        parts: list[str] = []
        for token in _WORD_RE.findall(text):
            if _WORD_ONLY_RE.match(token):
                key = token.lower()
                if lexicon and key in lexicon:
                    parts.append(lexicon[key])
                else:
                    parts.append(phonemizer.phonemize([token])[0].strip())
            else:
                parts.append(token)
        return "".join(parts)

    def _preprocess(self, wave: np.ndarray) -> torch.Tensor:
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self._to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEAN) / STD
        return mel_tensor

    def _compute_style(self, wav_path: Path) -> torch.Tensor:
        cache_key = str(wav_path)
        if cache_key in self.style_cache:
            return self.style_cache[cache_key]

        wave, sr = self._librosa.load(str(wav_path), sr=self.sample_rate)
        audio, _index = self._librosa.effects.trim(wave, top_db=30)
        if sr != self.sample_rate:
            audio = self._librosa.resample(audio, sr, self.sample_rate)
        mel_tensor = self._preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        style = torch.cat([ref_s, ref_p], dim=1)
        self.style_cache[cache_key] = style
        return style

    def generate(
        self,
        text: str,
        ref_wav_path: Path,
        alpha: float,
        beta: float,
        diffusion_steps: int,
        embedding_scale: float,
        f0_scale: float,
        phonemizer_lang: str | None = None,
        lexicon: dict[str, str] | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("Text is empty.")
        if seed is not None:
            _seed_everything(seed)

        phonemes = self._phonemize(text, phonemizer_lang, lexicon)
        tokens = " ".join(self._tokenize(phonemes))
        if self._text_symbols:
            tokens = "".join(ch for ch in tokens if ch in self._text_symbols)
        tokens = re.sub(r"\s+", " ", tokens).strip()
        if not tokens:
            raise ValueError("Text normalization produced an empty token stream.")
        token_ids = self._text_cleaner(tokens)
        token_ids.insert(0, 0)

        tokens_tensor = torch.LongTensor(token_ids).to(self.device).unsqueeze(0)

        ref_s = self._compute_style(ref_wav_path)
        style_dim = ref_s.shape[-1] // 2

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(self.device)
            text_mask = self._length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens_tensor, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens_tensor, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            noise = torch.randn((1, style_dim * 2)).unsqueeze(1).to(self.device)
            s_pred = self.sampler(
                noise=noise,
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, style_dim:]
            ref = s_pred[:, :style_dim]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :style_dim]
            s = beta * s + (1 - beta) * ref_s[:, style_dim:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            duration = torch.nan_to_num(duration, nan=1.0, posinf=1.0, neginf=1.0)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            if pred_dur.dim() == 0:
                pred_dur = pred_dur.unsqueeze(0)

            total_frames = int(torch.clamp(pred_dur.sum(), min=1).item())
            pred_aln_trg = torch.zeros(
                (input_lengths.item(), total_frames),
                device=self.device,
            )
            c_frame = 0
            for idx, dur in enumerate(pred_dur):
                dur_i = max(1, int(dur.item()))
                pred_aln_trg[idx, c_frame : c_frame + dur_i] = 1
                c_frame += dur_i

            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            f0_pred, n_pred = self.model.predictor.F0Ntrain(en, s)
            if f0_scale != 1.0:
                f0_pred = f0_pred * f0_scale
            f0_pred = torch.nan_to_num(f0_pred, nan=0.0, posinf=0.0, neginf=0.0)

            asr = (t_en @ pred_aln_trg.unsqueeze(0))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, f0_pred, n_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50]




def _normalize_lipsync_backend(backend: str | None, default: str = "musetalk") -> str:
    resolved_input = (
        backend.strip().lower()
        if isinstance(backend, str) and backend.strip()
        else default.strip().lower()
    )
    aliases = {
        "wav2lip": "wav2lip",
        "w2l": "wav2lip",
        "musetalk": "musetalk",
        "mt": "musetalk",
    }
    resolved = aliases.get(resolved_input)
    if not resolved:
        raise ValueError(
            f"Unsupported LIPSYNC_BACKEND={resolved_input}. Use one of: wav2lip, musetalk."
        )
    return resolved


def _runtime_lipsync_backend() -> str:
    # Default to MuseTalk unless the process is explicitly started with wav2lip.
    return _normalize_lipsync_backend(os.getenv("LIPSYNC_BACKEND", "musetalk"), default="musetalk")


def _resolve_lipsync_backend(requested_backend: str | None = None) -> str:
    runtime_backend = _runtime_lipsync_backend()
    if isinstance(requested_backend, str) and requested_backend.strip():
        req = _normalize_lipsync_backend(requested_backend, default=runtime_backend)
        if req != runtime_backend:
            logger.info(
                "component=lipsync op=backend_select status=locked runtime=%s requested=%s",
                runtime_backend,
                req,
            )
    return runtime_backend


def _build_lipsync_engine(backend: str) -> _LipSyncEngineProtocol:
    if backend == "musetalk":
        from src.musetalk_bridge import MuseTalkBridge

        return MuseTalkBridge()
    from src.lipsync_bridge import LipSyncBridge

    return LipSyncBridge()


def _dispose_lipsync_engine(engine: object) -> None:
    for method_name in ("close", "shutdown"):
        method = getattr(engine, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                logger.debug(
                    "component=lipsync op=engine_dispose status=close_error method=%s",
                    method_name,
                    exc_info=True,
                )
    del engine
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.debug(
                "component=lipsync op=engine_dispose status=cuda_empty_cache_error",
                exc_info=True,
            )


def _evict_lipsync_engines(keep_backend: str) -> None:
    for backend_name in list(_lipsync_engines.keys()):
        if backend_name == keep_backend:
            continue
        stale_engine = _lipsync_engines.pop(backend_name, None)
        if stale_engine is None:
            continue
        _dispose_lipsync_engine(stale_engine)
        logger.info(
            "component=lipsync op=engine_evict status=ok backend=%s keep=%s",
            backend_name,
            keep_backend,
        )
        warmed = globals().get("_warmed_lipsync_profiles")
        if isinstance(warmed, set):
            stale_keys = [k for k in warmed if isinstance(k, tuple) and len(k) == 3 and k[2] == backend_name]
            for key in stale_keys:
                warmed.discard(key)


def _get_lipsync_engine(requested_backend: str | None = None) -> _LipSyncEngineProtocol:
    with _lipsync_lock:
        desired_backend = _resolve_lipsync_backend(requested_backend)
        strict_isolation = os.getenv("LIPSYNC_STRICT_ISOLATION", "1") == "1"
        if strict_isolation:
            _evict_lipsync_engines(keep_backend=desired_backend)
        if desired_backend not in _lipsync_engines:
            try:
                _lipsync_engines[desired_backend] = _build_lipsync_engine(desired_backend)
                logger.info(
                    "component=lipsync op=engine_init status=ok backend=%s",
                    desired_backend,
                )
            except Exception:
                logger.exception(
                    "component=lipsync op=engine_init status=error backend=%s",
                    desired_backend,
                )
                raise
        return _lipsync_engines[desired_backend]


def _get_engine(model_path: Path, config_path: Path) -> StyleTTS2RepoEngine:
    key = (str(model_path), str(config_path))
    with _engine_lock:
        if key not in _engines:
            _engines[key] = StyleTTS2RepoEngine(model_path=model_path, config_path=config_path)
        return _engines[key]


def _warmup_engine(model_path: Path, config_path: Path) -> None:
    if os.getenv("STYLE_TTS2_DISABLE_WARMUP") == "1":
        return
    profile = _load_profile_defaults(model_path)
    ref_path = os.getenv("STYLE_TTS2_REF_WAV") or profile.get("ref_wav_path")
    if not ref_path:
        print("Warmup skipped: no reference wav available.")
        return
    ref_wav = Path(ref_path).expanduser()
    if not ref_wav.exists():
        print(f"Warmup skipped: ref wav not found: {ref_wav}")
        return

    try:
        engine = _get_engine(model_path, config_path)
        params = {
            "alpha": profile.get("alpha", DEFAULT_ALPHA),
            "beta": profile.get("beta", DEFAULT_BETA),
            "diffusion_steps": profile.get("diffusion_steps", DEFAULT_DIFFUSION_STEPS),
            "embedding_scale": profile.get("embedding_scale", DEFAULT_EMBEDDING_SCALE),
            "f0_scale": profile.get("f0_scale", DEFAULT_F0_SCALE),
        }
        text = os.getenv("STYLE_TTS2_WARMUP_TEXT", "warmup")
        engine.generate(
            text=text,
            ref_wav_path=ref_wav,
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            diffusion_steps=int(params["diffusion_steps"]),
            embedding_scale=float(params["embedding_scale"]),
            f0_scale=float(params["f0_scale"]),
            phonemizer_lang=profile.get("phonemizer_lang"),
            lexicon=None,
            seed=1234,
        )
        print("Warmup completed.")
    except Exception as exc:
        print(f"Warmup failed: {exc}")


_warmed_tts_profiles: set[tuple[str, str]] = set()
_warmed_lipsync_profiles: set[tuple[str, str, str]] = set()


def _warmup_lipsync(
    profile: str,
    profile_type: str,
    requested_backend: str | None = None,
) -> str:
    backend = _resolve_lipsync_backend(requested_backend)
    try:
        lipsync = _get_lipsync_engine(backend)
        with _lipsync_lock:
            lipsync.load_profile(profile, profile_type)
            dummy = np.zeros(int(0.4 * 16000), dtype=np.float32)
            lipsync.sync_chunk(dummy, fps=lipsync.fps)
        print(f"Lipsync warmup completed ({backend}).")
        return backend
    except Exception as exc:
        print(f"Lipsync warmup failed ({backend}): {exc}")
        raise


def _warmup_profile(
    profile: str,
    profile_type: str,
    requested_backend: str | None = None,
    force: bool = False,
) -> str | None:
    with _warmup_profile_lock:
        tts_key = (profile, profile_type)
        if force or tts_key not in _warmed_tts_profiles:
            try:
                model_path = _resolve_model_path(None, profile, profile_type)
                config_path = _resolve_config_path(model_path, None, profile, profile_type)
                _warmup_engine(model_path, config_path)
                _warmed_tts_profiles.add(tts_key)
            except HTTPException:
                logger.info(
                    "component=backend op=warmup_profile skip=tts profile=%s profile_type=%s reason=model_unset",
                    profile,
                    profile_type,
                )
            except Exception:
                logger.exception(
                    "component=backend op=warmup_profile skip=tts profile=%s profile_type=%s reason=error",
                    profile,
                    profile_type,
                )

        warmed_backend: str | None = None
        if profile_type == PROFILE_TYPE_AVATAR:
            backend = _resolve_lipsync_backend(requested_backend)
            lipsync_key = (profile, profile_type, backend)
            if force or lipsync_key not in _warmed_lipsync_profiles:
                warmed_backend = _warmup_lipsync(profile, profile_type, backend)
                _warmed_lipsync_profiles.add(lipsync_key)
            else:
                warmed_backend = backend
        return warmed_backend


@app.on_event("startup")
def _startup() -> None:
    try:
        warmup_text_normalizer()
        print("Text normalizer warmup completed.")
    except Exception as exc:
        logger.exception("component=backend op=text_normalizer_warmup status=error")
        print(f"Text normalizer warmup failed: {exc}")
    default_model = os.getenv("STYLE_TTS2_MODEL")
    if default_model:
        model_path = Path(default_model).expanduser()
        config_path = _resolve_config_path(model_path, os.getenv("STYLE_TTS2_CONFIG"))
        _get_engine(model_path, config_path)
        _warmup_engine(model_path, config_path)


@app.get("/lipsync_backend")
def lipsync_backend_status():
    return {
        "backend": _runtime_lipsync_backend(),
    }


@app.post("/warmup")
def warmup(req: WarmupRequest):
    profile = req.profile.strip()
    if not profile:
        raise HTTPException(status_code=400, detail="profile is required")
    profile_type = req.profile_type or PROFILE_TYPE_VOICE
    warmed_backend = _warmup_profile(
        profile,
        profile_type,
        requested_backend=req.lipsync_backend,
        force=bool(req.force),
    )
    return {
        "status": "ok",
        "profile": profile,
        "profile_type": profile_type,
        "lipsync_backend": warmed_backend,
    }


@app.post("/interrupt")
def interrupt_active_generation():
    interrupted = _interrupt_active_streams()
    return {"status": "ok", "interrupted_streams": interrupted}


@app.get("/profiles")
def profiles(profile_type: str | None = None):
    return {"profiles": _list_profiles(profile_type)}


@app.delete("/profiles/{profile_name}")
def delete_profile(profile_name: str, profile_type: str | None = None):
    profile = _sanitize_profile_name(profile_name, "profile_name")
    ptype = _normalize_profile_type(profile_type)
    with _active_stream_lock:
        if _active_stream_stop_events:
            raise HTTPException(
                status_code=409,
                detail="Cannot delete profile while generation is running. Stop generation first.",
            )
    paths = _profile_paths(profile, ptype)
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise HTTPException(status_code=404, detail="Profile not found")

    removed: list[str] = []
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        removed.append(str(path))

    _drop_warmup_cache_for_profile(profile, ptype)
    return {
        "status": "ok",
        "profile": profile,
        "profile_type": ptype,
        "removed_paths": removed,
    }


@app.patch("/profiles/{profile_name}")
def rename_profile(profile_name: str, req: ProfileRenameRequest):
    old_name = _sanitize_profile_name(profile_name, "profile_name")
    new_name = _sanitize_profile_name(req.new_name, "new_name")
    ptype = _normalize_profile_type(req.profile_type)
    if old_name == new_name:
        return {
            "status": "ok",
            "profile_type": ptype,
            "old_name": old_name,
            "new_name": new_name,
            "moved_paths": [],
        }
    with _active_stream_lock:
        if _active_stream_stop_events:
            raise HTTPException(
                status_code=409,
                detail="Cannot rename profile while generation is running. Stop generation first.",
            )

    source_paths = _profile_paths(old_name, ptype)
    existing_sources = [path for path in source_paths if path.exists()]
    if not existing_sources:
        raise HTTPException(status_code=404, detail="Profile not found")

    destination_pairs: list[tuple[Path, Path]] = []
    for source in source_paths:
        dest = source.parent / new_name
        if source.exists():
            destination_pairs.append((source, dest))
        if dest.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Target profile already exists at {dest}",
            )

    moved_paths: list[dict[str, str]] = []
    path_replacements: dict[str, str] = {}
    for source, dest in destination_pairs:
        source_resolved = str(source.resolve())
        shutil.move(str(source), str(dest))
        dest_resolved = str(dest.resolve())
        moved_paths.append({"from": source_resolved, "to": dest_resolved})
        path_replacements[source_resolved] = dest_resolved

    for _source, dest in destination_pairs:
        _rewrite_profile_refs_in_dir(
            dest,
            old_name=old_name,
            new_name=new_name,
            path_replacements=path_replacements,
        )

    _drop_warmup_cache_for_profile(old_name, ptype)
    _drop_warmup_cache_for_profile(new_name, ptype)
    return {
        "status": "ok",
        "profile_type": ptype,
        "old_name": old_name,
        "new_name": new_name,
        "moved_paths": moved_paths,
    }


@app.post("/upload")
def upload(
    profile: str = Form(...),
    file: UploadFile = File(...),
    profile_type: str = Form(PROFILE_TYPE_VOICE),
):
    if not profile:
        raise HTTPException(status_code=400, detail="profile is required")
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")
    dest_dir = raw_videos_dir(profile, profile_type)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename).name
    dest_path = dest_dir / filename
    with dest_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"saved_path": str(dest_path), "filename": filename}


@app.post("/upload_audio")
def upload_audio(
    profile: str = Form(...),
    file: UploadFile = File(...),
    profile_type: str = Form(PROFILE_TYPE_VOICE),
):
    if not profile:
        raise HTTPException(status_code=400, detail="profile is required")
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")
    dest_dir = raw_audio_dir(profile, profile_type)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename).name
    dest_path = dest_dir / filename
    with dest_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"saved_path": str(dest_path), "filename": filename}


@app.post("/preprocess")
def preprocess(req: PreprocessRequest):
    profile = req.profile.strip()
    if not profile:
        raise HTTPException(status_code=400, detail="profile is required")
    profile_type = req.profile_type or PROFILE_TYPE_VOICE
    raw_dir = raw_videos_dir(profile, profile_type)
    audio_dir = raw_audio_dir(profile, profile_type)
    filename = req.filename
    if not filename:
        if not raw_dir.exists():
            raise HTTPException(status_code=400, detail="No raw uploads found for profile.")
        candidates = sorted(raw_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise HTTPException(status_code=400, detail="No raw uploads found for profile.")
        filename = candidates[0].name
        print(f"Auto-selected raw upload: {filename}", flush=True)
    video_path = raw_dir / filename
    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"file not found: {video_path}")
    audio_path: Path | None = None
    if req.audio_filename:
        audio_path = audio_dir / req.audio_filename
        if not audio_path.exists():
            raise HTTPException(status_code=400, detail=f"audio file not found: {audio_path}")
    elif profile_type == PROFILE_TYPE_AVATAR:
        if not audio_dir.exists():
            raise HTTPException(status_code=400, detail="audio_filename is required for avatar preprocessing")
        candidates = sorted(audio_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise HTTPException(status_code=400, detail="audio_filename is required for avatar preprocessing")
        audio_path = candidates[0]
        print(f"Auto-selected audio upload: {audio_path.name}", flush=True)
    preprocess_script = "preprocess.py"
    command = []
    if profile_type == PROFILE_TYPE_AVATAR:
        preprocess_script = "preprocess_video.py"
        command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "src" / preprocess_script),
            "--video",
            str(video_path),
            *(["--audio", str(audio_path)] if audio_path else []),
            "--name",
            profile,
        ]
        if not req.bake_avatar:
            command.append("--no_bake_avatar")
        if req.avatar_fps is not None:
            command += ["--avatar_fps", str(req.avatar_fps)]
        if req.avatar_start_sec is not None:
            command += ["--avatar_start_sec", str(req.avatar_start_sec)]
        if req.avatar_loop_sec is not None:
            command += ["--avatar_loop_sec", str(req.avatar_loop_sec)]
        if req.avatar_loop_fade_sec is not None:
            command += ["--avatar_loop_fade_sec", str(req.avatar_loop_fade_sec)]
        if req.avatar_resize_factor is not None:
            command += ["--avatar_resize_factor", str(req.avatar_resize_factor)]
        if req.avatar_pads:
            command += ["--avatar_pads", req.avatar_pads]
        if req.avatar_batch_size is not None:
            command += ["--avatar_batch_size", str(req.avatar_batch_size)]
        if req.avatar_nosmooth:
            command.append("--avatar_nosmooth")
        if req.avatar_blur_background is not None and not req.avatar_blur_background:
            command.append("--avatar_no_blur_background")
        if req.avatar_blur_kernel is not None:
            command += ["--avatar_blur_kernel", str(req.avatar_blur_kernel)]
        if req.avatar_device:
            command += ["--avatar_device", req.avatar_device]
    else:
        command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "src" / preprocess_script),
            "--video",
            str(video_path),
            *(["--audio", str(audio_path)] if audio_path else []),
            "--name",
            profile,
        ]
    return StreamingResponse(
        _stream_subprocess(command, cwd=PROJECT_ROOT),
        media_type="text/plain",
    )


@app.post("/train")
def train(req: TrainRequest):
    profile = req.profile.strip()
    if not profile:
        raise HTTPException(status_code=400, detail="profile is required")
    profile_type = req.profile_type or PROFILE_TYPE_VOICE
    dataset_path = resolve_dataset_root(profile, profile_type)
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail=f"dataset not found: {dataset_path}")
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "src" / "train.py"),
        "--dataset_path",
        str(dataset_path),
        "--profile_type",
        profile_type,
    ]
    if req.batch_size:
        command += ["--batch_size", str(req.batch_size)]
    if req.epochs:
        command += ["--epochs", str(req.epochs)]
    if req.max_len:
        command += ["--max_len", str(req.max_len)]
    if req.auto_select_epoch:
        command.append("--auto_select_epoch")
        if req.select_thorough:
            command.append("--select_thorough")
    if req.auto_tune_profile:
        command.append("--auto_tune_profile")
    if req.auto_build_lexicon:
        command.append("--auto_build_lexicon")
    if not req.early_stop:
        command.append("--no_early_stop")
    def _train_stream() -> Iterator[str]:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            yield f"{line.rstrip()}\n"
        exit_code = process.wait()
        if exit_code != 0:
            logger.error(
                "component=backend op=train_subprocess status=error exit_code=%s profile=%s profile_type=%s cmd=%s",
                exit_code,
                profile,
                profile_type,
                command,
            )
            yield f"[process exited {exit_code}]\n"
            yield "ERROR: Subprocess failed. See logs above for details.\n"
            return
        yield f"[process exited {exit_code}]\n"
        yield "[warmup] starting...\n"
        try:
            _warmup_profile(profile, profile_type, force=True)
            yield "[warmup] done\n"
        except Exception as exc:
            logger.exception(
                "component=backend op=warmup_profile status=error profile=%s profile_type=%s",
                profile,
                profile_type,
            )
            yield f"[warmup] failed: {exc}\n"

    return StreamingResponse(_train_stream(), media_type="text/plain")


@app.post("/stream")
def stream(req: GenerateRequest):
    start_time = time.perf_counter()
    profile_type = req.profile_type or PROFILE_TYPE_VOICE
    model_path = _resolve_model_path(req.model_path, req.speaker, profile_type)
    config_path = _resolve_config_path(model_path, req.config_path, req.speaker, profile_type)
    ref_wav_path = _resolve_ref_wav(req.ref_wav_path, req.speaker, profile_type, model_path)
    profile_dir = resolve_dataset_root(req.speaker or "", profile_type) if req.speaker else None
    params = _resolve_inference_params(model_path, config_path, req, req.speaker, profile_type)
    phonemizer_lang = _resolve_phonemizer_lang(model_path, req, req.speaker, profile_type)
    lexicon_path = _resolve_lexicon_path(model_path, req, req.speaker, profile_type)
    lexicon = _load_lexicon(lexicon_path)

    engine = _get_engine(model_path, config_path)
    profile = _load_profile_defaults(model_path)
    max_chars = (
        req.max_chunk_chars
        if req.max_chunk_chars is not None
        else profile.get("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS)
    )
    max_words = (
        req.max_chunk_words
        if req.max_chunk_words is not None
        else profile.get("max_chunk_words", DEFAULT_MAX_CHUNK_WORDS)
    )
    pause_ms = (
        req.pause_ms
        if req.pause_ms is not None
        else profile.get("pause_ms", DEFAULT_PAUSE_MS)
    )
    pad_text = (
        req.pad_text
        if req.pad_text is not None
        else profile.get("pad_text", True)
    )
    pad_text_token = (
        req.pad_text_token
        if req.pad_text_token is not None
        else profile.get("pad_text_token", "...")
    )
    smart_trim_db = (
        req.smart_trim_db
        if req.smart_trim_db is not None
        else profile.get("smart_trim_db", 30.0)
    )
    smart_trim_pad_ms = (
        req.smart_trim_pad_ms
        if req.smart_trim_pad_ms is not None
        else profile.get("smart_trim_pad_ms", 50.0)
    )
    crossfade_ms = (
        req.crossfade_ms
        if req.crossfade_ms is not None
        else profile.get("crossfade_ms", 8.0)
    )
    clean_text = clean_text_for_tts(req.text)
    chunks = _split_text(clean_text, max_chars, max_words)
    if not chunks:
        raise HTTPException(status_code=400, detail="Text is empty.")
    seed = req.seed if req.seed is not None else profile.get("seed")
    if seed is None and len(chunks) > 1:
        seed = 1234
    pause = np.zeros(int(engine.sample_rate * (pause_ms / 1000.0)), dtype=np.float32)
    comma_pause_ms = max(50.0, float(pause_ms) * 0.35)
    comma_pause = np.zeros(int(engine.sample_rate * (comma_pause_ms / 1000.0)), dtype=np.float32)
    def _generator() -> Iterator[str]:
        tts_chunk_index = 0
        if seed is not None:
            _seed_everything(seed)
        for idx, chunk in enumerate(chunks):
            ref_for_chunk = ref_wav_path
            if pad_text:
                chunk = f"{pad_text_token} {chunk} {pad_text_token}"
            chunk_seed = seed
            audio = engine.generate(
                chunk,
                ref_wav_path=ref_for_chunk,
                alpha=params["alpha"],
                beta=params["beta"],
                diffusion_steps=params["diffusion_steps"],
                embedding_scale=params["embedding_scale"],
                f0_scale=params["f0_scale"],
                phonemizer_lang=phonemizer_lang,
                lexicon=lexicon,
                seed=chunk_seed,
            )
            tts_chunk_index += 1
            if smart_trim_db and smart_trim_db > 0:
                audio = _smart_vad_trim(
                    audio,
                    engine.sample_rate,
                    top_db=float(smart_trim_db),
                    pad_ms=float(smart_trim_pad_ms),
                )
            audio = _remove_dc(audio.astype(np.float32, copy=False))
            if idx < len(chunks) - 1:
                suffix = chunk.rstrip()
                if suffix.endswith((".", "!", "?")) and pause.size:
                    audio = np.concatenate([audio, pause])
                elif suffix.endswith((",", ";", ":")) and comma_pause.size:
                    audio = np.concatenate([audio, comma_pause])
            pitch_shift = (
                req.pitch_shift
                if req.pitch_shift is not None
                else profile.get("pitch_shift", 0.0)
            )
            if pitch_shift:
                audio = _apply_pitch_shift(audio, engine.sample_rate, pitch_shift)
            de_esser_cutoff = (
                req.de_esser_cutoff
                if req.de_esser_cutoff is not None
                else profile.get("de_esser_cutoff", 0.0)
            )
            de_esser_order = (
                req.de_esser_order
                if req.de_esser_order is not None
                else profile.get("de_esser_order", 2)
            )
            if de_esser_cutoff:
                audio = _apply_de_esser(audio, engine.sample_rate, de_esser_cutoff, int(de_esser_order))
            audio = _soft_clip(audio)
            audio = _fade_edges(audio, engine.sample_rate, fade_ms=5.0)
            wav_bytes = _audio_to_wav_bytes(audio, engine.sample_rate)
            payload = base64.b64encode(wav_bytes).decode("ascii")
            yield json.dumps(
                {
                    "chunk_index": idx,
                    "audio_base64": payload,
                    "sample_rate": engine.sample_rate,
                }
            ) + "\n"
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        yield json.dumps({"event": "done", "inference_ms": round(elapsed_ms, 2)}) + "\n"

    return StreamingResponse(_generator(), media_type="application/x-ndjson")

@app.post("/stream_avatar")
def stream_avatar(req: GenerateRequest):
    text_iter = iter([req.text])
    return _stream_avatar_from_text_iter(req, text_iter)



@app.post("/generate")
def generate(req: GenerateRequest):
    start_time = time.perf_counter()
    profile_type = req.profile_type or PROFILE_TYPE_VOICE
    model_path = _resolve_model_path(req.model_path, req.speaker, profile_type)

    config_path = _resolve_config_path(model_path, req.config_path, req.speaker, profile_type)
    ref_wav_path = _resolve_ref_wav(req.ref_wav_path, req.speaker, profile_type, model_path)
    profile_dir = resolve_dataset_root(req.speaker or "", profile_type) if req.speaker else None
    params = _resolve_inference_params(model_path, config_path, req, req.speaker, profile_type)
    phonemizer_lang = _resolve_phonemizer_lang(model_path, req, req.speaker, profile_type)
    lexicon_path = _resolve_lexicon_path(model_path, req, req.speaker, profile_type)
    lexicon = _load_lexicon(lexicon_path)

    try:
        engine = _get_engine(model_path, config_path)
        profile = _load_profile_defaults(model_path)
        max_chars = (
            req.max_chunk_chars
            if req.max_chunk_chars is not None
            else profile.get("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS)
        )
        max_words = (
            req.max_chunk_words
            if req.max_chunk_words is not None
            else profile.get("max_chunk_words", DEFAULT_MAX_CHUNK_WORDS)
        )
        pause_ms = (
            req.pause_ms
            if req.pause_ms is not None
            else profile.get("pause_ms", DEFAULT_PAUSE_MS)
        )
        pad_text = (
            req.pad_text
            if req.pad_text is not None
            else profile.get("pad_text", True)
        )
        pad_text_token = (
            req.pad_text_token
            if req.pad_text_token is not None
            else profile.get("pad_text_token", "...")
        )
        smart_trim_db = (
            req.smart_trim_db
            if req.smart_trim_db is not None
            else profile.get("smart_trim_db", 30.0)
        )
        smart_trim_pad_ms = (
            req.smart_trim_pad_ms
            if req.smart_trim_pad_ms is not None
            else profile.get("smart_trim_pad_ms", 50.0)
        )
        crossfade_ms = (
            req.crossfade_ms
            if req.crossfade_ms is not None
            else profile.get("crossfade_ms", 8.0)
        )
        clean_text = clean_text_for_tts(req.text)
        chunks = _split_text(clean_text, max_chars, max_words)
        if not chunks:
            raise HTTPException(status_code=400, detail="Text is empty.")
        parts: list[np.ndarray] = []
        pause = np.zeros(int(engine.sample_rate * (pause_ms / 1000.0)), dtype=np.float32)
        comma_pause_ms = max(50.0, float(pause_ms) * 0.35)
        comma_pause = np.zeros(int(engine.sample_rate * (comma_pause_ms / 1000.0)), dtype=np.float32)
        seed = req.seed if req.seed is not None else profile.get("seed")
        if seed is None and len(chunks) > 1:
            seed = 1234
        if seed is not None:
            _seed_everything(seed)
        tts_chunk_index = 0
        for idx, chunk in enumerate(chunks):
            ref_for_chunk = ref_wav_path
            if pad_text:
                chunk = f"{pad_text_token} {chunk} {pad_text_token}"
            chunk_seed = seed
            audio = engine.generate(
                chunk,
                ref_wav_path=ref_for_chunk,
                alpha=params["alpha"],
                beta=params["beta"],
                diffusion_steps=params["diffusion_steps"],
                embedding_scale=params["embedding_scale"],
                f0_scale=params["f0_scale"],
                phonemizer_lang=phonemizer_lang,
                lexicon=lexicon,
                seed=chunk_seed,
            )
            tts_chunk_index += 1
            if smart_trim_db and smart_trim_db > 0:
                audio = _smart_vad_trim(
                    audio,
                    engine.sample_rate,
                    top_db=float(smart_trim_db),
                    pad_ms=float(smart_trim_pad_ms),
                )
            audio = _remove_dc(audio.astype(np.float32, copy=False))
            parts.append(audio)
            if idx < len(chunks) - 1:
                suffix = chunk.rstrip()
                if suffix.endswith((".", "!", "?")) and pause.size:
                    parts.append(pause)
                elif suffix.endswith((",", ";", ":")) and comma_pause.size:
                    parts.append(comma_pause)
        audio = _apply_crossfade(parts, engine.sample_rate, crossfade_ms)
        pitch_shift = (
            req.pitch_shift
            if req.pitch_shift is not None
            else profile.get("pitch_shift", 0.0)
        )
        if pitch_shift:
            audio = _apply_pitch_shift(audio, engine.sample_rate, pitch_shift)
        de_esser_cutoff = (
            req.de_esser_cutoff
            if req.de_esser_cutoff is not None
            else profile.get("de_esser_cutoff", 0.0)
        )
        de_esser_order = (
            req.de_esser_order
            if req.de_esser_order is not None
            else profile.get("de_esser_order", 2)
        )
        if de_esser_cutoff:
            audio = _apply_de_esser(audio, engine.sample_rate, de_esser_cutoff, int(de_esser_order))
        audio = _soft_clip(audio)
        audio = _fade_edges(audio, engine.sample_rate, fade_ms=5.0)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    wav_bytes = _audio_to_wav_bytes(audio, engine.sample_rate)

    if req.return_base64:
        payload = base64.b64encode(wav_bytes).decode("ascii")
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return JSONResponse(
            {"audio_base64": payload, "inference_ms": round(elapsed_ms, 2)},
            headers={"X-Inference-Time-Ms": f"{elapsed_ms:.2f}"},
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Inference-Time-Ms": f"{elapsed_ms:.2f}"},
    )


@app.post("/chat")
def chat(req: GenerateRequest, request: Request):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    llm = _get_llm_service()
    llm_stream = llm.stream_response(req.text)
    # Stabilize LLM mode with conservative defaults if not provided.
    if req.alpha is None:
        req.alpha = 0.2
    if req.beta is None:
        req.beta = 0.5
    if req.embedding_scale is None:
        req.embedding_scale = 1.2
    if req.diffusion_steps is None:
        req.diffusion_steps = 10
    if req.seed is None:
        req.seed = 1234
    if req.pad_text is None:
        req.pad_text = True
    if req.smart_trim_db is None:
        req.smart_trim_db = 0.0
    if req.smart_trim_pad_ms is None:
        req.smart_trim_pad_ms = 0.0
    profile_type = req.profile_type or PROFILE_TYPE_VOICE
    binary_transport = _binary_stream_requested(request)
    mobile_profile = _mobile_stream_requested(request)
    if profile_type == PROFILE_TYPE_AVATAR or req.avatar_profile:
        return _stream_avatar_from_text_iter(
            req,
            llm_stream,
            binary_transport=binary_transport,
            mobile_profile=mobile_profile,
        )
    return _stream_voice_from_text_iter(
        req,
        llm_stream,
        binary_transport=binary_transport,
        mobile_profile=mobile_profile,
    )


@app.post("/speak")
def speak(req: GenerateRequest, request: Request):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    profile_type = req.profile_type or PROFILE_TYPE_VOICE
    binary_transport = _binary_stream_requested(request)
    mobile_profile = _mobile_stream_requested(request)
    if profile_type == PROFILE_TYPE_AVATAR or req.avatar_profile:
        return _stream_avatar_from_text_iter(
            req,
            iter([req.text]),
            binary_transport=binary_transport,
            mobile_profile=mobile_profile,
        )
    return _stream_voice_from_text_iter(
        req,
        iter([req.text]),
        binary_transport=binary_transport,
        mobile_profile=mobile_profile,
    )
