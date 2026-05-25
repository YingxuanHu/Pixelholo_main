import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def split_text(text: str, max_chars: int, max_words: int) -> list[str]:
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


def apply_pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
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


def apply_de_esser(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int) -> np.ndarray:
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


def soft_clip(audio: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    if audio.size == 0:
        return audio
    audio = audio.astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(np.abs(audio)))
    if max_val <= threshold:
        return audio
    return np.tanh(audio / threshold) * threshold


def sanitize_audio_for_playback(
    audio: np.ndarray,
    max_peak: float = 0.95,
    max_rms: float = 0.25,
    corrupt_nonfinite_fraction: float = 0.02,
    corrupt_clipped_fraction: float = 0.20,
    corrupt_noise_rms: float = 0.10,
    corrupt_zero_cross_fraction: float = 0.45,
) -> np.ndarray:
    """Prevent invalid or runaway samples from becoming full-scale PCM noise."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    finite_mask = np.isfinite(audio)
    nonfinite_fraction = 1.0 - (float(np.count_nonzero(finite_mask)) / float(audio.size))
    if nonfinite_fraction >= corrupt_nonfinite_fraction:
        return np.zeros_like(audio, dtype=np.float32)
    if not finite_mask.all():
        audio = audio.copy()
        audio[~finite_mask] = 0.0

    abs_audio = np.abs(audio)
    peak = float(np.max(abs_audio))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(audio, dtype=np.float32)

    audio64 = audio.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(audio64 * audio64)))
    if not np.isfinite(rms):
        return np.zeros_like(audio, dtype=np.float32)

    clipped_fraction = float(np.mean(abs_audio >= 0.995))
    if rms > max_rms and clipped_fraction >= corrupt_clipped_fraction:
        return np.zeros_like(audio, dtype=np.float32)
    if audio.size > 1:
        zero_cross_fraction = float(np.mean(np.signbit(audio[1:]) != np.signbit(audio[:-1])))
        if rms > corrupt_noise_rms and zero_cross_fraction >= corrupt_zero_cross_fraction:
            return np.zeros_like(audio, dtype=np.float32)

    gain = 1.0
    if peak > max_peak:
        gain = min(gain, max_peak / peak)
    if rms > max_rms:
        gain = min(gain, max_rms / rms)
    if gain < 1.0:
        audio = audio * np.float32(gain)

    return soft_clip(audio, threshold=max_peak).astype(np.float32, copy=False)


def remove_dc(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    audio = audio.astype(np.float32, copy=False)
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio - float(np.mean(audio))


def apply_crossfade(
    chunks: list[np.ndarray],
    sample_rate: int,
    crossfade_ms: float,
    fade_edges_ms: float | None = None,
) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)

    if fade_edges_ms is None:
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

    cross_len = int(sample_rate * (crossfade_ms / 1000.0))
    fade_len = int(sample_rate * (fade_edges_ms / 1000.0))

    def _edge_fade(audio: np.ndarray) -> np.ndarray:
        if fade_len <= 1:
            return audio
        effective_len = fade_len
        if effective_len * 2 > audio.size:
            effective_len = max(1, audio.size // 2)
        audio = audio.astype(np.float32, copy=False)
        fade_in = np.linspace(0.0, 1.0, effective_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, effective_len, dtype=np.float32)
        audio[:effective_len] *= fade_in
        audio[-effective_len:] *= fade_out
        return audio

    if cross_len < 2:
        return np.concatenate([_edge_fade(c) for c in chunks])

    faded = _edge_fade(chunks[0].astype(np.float32, copy=False))
    t = np.linspace(0.0, 1.0, cross_len, dtype=np.float32)
    fade_in = np.sin(t * (np.pi / 2.0))
    fade_out = np.cos(t * (np.pi / 2.0))
    for nxt in chunks[1:]:
        nxt = _edge_fade(nxt.astype(np.float32, copy=False))
        if faded.size < cross_len or nxt.size < cross_len:
            faded = np.concatenate([faded, nxt])
            continue
        tail = faded[-cross_len:] * fade_out
        head = nxt[:cross_len] * fade_in
        blended = tail + head
        faded = np.concatenate([faded[:-cross_len], blended, nxt[cross_len:]])
    return faded


def fade_edges(audio: np.ndarray, sample_rate: int, fade_ms: float = 5.0) -> np.ndarray:
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


def smart_vad_trim(
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


def trim_leading_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 30.0,
    frame_length: int = 1024,
    hop_length: int = 256,
    pad_ms: float = 12.0,
    max_trim_ms: float = 220.0,
) -> np.ndarray:
    try:
        import librosa
    except Exception:
        return audio
    if audio.size == 0 or top_db <= 0 or max_trim_ms <= 0:
        return audio
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0 or float(np.max(rms)) <= 1e-8:
        return audio
    rms_db = librosa.power_to_db(rms * rms, ref=np.max)
    non_silent = np.flatnonzero(rms_db > -top_db)
    if non_silent.size == 0:
        return audio
    start = librosa.frames_to_samples(non_silent[0], hop_length=hop_length)
    pad = int(sample_rate * (pad_ms / 1000.0))
    max_trim = int(sample_rate * (max_trim_ms / 1000.0))
    trim_to = min(max(0, start - pad), max_trim)
    if trim_to <= 0:
        return audio
    return audio[trim_to:]
