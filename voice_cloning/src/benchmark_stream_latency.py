from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUTS_DIR  # noqa: E402


PIXELHOLO_BINARY_STREAM_MEDIA_TYPE = "application/vnd.pixelholo.stream-v1"
PIXELHOLO_BINARY_MAGIC = b"PHS1"

DEFAULT_TEXT = (
    "The latency benchmark is running now. Please speak this short sentence "
    "smoothly while the avatar starts as quickly as possible."
)


@dataclass(frozen=True)
class ClientTiming:
    audio_start_delay_sec: float
    audio_chunk_lead_sec: float
    video_predecode_wait_ms: float
    waits_for_predecode_before_audio: bool


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    description: str
    request_overrides: dict[str, Any]
    client_timing: ClientTiming


CASES: dict[str, BenchmarkCase] = {
    "old_like": BenchmarkCase(
        name="old_like",
        description=(
            "Old-like baseline: no first MuseTalk window override and older web "
            "audio/video scheduling constants."
        ),
        request_overrides={
            "musetalk_preset": "realistic",
            "musetalk_stream_window_sec": 1.65,
            "musetalk_first_window_sec": 0.0,
            "musetalk_lookahead_sec": 0.28,
            "musetalk_temporal_smooth": 0.16,
            "musetalk_jpeg_quality": 95,
        },
        client_timing=ClientTiming(
            audio_start_delay_sec=0.52,
            audio_chunk_lead_sec=0.14,
            video_predecode_wait_ms=90.0,
            waits_for_predecode_before_audio=True,
        ),
    ),
    "current": BenchmarkCase(
        name="current",
        description=(
            "Current optimized path: early first MuseTalk window and current web "
            "audio/video scheduling constants."
        ),
        request_overrides={
            "musetalk_preset": "realistic",
            "musetalk_stream_window_sec": 1.65,
            "musetalk_first_window_sec": 0.55,
            "musetalk_lookahead_sec": 0.28,
            "musetalk_temporal_smooth": 0.20,
            "musetalk_jpeg_quality": 95,
        },
        client_timing=ClientTiming(
            audio_start_delay_sec=0.34,
            audio_chunk_lead_sec=0.08,
            video_predecode_wait_ms=45.0,
            waits_for_predecode_before_audio=False,
        ),
    ),
    "candidate_070": BenchmarkCase(
        name="candidate_070",
        description=(
            "Candidate path: current web scheduling with a 0.70s first MuseTalk "
            "window to test quality recovery versus startup latency."
        ),
        request_overrides={
            "musetalk_preset": "realistic",
            "musetalk_stream_window_sec": 1.65,
            "musetalk_first_window_sec": 0.70,
            "musetalk_lookahead_sec": 0.28,
            "musetalk_jpeg_quality": 95,
        },
        client_timing=ClientTiming(
            audio_start_delay_sec=0.34,
            audio_chunk_lead_sec=0.08,
            video_predecode_wait_ms=45.0,
            waits_for_predecode_before_audio=False,
        ),
    ),
    "current_smooth": BenchmarkCase(
        name="current_smooth",
        description=(
            "Candidate path: current 0.55s first MuseTalk window with slightly "
            "stronger temporal smoothing."
        ),
        request_overrides={
            "musetalk_preset": "realistic",
            "musetalk_stream_window_sec": 1.65,
            "musetalk_first_window_sec": 0.55,
            "musetalk_lookahead_sec": 0.28,
            "musetalk_temporal_smooth": 0.20,
            "musetalk_jpeg_quality": 95,
        },
        client_timing=ClientTiming(
            audio_start_delay_sec=0.34,
            audio_chunk_lead_sec=0.08,
            video_predecode_wait_ms=45.0,
            waits_for_predecode_before_audio=False,
        ),
    ),
    "current_lookahead_016": BenchmarkCase(
        name="current_lookahead_016",
        description=(
            "Candidate path: current first window and smoothing with shorter "
            "MuseTalk lookahead to test mouth/audio lag."
        ),
        request_overrides={
            "musetalk_preset": "realistic",
            "musetalk_stream_window_sec": 1.65,
            "musetalk_first_window_sec": 0.55,
            "musetalk_lookahead_sec": 0.16,
            "musetalk_temporal_smooth": 0.20,
            "musetalk_jpeg_quality": 95,
        },
        client_timing=ClientTiming(
            audio_start_delay_sec=0.34,
            audio_chunk_lead_sec=0.08,
            video_predecode_wait_ms=45.0,
            waits_for_predecode_before_audio=False,
        ),
    ),
    "candidate_085": BenchmarkCase(
        name="candidate_085",
        description=(
            "Candidate path: current web scheduling with a 0.85s first MuseTalk "
            "window to test a more conservative quality/latency balance."
        ),
        request_overrides={
            "musetalk_preset": "realistic",
            "musetalk_stream_window_sec": 1.65,
            "musetalk_first_window_sec": 0.85,
            "musetalk_lookahead_sec": 0.28,
            "musetalk_jpeg_quality": 95,
        },
        client_timing=ClientTiming(
            audio_start_delay_sec=0.34,
            audio_chunk_lead_sec=0.08,
            video_predecode_wait_ms=45.0,
            waits_for_predecode_before_audio=False,
        ),
    ),
    "candidate_085_smooth": BenchmarkCase(
        name="candidate_085_smooth",
        description=(
            "Candidate path: 0.85s first MuseTalk window with slightly stronger "
            "temporal smoothing to test whether center jitter improves."
        ),
        request_overrides={
            "musetalk_preset": "realistic",
            "musetalk_stream_window_sec": 1.65,
            "musetalk_first_window_sec": 0.85,
            "musetalk_lookahead_sec": 0.28,
            "musetalk_temporal_smooth": 0.20,
            "musetalk_jpeg_quality": 95,
        },
        client_timing=ClientTiming(
            audio_start_delay_sec=0.34,
            audio_chunk_lead_sec=0.08,
            video_predecode_wait_ms=45.0,
            waits_for_predecode_before_audio=False,
        ),
    ),
}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return round(float(statistics.median(finite)), 1)


def _median_digits(values: list[float], digits: int) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return round(float(statistics.median(finite)), digits)


def _pct_delta(old_value: float | None, new_value: float | None) -> float | None:
    if old_value is None or new_value is None or old_value <= 0:
        return None
    return round(((old_value - new_value) / old_value) * 100.0, 1)


def _signed_pct_delta(old_value: float | None, new_value: float | None) -> float | None:
    if old_value is None or new_value is None or abs(old_value) <= 1e-9:
        return None
    return round(((new_value - old_value) / abs(old_value)) * 100.0, 1)


def _estimated_client_audio_start_ms(first_media_ms: float, timing: ClientTiming) -> tuple[float, float]:
    """Estimate when browser audio starts after first media packet arrival.

    This mirrors the App.tsx scheduling model closely enough for comparison:
    old code waited for initial frame predecode before scheduling audio, while
    the current code schedules audio first and lets frame predecode continue in
    parallel. The true browser value still depends on decode time and device.
    """
    first_media_ms = max(0.0, float(first_media_ms))
    predecode_ms = (
        float(timing.video_predecode_wait_ms)
        if timing.waits_for_predecode_before_audio
        else 0.0
    )
    start_delay_remaining_ms = max(0.0, timing.audio_start_delay_sec * 1000.0 - first_media_ms)
    scheduling_lead_ms = max(timing.audio_chunk_lead_sec * 1000.0, start_delay_remaining_ms)
    client_added_ms = predecode_ms + scheduling_lead_ms
    return round(first_media_ms + client_added_ms, 1), round(client_added_ms, 1)


def _moving_average(values: Any, window: int) -> Any:
    import numpy as np

    if values.size == 0 or window <= 1:
        return values
    radius = window // 2
    padded = np.pad(values.astype(np.float32), (radius, radius), mode="edge")
    out = np.empty_like(values, dtype=np.float32)
    for idx in range(len(values)):
        out[idx] = float(np.mean(padded[idx : idx + window]))
    return out


def _safe_corr(left: Any, right: Any) -> float:
    import numpy as np

    if left.size < 3 or right.size < 3:
        return 0.0
    if float(np.std(left)) < 1e-6 or float(np.std(right)) < 1e-6:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _audio_envelope_by_frame(audio: Any, frame_count: int) -> Any:
    import numpy as np

    if frame_count <= 0 or audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    edges = np.linspace(0, len(audio), frame_count + 1).round().astype(np.int64)
    envelope = np.zeros(frame_count, dtype=np.float32)
    for idx in range(frame_count):
        start = int(edges[idx])
        end = max(start + 1, int(edges[idx + 1]))
        chunk = audio[start:end]
        envelope[idx] = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
    return envelope


def _best_lag_corr(audio_env: Any, mouth_aperture: Any, max_lag: int = 10) -> tuple[float, int]:
    if audio_env.size < 8 or mouth_aperture.size < 8:
        return 0.0, 0
    best_corr = -1.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left = audio_env[-lag:]
            right = mouth_aperture[: len(left)]
        elif lag > 0:
            left = audio_env[:-lag]
            right = mouth_aperture[lag:]
        else:
            left = audio_env
            right = mouth_aperture
        count = min(len(left), len(right))
        if count < 8:
            continue
        corr = _safe_corr(left[:count], right[:count])
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return max(0.0, best_corr), best_lag


def _decode_audio_chunks(audio_chunks: list[dict[str, Any]]) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    decoded: list[Any] = []
    sample_rate = 24000
    for chunk in audio_chunks:
        metadata = chunk["metadata"]
        payload = chunk["payload"]
        audio_format = metadata.get("audio_format", "wav")
        if audio_format == "pcm_s16le":
            channels = max(1, int(metadata.get("audio_channels") or 1))
            sample_rate = max(1, int(metadata.get("sample_rate") or sample_rate))
            audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio[: (len(audio) // channels) * channels]
                audio = audio.reshape(-1, channels).mean(axis=1)
        else:
            audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) == 2:
                audio = audio.mean(axis=1)
        decoded.append(np.asarray(audio, dtype=np.float32))
    if not decoded:
        return np.zeros(0, dtype=np.float32), sample_rate
    return np.concatenate(decoded), sample_rate


def _mouth_metrics(frames: list[Any], audio: Any) -> dict[str, Any]:
    import cv2
    import numpy as np

    try:
        import mediapipe as mp
    except Exception as exc:
        return {"landmark_error": f"mediapipe unavailable: {exc}", "landmark_valid_ratio": 0.0}

    if not frames:
        return {"landmark_error": "no frames", "landmark_valid_ratio": 0.0}

    mouth_left = 61
    mouth_right = 291
    upper_lip = 13
    lower_lip = 14
    face_left = 234
    face_right = 454
    centers: list[tuple[float, float]] = []
    apertures: list[float] = []
    valid = 0

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    try:
        for frame in frames:
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:
                centers.append((math.nan, math.nan))
                apertures.append(math.nan)
                continue
            valid += 1
            landmarks = result.multi_face_landmarks[0].landmark
            pts = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float32)
            mouth_w = float(np.linalg.norm(pts[mouth_right] - pts[mouth_left]))
            face_w = float(np.linalg.norm(pts[face_right] - pts[face_left]))
            scale = max(1.0, mouth_w, face_w * 0.28)
            mouth_center = (
                pts[mouth_left] + pts[mouth_right] + pts[upper_lip] + pts[lower_lip]
            ) / 4.0
            aperture = float(np.linalg.norm(pts[lower_lip] - pts[upper_lip]) / scale)
            centers.append((float(mouth_center[0] / scale), float(mouth_center[1] / scale)))
            apertures.append(aperture)
    finally:
        mesh.close()

    valid_ratio = valid / max(1, len(frames))
    centers_arr = np.array(centers, dtype=np.float32)
    apertures_arr = np.array(apertures, dtype=np.float32)
    finite_mask = np.isfinite(centers_arr).all(axis=1) & np.isfinite(apertures_arr)
    if int(finite_mask.sum()) < 8:
        return {
            "landmark_valid_ratio": round(valid_ratio, 4),
            "landmark_error": "too few valid mouth landmarks",
        }

    centers_valid = centers_arr[finite_mask]
    apertures_valid = apertures_arr[finite_mask]
    center_delta = np.linalg.norm(np.diff(centers_valid, axis=0), axis=1)
    aperture_delta = np.abs(np.diff(apertures_valid))
    aperture_smooth = _moving_average(apertures_valid, 5)
    aperture_highfreq = apertures_valid - aperture_smooth

    audio_env = _audio_envelope_by_frame(audio, len(apertures_arr))
    audio_env = audio_env[finite_mask]
    corr, lag = _best_lag_corr(audio_env, apertures_valid)
    center_jitter_p95 = float(np.percentile(center_delta, 95)) if center_delta.size else 0.0
    aperture_hf_rms = float(np.sqrt(np.mean(np.square(aperture_highfreq)))) if aperture_highfreq.size else 0.0
    stability_score = 100.0 - (center_jitter_p95 * 90.0) - (aperture_hf_rms * 160.0)

    return {
        "landmark_valid_ratio": round(valid_ratio, 4),
        "center_jitter_median": round(float(np.median(center_delta)) if center_delta.size else 0.0, 6),
        "center_jitter_p95": round(center_jitter_p95, 6),
        "aperture_delta_median": round(float(np.median(aperture_delta)) if aperture_delta.size else 0.0, 6),
        "aperture_delta_p95": round(float(np.percentile(aperture_delta, 95)) if aperture_delta.size else 0.0, 6),
        "aperture_highfreq_rms": round(aperture_hf_rms, 6),
        "audio_mouth_corr": round(corr, 4),
        "audio_mouth_best_lag_frames": int(lag),
        "audio_mouth_abs_lag_frames": int(abs(lag)),
        "stability_score": round(float(np.clip(stability_score, 0.0, 100.0)), 2),
    }


def _frame_motion_metrics(frames: list[Any], chunk_frame_counts: list[int]) -> dict[str, Any]:
    import cv2
    import numpy as np

    if len(frames) < 2:
        return {}
    grays = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        for frame in frames
    ]
    deltas = np.array(
        [float(np.mean(np.abs(grays[idx] - grays[idx - 1]))) for idx in range(1, len(grays))],
        dtype=np.float32,
    )
    if deltas.size == 0:
        return {}
    median = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - median)))
    spike_threshold = median + max(0.012, 4.0 * mad)
    boundary_deltas: list[float] = []
    cursor = 0
    for count in chunk_frame_counts[:-1]:
        cursor += int(count)
        if 0 < cursor < len(frames):
            boundary_deltas.append(float(deltas[cursor - 1]))

    out = {
        "frame_delta_median": round(median, 6),
        "frame_delta_p95": round(float(np.percentile(deltas, 95)), 6),
        "frame_delta_max": round(float(np.max(deltas)), 6),
        "frame_delta_spikes": int(np.sum(deltas > spike_threshold)),
        "frame_delta_spike_threshold": round(float(spike_threshold), 6),
    }
    if boundary_deltas:
        boundary = np.array(boundary_deltas, dtype=np.float32)
        out.update(
            {
                "chunk_boundary_delta_median": round(float(np.median(boundary)), 6),
                "chunk_boundary_delta_p95": round(float(np.percentile(boundary, 95)), 6),
                "chunk_boundary_delta_max": round(float(np.max(boundary)), 6),
                "chunk_boundaries": int(len(boundary_deltas)),
            }
        )
    return out


def _write_media(
    output_dir: Path,
    stem: str,
    frames: list[Any],
    fps: float,
    audio: Any,
    sample_rate: int,
) -> dict[str, str]:
    import cv2
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if audio.size:
        wav_path = output_dir / f"{stem}.wav"
        sf.write(wav_path, audio, sample_rate)
        paths["audio_path"] = str(wav_path)
    if frames:
        video_path = output_dir / f"{stem}.mp4"
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps or 25.0),
            (width, height),
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
        paths["video_path"] = str(video_path)
    return paths


def _read_binary_stream(
    response: requests.Response,
    started: float,
    *,
    collect_media: bool,
) -> dict[str, Any]:
    if collect_media:
        import cv2
        import numpy as np
    else:
        cv2 = None
        np = None

    pending = b""
    events: list[dict[str, Any]] = []
    audio_chunks: list[dict[str, Any]] = []
    frames_data: list[Any] = []
    wire_bytes = 0
    audio_bytes = 0
    frame_bytes = 0
    frame_count = 0
    first_media_ms = 0.0
    chunk_count = 0
    chunk_frame_counts: list[int] = []
    audio_duration_sec = 0.0
    fps_values: list[float] = []

    for raw in response.iter_content(chunk_size=65536):
        if not raw:
            continue
        wire_bytes += len(raw)
        pending += raw
        while len(pending) >= 12:
            if pending[:4] != PIXELHOLO_BINARY_MAGIC:
                raise RuntimeError(f"Invalid PixelHolo binary stream magic: {pending[:12]!r}")
            metadata_len, payload_len = struct.unpack(">II", pending[4:12])
            packet_len = 12 + metadata_len + payload_len
            if len(pending) < packet_len:
                break

            metadata = json.loads(pending[12 : 12 + metadata_len])
            payload = pending[12 + metadata_len : packet_len]
            pending = pending[packet_len:]
            events.append(metadata)

            event = metadata.get("event")
            if event == "error":
                raise RuntimeError(str(metadata.get("detail", "backend stream error")))
            if event != "chunk":
                continue

            if first_media_ms <= 0.0:
                first_media_ms = (time.perf_counter() - started) * 1000.0
            chunk_count += 1
            current_audio_bytes = int(metadata.get("audio_bytes_len") or 0)
            audio_bytes += current_audio_bytes
            frame_lengths = [int(length) for length in (metadata.get("frame_lengths") or [])]
            chunk_frame_counts.append(len(frame_lengths))
            frame_count += len(frame_lengths)
            frame_bytes += sum(frame_lengths)
            audio_duration_sec += float(metadata.get("duration_sec") or 0.0)
            if metadata.get("fps"):
                fps_values.append(float(metadata["fps"]))
            if len(payload) != payload_len:
                raise RuntimeError("Packet payload length mismatch.")
            if collect_media:
                audio_chunks.append(
                    {
                        "metadata": metadata,
                        "payload": payload[:current_audio_bytes],
                    }
                )
                frame_payload = payload[current_audio_bytes:]
                cursor = 0
                for frame_len in frame_lengths:
                    frame_bytes_data = frame_payload[cursor : cursor + frame_len]
                    cursor += frame_len
                    decoded = cv2.imdecode(np.frombuffer(frame_bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None:
                        frames_data.append(decoded)

    if pending:
        raise RuntimeError(f"Trailing undecoded binary bytes: {len(pending)}")

    done_events = [event for event in events if event.get("event") == "done"]
    metrics = {
        "events": events,
        "first_media_ms": round(first_media_ms, 1),
        "wire_bytes": wire_bytes,
        "audio_bytes": audio_bytes,
        "frame_bytes": frame_bytes,
        "chunks": chunk_count,
        "chunk_frame_counts": chunk_frame_counts,
        "frames": frame_count,
        "audio_duration_sec": round(audio_duration_sec, 3),
        "fps": _median(fps_values),
        "server_inference_ms": done_events[-1].get("inference_ms") if done_events else None,
    }
    if collect_media:
        metrics["audio_chunks_data"] = audio_chunks
        metrics["frames_data"] = frames_data
    return metrics


def _post_stream(
    *,
    api_base: str,
    endpoint: str,
    payload: dict[str, Any],
    audio_format: str,
    timeout: float,
    quality_metrics: bool,
    save_media: bool,
    media_dir: Path,
    media_stem: str,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Accept": PIXELHOLO_BINARY_STREAM_MEDIA_TYPE,
        "X-PixelHolo-Transport": "binary",
        "X-PixelHolo-Client": "web",
    }
    if audio_format:
        headers["X-PixelHolo-Audio-Format"] = audio_format

    started = time.perf_counter()
    with requests.post(
        f"{api_base.rstrip('/')}/{endpoint.lstrip('/')}",
        json=payload,
        headers=headers,
        stream=True,
        timeout=(15, timeout),
    ) as response:
        response.raise_for_status()
        stream_metrics = _read_binary_stream(
            response,
            started,
            collect_media=quality_metrics,
        )
    total_ms = (time.perf_counter() - started) * 1000.0
    stream_metrics.pop("events", None)
    stream_metrics["total_ms"] = round(total_ms, 1)
    if quality_metrics:
        try:
            audio, sample_rate = _decode_audio_chunks(stream_metrics.pop("audio_chunks_data", []))
            frames = stream_metrics.pop("frames_data", [])
            stream_metrics["decoded_audio_duration_sec"] = (
                round(float(len(audio) / sample_rate), 4) if sample_rate else 0.0
            )
            stream_metrics["decoded_frames"] = len(frames)
            stream_metrics.update(_mouth_metrics(frames, audio))
            stream_metrics.update(_frame_motion_metrics(frames, stream_metrics["chunk_frame_counts"]))
            if save_media:
                stream_metrics.update(
                    _write_media(
                        media_dir,
                        media_stem,
                        frames,
                        float(stream_metrics.get("fps") or 25.0),
                        audio,
                        sample_rate,
                    )
                )
        except Exception as exc:
            stream_metrics["quality_error"] = repr(exc)
    return stream_metrics


def _warmup(
    *,
    api_base: str,
    profile: str,
    timeout: float,
    avatar_max_frame_edge: int,
    case: BenchmarkCase,
) -> dict[str, Any]:
    payload = {
        "profile": profile,
        "profile_type": "avatar",
        "lipsync_backend": "musetalk",
        "include_llm": False,
        "musetalk_preset": case.request_overrides.get("musetalk_preset", "realistic"),
        "musetalk_stream_window_sec": case.request_overrides.get("musetalk_stream_window_sec"),
        "musetalk_lookahead_sec": case.request_overrides.get("musetalk_lookahead_sec"),
        "avatar_max_frame_edge": avatar_max_frame_edge,
    }
    started = time.perf_counter()
    response = requests.post(
        f"{api_base.rstrip('/')}/warmup",
        json=payload,
        timeout=(15, timeout),
    )
    response.raise_for_status()
    data = response.json()
    data["wall_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return data


def _build_payload(
    *,
    endpoint: str,
    profile: str,
    text: str,
    seed: int,
    avatar_max_frame_edge: int,
    llm_mode: str,
    case: BenchmarkCase,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "speaker": profile,
        "avatar_profile": profile,
        "profile_type": "avatar",
        "lipsync_backend": "musetalk",
        "seed": seed,
    }
    if endpoint.lstrip("/") == "chat":
        payload["llm_mode"] = llm_mode
    if avatar_max_frame_edge > 0:
        payload["avatar_max_frame_edge"] = avatar_max_frame_edge
    payload.update(case.request_overrides)
    return payload


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_case.setdefault(row["case"], []).append(row)

    cases: dict[str, dict[str, Any]] = {}
    for case_name, rows in by_case.items():
        case_summary = {
            "runs": len(rows),
            "median_first_media_ms": _median([row["first_media_ms"] for row in rows]),
            "median_estimated_web_audio_start_ms": _median(
                [row["estimated_web_audio_start_ms"] for row in rows]
            ),
            "median_total_ms": _median([row["total_ms"] for row in rows]),
            "median_server_inference_ms": _median(
                [
                    float(row["server_inference_ms"])
                    for row in rows
                    if row.get("server_inference_ms") is not None
                ]
            ),
            "median_chunks": _median([float(row["chunks"]) for row in rows]),
            "median_frames": _median([float(row["frames"]) for row in rows]),
            "median_wire_kb": _median([row["wire_bytes"] / 1024.0 for row in rows]),
        }
        quality_keys = (
            "landmark_valid_ratio",
            "center_jitter_p95",
            "aperture_delta_p95",
            "aperture_highfreq_rms",
            "audio_mouth_corr",
            "audio_mouth_best_lag_frames",
            "audio_mouth_abs_lag_frames",
            "stability_score",
            "frame_delta_p95",
            "frame_delta_spikes",
            "chunk_boundary_delta_p95",
            "chunk_boundary_delta_max",
        )
        for key in quality_keys:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            if values:
                case_summary[f"median_{key}"] = _median_digits(values, 6)
        cases[case_name] = case_summary

    old_case = cases.get("old_like")
    current_case = cases.get("current")
    deltas: dict[str, Any] = {}
    if old_case and current_case:
        for key in (
            "median_first_media_ms",
            "median_estimated_web_audio_start_ms",
            "median_total_ms",
            "median_server_inference_ms",
            "median_wire_kb",
        ):
            old_value = old_case.get(key)
            new_value = current_case.get(key)
            if old_value is None or new_value is None:
                continue
            deltas[f"{key}_improvement_pct"] = _pct_delta(float(old_value), float(new_value))
            deltas[f"{key}_delta_ms" if key.endswith("_ms") else f"{key}_delta"] = round(
                float(old_value) - float(new_value),
                1,
            )
        lower_is_better = (
            "median_center_jitter_p95",
            "median_aperture_delta_p95",
            "median_aperture_highfreq_rms",
            "median_audio_mouth_abs_lag_frames",
            "median_frame_delta_p95",
            "median_frame_delta_spikes",
            "median_chunk_boundary_delta_p95",
            "median_chunk_boundary_delta_max",
        )
        higher_is_better = (
            "median_landmark_valid_ratio",
            "median_audio_mouth_corr",
            "median_stability_score",
        )
        for key in lower_is_better:
            old_value = old_case.get(key)
            new_value = current_case.get(key)
            if old_value is None or new_value is None:
                continue
            deltas[f"{key}_improvement_pct"] = _pct_delta(float(old_value), float(new_value))
            deltas[f"{key}_delta"] = round(float(old_value) - float(new_value), 6)
        for key in higher_is_better:
            old_value = old_case.get(key)
            new_value = current_case.get(key)
            if old_value is None or new_value is None:
                continue
            deltas[f"{key}_improvement_pct"] = _signed_pct_delta(float(old_value), float(new_value))
            deltas[f"{key}_delta"] = round(float(new_value) - float(old_value), 6)

    return {"cases": cases, "deltas": deltas}


def _print_result(row: dict[str, Any]) -> None:
    print(
        "run={run} case={case:<8} first_media={first_media_ms:>7.1f}ms "
        "est_web_audio={estimated_web_audio_start_ms:>7.1f}ms "
        "total={total_ms:>7.1f}ms chunks={chunks:<2} frames={frames:<3} "
        "wire={wire_kb:>7.1f}KB stability={stability}".format(
            **row,
            wire_kb=row["wire_bytes"] / 1024.0,
            stability=row.get("stability_score", "-"),
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PixelHolo streaming latency by comparing old-like and "
            "current MuseTalk/web scheduling settings."
        )
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", choices=("speak", "chat"), default="speak")
    parser.add_argument("--profile", default="alvin1_video")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--llm-mode", default="legacy_fast")
    parser.add_argument("--cases", default="old_like,current")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--avatar-max-frame-edge", type=int, default=853)
    parser.add_argument("--audio-format", choices=("wav", "pcm_s16le"), default="wav")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--quality-metrics", action="store_true")
    parser.add_argument("--save-media", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    selected_case_names = _parse_csv(args.cases)
    unknown = [name for name in selected_case_names if name not in CASES]
    if unknown:
        raise SystemExit(f"Unknown benchmark case(s): {', '.join(unknown)}")

    output_dir = args.output_dir or (
        OUTPUTS_DIR / "latency_bench" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_cases = [CASES[name] for name in selected_case_names]
    warmup_results: list[dict[str, Any]] = []
    if not args.no_warmup:
        # Warm the current path once before measurement. Individual case runs are
        # then alternated so cache state is shared as fairly as possible.
        warmup_case = CASES["current"] if "current" in CASES else selected_cases[0]
        warmup_results.append(
            _warmup(
                api_base=args.api_base,
                profile=args.profile,
                timeout=args.timeout,
                avatar_max_frame_edge=args.avatar_max_frame_edge,
                case=warmup_case,
            )
        )

    results: list[dict[str, Any]] = []
    run_index = 0
    for repeat in range(1, max(1, args.repeats) + 1):
        for case in selected_cases:
            run_index += 1
            payload = _build_payload(
                endpoint=args.endpoint,
                profile=args.profile,
                text=args.text,
                seed=args.seed,
                avatar_max_frame_edge=args.avatar_max_frame_edge,
                llm_mode=args.llm_mode,
                case=case,
            )
            metrics = _post_stream(
                api_base=args.api_base,
                endpoint=args.endpoint,
                payload=payload,
                audio_format=args.audio_format,
                timeout=args.timeout,
                quality_metrics=args.quality_metrics,
                save_media=args.save_media,
                media_dir=output_dir / "media",
                media_stem=f"{run_index:02d}_{case.name}_repeat{repeat}",
            )
            estimated_start_ms, client_added_ms = _estimated_client_audio_start_ms(
                metrics["first_media_ms"],
                case.client_timing,
            )
            row = {
                "run": run_index,
                "repeat": repeat,
                "case": case.name,
                "description": case.description,
                "endpoint": args.endpoint,
                "profile": args.profile,
                "text_chars": len(args.text),
                "audio_format": args.audio_format,
                "estimated_web_audio_start_ms": estimated_start_ms,
                "estimated_client_added_ms": client_added_ms,
                **metrics,
                "request_overrides": case.request_overrides,
            }
            results.append(row)
            _print_result(row)

    summary = _summarize(results)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "api_base": args.api_base,
        "endpoint": args.endpoint,
        "profile": args.profile,
        "text": args.text,
        "audio_format": args.audio_format,
        "warmup": warmup_results,
        "summary": summary,
        "results": results,
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Report written to {report_path}", flush=True)


if __name__ == "__main__":
    main()
