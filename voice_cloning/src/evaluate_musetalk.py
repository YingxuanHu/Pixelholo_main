from __future__ import annotations

import argparse
import csv
import io
import json
import math
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    AVATAR_PROFILE_DIRNAME,
    OUTPUTS_DIR,
    RAW_VIDEOS_DIRNAME,
    workspace_data_root,
)


PIXELHOLO_BINARY_STREAM_MEDIA_TYPE = "application/vnd.pixelholo.stream-v1"
PIXELHOLO_BINARY_MAGIC = b"PHS1"

DEFAULT_TEXTS = [
    "Peter packed a purple paper parcel, then placed it carefully beside the bright blue box.",
    "Fifty five vivid voices rise and fall naturally while I speak clearly, calmly, and comfortably.",
    "I am ready to begin. After a brief pause, I continue speaking in the same natural voice.",
]


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _decode_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _decode_audio_chunk(payload: bytes, metadata: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Decode a PHS1 audio payload without assuming a particular transport format."""
    audio_format = str(metadata.get("audio_format") or "wav").strip().lower()
    if audio_format not in {"pcm", "pcm_s16le", "s16le", "raw"}:
        return _decode_wav_bytes(payload)

    sample_rate = max(1, int(metadata.get("sample_rate") or 24000))
    channels = max(1, int(metadata.get("audio_channels") or 1))
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1) if usable else np.zeros(0, dtype=np.float32)
    return samples, sample_rate


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values
    radius = window // 2
    padded = np.pad(values.astype(np.float32), (radius, radius), mode="edge")
    out = np.empty_like(values, dtype=np.float32)
    for idx in range(len(values)):
        out[idx] = float(np.mean(padded[idx : idx + window]))
    return out


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or right.size < 3:
        return 0.0
    if float(np.std(left)) < 1e-6 or float(np.std(right)) < 1e-6:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _audio_envelope_by_frame(audio: np.ndarray, frame_count: int) -> np.ndarray:
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


def _best_lag_corr(audio_env: np.ndarray, mouth_aperture: np.ndarray, max_lag: int = 10) -> tuple[float, int]:
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


def _sustained_silence_mask(
    audio_envelope: np.ndarray,
    *,
    threshold: float = 0.006,
    min_frames: int = 7,
) -> np.ndarray:
    """Identify only sustained, low-energy audio regions.

    A frame-by-frame RMS threshold would wrongly classify quiet consonants and
    normal word boundaries as silence.  This intentionally requires about
    280 ms at 25 fps before a region contributes to the pause-stability
    metric.
    """
    quiet = np.asarray(audio_envelope, dtype=np.float32) <= float(threshold)
    stable = np.zeros_like(quiet, dtype=bool)
    start = 0
    while start < len(quiet):
        if not quiet[start]:
            start += 1
            continue
        end = start + 1
        while end < len(quiet) and quiet[end]:
            end += 1
        if end - start >= max(1, int(min_frames)):
            stable[start:end] = True
        start = end
    return stable


def _mouth_roi(gray: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """Return a scale-normalized lower-mouth patch for detail and motion checks."""
    mouth_left, mouth_right, upper_lip, lower_lip = 61, 291, 13, 14
    mouth_width = float(np.linalg.norm(points[mouth_right] - points[mouth_left]))
    if mouth_width < 2.0:
        return None
    center = (points[mouth_left] + points[mouth_right] + points[upper_lip] + points[lower_lip]) / 4.0
    height, width = gray.shape[:2]
    x1 = max(0, int(round(center[0] - mouth_width * 0.78)))
    x2 = min(width, int(round(center[0] + mouth_width * 0.78)))
    y1 = max(0, int(round(center[1] - mouth_width * 0.45)))
    y2 = min(height, int(round(center[1] + mouth_width * 0.62)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return cv2.resize(gray[y1:y2, x1:x2], (96, 64), interpolation=cv2.INTER_AREA)


def _motion_summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float32)
    if finite.size == 0:
        return {"median": None, "p95": None}
    return {
        "median": round(float(np.median(finite)), 6),
        "p95": round(float(np.percentile(finite, 95)), 6),
    }


def _mouth_metrics(
    frames: list[np.ndarray],
    audio: np.ndarray | None,
    *,
    fps: float = 25.0,
) -> dict[str, Any]:
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
    mouth_rois: list[np.ndarray | None] = []
    sharpness: list[float] = []
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
                mouth_rois.append(None)
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
            roi = _mouth_roi(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pts)
            mouth_rois.append(roi)
            if roi is not None:
                sharpness.append(float(cv2.Laplacian(roi, cv2.CV_32F).var()))
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

    audio_env_all = _audio_envelope_by_frame(audio, len(apertures_arr)) if audio is not None else np.zeros(0, dtype=np.float32)
    audio_env = audio_env_all[finite_mask] if audio_env_all.size else np.zeros(0, dtype=np.float32)
    corr, lag = _best_lag_corr(audio_env, apertures_valid)
    center_jitter_p95 = float(np.percentile(center_delta, 95)) if center_delta.size else 0.0
    aperture_hf_rms = float(np.sqrt(np.mean(np.square(aperture_highfreq)))) if aperture_highfreq.size else 0.0
    # Keep the score useful as a coarse ranking signal. The raw jitter metrics
    # remain the source of truth; this combines them into a 0-100 sortable value.
    stability_score = 100.0 - (center_jitter_p95 * 90.0) - (aperture_hf_rms * 160.0)

    pair_motion: list[float] = [math.nan]
    for previous, current in zip(mouth_rois, mouth_rois[1:]):
        if previous is None or current is None:
            pair_motion.append(math.nan)
            continue
        pair_motion.append(float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))) / 255.0))

    silence_mask = _sustained_silence_mask(
        audio_env_all,
        min_frames=max(1, int(round(max(0.24, 7.0 / max(1.0, fps)) * fps))),
    ) if audio_env_all.size else np.zeros(len(frames), dtype=bool)
    silence_motion = [
        pair_motion[index]
        for index in range(1, min(len(pair_motion), len(silence_mask)))
        if silence_mask[index] and silence_mask[index - 1]
    ]
    voiced_motion = [
        pair_motion[index]
        for index in range(1, min(len(pair_motion), len(silence_mask)))
        if not silence_mask[index] and not silence_mask[index - 1]
    ]
    silence_summary = _motion_summary(silence_motion)
    voiced_summary = _motion_summary(voiced_motion)

    return {
        "landmark_valid_ratio": round(valid_ratio, 4),
        "center_jitter_median": round(float(np.median(center_delta)) if center_delta.size else 0.0, 6),
        "center_jitter_p95": round(center_jitter_p95, 6),
        "aperture_delta_median": round(float(np.median(aperture_delta)) if aperture_delta.size else 0.0, 6),
        "aperture_delta_p95": round(float(np.percentile(aperture_delta, 95)) if aperture_delta.size else 0.0, 6),
        "aperture_highfreq_rms": round(aperture_hf_rms, 6),
        "audio_mouth_corr": round(corr, 4),
        "audio_mouth_best_lag_frames": int(lag),
        "audio_mouth_best_lag_ms": round(float(lag) * 1000.0 / max(1.0, float(fps)), 1),
        "audio_sustained_silence_frames": int(np.sum(silence_mask)),
        "audio_sustained_silence_ratio": round(float(np.mean(silence_mask)) if silence_mask.size else 0.0, 4),
        "silence_mouth_motion_median": silence_summary["median"],
        "silence_mouth_motion_p95": silence_summary["p95"],
        "voiced_mouth_motion_median": voiced_summary["median"],
        "voiced_mouth_motion_p95": voiced_summary["p95"],
        "mouth_sharpness_laplacian_median": round(float(np.median(sharpness)) if sharpness else 0.0, 3),
        "mouth_sharpness_laplacian_p95": round(float(np.percentile(sharpness, 95)) if sharpness else 0.0, 3),
        "stability_score": round(float(np.clip(stability_score, 0.0, 100.0)), 2),
    }


def _resize_max_edge(frame: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0 or max(frame.shape[:2]) <= max_edge:
        return frame
    height, width = frame.shape[:2]
    scale = float(max_edge) / float(max(height, width))
    return cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _source_visual_metrics(source_video: Path, *, max_edge: int, sample_frames: int) -> dict[str, Any]:
    """Measure source-video face and mouth detail at the generated frame scale.

    This is a detail-retention baseline, not a ground-truth realism metric.
    The reference clip and generated answer have different speech content, so
    frame-level image similarity would be misleading.
    """
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        return {"source_video": str(source_video), "source_error": "could not open video"}
    reported_frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: list[np.ndarray] = []
    if reported_frame_count > 0:
        indexes = np.linspace(
            0,
            reported_frame_count - 1,
            max(1, min(sample_frames, reported_frame_count)),
        ).round().astype(int)
        for index in np.unique(indexes):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(_resize_max_edge(frame, max_edge))
    else:
        # Some browser-recorded WebM clips have no reliable frame count in
        # OpenCV. Read sequentially and sample a bounded prefix in that case.
        while len(frames) < sample_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append(_resize_max_edge(frame, max_edge))
    capture.release()
    metrics = _mouth_metrics(frames, None, fps=source_fps)
    metrics.update(
        {
            "source_video": str(source_video),
            "source_frames_sampled": len(frames),
            "source_fps": round(source_fps, 3),
            "source_frame_count": reported_frame_count or None,
        }
    )
    return metrics


def _resolve_source_video(profile: str, workspace_id: str | None, explicit_path: str | None) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        return candidate if candidate.is_file() else None
    profile_dir = workspace_data_root(workspace_id) / AVATAR_PROFILE_DIRNAME / profile / RAW_VIDEOS_DIRNAME
    candidates = sorted(
        path for path in profile_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    ) if profile_dir.exists() else []
    return candidates[0] if candidates else None


def _numeric_summary(values: list[Any]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not numeric:
        return {"n": 0, "median": None, "p95": None, "minimum": None, "maximum": None}
    return {
        "n": len(numeric),
        "median": round(float(np.median(numeric)), 6),
        "p95": round(float(np.percentile(numeric, 95)), 6),
        "minimum": round(float(np.min(numeric)), 6),
        "maximum": round(float(np.max(numeric)), 6),
    }


QUALITY_SUMMARY_KEYS = (
    "landmark_valid_ratio",
    "audio_mouth_corr",
    "audio_mouth_best_lag_ms",
    "center_jitter_p95",
    "aperture_highfreq_rms",
    "silence_mouth_motion_p95",
    "voiced_mouth_motion_p95",
    "mouth_sharpness_laplacian_median",
    "stability_score",
    "first_media_ms",
    "total_ms",
)


def _quality_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: _numeric_summary([row.get(key) for row in results])
        for key in QUALITY_SUMMARY_KEYS
    }


def _write_media(output_dir: Path, stem: str, frames: list[np.ndarray], fps: float, audio: np.ndarray, sample_rate: int) -> dict[str, str]:
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray], int, float]:
    pending = b""
    events: list[dict[str, Any]] = []
    audio_chunks: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    wire_bytes = 0
    first_media_ms = 0.0
    for raw in response.iter_content(chunk_size=65536):
        if not raw:
            continue
        wire_bytes += len(raw)
        pending += raw
        while len(pending) >= 12:
            if pending[:4] != PIXELHOLO_BINARY_MAGIC:
                raise RuntimeError(f"Invalid PixelHolo packet magic: {pending[:12]!r}")
            metadata_len, payload_len = struct.unpack(">II", pending[4:12])
            packet_len = 12 + metadata_len + payload_len
            if len(pending) < packet_len:
                break
            metadata = json.loads(pending[12 : 12 + metadata_len])
            payload = pending[12 + metadata_len : packet_len]
            pending = pending[packet_len:]
            events.append(metadata)
            if metadata.get("event") == "error":
                raise RuntimeError(str(metadata.get("detail", "backend stream error")))
            if metadata.get("event") != "chunk":
                continue
            if first_media_ms <= 0.0:
                first_media_ms = (time.perf_counter() - started) * 1000.0
            audio_len = int(metadata.get("audio_bytes_len") or 0)
            audio_chunks.append({"metadata": metadata, "payload": payload[:audio_len]})
            frame_payload = payload[audio_len:]
            cursor = 0
            for frame_len in metadata.get("frame_lengths") or []:
                frame_bytes = frame_payload[cursor : cursor + int(frame_len)]
                cursor += int(frame_len)
                decoded = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None:
                    frames.append(decoded)
    if pending:
        raise RuntimeError(f"Trailing undecoded binary bytes: {len(pending)}")
    return events, audio_chunks, frames, wire_bytes, first_media_ms


def _run_case(
    *,
    api_base: str,
    profile: str,
    workspace_id: str | None,
    text: str,
    preset: str,
    coord_source: str,
    seed: int,
    avatar_max_frame_edge: int,
    face_scale: float | None,
    temporal_smooth: float | None,
    lookahead_sec: float | None,
    mouth_mask_bottom_ratio: float | None,
    timeout: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "speaker": profile,
        "avatar_profile": profile,
        "profile_type": "avatar",
        "lipsync_backend": "musetalk",
        "musetalk_preset": preset,
        "musetalk_coord_source": coord_source,
        "seed": seed,
    }
    if avatar_max_frame_edge > 0:
        payload["avatar_max_frame_edge"] = avatar_max_frame_edge
    if face_scale is not None:
        payload["musetalk_face_scale"] = face_scale
    if temporal_smooth is not None:
        payload["musetalk_temporal_smooth"] = temporal_smooth
    if lookahead_sec is not None:
        payload["musetalk_lookahead_sec"] = lookahead_sec
    if mouth_mask_bottom_ratio is not None:
        payload["musetalk_mouth_mask_bottom_ratio"] = mouth_mask_bottom_ratio
    headers = {
        "Content-Type": "application/json",
        "Accept": PIXELHOLO_BINARY_STREAM_MEDIA_TYPE,
        "X-PixelHolo-Transport": "binary",
        "X-PixelHolo-Client": "web",
    }
    if workspace_id:
        headers["X-PixelHolo-Workspace"] = workspace_id
    started = time.perf_counter()
    with requests.post(
        f"{api_base.rstrip('/')}/speak",
        json=payload,
        headers=headers,
        stream=True,
        timeout=(10, timeout),
    ) as response:
        response.raise_for_status()
        events, audio_chunks, frames, wire_bytes, first_media_ms = _read_binary_stream(response, started)
    total_ms = (time.perf_counter() - started) * 1000.0

    sample_rate = 24000
    decoded_audio: list[np.ndarray] = []
    for chunk in audio_chunks:
        audio, sample_rate = _decode_audio_chunk(chunk["payload"], chunk["metadata"])
        decoded_audio.append(audio)
    audio = np.concatenate(decoded_audio) if decoded_audio else np.zeros(0, dtype=np.float32)

    fps_values = [float(event.get("fps")) for event in events if event.get("fps")]
    fps = float(np.median(fps_values)) if fps_values else 25.0
    done_events = [event for event in events if event.get("event") == "done"]
    chunk_events = [event for event in events if event.get("event") == "chunk"]
    metrics = _mouth_metrics(frames, audio, fps=fps)
    return {
        "profile": profile,
        "preset": preset,
        "coord_source": coord_source,
        "face_scale": face_scale,
        "temporal_smooth": temporal_smooth,
        "lookahead_sec": lookahead_sec,
        "mouth_mask_bottom_ratio": mouth_mask_bottom_ratio,
        "text": text,
        "total_ms": round(total_ms, 1),
        "server_inference_ms": done_events[-1].get("inference_ms") if done_events else None,
        "first_media_ms": round(first_media_ms, 1),
        "wire_bytes": wire_bytes,
        "chunks": len(chunk_events),
        "frames": len(frames),
        "audio_duration_sec": round(float(len(audio) / sample_rate), 4) if sample_rate else 0.0,
        "fps": round(fps, 3),
        "sample_rate": sample_rate,
        "audio": audio,
        "frames_data": frames,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure streamed MuseTalk quality proxies and source-detail retention."
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--profile", default="alvin1_video")
    parser.add_argument(
        "--workspace-id",
        help="Workspace that owns the explicitly selected test profile.",
    )
    parser.add_argument("--presets", default="low_latency,balanced,stable")
    parser.add_argument("--coord-sources", default="legacy")
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Sequential repeats per text and renderer configuration (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--avatar-max-frame-edge", type=int, default=512)
    parser.add_argument("--face-scale", type=float)
    parser.add_argument("--temporal-smooth", type=float)
    parser.add_argument("--lookahead-sec", type=float)
    parser.add_argument("--mouth-mask-bottom-ratio", type=float)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-save-media", action="store_true")
    parser.add_argument(
        "--source-video",
        help="Optional profile source clip. If omitted, uses raw_videos/<profile> in the selected workspace.",
    )
    parser.add_argument(
        "--source-sample-frames",
        type=int,
        default=60,
        help="Evenly spaced source frames used for source-detail baseline (default: 60).",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    presets = _parse_csv_arg(args.presets)
    coord_sources = _parse_csv_arg(args.coord_sources)
    texts = args.texts or DEFAULT_TEXTS
    output_dir = args.output_dir or (
        OUTPUTS_DIR / "musetalk_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    repeats = max(1, int(args.repeats))

    source_video = _resolve_source_video(args.profile, args.workspace_id, args.source_video)
    source_metrics = (
        _source_visual_metrics(
            source_video,
            max_edge=max(0, int(args.avatar_max_frame_edge)),
            sample_frames=max(1, int(args.source_sample_frames)),
        )
        if source_video is not None
        else {
            "source_video": None,
            "source_error": "no source video was found; lower-face detail baseline skipped",
        }
    )

    if not args.no_warmup:
        for preset in presets:
            for coord_source in coord_sources:
                payload = {
                    "profile": args.profile,
                    "profile_type": "avatar",
                    "lipsync_backend": "musetalk",
                    "musetalk_preset": preset,
                    "musetalk_coord_source": coord_source,
                    "include_llm": False,
                }
                response = requests.post(
                    f"{args.api_base.rstrip('/')}/warmup",
                    json=payload,
                    headers=(
                        {"X-PixelHolo-Workspace": args.workspace_id}
                        if args.workspace_id
                        else None
                    ),
                    timeout=(10, args.timeout),
                )
                response.raise_for_status()

    results: list[dict[str, Any]] = []
    for preset in presets:
        for coord_source in coord_sources:
            for repeat in range(1, repeats + 1):
                for idx, text in enumerate(texts, start=1):
                    case = _run_case(
                        api_base=args.api_base,
                        profile=args.profile,
                        workspace_id=args.workspace_id,
                        text=text,
                        preset=preset,
                        coord_source=coord_source,
                        seed=args.seed,
                        avatar_max_frame_edge=args.avatar_max_frame_edge,
                        face_scale=args.face_scale,
                        temporal_smooth=args.temporal_smooth,
                        lookahead_sec=args.lookahead_sec,
                        mouth_mask_bottom_ratio=args.mouth_mask_bottom_ratio,
                        timeout=args.timeout,
                    )
                    case["repeat"] = repeat
                    case["text_index"] = idx
                    stem = f"{args.profile}_{preset}_{coord_source}_r{repeat}_{idx}"
                    if not args.no_save_media:
                        media_paths = _write_media(
                            output_dir,
                            stem,
                            case.pop("frames_data"),
                            float(case["fps"]),
                            case.pop("audio"),
                            int(case["sample_rate"]),
                        )
                        case.update(media_paths)
                    else:
                        case.pop("frames_data")
                        case.pop("audio")
                    results.append(case)
                    print(
                        f"{stem}: total={case['total_ms']}ms frames={case['frames']} "
                        f"lag={case.get('audio_mouth_best_lag_ms')}ms "
                        f"jitter_p95={case.get('center_jitter_p95')} "
                        f"corr={case.get('audio_mouth_corr')} score={case.get('stability_score')}",
                        flush=True,
                    )

    json_path = output_dir / "report.json"
    json_path.write_text(
        json.dumps(
            {
                "method": {
                    "scope": "Automated quality proxies for the actual PHS1 stream. These do not replace a held-out human realism study.",
                    "lip_sync_proxy": "Correlation and best lag between generated-audio RMS envelope and MediaPipe mouth aperture.",
                    "stability": "MediaPipe mouth-center jitter and high-frequency aperture variation.",
                    "pause_stability": "Mouth-patch motion only within sustained low-energy generated-audio regions.",
                    "detail_retention": "Laplacian variance of a normalized mouth crop, compared descriptively with the profile source at the same maximum frame edge. This is not a perceptual-realism score.",
                },
                "configuration": {
                    "api_base": args.api_base,
                    "profile": args.profile,
                    "workspace_id": args.workspace_id,
                    "presets": presets,
                    "coord_sources": coord_sources,
                    "texts": texts,
                    "repeats": repeats,
                    "seed": args.seed,
                    "avatar_max_frame_edge": args.avatar_max_frame_edge,
                },
                "source_metrics": source_metrics,
                "summary": _quality_summary(results),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_path = output_dir / "report.csv"
    fieldnames = sorted({key for row in results for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Report written to {json_path}")
    print(f"CSV written to {csv_path}")


if __name__ == "__main__":
    main()
