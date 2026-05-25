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

from config import OUTPUTS_DIR  # noqa: E402


PIXELHOLO_BINARY_STREAM_MEDIA_TYPE = "application/vnd.pixelholo.stream-v1"
PIXELHOLO_BINARY_MAGIC = b"PHS1"

DEFAULT_TEXTS = [
    "The latency test is running now. Watch the lips closely while I speak these words.",
    "Please compare the mouth stability when I say thirty three, fifty five, and ninety nine.",
]


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _decode_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


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


def _mouth_metrics(frames: list[np.ndarray], audio: np.ndarray) -> dict[str, Any]:
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
    # Keep the score useful as a coarse ranking signal. The raw jitter metrics
    # remain the source of truth; this combines them into a 0-100 sortable value.
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
        "stability_score": round(float(np.clip(stability_score, 0.0, 100.0)), 2),
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
) -> tuple[list[dict[str, Any]], list[bytes], list[np.ndarray], int, float]:
    pending = b""
    events: list[dict[str, Any]] = []
    audio_chunks: list[bytes] = []
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
            audio_chunks.append(payload[:audio_len])
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
    text: str,
    preset: str,
    coord_source: str,
    seed: int,
    avatar_max_frame_edge: int,
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
    headers = {
        "Content-Type": "application/json",
        "Accept": PIXELHOLO_BINARY_STREAM_MEDIA_TYPE,
        "X-PixelHolo-Transport": "binary",
        "X-PixelHolo-Client": "web",
    }
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
    for wav_bytes in audio_chunks:
        audio, sample_rate = _decode_wav_bytes(wav_bytes)
        decoded_audio.append(audio)
    audio = np.concatenate(decoded_audio) if decoded_audio else np.zeros(0, dtype=np.float32)

    fps_values = [float(event.get("fps")) for event in events if event.get("fps")]
    fps = float(np.median(fps_values)) if fps_values else 25.0
    done_events = [event for event in events if event.get("event") == "done"]
    chunk_events = [event for event in events if event.get("event") == "chunk"]
    metrics = _mouth_metrics(frames, audio)
    return {
        "profile": profile,
        "preset": preset,
        "coord_source": coord_source,
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
    parser = argparse.ArgumentParser(description="Compare MuseTalk latency, jitter, and sync proxy metrics.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--profile", default="alvin1_video")
    parser.add_argument("--presets", default="low_latency,balanced,stable")
    parser.add_argument("--coord-sources", default="legacy")
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--avatar-max-frame-edge", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-save-media", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    presets = _parse_csv_arg(args.presets)
    coord_sources = _parse_csv_arg(args.coord_sources)
    texts = args.texts or DEFAULT_TEXTS
    output_dir = args.output_dir or (
        OUTPUTS_DIR / "musetalk_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

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
                    timeout=(10, args.timeout),
                )
                response.raise_for_status()

    results: list[dict[str, Any]] = []
    for preset in presets:
        for coord_source in coord_sources:
            for idx, text in enumerate(texts, start=1):
                case = _run_case(
                    api_base=args.api_base,
                    profile=args.profile,
                    text=text,
                    preset=preset,
                    coord_source=coord_source,
                    seed=args.seed,
                    avatar_max_frame_edge=args.avatar_max_frame_edge,
                    timeout=args.timeout,
                )
                stem = f"{args.profile}_{preset}_{coord_source}_{idx}"
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
                    f"jitter_p95={case.get('center_jitter_p95')} "
                    f"corr={case.get('audio_mouth_corr')} score={case.get('stability_score')}",
                    flush=True,
                )

    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
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
