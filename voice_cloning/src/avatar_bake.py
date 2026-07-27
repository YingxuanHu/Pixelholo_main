import argparse
import hashlib
import json
import os
import pickle
import sys
import math
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
import torch

# --- 1. REMBG IMPORT ---
try:
    from rembg import new_session, remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Config Loader (Safe Fallback)
try:
    from config import (
        LIP_SYNCING_DIR,
        PROFILE_TYPE_AVATAR,
        PROFILE_TYPE_VOICE,
        avatar_cache_dir,
    )
except ImportError:
    LIP_SYNCING_DIR = PROJECT_ROOT
    PROFILE_TYPE_AVATAR = "avatar"
    PROFILE_TYPE_VOICE = "voice"
    def avatar_cache_dir(profile, type_): return PROJECT_ROOT / "cache" / profile / type_


ProgressReporter = Callable[[float, str], None]


def _report_progress(progress: ProgressReporter | None, fraction: float, activity: str) -> None:
    if progress:
        progress(min(1.0, max(0.0, fraction)), activity)


def _smooth_boxes(boxes: np.ndarray, window: int = 5) -> np.ndarray:
    if boxes.size == 0: return boxes
    smoothed = boxes.copy().astype(np.float32)
    for i in range(len(smoothed)):
        start = max(0, i - window + 1)
        smoothed[i] = smoothed[start : i + 1].mean(axis=0)
    return smoothed.round().astype(np.int32)


def _apply_loop_crossfade(frames: list[np.ndarray], fade_frames: int) -> list[np.ndarray]:
    if fade_frames <= 0 or fade_frames * 2 >= len(frames): return frames
    total = len(frames)
    for i in range(fade_frames):
        alpha = float(i + 1) / float(fade_frames)
        tail_idx = total - fade_frames + i
        head_idx = i
        blended = cv2.addWeighted(frames[tail_idx], 1.0 - alpha, frames[head_idx], alpha, 0)
        frames[tail_idx] = blended
    return frames


def _center_crop_3_4(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target_ratio = 3 / 4
    current_ratio = width / height
    if abs(current_ratio - target_ratio) < 1e-3: return frame
    
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        x0 = max(0, (width - new_width) // 2)
        return frame[:, x0:x0 + new_width]
    
    new_height = int(width / target_ratio)
    y0 = max(0, (height - new_height) // 2)
    return frame[y0:y0 + new_height, :]


# Tight crop around detected face box while preserving 3:4 aspect ratio.
def _tight_crop_to_face(
    frames: list[np.ndarray],
    coords: np.ndarray,
    *,
    scale: float = 2.0,
    target_ratio: float = 3 / 4,
) -> tuple[list[np.ndarray], np.ndarray]:
    cropped_frames: list[np.ndarray] = []
    cropped_coords: list[list[int]] = []

    for frame, box in zip(frames, coords):
        h, w = frame.shape[:2]
        y1, y2, x1, x2 = box
        face_h = max(1, y2 - y1)
        face_w = max(1, x2 - x1)

        desired_h = int(face_h * scale)
        desired_w = int(face_w * scale)

        crop_h = max(desired_h, int(desired_w / target_ratio))
        crop_w = int(crop_h * target_ratio)

        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / target_ratio)
        if crop_h > h:
            crop_h = h
            crop_w = int(crop_h * target_ratio)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        x0 = max(0, min(cx - crop_w // 2, w - crop_w))
        y0 = max(0, min(cy - crop_h // 2, h - crop_h))

        cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
        cropped_frames.append(cropped)

        new_box = [
            max(0, y1 - y0),
            min(crop_h, y2 - y0),
            max(0, x1 - x0),
            min(crop_w, x2 - x0),
        ]
        cropped_coords.append(new_box)

    return cropped_frames, np.array(cropped_coords, dtype=np.int32)


def _resize_frames_and_coords(
    frames: list[np.ndarray], coords: np.ndarray, target_size: tuple[int, int]
) -> tuple[list[np.ndarray], np.ndarray]:
    target_w, target_h = target_size
    resized_frames: list[np.ndarray] = []
    resized_coords: list[list[int]] = []

    for frame, box in zip(frames, coords):
        h, w = frame.shape[:2]
        if (w, h) != (target_w, target_h):
            frame = cv2.resize(frame, (target_w, target_h))
            scale_x = target_w / max(1, w)
            scale_y = target_h / max(1, h)
            y1, y2, x1, x2 = box
            box = [
                int(round(y1 * scale_y)),
                int(round(y2 * scale_y)),
                int(round(x1 * scale_x)),
                int(round(x2 * scale_x)),
            ]
        resized_frames.append(frame)
        resized_coords.append(box)

    return resized_frames, np.array(resized_coords, dtype=np.int32)
# --- 2. HIGH-QUALITY BLUR FUNCTION (REMBG) ---
def _blur_background_with_rembg(
    frames: list[np.ndarray],
    blur_kernel: int,
    progress: ProgressReporter | None = None,
) -> list[np.ndarray]:
    if not REMBG_AVAILABLE:
        print("\n[WARNING] 'rembg' library not found.")
        print("   -> Skipping blur. To fix: pip install rembg\n")
        _report_progress(progress, 1.0, "Background refinement skipped")
        return frames

    print(f"   ...blurring {len(frames)} frames using Rembg (High Quality)...")
    
    # Initialize session once (much faster than re-loading per frame)
    # 'u2net_human_seg' is optimized specifically for human bodies/hair
    try:
        session = new_session("u2net_human_seg")
    except Exception as e:
        print(f"[WARN] Could not load 'u2net_human_seg', falling back to default. Error: {e}")
        session = new_session("u2net")

    k = max(3, int(blur_kernel) // 2 * 2 + 1)
    output_frames: list[np.ndarray] = []
    
    total_frames = max(1, len(frames))
    report_every = max(1, total_frames // 24)
    for i, frame in enumerate(frames):
        # 1. Prepare Input (OpenCV is BGR, Rembg needs RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Get Mask
        # only_mask=True returns a single channel alpha mask (0-255)
        mask = remove(rgb_frame, session=session, only_mask=True)
        
        # 3. Normalize Mask (0.0 to 1.0)
        mask_float = np.array(mask) / 255.0
        
        # Stack to 3 channels to match image shape (H, W, 3)
        mask_3d = np.stack((mask_float,) * 3, axis=-1)

        # 4. Create Blurred Version
        blurred = cv2.GaussianBlur(frame, (k, k), 0)
        
        # 5. Composite (Soft Blend)
        # Pixel = (Original * Mask) + (Blurred * (1-Mask))
        composite = (frame.astype(np.float32) * mask_3d + 
                     blurred.astype(np.float32) * (1.0 - mask_3d)).astype(np.uint8)
        
        output_frames.append(composite)
        
        # Progress indicator
        if i % 5 == 0:
            print(f"   Processed {i}/{len(frames)} frames", end="\r")
        if (i + 1) % report_every == 0 or i + 1 == total_frames:
            _report_progress(
                progress,
                (i + 1) / total_frames,
                "Refining the background",
            )
            
    print("") # Clear line
    return output_frames
# ---------------------------------------------


def _load_detector(device: str = "cuda"):
    lip_dir = LIP_SYNCING_DIR / "lib" / "Wav2Lip"
    if lip_dir.exists():
        sys.path.insert(0, str(lip_dir))
    try:
        import face_detection 
        return face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, flip_input=False, device=device
        )
    except ImportError:
        print("[WARNING] Wav2Lip face_detection not found. Falling back to simple cache.")
        return None


def _sanitize_xyxy(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))
    if x2 <= x1:
        x1, x2 = 0, width
    if y2 <= y1:
        y1, y2 = 0, height
    return x1, y1, x2, y2


def _legacy_coords_to_xyxy(
    coords_y1y2x1x2: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert the detector's persisted full-face boxes to MuseTalk xyxy boxes.

    `coords.npy` is stored in Wav2Lip's y1/y2/x1/x2 order.  MuseTalk itself
    expects the complete detected face, not the tighter lower-face landmark
    box used for an earlier visual experiment.
    """
    converted: list[list[int]] = []
    for y1, y2, x1, x2 in np.asarray(coords_y1y2x1x2, dtype=np.int32):
        converted.append(list(_sanitize_xyxy((int(x1), int(y1), int(x2), int(y2)), width, height)))
    return np.asarray(converted, dtype=np.int32)


def _align_mask_to_crop(
    mask: np.ndarray,
    crop_box: list[int],
    frame_w: int,
    frame_h: int,
) -> tuple[np.ndarray, list[int]]:
    x_s, y_s, x_e, y_e = [int(v) for v in crop_box]
    if x_e <= x_s or y_e <= y_s:
        return np.zeros((1, 1), dtype=np.uint8), [0, 0, 1, 1]

    cx1 = max(0, x_s)
    cy1 = max(0, y_s)
    cx2 = min(frame_w, x_e)
    cy2 = min(frame_h, y_e)
    if cx2 <= cx1 or cy2 <= cy1:
        return np.zeros((1, 1), dtype=np.uint8), [0, 0, 1, 1]

    off_x = cx1 - x_s
    off_y = cy1 - y_s
    want_h = cy2 - cy1
    want_w = cx2 - cx1
    mask = np.asarray(mask)
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    mask = mask[max(0, off_y) : max(0, off_y) + want_h, max(0, off_x) : max(0, off_x) + want_w]
    if mask.shape[:2] != (want_h, want_w):
        mask = cv2.resize(mask, (want_w, want_h), interpolation=cv2.INTER_LINEAR)
    return mask.astype(np.uint8, copy=False), [cx1, cy1, cx2, cy2]


def _smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size == 0:
        return values
    out = values.astype(np.float32).copy()
    for i in range(len(out)):
        start = max(0, i - window + 1)
        out[i] = out[start : i + 1].mean(axis=0)
    return out


def _build_musetalk_boxes(
    frames: np.ndarray,
    base_coords_y1y2x1x2: np.ndarray,
) -> np.ndarray:
    frame_h, frame_w = frames.shape[1:3]
    try:
        import mediapipe as mp
    except Exception:
        mp = None

    # Landmarks used for stable lower-face ROI.
    lm_mouth_left = 61
    lm_mouth_right = 291
    lm_upper_lip = 13
    lm_lower_lip = 14
    lm_chin = 152
    lm_nose = 1

    centers_sizes: list[tuple[float, float, float]] = []
    for frame, base in zip(frames, base_coords_y1y2x1x2):
        y1, y2, x1, x2 = [int(v) for v in base]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        base_size = float(max(bw, bh))
        fallback_cx = x1 + bw * 0.5
        fallback_cy = y1 + bh * 0.62
        fallback_size = base_size * 0.78

        cx, cy, size = fallback_cx, fallback_cy, fallback_size
        if mp is not None:
            if not hasattr(_build_musetalk_boxes, "_mesh"):
                _build_musetalk_boxes._mesh = mp.solutions.face_mesh.FaceMesh(  # type: ignore[attr-defined]
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
            mesh = _build_musetalk_boxes._mesh  # type: ignore[attr-defined]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark
                pts = np.array([[lm.x * frame_w, lm.y * frame_h] for lm in lms], dtype=np.float32)
                mouth_l = pts[lm_mouth_left]
                mouth_r = pts[lm_mouth_right]
                mouth_u = pts[lm_upper_lip]
                mouth_d = pts[lm_lower_lip]
                chin = pts[lm_chin]
                nose = pts[lm_nose]

                mouth_center = (mouth_l + mouth_r + mouth_u + mouth_d) / 4.0
                mouth_w = float(np.linalg.norm(mouth_r - mouth_l))
                lower_h = float(max(12.0, chin[1] - nose[1]))

                cx = float(mouth_center[0])
                cy = float(mouth_center[1] + 0.18 * max(8.0, chin[1] - mouth_center[1]))
                size = max(mouth_w * 2.55, lower_h * 1.2)
                size = float(np.clip(size, base_size * 0.58, base_size * 1.12))

        centers_sizes.append((cx, cy, size))

    arr = np.array(centers_sizes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.int32)

    median_size = float(np.median(arr[:, 2]))
    if median_size > 1.0:
        arr[:, 2] = np.clip(arr[:, 2], median_size * 0.85, median_size * 1.22)
    arr = _smooth_series(arr, window=5)

    coords_xyxy: list[list[int]] = []
    for cx, cy, size in arr:
        half = int(round(size / 2.0))
        nx1 = int(round(cx)) - half
        ny1 = int(round(cy)) - half
        nx2 = nx1 + int(round(size))
        ny2 = ny1 + int(round(size))

        if nx1 < 0:
            nx2 += -nx1
            nx1 = 0
        if ny1 < 0:
            ny2 += -ny1
            ny1 = 0
        if nx2 > frame_w:
            nx1 -= (nx2 - frame_w)
            nx2 = frame_w
        if ny2 > frame_h:
            ny1 -= (ny2 - frame_h)
            ny2 = frame_h
        nx1, ny1, nx2, ny2 = _sanitize_xyxy((nx1, ny1, nx2, ny2), frame_w, frame_h)
        coords_xyxy.append([nx1, ny1, nx2, ny2])

    return np.array(coords_xyxy, dtype=np.int32)


def _musetalk_runtime_cache_settings() -> dict[str, float | int | str]:
    """Settings that must match MuseTalkBridge._cache_manifest exactly.

    Avatar preparation runs in its own process.  Keeping this small manifest
    contract here lets preparation persist assets that the inference worker can
    load directly instead of rebuilding the 500-frame latent/mask cache on the
    first response.
    """
    return {
        "cache_version": int(os.getenv("MUSE_TALK_CACHE_VERSION", "10")),
        "parsing_mode": os.getenv("MUSE_TALK_PARSING_MODE", "raw"),
        "coord_expand_x": float(os.getenv("MUSE_TALK_COORD_EXPAND_X", "0.08")),
        "coord_expand_up": float(os.getenv("MUSE_TALK_COORD_EXPAND_UP", "0.04")),
        "coord_expand_down": float(os.getenv("MUSE_TALK_COORD_EXPAND_DOWN", "0.18")),
        "upper_boundary_ratio": float(os.getenv("MUSE_TALK_UPPER_BOUNDARY_RATIO", "0.5")),
        "blend_expand": float(os.getenv("MUSE_TALK_BLEND_EXPAND", "1.2")),
        "alpha_blur_ratio": float(os.getenv("MUSE_TALK_ALPHA_BLUR_RATIO", "0.035")),
        "vignette_margin_ratio": float(os.getenv("MUSE_TALK_VIGNETTE_MARGIN_RATIO", "0.02")),
        "alpha_gamma": float(os.getenv("MUSE_TALK_ALPHA_GAMMA", "0.82")),
    }


def _expand_musetalk_coord(
    coord: tuple[int, int, int, int],
    width: int,
    height: int,
    settings: dict[str, float | int | str],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = coord
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(round(box_w * float(settings["coord_expand_x"])))
    pad_up = int(round(box_h * float(settings["coord_expand_up"])))
    pad_down = int(round(box_h * float(settings["coord_expand_down"])))
    return _sanitize_xyxy(
        (x1 - pad_x, y1 - pad_up, x2 + pad_x, y2 + pad_down),
        width,
        height,
    )


def bake_musetalk_assets(
    cache_dir: Path,
    frames: np.ndarray,
    base_coords_y1y2x1x2: np.ndarray,
    progress: ProgressReporter | None = None,
) -> None:
    if frames.ndim != 4 or base_coords_y1y2x1x2.ndim != 2:
        raise ValueError("Invalid frames/coords layout for MuseTalk baking.")
    if len(frames) == 0 or len(frames) != len(base_coords_y1y2x1x2):
        raise ValueError("Frames/coords mismatch for MuseTalk baking.")

    musetalk_dir = LIP_SYNCING_DIR / "lib" / "MuseTalk"
    if not musetalk_dir.exists():
        raise FileNotFoundError(f"MuseTalk repo not found at {musetalk_dir}")
    if str(musetalk_dir) not in sys.path:
        sys.path.insert(0, str(musetalk_dir))

    from musetalk.models.vae import VAE
    from musetalk.utils.blending import get_image_prepare_material
    from musetalk.utils.face_parsing import FaceParsing

    models_dir = Path(os.getenv("MUSE_TALK_MODELS_DIR", str(musetalk_dir / "models")))
    vae_dir = Path(os.getenv("MUSE_TALK_VAE_DIR", str(models_dir / "sd-vae")))
    face_parse_model = Path(
        os.getenv(
            "MUSE_TALK_FACE_PARSE_MODEL",
            str(models_dir / "face-parse-bisent" / "79999_iter.pth"),
        )
    )
    face_parse_resnet = Path(
        os.getenv(
            "MUSE_TALK_FACE_PARSE_RESNET",
            str(models_dir / "face-parse-bisent" / "resnet18-5c106cde.pth"),
        )
    )
    for path in (vae_dir, face_parse_model, face_parse_resnet):
        if not path.exists():
            raise FileNotFoundError(f"MuseTalk asset missing: {path}")

    class _FaceParsingWithPaths(FaceParsing):
        def __init__(self, resnet_path: Path, model_path: Path) -> None:
            self._resnet_path = str(resnet_path)
            self._model_path = str(model_path)
            super().__init__()

        def model_init(self, resnet_path=None, model_pth=None):
            return super().model_init(
                resnet_path=self._resnet_path,
                model_pth=self._model_path,
            )

    print("   Baking MuseTalk assets (landmark-stabilized)...")
    _report_progress(progress, 0.02, "Loading lip-sync models")
    vae = VAE(model_path=str(vae_dir))
    face_parser = _FaceParsingWithPaths(face_parse_resnet, face_parse_model)
    # Use the same mask, expansion, and manifest settings as the long-lived
    # inference worker.  Previously this function wrote similar-looking files
    # without the runtime manifest and with a different face expansion, so
    # MuseTalk discarded them and rebuilt every latent on the first request.
    settings = _musetalk_runtime_cache_settings()
    frame_h, frame_w = frames.shape[1:3]

    # MuseTalk's latent and mask cache must be prepared from the same
    # full-face coordinate source used at inference.  The landmark-stabilized
    # lower-face box is kept separately for diagnostics, but using it as the
    # primary crop strips model context and can make generated lips appear
    # unchanged from the recorded source video.
    primary_coords_xyxy = _legacy_coords_to_xyxy(base_coords_y1y2x1x2, frame_w, frame_h)
    landmark_coords_xyxy = _build_musetalk_boxes(frames, base_coords_y1y2x1x2)
    np.save(cache_dir / "musetalk_coords.npy", landmark_coords_xyxy)
    _report_progress(progress, 0.08, "Aligning face frames")

    latents: list[torch.Tensor] = []
    mask_arrays: list[np.ndarray] = []
    mask_crop_boxes: list[list[int]] = []

    total_frames = max(1, len(frames))
    report_every = max(1, total_frames // 24)
    for index, (frame, coord) in enumerate(zip(frames, primary_coords_xyxy), start=1):
        raw_coord = _sanitize_xyxy(tuple(int(v) for v in coord), frame_w, frame_h)
        x1, y1, x2, y2 = _expand_musetalk_coord(
            raw_coord,
            frame_w,
            frame_h,
            settings,
        )

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
            x1, y1, x2, y2 = 0, 0, frame_w, frame_h
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents.append(vae.get_latents_for_unet(resized).detach().cpu())

        mask, crop_box = get_image_prepare_material(
            frame,
            [x1, y1, x2, y2],
            upper_boundary_ratio=float(settings["upper_boundary_ratio"]),
            expand=float(settings["blend_expand"]),
            fp=face_parser,
            mode=str(settings["parsing_mode"]),
        )
        aligned_mask, aligned_crop_box = _align_mask_to_crop(mask, crop_box, frame_w, frame_h)
        mask_arrays.append(aligned_mask)
        mask_crop_boxes.append([int(v) for v in aligned_crop_box])
        if index % report_every == 0 or index == total_frames:
            _report_progress(
                progress,
                0.08 + 0.87 * (index / total_frames),
                "Preparing lip-sync frames",
            )

    _report_progress(progress, 0.96, "Saving lip-sync assets")
    # No suffix is the runtime cache selected by the default full-face
    # `legacy` coordinate source.  Preparing it here prevents a long cache
    # rebuild the first time a newly created avatar is selected.
    cache_suffix = ""
    latents_path = cache_dir / f"musetalk_latents{cache_suffix}.pt"
    masks_path = cache_dir / f"musetalk_masks{cache_suffix}.pkl"
    runtime_meta_path = cache_dir / f"musetalk_runtime_meta{cache_suffix}.json"
    torch.save(latents, latents_path)
    with masks_path.open("wb") as handle:
        pickle.dump(
            {
                "cache_version": int(settings["cache_version"]),
                "coord_format": "xyxy",
                "coords_sha1": hashlib.sha1(
                    primary_coords_xyxy.astype(np.int32, copy=False).tobytes()
                ).hexdigest(),
                "parsing_mode": str(settings["parsing_mode"]),
                "mask_arrays": mask_arrays,
                "mask_crop_boxes": mask_crop_boxes,
            },
            handle,
        )
    runtime_meta_path.write_text(
        json.dumps(
            {
                "cache_version": int(settings["cache_version"]),
                "coord_format": "xyxy",
                "coords_sha1": hashlib.sha1(
                    primary_coords_xyxy.astype(np.int32, copy=False).tobytes()
                ).hexdigest(),
                "parsing_mode": str(settings["parsing_mode"]),
                "coord_expand_x": float(settings["coord_expand_x"]),
                "coord_expand_up": float(settings["coord_expand_up"]),
                "coord_expand_down": float(settings["coord_expand_down"]),
                "upper_boundary_ratio": float(settings["upper_boundary_ratio"]),
                "blend_expand": float(settings["blend_expand"]),
                "alpha_blur_ratio": float(settings["alpha_blur_ratio"]),
                "vignette_margin_ratio": float(settings["vignette_margin_ratio"]),
                "alpha_gamma": float(settings["alpha_gamma"]),
                "frame_shape": list(frames.shape),
            },
            indent=2,
        )
    )
    _report_progress(progress, 1.0, "Lip-sync frames prepared")


def bake_avatar(
    profile: str,
    video_path: Path,
    profile_type: str = PROFILE_TYPE_AVATAR,
    fps: float = 25.0,
    start_sec: float = 0.0,
    loop_sec: float = 20.0,
    loop_fade_sec: float = 0.15,
    resize_factor: int = 1,
    pads: tuple[int, int, int, int] = (0, 10, 0, 0),
    batch_size: int = 16,
    nosmooth: bool = False,
    blur_background: bool = True,
    blur_kernel: int = 75,
    face_crop_scale: float = 0.0,
    device: str = "cuda",
    progress: ProgressReporter | None = None,
) -> Path:
    cache_dir = avatar_cache_dir(profile, profile_type)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    reported_src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    # MediaRecorder WebM files commonly expose their millisecond time base as
    # an apparent 1000 FPS.  Treat implausible metadata as unknown; sampling
    # every 40th frame otherwise collapses a 20-second recording to a handful
    # of frames and makes the avatar loop visibly jumpy.
    src_fps = reported_src_fps if 5.0 <= reported_src_fps <= 120.0 else (fps or 25.0)
    fps = fps or src_fps or 25.0
    frame_interval = max(1, int(round(src_fps / fps))) if src_fps else 1
    
    start_frame = int(round(start_sec * src_fps))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    max_frames = int(loop_sec * fps) if loop_sec > 0 else None

    print(f"   Reading video: {video_path}")
    _report_progress(progress, 0.0, "Reading your source video")
    frames: list[np.ndarray] = []
    frame_index = start_frame
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    available_source_frames = max(0, source_frame_count - start_frame)
    expected_frames = math.ceil(available_source_frames / frame_interval) if available_source_frames else 0
    if max_frames:
        expected_frames = min(expected_frames, max_frames) if expected_frames else max_frames
    report_every = max(1, expected_frames // 24) if expected_frames else 24
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if (frame_index - start_frame) % frame_interval != 0:
            frame_index += 1
            continue
            
        if resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))
            
        frame = _center_crop_3_4(frame)
        frames.append(frame)
        frame_index += 1
        if len(frames) % report_every == 0:
            read_fraction = min(1.0, len(frames) / expected_frames) if expected_frames else 0.0
            _report_progress(progress, 0.08 * read_fraction, "Reading your source video")
        
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    if not frames:
        raise RuntimeError("No frames extracted from video.")
    _report_progress(progress, 0.08, "Source video read")

    # 1. Detect Faces
    print("   Detecting faces...")
    detector = _load_detector(device)
    coords_arr = []
    
    if detector:
        preds = []
        total_batches = max(1, math.ceil(len(frames) / max(1, batch_size)))
        for batch_number, i in enumerate(range(0, len(frames), batch_size), start=1):
            batch = np.array(frames[i : i + batch_size])
            preds.extend(detector.get_detections_for_batch(batch))
            _report_progress(
                progress,
                0.08 + 0.24 * (batch_number / total_batches),
                "Finding your face in each frame",
            )

        coords: list[list[int]] = []
        last_good = None
        top_pad, bottom_pad, left_pad, right_pad = pads
        
        for pred, frame in zip(preds, frames):
            if pred is None:
                if last_good is None:
                    h, w = frame.shape[:2]
                    last_good = [h//4, h//4*3, w//4, w//4*3] 
                coords.append(last_good)
                continue
            x1, y1, x2, y2 = pred[:4]
            y1 = max(0, int(y1) - top_pad)
            y2 = min(frame.shape[0], int(y2) + bottom_pad)
            x1 = max(0, int(x1) - left_pad)
            x2 = min(frame.shape[1], int(x2) + right_pad)
            box = [y1, y2, x1, x2]
            coords.append(box)
            last_good = box
            
        coords_arr = np.array(coords, dtype=np.int32)
        if not nosmooth:
            coords_arr = _smooth_boxes(coords_arr, window=5)
    else:
        print("   [WARN] No face detector. Using full frame.")
        h, w = frames[0].shape[:2]
        coords_arr = np.array([[0, h, 0, w]] * len(frames), dtype=np.int32)
    _report_progress(progress, 0.34, "Face framing prepared")

    # 2. Optional tight crop around face.
    # Default is disabled to preserve the original center 3:4 framing from uploaded video.
    if face_crop_scale and face_crop_scale > 1.0:
        frames, coords_arr = _tight_crop_to_face(frames, coords_arr, scale=face_crop_scale)
        if frames:
            base_h, base_w = frames[0].shape[:2]
            frames, coords_arr = _resize_frames_and_coords(frames, coords_arr, (base_w, base_h))

    # 3. Blur using Rembg
    if blur_background:
        frames = _blur_background_with_rembg(
            frames,
            blur_kernel,
            progress=lambda fraction, activity: _report_progress(
                progress,
                0.34 + 0.20 * fraction,
                activity,
            ),
        )
    else:
        _report_progress(progress, 0.54, "Background refinement skipped")

    # 4. Loop Crossfade
    if loop_fade_sec and fps:
        fade_frames = int(round(loop_fade_sec * fps))
        frames = _apply_loop_crossfade(frames, fade_frames)
    _report_progress(progress, 0.56, "Smoothing the video loop")

    np.save(cache_dir / "frames.npy", np.array(frames, dtype=np.uint8))
    np.save(cache_dir / "coords.npy", coords_arr)
    _report_progress(progress, 0.59, "Saving face frames")

    meta = {
        "profile": profile,
        "source_video": str(video_path),
        "fps": float(fps),
        "frame_count": len(frames),
        "center_crop_3_4": True,
        "tight_face_crop_enabled": bool(face_crop_scale and face_crop_scale > 1.0),
        "face_crop_scale": float(face_crop_scale),
        "width": int(frames[0].shape[1]),
        "height": int(frames[0].shape[0]),
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Build MuseTalk assets offline to keep runtime inference deterministic.
    try:
        bake_musetalk_assets(
            cache_dir=cache_dir,
            frames=np.array(frames, dtype=np.uint8),
            base_coords_y1y2x1x2=coords_arr,
            progress=lambda fraction, activity: _report_progress(
                progress,
                0.59 + 0.40 * fraction,
                activity,
            ),
        )
    except Exception as exc:
        print(f"[WARN] MuseTalk asset bake failed, runtime fallback will be used: {exc}")

    _report_progress(progress, 1.0, "Face frames prepared")
    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--profile_type", default=PROFILE_TYPE_AVATAR)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--loop_sec", type=float, default=20.0)
    parser.add_argument("--loop_fade_sec", type=float, default=0.15)
    parser.add_argument("--resize_factor", type=int, default=1)
    parser.add_argument("--pads", type=str, default="0 10 0 0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--nosmooth", action="store_true")
    
    # Flags for blurring
    parser.add_argument("--no_blur_background", action="store_true")
    parser.add_argument("--blur_kernel", type=int, default=55)
    parser.add_argument(
        "--face_crop_scale",
        type=float,
        default=0.0,
        help="Optional face-tight crop scale (>1 enables). Default 0 keeps center 3:4 crop.",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    pads = tuple(int(p) for p in args.pads.split())

    cache_dir = bake_avatar(
        profile=args.profile,
        video_path=args.video,
        profile_type=args.profile_type,
        fps=args.fps,
        start_sec=args.start_sec,
        loop_sec=args.loop_sec,
        loop_fade_sec=args.loop_fade_sec,
        resize_factor=args.resize_factor,
        pads=pads,
        batch_size=args.batch_size,
        nosmooth=args.nosmooth,
        blur_background=not args.no_blur_background,
        blur_kernel=args.blur_kernel,
        face_crop_scale=args.face_crop_scale,
        device=args.device,
    )
    print(f"Avatar successfully cached at: {cache_dir}")

if __name__ == "__main__":
    main()
