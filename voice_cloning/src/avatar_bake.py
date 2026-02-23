import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

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
def _blur_background_with_rembg(frames: list[np.ndarray], blur_kernel: int) -> list[np.ndarray]:
    if not REMBG_AVAILABLE:
        print("\n[WARNING] 'rembg' library not found.")
        print("   -> Skipping blur. To fix: pip install rembg\n")
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


def bake_avatar(
    profile: str,
    video_path: Path,
    profile_type: str = PROFILE_TYPE_AVATAR,
    fps: float = 25.0,
    start_sec: float = 0.0,
    loop_sec: float = 10.0,
    loop_fade_sec: float = 0.0,
    resize_factor: int = 1,
    pads: tuple[int, int, int, int] = (0, 10, 0, 0),
    batch_size: int = 16,
    nosmooth: bool = False,
    blur_background: bool = True,
    blur_kernel: int = 75,
    face_crop_scale: float = 0.0,
    device: str = "cuda",
) -> Path:
    cache_dir = avatar_cache_dir(profile, profile_type)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    fps = fps or src_fps or 25.0
    frame_interval = max(1, int(round(src_fps / fps))) if src_fps else 1
    
    start_frame = int(round(start_sec * src_fps))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    max_frames = int(loop_sec * fps) if loop_sec > 0 else None

    print(f"   Reading video: {video_path}")
    frames: list[np.ndarray] = []
    frame_index = start_frame
    
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
        
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    if not frames:
        raise RuntimeError("No frames extracted from video.")

    # 1. Detect Faces
    print("   Detecting faces...")
    detector = _load_detector(device)
    coords_arr = []
    
    if detector:
        preds = []
        for i in range(0, len(frames), batch_size):
            batch = np.array(frames[i : i + batch_size])
            preds.extend(detector.get_detections_for_batch(batch))

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

    # 2. Optional tight crop around face.
    # Default is disabled to preserve the original center 3:4 framing from uploaded video.
    if face_crop_scale and face_crop_scale > 1.0:
        frames, coords_arr = _tight_crop_to_face(frames, coords_arr, scale=face_crop_scale)
        if frames:
            base_h, base_w = frames[0].shape[:2]
            frames, coords_arr = _resize_frames_and_coords(frames, coords_arr, (base_w, base_h))

    # 3. Blur using Rembg
    if blur_background:
        frames = _blur_background_with_rembg(frames, blur_kernel)

    # 4. Loop Crossfade
    if loop_fade_sec and fps:
        fade_frames = int(round(loop_fade_sec * fps))
        frames = _apply_loop_crossfade(frames, fade_frames)

    np.save(cache_dir / "frames.npy", np.array(frames, dtype=np.uint8))
    np.save(cache_dir / "coords.npy", coords_arr)

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
    return cache_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--profile_type", default=PROFILE_TYPE_AVATAR)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--loop_sec", type=float, default=10.0)
    parser.add_argument("--loop_fade_sec", type=float, default=0.0)
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
    print(f"✅ Avatar successfully cached at: {cache_dir}")

if __name__ == "__main__":
    main()
