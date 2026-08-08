from __future__ import annotations

import hashlib
import gc
import json
import logging
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor, WhisperModel

from config import LIP_SYNCING_DIR, PROFILE_TYPE_AVATAR, avatar_cache_dir

logger = logging.getLogger("pixelholo.musetalk")


def _odd_kernel(size: int) -> int:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return size


def _env_int(name: str, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "component=musetalk op=env_config status=invalid name=%s value=%s default=%s",
            name,
            raw_value,
            default,
        )
        return default
    value = max(minimum, value)
    if maximum is not None:
        value = min(value, maximum)
    return value


class MuseTalkBridge:
    def __init__(
        self,
        device: str | None = None,
        batch_size: int | None = None,
        parsing_mode: str | None = None,
        audio_padding_left: int | None = None,
        audio_padding_right: int | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.batch_size = int(batch_size or os.getenv("MUSE_TALK_BATCH_SIZE", "24"))
        # CUDA selects and compiles kernels by tensor shape. A final short
        # batch (for example 18 frames after several 24-frame batches) can
        # therefore incur a multi-second first-use stall even though the
        # renderer is otherwise warm. Pad only that final execution batch and
        # discard its duplicated outputs so every live request uses the same
        # shape as the prepared 24-frame path.
        self.static_batch_padding = os.getenv("MUSE_TALK_STATIC_BATCH_PADDING", "1") != "0"
        # Keep the replacement mask to MuseTalk's native face/lip segmentation.
        # The custom `jaw` dilation can overwrite the chin with the model's
        # lower-resolution reconstruction, which reads as a blurry jawline.
        self.parsing_mode = parsing_mode or os.getenv("MUSE_TALK_PARSING_MODE", "raw")
        self.audio_padding_left = int(
            audio_padding_left or os.getenv("MUSE_TALK_AUDIO_PADDING_LEFT", "2")
        )
        self.audio_padding_right = int(
            audio_padding_right or os.getenv("MUSE_TALK_AUDIO_PADDING_RIGHT", "2")
        )

        self.frame_idx = 0
        self.frame_accumulator = 0.0
        self.infer_frame_accumulator = 0.0
        self.fps = 25.0
        self.audio_history = np.array([], dtype=np.float32)
        self.default_audio_history_sec = float(os.getenv("MUSE_TALK_AUDIO_HISTORY_SEC", "2.0"))
        self.audio_history_sec = self.default_audio_history_sec
        # Keep default infer fps equal to output/avatar fps to avoid apparent slow-motion
        # (low infer fps + frame upsampling reduces base-frame progression speed).
        self.default_infer_fps = float(os.getenv("MUSE_TALK_INFER_FPS", "25.0"))
        self.infer_fps = self.default_infer_fps
        # MuseTalk conditions on a full-face crop.  The baked lower-face track
        # is useful for controlled diagnostics, but it can reduce the amount
        # of generated mouth motion.  Keep full-face conditioning as the
        # public default and address chin stability at compositing time.
        self.default_coord_source = self._normalize_coord_source(
            os.getenv("MUSE_TALK_COORD_SOURCE", "legacy")
        )
        self.coord_source = self.default_coord_source

        # Keep MuseTalk box expansion conservative. This avoids chin cut lines without
        # shifting the mouth target area too far from the model's expected region.
        self.coord_expand_x = float(os.getenv("MUSE_TALK_COORD_EXPAND_X", "0.08"))
        self.coord_expand_up = float(os.getenv("MUSE_TALK_COORD_EXPAND_UP", "0.04"))
        self.coord_expand_down = float(os.getenv("MUSE_TALK_COORD_EXPAND_DOWN", "0.18"))

        # Blend settings tuned to avoid visible rectangular seams while keeping mouth core opaque.
        self.upper_boundary_ratio = float(os.getenv("MUSE_TALK_UPPER_BOUNDARY_RATIO", "0.5"))
        self.blend_expand = float(os.getenv("MUSE_TALK_BLEND_EXPAND", "1.2"))
        # Slightly shrink generated face patch inside the target box to avoid oversized mouth appearance.
        self.default_face_scale = float(os.getenv("MUSE_TALK_FACE_SCALE", "0.96"))
        self.face_scale = self.default_face_scale
        self.alpha_blur_ratio = float(os.getenv("MUSE_TALK_ALPHA_BLUR_RATIO", "0.035"))
        self.vignette_margin_ratio = float(os.getenv("MUSE_TALK_VIGNETTE_MARGIN_RATIO", "0.02"))
        self.alpha_gamma = float(os.getenv("MUSE_TALK_ALPHA_GAMMA", "0.82"))
        # MuseTalk reconstructs the mouth at 256px and the result is then expanded
        # into the portrait frame.  A moderate luminance-only sharpen restores some
        # lip/teeth edge definition without introducing ringing around the mouth.
        self.default_detail_sharpen = float(os.getenv("MUSE_TALK_DETAIL_SHARPEN", "0.70"))
        self.detail_sharpen = self.default_detail_sharpen
        # Preserve the complete generated mouth, including the lower lip,
        # while fading back to the source before the bottom of the chin.  A
        # boundary near 0.65 crosses the mouth on normal full-face crops.  It
        # leaves the source lower lip visible underneath MuseTalk's upper lip,
        # so the displayed articulation follows the reference recording
        # instead of the generated audio.  The stable runtime coordinate track
        # below prevents the old stream-window chin jump without cutting the
        # generated viseme in half.
        self.default_mouth_mask_bottom_ratio = float(
            os.getenv("MUSE_TALK_MOUTH_MASK_BOTTOM_RATIO", "0.84")
        )
        self.mouth_mask_bottom_ratio = self.default_mouth_mask_bottom_ratio
        self.mouth_mask_bottom_feather = float(
            os.getenv("MUSE_TALK_MOUTH_MASK_BOTTOM_FEATHER", "0.08")
        )
        self.color_match_strength = float(os.getenv("MUSE_TALK_COLOR_MATCH_STRENGTH", "0.65"))
        # Version 10 is the cache contract shared with avatar_bake.py.  Earlier
        # prepared assets did not have the runtime manifest, so the first
        # response rebuilt them despite preprocessing having completed.
        self.cache_version = int(os.getenv("MUSE_TALK_CACHE_VERSION", "10"))
        self.mask_stabilize = os.getenv("MUSE_TALK_MASK_STABILIZE", "1") != "0"
        self.mask_stabilize_window = int(os.getenv("MUSE_TALK_MASK_STABILIZE_WINDOW", "5"))
        # Excess temporal averaging softens lip edges and teeth.  A small
        # amount removes flicker without turning moving mouths into a blur.
        self.default_temporal_smooth = float(os.getenv("MUSE_TALK_TEMPORAL_SMOOTH", "0.025"))
        self.temporal_smooth = self.default_temporal_smooth
        # MuseTalk uses a sliding audio window, which can turn quiet audio into
        # tiny lip movement. Hold the last rendered portrait for sustained
        # silence so natural conversational pauses are genuinely still.
        self.silence_rms_threshold = float(
            # Keep this deliberately conservative.  A short low-energy
            # consonant or word boundary is still speech and needs MuseTalk's
            # viseme.  The earlier 120 ms / 0.006 rule held regular speech
            # frames, which made the displayed mouth fall behind the audio.
            np.clip(float(os.getenv("MUSE_TALK_SILENCE_RMS_THRESHOLD", "0.004")), 0.0005, 0.05)
        )
        self.silence_min_duration_ms = int(
            np.clip(int(os.getenv("MUSE_TALK_SILENCE_MIN_DURATION_MS", "280")), 40, 800)
        )
        # Do not replace generated visemes with held frames in the production
        # path.  Even a carefully tuned RMS gate cannot distinguish every
        # quiet phoneme and word boundary from a conversational pause.  This
        # was added after the original working stream and is therefore opt-in
        # for controlled diagnostics only.
        self.suppress_silence_motion = os.getenv(
            "MUSE_TALK_SUPPRESS_SILENCE_MOTION", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.default_runtime_max_frame_edge = _env_int(
            "MUSE_TALK_RUNTIME_MAX_FRAME_EDGE",
            0,
            minimum=0,
            maximum=1280,
        )
        self.runtime_max_frame_edge = self.default_runtime_max_frame_edge

        self.frames: np.ndarray | None = None
        self.coords_xyxy: np.ndarray | None = None
        self._latents: list[torch.Tensor] = []
        self._mask_arrays: list[np.ndarray] = []
        self._mask_crop_boxes: list[list[int]] = []
        self._blend_alphas: list[np.ndarray] = []
        self._loaded_cache_dir: Path | None = None
        self._loaded_coord_source: str | None = None
        self._loaded_runtime_max_frame_edge = 0
        self._loaded_cache_signature: tuple[tuple[str, int, int] | None, ...] | None = None
        self._coord_sha1: str | None = None
        self._vignette_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
        self._prev_generated_face: np.ndarray | None = None
        # The source clip is naturally moving.  Re-inserting a source frame
        # during a quiet region can therefore make the jaw jump between the
        # generated and source geometry.  Keep the last composited portrait
        # so a genuine conversational pause is visually still.
        self._last_rendered_frame: np.ndarray | None = None

        self.musetalk_dir = LIP_SYNCING_DIR / "lib" / "MuseTalk"
        if not self.musetalk_dir.exists():
            raise FileNotFoundError(f"MuseTalk repo not found at {self.musetalk_dir}")
        if str(self.musetalk_dir) not in sys.path:
            sys.path.insert(0, str(self.musetalk_dir))

        from musetalk.models.unet import PositionalEncoding, UNet
        from musetalk.models.vae import VAE
        from musetalk.utils.blending import get_image_prepare_material
        from musetalk.utils.face_parsing import FaceParsing
        from musetalk.utils.utils import datagen

        models_dir = Path(os.getenv("MUSE_TALK_MODELS_DIR", str(self.musetalk_dir / "models")))
        unet_config = Path(
            os.getenv("MUSE_TALK_UNET_CONFIG", str(models_dir / "musetalkV15" / "musetalk.json"))
        )
        unet_model = Path(
            os.getenv("MUSE_TALK_UNET_MODEL", str(models_dir / "musetalkV15" / "unet.pth"))
        )
        vae_dir = Path(os.getenv("MUSE_TALK_VAE_DIR", str(models_dir / "sd-vae")))
        whisper_dir = Path(os.getenv("MUSE_TALK_WHISPER_DIR", str(models_dir / "whisper")))
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
        for path in (unet_config, unet_model, vae_dir, whisper_dir, face_parse_model, face_parse_resnet):
            if not path.exists():
                raise FileNotFoundError(f"MuseTalk asset missing: {path}")

        self._datagen = datagen
        self._get_image_prepare_material = get_image_prepare_material

        self.vae = VAE(model_path=str(vae_dir))
        self.unet = UNet(
            unet_config=str(unet_config),
            model_path=str(unet_model),
            device=torch.device(self.device),
        )
        self.pe = PositionalEncoding(d_model=384).to(self.device)

        use_fp16 = os.getenv("MUSE_TALK_FP16", "1") == "1" and self.device.startswith("cuda")
        if use_fp16:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()

        self.weight_dtype = self.unet.model.dtype
        self.timesteps = torch.tensor([0], device=self.device)

        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(str(whisper_dir))
        self.whisper = WhisperModel.from_pretrained(str(whisper_dir))
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        class _FaceParsingWithPaths(FaceParsing):
            def __init__(
                self,
                resnet_path: Path,
                model_path: Path,
                left_cheek_width: int = 80,
                right_cheek_width: int = 80,
            ) -> None:
                self._resnet_path = str(resnet_path)
                self._model_path = str(model_path)
                super().__init__(
                    left_cheek_width=left_cheek_width,
                    right_cheek_width=right_cheek_width,
                )

            def model_init(self, resnet_path=None, model_pth=None):
                return super().model_init(
                    resnet_path=self._resnet_path,
                    model_pth=self._model_path,
                )

        self.face_parser = _FaceParsingWithPaths(face_parse_resnet, face_parse_model)

    @staticmethod
    def _normalize_coord_source(value: str | None) -> str:
        source = (value or "legacy").strip().lower().replace("-", "_")
        aliases = {
            "default": "legacy",
            "coords": "legacy",
            "runtime": "legacy",
            "musetalk": "baked",
            "mt": "baked",
            "musetalk_coords": "baked",
            "landmark": "baked",
            "landmark_stabilized": "baked",
        }
        source = aliases.get(source, source)
        if source not in {"legacy", "baked", "auto"}:
            logger.warning(
                "component=musetalk op=coord_source status=invalid value=%s fallback=legacy",
                value,
            )
            return "legacy"
        return source

    def configure_for_request(
        self,
        *,
        preset: str | None = None,
        coord_source: str | None = None,
        face_scale: float | None = None,
        temporal_smooth: float | None = None,
        detail_sharpen: float | None = None,
        mouth_mask_bottom_ratio: float | None = None,
        infer_fps: float | None = None,
        audio_history_sec: float | None = None,
    ) -> None:
        """Reset request-local rendering settings before a profile is loaded.

        The engine is shared by profiles, so stale values from the previous
        request must never affect a new avatar.  The preset controls only
        compositing behaviour; it does not change the underlying MuseTalk
        checkpoint or add a second inference pass.
        """
        preset_name = (preset or "realistic").strip().lower().replace("-", "_")
        preset_defaults = {
            "realistic": ("legacy", 0.025, 0.96, 0.70),
            "balanced": ("legacy", 0.045, 0.97, 0.64),
            "low_latency": ("legacy", 0.0, 0.98, 0.60),
            "stable": ("legacy", 0.09, 0.96, 0.62),
        }
        preset_coord_source, default_smooth, default_scale, default_sharpen = preset_defaults.get(
            preset_name,
            preset_defaults["realistic"],
        )
        # A request-level source is intentionally supported for controlled
        # quality evaluation.  It lets the evaluator compare the same profile,
        # prompt, and checkpoint with the only variable being the face track.
        # The server setting remains the production-wide override.
        requested_coord_source = coord_source or os.getenv(
            "MUSE_TALK_COORD_SOURCE",
            preset_coord_source,
        )
        self.coord_source = self._normalize_coord_source(requested_coord_source)
        self.temporal_smooth = float(
            np.clip(
                self.default_temporal_smooth if preset_name == "realistic" else default_smooth,
                0.0,
                0.35,
            )
        )
        self.face_scale = float(
            np.clip(
                self.default_face_scale if preset_name == "realistic" else default_scale,
                0.75,
                1.15,
            )
        )
        self.detail_sharpen = float(
            np.clip(
                self.default_detail_sharpen if preset_name == "realistic" else default_sharpen,
                0.0,
                0.95,
            )
        )
        self.mouth_mask_bottom_ratio = self.default_mouth_mask_bottom_ratio
        self.infer_fps = self.default_infer_fps
        self.audio_history_sec = self.default_audio_history_sec

        if face_scale is not None:
            self.face_scale = float(np.clip(float(face_scale), 0.75, 1.15))
        if temporal_smooth is not None:
            self.temporal_smooth = float(np.clip(float(temporal_smooth), 0.0, 0.35))
        if detail_sharpen is not None:
            self.detail_sharpen = float(np.clip(float(detail_sharpen), 0.0, 0.95))
        if mouth_mask_bottom_ratio is not None:
            self.mouth_mask_bottom_ratio = float(
                np.clip(float(mouth_mask_bottom_ratio), 0.45, 0.85)
            )
        if infer_fps is not None:
            self.infer_fps = float(np.clip(float(infer_fps), 6.0, 30.0))
        if audio_history_sec is not None:
            self.audio_history_sec = float(np.clip(float(audio_history_sec), 0.5, 6.0))

    @staticmethod
    def _coords_sha1(coords: np.ndarray) -> str:
        return hashlib.sha1(coords.astype(np.int32, copy=False).tobytes()).hexdigest()

    @staticmethod
    def _coord_valid_ratio(coords: np.ndarray, width: int, height: int, fmt: str) -> float:
        if coords.size == 0:
            return 0.0
        arr = coords.astype(np.int32, copy=False)
        if fmt == "y1y2x1x2":
            y1, y2, x1, x2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        else:
            x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        valid = (
            (x2 > x1)
            & (y2 > y1)
            & (x1 >= 0)
            & (y1 >= 0)
            & (x2 <= width)
            & (y2 <= height)
        )
        return float(valid.mean())

    @classmethod
    def _detect_coord_format(cls, coords: np.ndarray, width: int, height: int) -> str:
        yx = cls._coord_valid_ratio(coords, width, height, "y1y2x1x2")
        xy = cls._coord_valid_ratio(coords, width, height, "xyxy")
        return "y1y2x1x2" if yx >= xy else "xyxy"

    @staticmethod
    def _sanitize_xyxy(coord: np.ndarray | list[int], width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(v) for v in coord]
        x1 = max(0, min(x1, width - 1))
        x2 = max(1, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(1, min(y2, height))
        if x2 <= x1:
            x1, x2 = 0, width
        if y2 <= y1:
            y1, y2 = 0, height
        return x1, y1, x2, y2

    def _convert_coords_to_xyxy(
        self,
        coords: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, str]:
        fmt = self._detect_coord_format(coords, width, height)
        arr = coords.astype(np.int32, copy=False)
        if fmt == "y1y2x1x2":
            y1, y2, x1, x2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            xyxy = np.stack([x1, y1, x2, y2], axis=1)
        else:
            xyxy = arr.copy()
        clean = np.array(
            [self._sanitize_xyxy(c, width, height) for c in xyxy],
            dtype=np.int32,
        )
        return clean, fmt

    def _expand_coord(self, coord: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = coord
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(round(bw * self.coord_expand_x))
        pad_up = int(round(bh * self.coord_expand_up))
        pad_down = int(round(bh * self.coord_expand_down))
        nx1 = max(0, x1 - pad_x)
        nx2 = min(width, x2 + pad_x)
        ny1 = max(0, y1 - pad_up)
        ny2 = min(height, y2 + pad_down)
        if nx2 <= nx1:
            nx1, nx2 = x1, x2
        if ny2 <= ny1:
            ny1, ny2 = y1, y2
        return nx1, ny1, nx2, ny2

    @classmethod
    def _smooth_xyxy_boxes(
        cls,
        boxes: np.ndarray,
        width: int,
        height: int,
        window: int,
    ) -> np.ndarray:
        if boxes.size == 0:
            return boxes.astype(np.int32, copy=True)
        window = max(1, int(window))
        if len(boxes) < 3 or window <= 1:
            return boxes.astype(np.int32, copy=True)

        arr = boxes.astype(np.float32, copy=False)
        centers_sizes = np.column_stack(
            (
                (arr[:, 0] + arr[:, 2]) * 0.5,
                (arr[:, 1] + arr[:, 3]) * 0.5,
                np.maximum(1.0, arr[:, 2] - arr[:, 0]),
                np.maximum(1.0, arr[:, 3] - arr[:, 1]),
            )
        )
        radius = window // 2
        padded = np.pad(centers_sizes, ((radius, radius), (0, 0)), mode="edge")
        smoothed = np.empty_like(centers_sizes)
        for idx in range(len(smoothed)):
            smoothed[idx] = padded[idx : idx + window].mean(axis=0)

        out: list[tuple[int, int, int, int]] = []
        for cx, cy, bw, bh in smoothed:
            x1 = int(round(cx - bw * 0.5))
            y1 = int(round(cy - bh * 0.5))
            x2 = int(round(cx + bw * 0.5))
            y2 = int(round(cy + bh * 0.5))
            out.append(cls._sanitize_xyxy([x1, y1, x2, y2], width, height))
        return np.array(out, dtype=np.int32)

    @staticmethod
    def _remap_mask_to_crop(
        mask: np.ndarray,
        old_box: list[int] | np.ndarray,
        new_box: list[int] | np.ndarray,
    ) -> np.ndarray:
        ox1, oy1, ox2, oy2 = [int(v) for v in old_box]
        nx1, ny1, nx2, ny2 = [int(v) for v in new_box]
        new_w = max(1, nx2 - nx1)
        new_h = max(1, ny2 - ny1)
        canvas = np.zeros((new_h, new_w), dtype=np.uint8)
        if ox2 <= ox1 or oy2 <= oy1:
            return canvas

        mask_arr = np.asarray(mask)
        if mask_arr.ndim != 2:
            mask_arr = np.squeeze(mask_arr)
        old_w = max(1, ox2 - ox1)
        old_h = max(1, oy2 - oy1)
        if mask_arr.shape[:2] != (old_h, old_w):
            mask_arr = cv2.resize(mask_arr.astype(np.uint8), (old_w, old_h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_arr = mask_arr.astype(np.uint8, copy=False)

        ix1 = max(ox1, nx1)
        iy1 = max(oy1, ny1)
        ix2 = min(ox2, nx2)
        iy2 = min(oy2, ny2)
        if ix2 <= ix1 or iy2 <= iy1:
            return canvas

        src_x1 = ix1 - ox1
        src_y1 = iy1 - oy1
        src_x2 = ix2 - ox1
        src_y2 = iy2 - oy1
        dst_x1 = ix1 - nx1
        dst_y1 = iy1 - ny1
        dst_x2 = ix2 - nx1
        dst_y2 = iy2 - ny1
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = mask_arr[src_y1:src_y2, src_x1:src_x2]
        return canvas

    def _stabilize_mask_sequence(
        self,
        mask_arrays: list[np.ndarray],
        mask_crop_boxes: list[list[int]],
        frame_w: int,
        frame_h: int,
    ) -> tuple[list[np.ndarray], list[list[int]]]:
        if not self.mask_stabilize or len(mask_arrays) < 3 or len(mask_arrays) != len(mask_crop_boxes):
            return mask_arrays, mask_crop_boxes

        old_boxes = np.array(mask_crop_boxes, dtype=np.int32)
        new_boxes = self._smooth_xyxy_boxes(
            old_boxes,
            frame_w,
            frame_h,
            self.mask_stabilize_window,
        )
        remapped_masks = [
            self._remap_mask_to_crop(mask, old_box, new_box)
            for mask, old_box, new_box in zip(mask_arrays, old_boxes, new_boxes)
        ]
        return remapped_masks, [[int(v) for v in box] for box in new_boxes.tolist()]

    @staticmethod
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

    def _build_vignette(self, h: int, w: int) -> np.ndarray:
        ratio = max(0.0, self.vignette_margin_ratio)
        if ratio <= 0:
            return np.ones((h, w), dtype=np.float32)
        margin_y = int(round(h * ratio))
        margin_x = int(round(w * ratio))
        margin_y = max(1, min(margin_y, h // 2))
        margin_x = max(1, min(margin_x, w // 2))
        key = (h, w, margin_y, margin_x)
        cached = self._vignette_cache.get(key)
        if cached is not None:
            return cached

        ramp_y = np.ones(h, dtype=np.float32)
        ramp_x = np.ones(w, dtype=np.float32)
        edge_y = np.linspace(0.0, 1.0, margin_y, endpoint=False, dtype=np.float32)
        edge_x = np.linspace(0.0, 1.0, margin_x, endpoint=False, dtype=np.float32)
        ramp_y[:margin_y] = edge_y
        ramp_y[-margin_y:] = edge_y[::-1]
        ramp_x[:margin_x] = edge_x
        ramp_x[-margin_x:] = edge_x[::-1]
        vignette = np.outer(ramp_y, ramp_x).astype(np.float32)
        self._vignette_cache[key] = vignette
        return vignette

    def _prepare_alpha(self, mask: np.ndarray) -> np.ndarray:
        alpha = np.asarray(mask, dtype=np.float32) / 255.0
        h, w = alpha.shape[:2]
        blur = _odd_kernel(int(round(min(h, w) * self.alpha_blur_ratio)))
        if blur > 1:
            alpha = cv2.GaussianBlur(alpha, (blur, blur), 0)
        gamma = max(0.6, min(1.4, float(self.alpha_gamma)))
        if abs(gamma - 1.0) > 1e-3:
            alpha = np.power(np.clip(alpha, 0.0, 1.0), gamma)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha *= self._build_vignette(h, w)
        alpha[alpha < 0.01] = 0.0
        alpha[0, :] = 0.0
        alpha[-1, :] = 0.0
        alpha[:, 0] = 0.0
        alpha[:, -1] = 0.0
        return np.clip(alpha, 0.0, 1.0)

    @staticmethod
    def _match_mean_color(target: np.ndarray, source: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if target.size == 0 or source.size == 0:
            return source
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 1e-6:
            return source

        t_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        s_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)

        out_lab = s_lab.copy()
        for channel in range(3):
            t_ch = t_lab[:, :, channel]
            s_ch = s_lab[:, :, channel]
            t_mean = float(np.mean(t_ch))
            s_mean = float(np.mean(s_ch))
            t_std = float(np.std(t_ch)) + 1e-5
            s_std = float(np.std(s_ch)) + 1e-5
            gain = float(np.clip(t_std / s_std, 0.7, 1.3))
            mapped = (s_ch - s_mean) * gain + t_mean
            out_lab[:, :, channel] = mapped

        corrected = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        if strength >= 0.999:
            return corrected
        mixed = cv2.addWeighted(corrected, strength, source, 1.0 - strength, 0.0)
        return np.clip(mixed, 0, 255).astype(np.uint8)

    @staticmethod
    def _adaptive_unsharp_luma(image_bgr: np.ndarray, amount: float) -> np.ndarray:
        amount = float(np.clip(amount, 0.0, 0.8))
        if amount <= 1e-4:
            return image_bgr
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_f = l.astype(np.float32)
        blur = cv2.GaussianBlur(l_f, (0, 0), 1.05)
        sharp = cv2.addWeighted(l_f, 1.0 + amount, blur, -amount, 0.0)
        l_out = np.clip(sharp, 0, 255).astype(np.uint8)
        out = cv2.merge((l_out, a, b))
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    def _restrict_chin_blend(
        self,
        alpha: np.ndarray,
        face_box: tuple[int, int, int, int],
        crop_box: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Fade the generated face before it reaches the source chin."""
        if alpha.size == 0:
            return alpha
        _x1, y1, _x2, y2 = face_box
        _cx1, cy1, _cx2, _cy2 = crop_box
        face_height = max(1, int(y2 - y1))
        bottom_ratio = float(np.clip(self.mouth_mask_bottom_ratio, 0.65, 1.05))
        feather_ratio = float(np.clip(self.mouth_mask_bottom_feather, 0.01, 0.25))
        bottom = float(y1 - cy1) + face_height * bottom_ratio
        feather = max(2.0, face_height * feather_ratio)
        rows = np.arange(alpha.shape[0], dtype=np.float32)
        gate = np.ones_like(rows)
        fade_start = bottom - feather
        fade_end = bottom + feather
        gate[rows >= fade_end] = 0.0
        fading = (rows > fade_start) & (rows < fade_end)
        gate[fading] = (fade_end - rows[fading]) / max(1.0, fade_end - fade_start)
        return np.clip(alpha * gate[:, None], 0.0, 1.0)

    def _cache_manifest(
        self,
        coords_sha1: str,
        frame_shape: tuple[int, ...],
    ) -> dict[str, object]:
        return {
            "cache_version": self.cache_version,
            "coord_format": "xyxy",
            "coords_sha1": coords_sha1,
            "parsing_mode": self.parsing_mode,
            "coord_expand_x": self.coord_expand_x,
            "coord_expand_up": self.coord_expand_up,
            "coord_expand_down": self.coord_expand_down,
            "upper_boundary_ratio": self.upper_boundary_ratio,
            "blend_expand": self.blend_expand,
            "alpha_blur_ratio": self.alpha_blur_ratio,
            "vignette_margin_ratio": self.vignette_margin_ratio,
            "alpha_gamma": self.alpha_gamma,
            "frame_shape": list(frame_shape),
        }

    def _runtime_edge(self, frame_w: int | None = None, frame_h: int | None = None) -> int:
        edge = max(0, min(int(self.runtime_max_frame_edge or 0), 1280))
        if edge <= 0 or frame_w is None or frame_h is None:
            return edge
        if max(frame_w, frame_h) <= edge:
            return 0
        return edge

    @staticmethod
    def _runtime_cache_meta(
        frames_path: Path,
        coords_path: Path,
        coord_source: str,
        max_frame_edge: int,
    ) -> dict[str, object]:
        frames_stat = frames_path.stat()
        coords_stat = coords_path.stat()
        return {
            "cache_version": 1,
            "source_frames": frames_path.name,
            "source_frames_size": frames_stat.st_size,
            "source_frames_mtime_ns": frames_stat.st_mtime_ns,
            "source_coords": coords_path.name,
            "source_coords_size": coords_stat.st_size,
            "source_coords_mtime_ns": coords_stat.st_mtime_ns,
            "coord_source": coord_source,
            "max_frame_edge": int(max_frame_edge),
        }

    @staticmethod
    def _scale_coords(coords_xyxy: np.ndarray, scale: float, frame_w: int, frame_h: int) -> np.ndarray:
        scaled = np.rint(coords_xyxy.astype(np.float32, copy=False) * float(scale)).astype(np.int32)
        scaled[:, 0] = np.clip(scaled[:, 0], 0, max(0, frame_w - 1))
        scaled[:, 1] = np.clip(scaled[:, 1], 0, max(0, frame_h - 1))
        scaled[:, 2] = np.clip(scaled[:, 2], 1, frame_w)
        scaled[:, 3] = np.clip(scaled[:, 3], 1, frame_h)
        too_narrow = scaled[:, 2] <= scaled[:, 0]
        too_short = scaled[:, 3] <= scaled[:, 1]
        scaled[too_narrow, 2] = np.minimum(frame_w, scaled[too_narrow, 0] + 1)
        scaled[too_short, 3] = np.minimum(frame_h, scaled[too_short, 1] + 1)
        return scaled

    def _load_or_build_runtime_frame_cache(
        self,
        cache_dir: Path,
        frames_path: Path,
        coords_path: Path,
        raw_coords: np.ndarray,
        coord_source: str,
        max_frame_edge: int,
    ) -> tuple[np.ndarray, np.ndarray, str, int]:
        original_frames = np.load(frames_path, mmap_mode="r")
        if original_frames.ndim != 4 or len(original_frames) == 0:
            raise ValueError(f"Invalid avatar cache layout in {cache_dir}")
        frame_h, frame_w = original_frames.shape[1:3]
        coords_xyxy, source_fmt = self._convert_coords_to_xyxy(raw_coords, frame_w, frame_h)
        previous_edge = self.runtime_max_frame_edge
        self.runtime_max_frame_edge = max_frame_edge
        try:
            runtime_edge = self._runtime_edge(frame_w, frame_h)
        finally:
            self.runtime_max_frame_edge = previous_edge
        if runtime_edge <= 0:
            return np.asarray(original_frames), coords_xyxy, source_fmt, 0

        scale = float(runtime_edge) / float(max(frame_w, frame_h))
        runtime_w = max(1, int(round(frame_w * scale)))
        runtime_h = max(1, int(round(frame_h * scale)))
        suffix = f"{coord_source}_edge{runtime_edge}"
        runtime_frames_path = cache_dir / f"musetalk_frames_{suffix}.npy"
        runtime_coords_path = cache_dir / f"musetalk_coords_xyxy_{suffix}.npy"
        runtime_meta_path = cache_dir / f"musetalk_frames_{suffix}.json"
        expected_meta = self._runtime_cache_meta(frames_path, coords_path, coord_source, runtime_edge)
        expected_meta.update(
            {
                "source_shape": list(original_frames.shape),
                "runtime_shape": [len(original_frames), runtime_h, runtime_w, original_frames.shape[3]],
                "scale": round(scale, 8),
                "source_coord_format": source_fmt,
            }
        )

        if runtime_frames_path.exists() and runtime_coords_path.exists() and runtime_meta_path.exists():
            try:
                runtime_meta = json.loads(runtime_meta_path.read_text())
                if all(runtime_meta.get(key) == value for key, value in expected_meta.items()):
                    frames = np.load(runtime_frames_path)
                    coords = np.load(runtime_coords_path).astype(np.int32, copy=False)
                    if (
                        frames.ndim == 4
                        and coords.ndim == 2
                        and len(frames) == len(original_frames)
                        and frames.shape[1] == runtime_h
                        and frames.shape[2] == runtime_w
                    ):
                        return frames, coords, source_fmt, runtime_edge
            except Exception:
                logger.debug(
                    "component=musetalk op=runtime_frame_cache status=load_error path=%s",
                    runtime_frames_path,
                    exc_info=True,
                )

        logger.info(
            "component=musetalk op=runtime_frame_cache status=build edge=%s source_shape=%s runtime_shape=%s",
            runtime_edge,
            tuple(original_frames.shape),
            (len(original_frames), runtime_h, runtime_w, original_frames.shape[3]),
        )
        resized_frames = np.empty(
            (len(original_frames), runtime_h, runtime_w, original_frames.shape[3]),
            dtype=np.uint8,
        )
        for idx, frame in enumerate(original_frames):
            resized_frames[idx] = cv2.resize(frame, (runtime_w, runtime_h), interpolation=cv2.INTER_AREA)
        scaled_coords = self._scale_coords(coords_xyxy, scale, runtime_w, runtime_h)
        np.save(runtime_frames_path, resized_frames)
        np.save(runtime_coords_path, scaled_coords)
        runtime_meta_path.write_text(json.dumps(expected_meta, indent=2))
        return resized_frames, scaled_coords, source_fmt, runtime_edge

    def _load_cached_assets(
        self,
        latents_path: Path,
        masks_path: Path,
        runtime_meta_path: Path,
        frames: np.ndarray,
        coords_xyxy: np.ndarray,
    ) -> tuple[list[torch.Tensor], list[np.ndarray], list[list[int]]] | None:
        if not (latents_path.exists() and masks_path.exists() and runtime_meta_path.exists()):
            return None
        try:
            runtime_meta = json.loads(runtime_meta_path.read_text())
            expected = self._cache_manifest(self._coords_sha1(coords_xyxy), tuple(frames.shape))
            for key, value in expected.items():
                if runtime_meta.get(key) != value:
                    return None

            loaded_latents = torch.load(latents_path, map_location="cpu")
            with masks_path.open("rb") as handle:
                payload = pickle.load(handle)
            if not (
                isinstance(loaded_latents, list)
                and isinstance(payload, dict)
                and isinstance(payload.get("mask_arrays"), list)
                and isinstance(payload.get("mask_crop_boxes"), list)
                and len(loaded_latents) == len(frames)
                and len(payload["mask_arrays"]) == len(frames)
                and len(payload["mask_crop_boxes"]) == len(frames)
            ):
                return None
            return loaded_latents, payload["mask_arrays"], payload["mask_crop_boxes"]
        except Exception:
            return None

    def _build_profile_cache(
        self,
        profile: str,
        profile_type: str,
        cache_dir: Path,
        frames: np.ndarray,
        coords_xyxy: np.ndarray,
        latents_path: Path,
        masks_path: Path,
        runtime_meta_path: Path,
    ) -> tuple[list[torch.Tensor], list[np.ndarray], list[list[int]]]:
        logger.info(
            "component=musetalk op=build_profile_cache status=started profile=%s profile_type=%s frames=%d",
            profile,
            profile_type,
            len(frames),
        )

        latents: list[torch.Tensor] = []
        mask_arrays: list[np.ndarray] = []
        mask_crop_boxes: list[list[int]] = []
        frame_h, frame_w = frames.shape[1:3]

        for frame, raw_coord in zip(frames, coords_xyxy):
            x1, y1, x2, y2 = self._sanitize_xyxy(raw_coord, frame_w, frame_h)
            x1, y1, x2, y2 = self._expand_coord((x1, y1, x2, y2), frame_w, frame_h)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame
                x1, y1, x2, y2 = 0, 0, frame_w, frame_h
            resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latent = self.vae.get_latents_for_unet(resized).detach().cpu()
            latents.append(latent)

            mask, crop_box = self._get_image_prepare_material(
                frame,
                [x1, y1, x2, y2],
                upper_boundary_ratio=self.upper_boundary_ratio,
                expand=self.blend_expand,
                fp=self.face_parser,
                mode=self.parsing_mode,
            )
            aligned_mask, aligned_crop_box = self._align_mask_to_crop(mask, crop_box, frame_w, frame_h)
            mask_arrays.append(aligned_mask)
            mask_crop_boxes.append([int(v) for v in aligned_crop_box])

        torch.save(latents, latents_path)
        with masks_path.open("wb") as handle:
            pickle.dump(
                {
                    "cache_version": self.cache_version,
                    "coord_format": "xyxy",
                    "coords_sha1": self._coord_sha1,
                    "parsing_mode": self.parsing_mode,
                    "coord_expand_x": self.coord_expand_x,
                    "coord_expand_up": self.coord_expand_up,
                    "coord_expand_down": self.coord_expand_down,
                    "upper_boundary_ratio": self.upper_boundary_ratio,
                    "blend_expand": self.blend_expand,
                    "face_scale": self.face_scale,
                    "mask_arrays": mask_arrays,
                    "mask_crop_boxes": mask_crop_boxes,
                },
                handle,
            )
        runtime_meta_path.write_text(
            json.dumps(self._cache_manifest(self._coord_sha1, tuple(frames.shape)), indent=2)
        )

        logger.info(
            "component=musetalk op=build_profile_cache status=done profile=%s profile_type=%s",
            profile,
            profile_type,
        )
        return latents, mask_arrays, mask_crop_boxes

    def _clear_loaded_profile_assets(self) -> None:
        self.frames = None
        self.coords_xyxy = None
        self._latents = []
        self._mask_arrays = []
        self._mask_crop_boxes = []
        self._blend_alphas = []
        self.audio_history = np.array([], dtype=np.float32)
        self._loaded_cache_dir = None
        self._loaded_coord_source = None
        self._loaded_runtime_max_frame_edge = 0
        self._loaded_cache_signature = None
        self._coord_sha1 = None
        self._prev_generated_face = None
        self._last_rendered_frame = None
        self.frame_idx = 0
        self.frame_accumulator = 0.0
        self.infer_frame_accumulator = 0.0
        self._vignette_cache.clear()
        gc.collect()

    def load_profile(self, profile: str, profile_type: str = PROFILE_TYPE_AVATAR) -> None:
        cache_dir = avatar_cache_dir(profile, profile_type)
        frames_path = cache_dir / "frames.npy"
        legacy_coords_path = cache_dir / "coords.npy"
        baked_coords_path = cache_dir / "musetalk_coords.npy"
        coord_source = self._normalize_coord_source(self.coord_source)
        if coord_source == "auto":
            coord_source = "baked" if baked_coords_path.exists() else "legacy"
        coords_path = baked_coords_path if coord_source == "baked" and baked_coords_path.exists() else legacy_coords_path
        if coord_source == "baked" and coords_path != baked_coords_path:
            logger.warning(
                "component=musetalk op=load_profile status=coord_fallback requested=baked fallback=legacy profile=%s profile_type=%s",
                profile,
                profile_type,
            )
            coord_source = "legacy"
        meta_path = cache_dir / "meta.json"
        if not frames_path.exists() or not coords_path.exists():
            raise FileNotFoundError(
                f"Avatar cache missing for {profile}. Run preprocess with avatar baking."
            )

        cache_signature = tuple(
            (
                str(path.name),
                int(path.stat().st_mtime_ns),
                int(path.stat().st_size),
            )
            if path.exists()
            else None
            for path in (
                frames_path,
                coords_path,
                meta_path,
                cache_dir / "musetalk_runtime_meta.json",
                cache_dir / "musetalk_runtime_meta_baked.json",
            )
        )
        if (
            self._loaded_cache_dir == cache_dir
            and self._loaded_cache_signature == cache_signature
            and self._loaded_coord_source == coord_source
            and self._loaded_runtime_max_frame_edge == self._runtime_edge()
            and self.frames is not None
            and self._latents
        ):
            self.frame_idx = 0
            self.frame_accumulator = 0.0
            self.infer_frame_accumulator = 0.0
            self.audio_history = np.array([], dtype=np.float32)
            self._prev_generated_face = None
            return

        if self.frames is not None or self._latents or self._mask_arrays:
            logger.info(
                "component=musetalk op=profile_assets_clear status=before_load previous_cache=%s next_cache=%s",
                self._loaded_cache_dir,
                cache_dir,
            )
            self._clear_loaded_profile_assets()

        raw_coords = np.load(coords_path).astype(np.int32)
        frames, coords_xyxy, source_fmt, runtime_edge = self._load_or_build_runtime_frame_cache(
            cache_dir,
            frames_path,
            coords_path,
            raw_coords,
            coord_source,
            self._runtime_edge(),
        )
        if frames.ndim != 4 or raw_coords.ndim != 2 or len(frames) == 0:
            raise ValueError(f"Invalid avatar cache layout in {cache_dir}")
        if len(frames) != len(raw_coords):
            raise ValueError(
                f"Avatar cache mismatch in {cache_dir}: frames={len(frames)} coords={len(raw_coords)}"
            )

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if isinstance(meta, dict) and "fps" in meta:
                    self.fps = float(meta.get("fps", self.fps))
            except Exception:
                logger.warning(
                    "component=musetalk op=load_profile fallback=default_fps reason=invalid_meta profile=%s profile_type=%s",
                    profile,
                    profile_type,
                )

        frame_h, frame_w = frames.shape[1:3]
        self._coord_sha1 = self._coords_sha1(coords_xyxy)
        logger.info(
            "component=musetalk op=load_profile status=coords profile=%s profile_type=%s coord_source=%s source_format=%s runtime_edge=%s frame_shape=%s",
            profile,
            profile_type,
            coord_source,
            source_fmt,
            runtime_edge or "full",
            tuple(frames.shape),
        )

        suffix_parts: list[str] = []
        if coord_source != "legacy":
            suffix_parts.append(coord_source)
        if runtime_edge > 0:
            suffix_parts.append(f"edge{runtime_edge}")
        cache_suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)
        latents_path = cache_dir / f"musetalk_latents{cache_suffix}.pt"
        masks_path = cache_dir / f"musetalk_masks{cache_suffix}.pkl"
        runtime_meta_path = cache_dir / f"musetalk_runtime_meta{cache_suffix}.json"

        cached = self._load_cached_assets(
            latents_path,
            masks_path,
            runtime_meta_path,
            frames,
            coords_xyxy,
        )
        if cached is None:
            logger.info(
                "component=musetalk op=profile_cache status=miss profile=%s profile_type=%s source=%s",
                profile,
                profile_type,
                coord_source,
            )
            latents, mask_arrays, mask_crop_boxes = self._build_profile_cache(
                profile,
                profile_type,
                cache_dir,
                frames,
                coords_xyxy,
                latents_path,
                masks_path,
                runtime_meta_path,
            )
            # The manifest did not exist when ``cache_signature`` was first
            # collected above.  Refresh it now that the cache build has
            # persisted the three runtime artifacts.  Without this refresh,
            # the very first real request after warm-up reloads every latent
            # and mask from disk once more before subsequent requests become
            # genuinely hot.
            cache_signature = tuple(
                (
                    str(path.name),
                    int(path.stat().st_mtime_ns),
                    int(path.stat().st_size),
                )
                if path.exists()
                else None
                for path in (
                    frames_path,
                    coords_path,
                    meta_path,
                    cache_dir / "musetalk_runtime_meta.json",
                    cache_dir / "musetalk_runtime_meta_baked.json",
                )
            )
        else:
            logger.info(
                "component=musetalk op=profile_cache status=hit profile=%s profile_type=%s source=%s",
                profile,
                profile_type,
                coord_source,
            )
            latents, mask_arrays, mask_crop_boxes = cached

        mask_arrays, mask_crop_boxes = self._stabilize_mask_sequence(
            mask_arrays,
            mask_crop_boxes,
            frame_w,
            frame_h,
        )
        # The cached mask boxes were already stabilized, but the generated
        # MuseTalk face was still placed using the raw detector coordinates
        # below. That mismatch makes the lower-face patch shift its scale and
        # position from frame to frame, which is most visible as a chin that
        # flips between two shapes. Use the same short temporal track for the
        # compositor's face placement while retaining the raw coordinates for
        # cache validation and latent lookup above.
        runtime_coords_xyxy = (
            self._smooth_xyxy_boxes(
                coords_xyxy,
                frame_w,
                frame_h,
                self.mask_stabilize_window,
            )
            if self.mask_stabilize and len(coords_xyxy) >= 3
            else coords_xyxy
        )

        self.frames = frames
        self.coords_xyxy = runtime_coords_xyxy
        self._latents = latents
        self._mask_arrays = mask_arrays
        self._mask_crop_boxes = mask_crop_boxes
        self._blend_alphas = [self._prepare_alpha(mask) for mask in self._mask_arrays]
        self._loaded_cache_dir = cache_dir
        self._loaded_coord_source = coord_source
        self._loaded_runtime_max_frame_edge = self._runtime_edge()
        self._loaded_cache_signature = cache_signature
        self.frame_idx = 0
        self.frame_accumulator = 0.0
        self.infer_frame_accumulator = 0.0
        self.audio_history = np.array([], dtype=np.float32)
        self._prev_generated_face = None
        self._last_rendered_frame = None

    def _pause_frame(self, source_frame: np.ndarray) -> np.ndarray:
        """Return a stable portrait for a sustained silent audio frame.

        MuseTalk's source video continues to move independently of the new
        TTS audio.  Holding the most recently rendered result prevents that
        original motion from leaking back into the jaw during a pause.  A
        source frame is used only before the first successful composition.
        """
        previous = self._last_rendered_frame
        if previous is not None and previous.shape == source_frame.shape:
            return previous.copy()
        return source_frame

    def _smooth_generated_face(self, generated_face: np.ndarray) -> np.ndarray:
        strength = float(np.clip(self.temporal_smooth, 0.0, 0.35))
        current = generated_face.astype(np.uint8, copy=False)
        if strength <= 1e-4:
            self._prev_generated_face = current.copy()
            return current

        previous = self._prev_generated_face
        if previous is None:
            self._prev_generated_face = current.copy()
            return current
        if previous.shape[:2] != current.shape[:2]:
            previous = cv2.resize(previous, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_LINEAR)
        blended = cv2.addWeighted(current, 1.0 - strength, previous, strength, 0.0)
        self._prev_generated_face = blended.copy()
        return blended

    def _compute_target_frames(self, audio_16k: np.ndarray, fps: float) -> int:
        expected_frames = (len(audio_16k) / 16000.0) * fps
        total_frames = expected_frames + self.frame_accumulator
        target_frames = int(total_frames)
        self.frame_accumulator = total_frames - target_frames
        return target_frames

    def _compute_infer_target_frames(self, audio_16k: np.ndarray, infer_fps: float) -> int:
        expected_frames = (len(audio_16k) / 16000.0) * infer_fps
        total_frames = expected_frames + self.infer_frame_accumulator
        target_frames = int(total_frames)
        self.infer_frame_accumulator = total_frames - target_frames
        return target_frames

    @staticmethod
    def _normalize_audio(audio_16k: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio_16k, dtype=np.float32).flatten()
        if audio.size == 0:
            return audio
        max_abs = float(np.max(np.abs(audio)))
        if max_abs > 1.5:
            audio = audio / 32768.0
        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def _stable_silence_mask(
        audio_16k: np.ndarray,
        frame_count: int,
        fps: float,
        *,
        rms_threshold: float,
        min_duration_ms: int,
    ) -> np.ndarray:
        """Return one ``True`` value for every sustained silent video frame.

        Short low-energy gaps are normal articulation and should remain under
        MuseTalk control. A longer quiet run is a conversational pause, where
        holding the stable rendered portrait is more natural than a
        hallucinated mouth shape.
        """
        count = max(0, int(frame_count))
        if count == 0:
            return np.zeros(0, dtype=bool)
        audio = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return np.ones(count, dtype=bool)

        frame_samples = max(1.0, 16000.0 / max(1.0, float(fps)))
        threshold = max(1e-6, float(rms_threshold))
        quiet = np.zeros(count, dtype=bool)
        for index in range(count):
            start = min(audio.size, int(math.floor(index * frame_samples)))
            end = min(audio.size, max(start + 1, int(math.ceil((index + 1) * frame_samples))))
            if start >= audio.size:
                quiet[index] = True
                continue
            window = audio[start:end]
            rms = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
            quiet[index] = rms < threshold

        min_frames = max(
            1,
            int(math.ceil((max(1, int(min_duration_ms)) / 1000.0) * max(1.0, float(fps)))),
        )
        stable = np.zeros(count, dtype=bool)
        run_start = 0
        while run_start < count:
            if not quiet[run_start]:
                run_start += 1
                continue
            run_end = run_start + 1
            while run_end < count and quiet[run_end]:
                run_end += 1
            if run_end - run_start >= min_frames:
                stable[run_start:run_end] = True
            run_start = run_end
        return stable

    def _silence_hold_enabled(self, requested: bool | None) -> bool:
        """Resolve the optional pause experiment without changing normal sync.

        ``None`` is what the public streaming pipeline passes.  It must use
        the bridge's opt-in setting rather than silently re-enabling frame
        holding for regular generated speech.
        """
        return self.suppress_silence_motion if requested is None else bool(requested)

    def _extract_whisper_prompts(
        self,
        audio_buffer: np.ndarray,
        fps: float,
        target_frames: int,
        start_frame_offset: int,
    ) -> torch.Tensor:
        if target_frames <= 0:
            return torch.empty((0, 0, 0), device=self.device)

        segment_len = 30 * 16000
        segments = [
            audio_buffer[i : i + segment_len] for i in range(0, len(audio_buffer), segment_len)
        ] or [audio_buffer]

        hidden_states = []
        with torch.no_grad():
            for segment in segments:
                input_features = self.whisper_feature_extractor(
                    segment,
                    return_tensors="pt",
                    sampling_rate=16000,
                ).input_features
                input_features = input_features.to(self.device, dtype=self.weight_dtype)
                audio_feats = self.whisper.encoder(
                    input_features,
                    output_hidden_states=True,
                ).hidden_states
                hidden_states.append(torch.stack(audio_feats, dim=2))

        whisper_feature = torch.cat(hidden_states, dim=1)

        audio_fps = 50
        whisper_idx_multiplier = audio_fps / max(1.0, float(fps))
        actual_length = max(1, int(math.floor((len(audio_buffer) / 16000.0) * audio_fps)))
        whisper_feature = whisper_feature[:, :actual_length, ...]

        pad_unit = math.ceil(whisper_idx_multiplier)
        left = self.audio_padding_left
        right = self.audio_padding_right
        whisper_feature = torch.cat(
            [
                torch.zeros_like(whisper_feature[:, : pad_unit * left]),
                whisper_feature,
                torch.zeros_like(whisper_feature[:, : pad_unit * 3 * right]),
            ],
            dim=1,
        )

        audio_feature_len = 2 * (left + right + 1)
        prompts = []
        for i in range(target_frames):
            frame_idx = start_frame_offset + i
            audio_idx = int(math.floor(frame_idx * whisper_idx_multiplier))
            clip = whisper_feature[:, audio_idx : audio_idx + audio_feature_len]
            if clip.shape[1] < audio_feature_len:
                pad = torch.zeros_like(whisper_feature[:, : audio_feature_len - clip.shape[1]])
                clip = torch.cat([clip, pad], dim=1)
            prompts.append(clip)

        audio_prompts = torch.cat(prompts, dim=0)
        return rearrange(audio_prompts, "b c h w -> b (c h) w")

    def _frame_iter(
        self,
        count: int,
    ) -> Iterable[tuple[np.ndarray, list[int], torch.Tensor, np.ndarray, list[int], np.ndarray]]:
        if self.frames is None or self.coords_xyxy is None:
            return
        for _ in range(count):
            idx = self.frame_idx % len(self.frames)
            frame = self.frames[idx].copy()
            h, w = frame.shape[:2]
            base_coord = self._sanitize_xyxy(self.coords_xyxy[idx], w, h)
            # Use the same expanded box at runtime as cache-build time.
            # Mismatch here causes warped lips, ghost jawlines, and rectangular seams.
            coord = list(self._expand_coord(base_coord, w, h))
            latent = self._latents[idx]
            mask = self._mask_arrays[idx]
            mask_crop = self._mask_crop_boxes[idx]
            alpha = self._blend_alphas[idx]
            self.frame_idx += 1
            yield frame, coord, latent, mask, mask_crop, alpha

    def _pad_runtime_batch(
        self,
        whisper_batch: torch.Tensor,
        latent_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Pad a short final model batch without changing visible frames.

        MuseTalk's U-Net is sample-independent in evaluation mode. Repeating
        the final sample only gives CUDA a stable execution shape. Callers use
        the returned logical size to ignore the synthetic outputs.
        """
        logical_size = int(len(latent_batch))
        target_size = int(self.batch_size)
        if (
            not self.static_batch_padding
            or logical_size <= 0
            or logical_size >= target_size
            or target_size > 32
        ):
            return whisper_batch, latent_batch, logical_size

        pad_size = target_size - logical_size
        whisper_padding = whisper_batch[-1:].expand(pad_size, *whisper_batch.shape[1:])
        latent_padding = latent_batch[-1:].expand(pad_size, *latent_batch.shape[1:])
        return (
            torch.cat((whisper_batch, whisper_padding), dim=0),
            torch.cat((latent_batch, latent_padding), dim=0),
            logical_size,
        )

    def _blend_frame(
        self,
        frame: np.ndarray,
        generated_face: np.ndarray,
        face_box: list[int],
        crop_box: list[int],
        alpha: np.ndarray,
    ) -> np.ndarray:
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = self._sanitize_xyxy(face_box, frame_w, frame_h)
        cx1, cy1, cx2, cy2 = [int(v) for v in crop_box]
        cx1 = max(0, min(cx1, frame_w - 1))
        cy1 = max(0, min(cy1, frame_h - 1))
        cx2 = max(1, min(cx2, frame_w))
        cy2 = max(1, min(cy2, frame_h))
        if cx2 <= cx1 or cy2 <= cy1:
            return frame
        if x2 <= x1 or y2 <= y1:
            return frame

        original_patch = frame[cy1:cy2, cx1:cx2].astype(np.float32, copy=False)
        patch_h, patch_w = original_patch.shape[:2]
        if patch_h <= 1 or patch_w <= 1:
            return frame

        local_x1 = max(0, x1 - cx1)
        local_y1 = max(0, y1 - cy1)
        local_x2 = min(patch_w, x2 - cx1)
        local_y2 = min(patch_h, y2 - cy1)
        if local_x2 <= local_x1 or local_y2 <= local_y1:
            return frame

        generated_face = self._smooth_generated_face(generated_face)
        resized = cv2.resize(
            generated_face,
            (local_x2 - local_x1, local_y2 - local_y1),
            interpolation=cv2.INTER_LANCZOS4,
        )
        if self.detail_sharpen > 1e-4:
            # Apply stronger sharpening only when 256px output is upscaled substantially.
            scale_x = float(local_x2 - local_x1) / 256.0
            scale_y = float(local_y2 - local_y1) / 256.0
            upscale = max(scale_x, scale_y)
            extra = max(0.0, upscale - 1.0) * 0.22
            amount = float(np.clip(self.detail_sharpen + extra, 0.0, 0.8))
            resized = self._adaptive_unsharp_luma(resized, amount)
        place_x1, place_y1 = local_x1, local_y1
        place_x2, place_y2 = local_x2, local_y2

        face_scale = float(self.face_scale)
        if 0.5 <= face_scale < 0.999:
            curr_w = local_x2 - local_x1
            curr_h = local_y2 - local_y1
            scaled_w = max(1, int(round(curr_w * face_scale)))
            scaled_h = max(1, int(round(curr_h * face_scale)))
            if scaled_w < curr_w or scaled_h < curr_h:
                resized = cv2.resize(
                    resized,
                    (scaled_w, scaled_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                off_x = max(0, (curr_w - scaled_w) // 2)
                off_y = max(0, (curr_h - scaled_h) // 2)
                place_x1 = local_x1 + off_x
                place_y1 = local_y1 + off_y
                place_x2 = place_x1 + scaled_w
                place_y2 = place_y1 + scaled_h

        gx1, gy1 = cx1 + place_x1, cy1 + place_y1
        gx2, gy2 = cx1 + place_x2, cy1 + place_y2
        roi_orig = frame[gy1:gy2, gx1:gx2]
        if roi_orig.shape[:2] == resized.shape[:2]:
            resized = self._match_mean_color(
                roi_orig,
                resized,
                strength=self.color_match_strength,
            )

        composed_patch = original_patch.copy()
        composed_patch[place_y1:place_y2, place_x1:place_x2] = resized.astype(np.float32)

        if alpha.shape[:2] != (patch_h, patch_w):
            alpha = cv2.resize(alpha, (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        alpha = self._restrict_chin_blend(alpha, (x1, y1, x2, y2), (cx1, cy1, cx2, cy2))
        alpha_3 = alpha[:, :, None]
        blended_patch = (composed_patch * alpha_3) + (original_patch * (1.0 - alpha_3))

        out = frame.copy()
        out[cy1:cy2, cx1:cx2] = np.clip(blended_patch, 0, 255).astype(np.uint8)
        return out

    def sync_chunk(
        self,
        audio_16k: np.ndarray,
        fps: float | None = None,
        lookahead_16k: np.ndarray | None = None,
        *,
        suppress_silence_motion: bool | None = None,
    ) -> list[np.ndarray]:
        timing_enabled = os.getenv("MUSE_TALK_PHASE_TIMING", "0") == "1"
        started_at = time.perf_counter() if timing_enabled else 0.0
        prepare_ms = 0.0
        whisper_ms = 0.0
        model_ms = 0.0
        blend_ms = 0.0
        model_batch_ms: list[float] = []
        model_batch_sizes: list[int] = []
        model_batch_shapes: list[tuple[int, ...]] = []
        if self.frames is None or self.coords_xyxy is None:
            raise RuntimeError("Avatar cache not loaded.")
        if audio_16k.size == 0:
            return []

        audio_chunk = self._normalize_audio(audio_16k)
        if audio_chunk.size == 0:
            return []
        lookahead_chunk = (
            self._normalize_audio(lookahead_16k)
            if lookahead_16k is not None and np.asarray(lookahead_16k).size > 0
            else np.array([], dtype=np.float32)
        )

        output_fps = float(fps or self.fps or 25.0)
        output_target_frames = self._compute_target_frames(audio_chunk, output_fps)
        if output_target_frames <= 0:
            return []
        infer_fps = min(output_fps, self.infer_fps if self.infer_fps > 0 else output_fps)
        infer_target_frames = self._compute_infer_target_frames(audio_chunk, infer_fps)
        if infer_target_frames <= 0:
            infer_target_frames = 1
        # MuseTalk's audio-conditioned frames are the synchronization source.
        # Holding frames while the TTS is still speaking makes the displayed
        # mouth lag its audio, so production defaults to the historical
        # all-viseme path.  An explicit request or diagnostic environment flag
        # can still enable the pause experiment when it is being evaluated.
        use_silence_hold = self._silence_hold_enabled(suppress_silence_motion)
        silence_frames = (
            self._stable_silence_mask(
                audio_chunk,
                infer_target_frames,
                infer_fps,
                rms_threshold=self.silence_rms_threshold,
                min_duration_ms=self.silence_min_duration_ms,
            )
            if use_silence_hold
            else np.zeros(infer_target_frames, dtype=bool)
        )

        history_append = audio_chunk
        lookahead_infer_frames = 0.0
        if lookahead_chunk.size:
            lookahead_infer_frames = (len(lookahead_chunk) / 16000.0) * infer_fps

        self.audio_history = np.concatenate([self.audio_history, history_append])
        max_history = int(max(0.5, self.audio_history_sec) * 16000)
        if len(self.audio_history) > max_history + len(history_append):
            self.audio_history = self.audio_history[-(max_history + len(history_append)) :]

        # Lookahead is context for the current inference window only. Do not store it
        # in the rolling history, because the same samples become real audio on the
        # next call and double-counting them pushes mouth motion ahead of speech.
        audio_context = (
            np.concatenate([self.audio_history, lookahead_chunk])
            if lookahead_chunk.size
            else self.audio_history
        )

        total_buffer_frames = (len(audio_context) / 16000.0) * infer_fps
        start_frame_offset = max(
            0,
            int(math.floor(total_buffer_frames - lookahead_infer_frames - infer_target_frames)),
        )
        frames: list[np.ndarray] = []
        coords: list[list[int]] = []
        latents: list[torch.Tensor] = []
        mask_crop_boxes: list[list[int]] = []
        alphas: list[np.ndarray] = []
        for frame, coord, latent, _mask, mask_crop, alpha in self._frame_iter(infer_target_frames):
            frames.append(frame)
            coords.append(coord)
            latents.append(latent)
            mask_crop_boxes.append(mask_crop)
            alphas.append(alpha)
        if timing_enabled:
            prepare_ms = (time.perf_counter() - started_at) * 1000.0

        output_frames: list[np.ndarray] = []
        if bool(np.all(silence_frames)):
            # A silent request is used during warm-up and can also occur at the
            # end of a spoken answer. Before any composition this falls back to
            # the source frame. Otherwise it holds the prior portrait rather
            # than reintroducing source-video mouth motion.
            self._prev_generated_face = None
            output_frames = [self._pause_frame(frame) for frame in frames]
        else:
            whisper_started_at = time.perf_counter() if timing_enabled else 0.0
            whisper_chunks = self._extract_whisper_prompts(
                audio_context,
                infer_fps,
                infer_target_frames,
                start_frame_offset,
            )
            if timing_enabled:
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                whisper_ms = (time.perf_counter() - whisper_started_at) * 1000.0
            if whisper_chunks.numel() == 0:
                return []
            cursor = 0
            model_started_at = time.perf_counter() if timing_enabled else 0.0
            with torch.inference_mode():
                gen = self._datagen(
                    whisper_chunks,
                    latents,
                    batch_size=self.batch_size,
                    device=self.device,
                )
                for whisper_batch, latent_batch in gen:
                    batch_started_at = time.perf_counter() if timing_enabled else 0.0
                    whisper_batch, latent_batch, batch_size = self._pad_runtime_batch(
                        whisper_batch,
                        latent_batch,
                    )
                    whisper_batch = whisper_batch.to(self.device, dtype=self.weight_dtype)
                    audio_feature_batch = self.pe(whisper_batch)
                    latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                    pred_latents = self.unet.model(
                        latent_batch,
                        self.timesteps,
                        encoder_hidden_states=audio_feature_batch,
                    ).sample
                    pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                    recon = self.vae.decode_latents(pred_latents)

                    if timing_enabled and self.device.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if timing_enabled:
                        model_batch_ms.append((time.perf_counter() - batch_started_at) * 1000.0)
                        model_batch_sizes.append(batch_size)
                        model_batch_shapes.append(tuple(int(value) for value in whisper_batch.shape))
                    if timing_enabled:
                        batch_blend_started_at = time.perf_counter()

                    for res_frame in recon[:batch_size]:
                        frame = frames[cursor]
                        if silence_frames[cursor]:
                            # Do not re-introduce the moving source clip through
                            # a pause.  Hold the prior composite instead, so the
                            # mouth and jaw do not twitch while the audio is quiet.
                            self._prev_generated_face = None
                            output_frames.append(self._pause_frame(frame))
                            cursor += 1
                            continue
                        coord = coords[cursor]
                        crop_box = mask_crop_boxes[cursor]
                        alpha = alphas[cursor]
                        try:
                            blended = self._blend_frame(frame, res_frame, coord, crop_box, alpha)
                            output_frames.append(blended)
                            self._last_rendered_frame = blended.copy()
                        except Exception:
                            logger.exception(
                                "component=musetalk op=blend_frame status=error idx=%s",
                                cursor,
                            )
                            output_frames.append(frame)
                        cursor += 1
                    if timing_enabled:
                        blend_ms += (time.perf_counter() - batch_blend_started_at) * 1000.0
            if timing_enabled:
                model_ms = max(0.0, (time.perf_counter() - model_started_at) * 1000.0 - blend_ms)

        # Never make a compositing failure look like a successful lip-sync
        # response.  Returning only untouched source frames is particularly
        # misleading to users because the audio continues normally while the
        # video appears to be a slow source-video loop.
        if frames and len(output_frames) == len(frames):
            untouched = sum(
                int(np.array_equal(output, source))
                for output, source in zip(output_frames, frames)
            )
            if untouched == len(frames) and not bool(np.all(silence_frames)):
                raise RuntimeError(
                    "MuseTalk could not composite generated face frames; refusing to stream the source video as lip-sync output."
                )

        if output_target_frames > 0 and output_frames and output_target_frames != len(output_frames):
            idx = np.linspace(0, len(output_frames) - 1, output_target_frames)
            idx = np.clip(np.round(idx).astype(np.int32), 0, len(output_frames) - 1)
            output_frames = [output_frames[i] for i in idx.tolist()]

        if timing_enabled:
            total_ms = (time.perf_counter() - started_at) * 1000.0
            if total_ms >= 500.0:
                logger.info(
                    "component=musetalk op=sync_chunk_timing total_ms=%.2f prepare_ms=%.2f whisper_ms=%.2f model_ms=%.2f blend_ms=%.2f frames=%d audio_sec=%.3f batch_sizes=%s batch_shapes=%s batch_model_ms=%s",
                    total_ms,
                    prepare_ms,
                    whisper_ms,
                    model_ms,
                    blend_ms,
                    len(output_frames),
                    len(audio_chunk) / 16000.0,
                    model_batch_sizes,
                    model_batch_shapes,
                    [round(value, 2) for value in model_batch_ms],
                )
        return output_frames

    def close(self) -> None:
        self._clear_loaded_profile_assets()
        for attr in (
            "face_parser",
            "whisper",
            "whisper_feature_extractor",
            "timesteps",
            "pe",
            "unet",
            "vae",
            "_datagen",
            "_get_image_prepare_material",
        ):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass
