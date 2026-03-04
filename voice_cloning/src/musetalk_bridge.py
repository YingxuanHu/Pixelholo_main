from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import sys
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
        self.parsing_mode = parsing_mode or os.getenv("MUSE_TALK_PARSING_MODE", "jaw")
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
        self.audio_history_sec = float(os.getenv("MUSE_TALK_AUDIO_HISTORY_SEC", "2.0"))
        # Keep default infer fps equal to output/avatar fps to avoid apparent slow-motion
        # (low infer fps + frame upsampling reduces base-frame progression speed).
        self.infer_fps = float(os.getenv("MUSE_TALK_INFER_FPS", "25.0"))

        # Keep MuseTalk box expansion conservative. This avoids chin cut lines without
        # shifting the mouth target area too far from the model's expected region.
        self.coord_expand_x = float(os.getenv("MUSE_TALK_COORD_EXPAND_X", "0.08"))
        self.coord_expand_up = float(os.getenv("MUSE_TALK_COORD_EXPAND_UP", "0.04"))
        self.coord_expand_down = float(os.getenv("MUSE_TALK_COORD_EXPAND_DOWN", "0.18"))

        # Blend settings tuned to avoid visible rectangular seams while keeping mouth core opaque.
        self.upper_boundary_ratio = float(os.getenv("MUSE_TALK_UPPER_BOUNDARY_RATIO", "0.5"))
        self.blend_expand = float(os.getenv("MUSE_TALK_BLEND_EXPAND", "1.2"))
        # Slightly shrink generated face patch inside the target box to avoid oversized mouth appearance.
        self.face_scale = float(os.getenv("MUSE_TALK_FACE_SCALE", "0.98"))
        self.alpha_blur_ratio = float(os.getenv("MUSE_TALK_ALPHA_BLUR_RATIO", "0.035"))
        self.vignette_margin_ratio = float(os.getenv("MUSE_TALK_VIGNETTE_MARGIN_RATIO", "0.02"))
        self.alpha_gamma = float(os.getenv("MUSE_TALK_ALPHA_GAMMA", "0.82"))
        self.detail_sharpen = float(os.getenv("MUSE_TALK_DETAIL_SHARPEN", "0.28"))
        self.color_match_strength = float(os.getenv("MUSE_TALK_COLOR_MATCH_STRENGTH", "0.65"))
        self.cache_version = int(os.getenv("MUSE_TALK_CACHE_VERSION", "9"))

        self.frames: np.ndarray | None = None
        self.coords_xyxy: np.ndarray | None = None
        self._latents: list[torch.Tensor] = []
        self._mask_arrays: list[np.ndarray] = []
        self._mask_crop_boxes: list[list[int]] = []
        self._blend_alphas: list[np.ndarray] = []
        self._loaded_cache_dir: Path | None = None
        self._coord_sha1: str | None = None
        self._vignette_cache: dict[tuple[int, int, int, int], np.ndarray] = {}

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
            "face_scale": self.face_scale,
            "alpha_blur_ratio": self.alpha_blur_ratio,
            "vignette_margin_ratio": self.vignette_margin_ratio,
            "alpha_gamma": self.alpha_gamma,
            "detail_sharpen": self.detail_sharpen,
            "color_match_strength": self.color_match_strength,
            "frame_shape": list(frame_shape),
        }

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

    def load_profile(self, profile: str, profile_type: str = PROFILE_TYPE_AVATAR) -> None:
        cache_dir = avatar_cache_dir(profile, profile_type)
        frames_path = cache_dir / "frames.npy"
        coords_path = cache_dir / "coords.npy"
        meta_path = cache_dir / "meta.json"
        if not frames_path.exists() or not coords_path.exists():
            raise FileNotFoundError(
                f"Avatar cache missing for {profile}. Run preprocess with avatar baking."
            )

        if self._loaded_cache_dir == cache_dir and self.frames is not None and self._latents:
            self.frame_idx = 0
            self.audio_history = np.array([], dtype=np.float32)
            return

        frames = np.load(frames_path)
        raw_coords = np.load(coords_path).astype(np.int32)
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
        coords_xyxy, source_fmt = self._convert_coords_to_xyxy(raw_coords, frame_w, frame_h)
        self._coord_sha1 = self._coords_sha1(coords_xyxy)
        logger.info(
            "component=musetalk op=load_profile status=coords profile=%s profile_type=%s source_format=%s",
            profile,
            profile_type,
            source_fmt,
        )

        latents_path = cache_dir / "musetalk_latents.pt"
        masks_path = cache_dir / "musetalk_masks.pkl"
        runtime_meta_path = cache_dir / "musetalk_runtime_meta.json"

        cached = self._load_cached_assets(
            latents_path,
            masks_path,
            runtime_meta_path,
            frames,
            coords_xyxy,
        )
        if cached is None:
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
        else:
            latents, mask_arrays, mask_crop_boxes = cached

        self.frames = frames
        self.coords_xyxy = coords_xyxy
        self._latents = latents
        self._mask_arrays = mask_arrays
        self._mask_crop_boxes = mask_crop_boxes
        self._blend_alphas = [self._prepare_alpha(mask) for mask in self._mask_arrays]
        self._loaded_cache_dir = cache_dir
        self.frame_idx = 0
        self.frame_accumulator = 0.0
        self.infer_frame_accumulator = 0.0
        self.audio_history = np.array([], dtype=np.float32)

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

        resized = cv2.resize(
            generated_face.astype(np.uint8),
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
    ) -> list[np.ndarray]:
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

        history_append = audio_chunk
        lookahead_infer_frames = 0.0
        if lookahead_chunk.size:
            history_append = np.concatenate([history_append, lookahead_chunk])
            lookahead_infer_frames = (len(lookahead_chunk) / 16000.0) * infer_fps

        self.audio_history = np.concatenate([self.audio_history, history_append])
        max_history = int(max(0.5, self.audio_history_sec) * 16000)
        if len(self.audio_history) > max_history + len(history_append):
            self.audio_history = self.audio_history[-(max_history + len(history_append)) :]

        total_buffer_frames = (len(self.audio_history) / 16000.0) * infer_fps
        start_frame_offset = max(
            0,
            int(math.floor(total_buffer_frames - lookahead_infer_frames - infer_target_frames)),
        )
        whisper_chunks = self._extract_whisper_prompts(
            self.audio_history,
            infer_fps,
            infer_target_frames,
            start_frame_offset,
        )
        if whisper_chunks.numel() == 0:
            return []

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

        output_frames: list[np.ndarray] = []
        cursor = 0
        with torch.inference_mode():
            gen = self._datagen(
                whisper_chunks,
                latents,
                batch_size=self.batch_size,
                device=self.device,
            )
            for whisper_batch, latent_batch in gen:
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

                for res_frame in recon:
                    frame = frames[cursor]
                    coord = coords[cursor]
                    crop_box = mask_crop_boxes[cursor]
                    alpha = alphas[cursor]
                    try:
                        blended = self._blend_frame(frame, res_frame, coord, crop_box, alpha)
                        output_frames.append(blended)
                    except Exception:
                        logger.exception(
                            "component=musetalk op=blend_frame status=error idx=%s",
                            cursor,
                        )
                        output_frames.append(frame)
                    cursor += 1

        if output_target_frames > 0 and output_frames and output_target_frames != len(output_frames):
            idx = np.linspace(0, len(output_frames) - 1, output_target_frames)
            idx = np.clip(np.round(idx).astype(np.int32), 0, len(output_frames) - 1)
            output_frames = [output_frames[i] for i in idx.tolist()]

        return output_frames
