import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from config import LIP_SYNCING_DIR, PROFILE_TYPE_AVATAR, avatar_cache_dir

logger = logging.getLogger("pixelholo.lipsync")


class LipSyncBridge:
    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: str | None = None,
        img_size: int = 96,
        wav2lip_batch_size: int = 128,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.frame_idx = 0
        self.frame_accumulator = 0.0
        self.frames: np.ndarray | None = None
        self.coords: np.ndarray | None = None
        self.fps: float = 25.0
        self._mask_cache: dict[tuple[int, int, int], np.ndarray] = {}

        lip_dir = LIP_SYNCING_DIR / "lib" / "Wav2Lip"
        if not lip_dir.exists():
            raise FileNotFoundError(f"Wav2Lip repo not found at {lip_dir}")
        hparams_path = lip_dir / "hparams.py"
        if not hparams_path.exists():
            raise FileNotFoundError(f"Wav2Lip hparams not found at {hparams_path}")
        hparams_spec = importlib.util.spec_from_file_location("hparams", hparams_path)
        if hparams_spec is None or hparams_spec.loader is None:
            raise ImportError("Failed to create module spec for Wav2Lip hparams.")
        hparams_module = importlib.util.module_from_spec(hparams_spec)
        sys.modules["hparams"] = hparams_module
        hparams_spec.loader.exec_module(hparams_module)

        models_init = lip_dir / "models" / "__init__.py"
        if not models_init.exists():
            raise FileNotFoundError(f"Wav2Lip models not found at {models_init}")
        models_spec = importlib.util.spec_from_file_location(
            "wav2lip_models",
            models_init,
            submodule_search_locations=[str(lip_dir / "models")],
        )
        if models_spec is None or models_spec.loader is None:
            raise ImportError("Failed to create module spec for Wav2Lip models.")
        wav2lip_models = importlib.util.module_from_spec(models_spec)
        sys.modules["wav2lip_models"] = wav2lip_models
        models_spec.loader.exec_module(wav2lip_models)

        audio_spec = importlib.util.spec_from_file_location("wav2lip_audio", lip_dir / "audio.py")
        if audio_spec is None or audio_spec.loader is None:
            raise ImportError("Failed to create module spec for Wav2Lip audio.")
        wav2lip_audio = importlib.util.module_from_spec(audio_spec)
        sys.modules["wav2lip_audio"] = wav2lip_audio
        audio_spec.loader.exec_module(wav2lip_audio)

        self._wav_audio = wav2lip_audio
        self.model = wav2lip_models.Wav2Lip().to(self.device)

        ckpt = checkpoint_path or (LIP_SYNCING_DIR / "models" / "wav2lip_gan.pth")
        if not ckpt.exists():
            raise FileNotFoundError(f"Wav2Lip checkpoint not found: {ckpt}")
        checkpoint = torch.load(ckpt, map_location=self.device)
        state = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        self.model.load_state_dict(state)
        self.model.eval()

    def load_profile(self, profile: str, profile_type: str = PROFILE_TYPE_AVATAR) -> None:
        cache_dir = avatar_cache_dir(profile, profile_type)
        frames_path = cache_dir / "frames.npy"
        coords_path = cache_dir / "coords.npy"
        meta_path = cache_dir / "meta.json"
        if not frames_path.exists() or not coords_path.exists():
            raise FileNotFoundError(
                f"Avatar cache missing for {profile}. Run preprocess with avatar baking."
            )
        self.frames = np.load(frames_path)
        self.coords = np.load(coords_path)
        if meta_path.exists():
            meta = meta_path.read_text()
            try:
                import json

                data = json.loads(meta)
                if isinstance(data, dict) and "fps" in data:
                    self.fps = float(data.get("fps", self.fps))
                else:
                    logger.warning(
                        "component=lipsync op=load_profile fallback=default_fps reason=missing_fps profile=%s profile_type=%s",
                        profile,
                        profile_type,
                    )
            except Exception:
                logger.warning(
                    "component=lipsync op=load_profile fallback=default_fps reason=invalid_meta profile=%s profile_type=%s",
                    profile,
                    profile_type,
                )
        else:
            logger.warning(
                "component=lipsync op=load_profile fallback=default_fps reason=meta_missing profile=%s profile_type=%s",
                profile,
                profile_type,
            )
        self.frame_idx = 0

    def _mel_chunks(self, audio_16k: np.ndarray, fps: float | None = None) -> list[np.ndarray]:
        mel = self._wav_audio.melspectrogram(audio_16k)
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError("Mel contains NaN.")
        mel_step_size = 16
        mel_chunks = []
        mel_idx_multiplier = 80.0 / float(fps or self.fps or 25.0)
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > mel.shape[1]:
                mel_chunks.append(mel[:, mel.shape[1] - mel_step_size :])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        return mel_chunks

    def _batch_iter(
        self,
        frames: list[np.ndarray],
        coords: list[list[int]],
        mels: list[np.ndarray],
    ) -> Iterable[tuple[np.ndarray, np.ndarray, list[np.ndarray], list[list[int]]]]:
        img_batch, mel_batch, frame_batch, coord_batch = [], [], [], []
        for frame, coord, mel in zip(frames, coords, mels):
            y1, y2, x1, x2 = coord
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (self.img_size, self.img_size))
            img_batch.append(face)
            mel_batch.append(mel)
            frame_batch.append(frame)
            coord_batch.append(coord)
            if len(img_batch) >= self.wav2lip_batch_size:
                yield self._prepare_batch(img_batch, mel_batch, frame_batch, coord_batch)
                img_batch, mel_batch, frame_batch, coord_batch = [], [], [], []
        if img_batch:
            yield self._prepare_batch(img_batch, mel_batch, frame_batch, coord_batch)

    def _prepare_batch(self, img_batch, mel_batch, frame_batch, coord_batch):
        img_batch = np.asarray(img_batch)
        mel_batch = np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, self.img_size // 2 :] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        return img_batch, mel_batch, frame_batch, coord_batch

    def _soft_mask(self, h: int, w: int, feather: int) -> np.ndarray:
        key = (h, w, feather)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        feather = max(1, min(feather, min(h, w) // 2))
        mask = np.ones((h, w), dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, feather, endpoint=False, dtype=np.float32)
        mask[:feather, :] *= ramp[:, None]
        mask[-feather:, :] *= ramp[::-1][:, None]
        mask[:, :feather] = np.minimum(mask[:, :feather], ramp[None, :])
        mask[:, -feather:] = np.minimum(mask[:, -feather:], ramp[::-1][None, :])
        mask = mask[:, :, None]
        self._mask_cache[key] = mask
        return mask

    @staticmethod
    def _match_mean_color(target: np.ndarray, source: np.ndarray) -> np.ndarray:
        if target.size == 0 or source.size == 0:
            return source
        t_mean = target.reshape(-1, 3).mean(axis=0)
        s_mean = source.reshape(-1, 3).mean(axis=0)
        diff = t_mean - s_mean
        corrected = source.astype(np.float32) + diff
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def sync_chunk(self, audio_16k: np.ndarray, fps: float | None = None) -> list[np.ndarray]:
        if self.frames is None or self.coords is None:
            raise RuntimeError("Avatar cache not loaded.")
        if audio_16k.size == 0:
            return []
        fps = fps or self.fps or 25.0
        mel_chunks = self._mel_chunks(audio_16k, fps=fps)
        if not mel_chunks:
            return []
        expected_frames = (len(audio_16k) / 16000.0) * fps
        total_frames = expected_frames + self.frame_accumulator
        target_frames = int(total_frames)
        self.frame_accumulator = total_frames - target_frames
        if target_frames <= 0:
            return []
        if len(mel_chunks) > target_frames:
            mel_chunks = mel_chunks[:target_frames]
        elif len(mel_chunks) < target_frames:
            mel_chunks.extend([mel_chunks[-1]] * (target_frames - len(mel_chunks)))
        frames = []
        coords = []
        for _ in mel_chunks:
            idx = self.frame_idx % len(self.frames)
            frames.append(self.frames[idx].copy())
            coords.append(self.coords[idx].tolist())
            self.frame_idx += 1

        output_frames: list[np.ndarray] = []
        for img_batch, mel_batch, frame_batch, coord_batch in self._batch_iter(
            frames, coords, mel_chunks
        ):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            for p, f, c in zip(pred, frame_batch, coord_batch):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
                roi = f[y1:y2, x1:x2].astype(np.uint8, copy=False)
                p = self._match_mean_color(roi, p)
                h, w = p.shape[:2]
                feather = max(6, min(20, min(h, w) // 4))
                mask = self._soft_mask(h, w, feather)
                blended = (p.astype(np.float32) * mask) + (roi.astype(np.float32) * (1.0 - mask))
                f[y1:y2, x1:x2] = blended.astype(np.uint8)
                output_frames.append(f)
        return output_frames
