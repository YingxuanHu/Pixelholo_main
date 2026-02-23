import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CLIP_DBFS,
    DEFAULT_MAX_NO_SPEECH_PROB,
    DEFAULT_MERGE_GAP_SEC,
    DEFAULT_MODEL_SIZE,
    DEFAULT_MIN_AVG_LOGPROB,
    DEFAULT_MIN_CHUNK_DBFS,
    DEFAULT_MIN_WORDS,
    DEFAULT_MIN_SPEECH_RATIO,
    DEFAULT_AVATAR_FPS,
    DEFAULT_AVATAR_PADS,
    DEFAULT_AVATAR_NOSMOOTH,
    PROFILE_TYPE_AVATAR,
)
from src.preprocess import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a video into avatar-ready chunks.")
    parser.add_argument("--video", required=True, type=Path, help="Path to input video (.mp4)")
    parser.add_argument("--audio", type=Path, default=None, help="Optional audio file for training")
    parser.add_argument("--name", required=True, help="Speaker name")
    parser.add_argument(
        "--profile_type",
        choices=[PROFILE_TYPE_AVATAR],
        default=PROFILE_TYPE_AVATAR,
        help="Profile type for avatar preprocessing.",
    )
    parser.add_argument("--model_size", default=DEFAULT_MODEL_SIZE, help="faster-whisper model size")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device for faster-whisper")
    parser.add_argument("--compute_type", default=DEFAULT_COMPUTE_TYPE, help="Compute type for faster-whisper")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Whisper language code")
    parser.add_argument("--no_vad", action="store_true", help="Disable whisper VAD filtering")
    parser.add_argument("--min_avg_logprob", type=float, default=DEFAULT_MIN_AVG_LOGPROB)
    parser.add_argument("--max_no_speech_prob", type=float, default=DEFAULT_MAX_NO_SPEECH_PROB)
    parser.add_argument("--min_words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--merge_gap_sec", type=float, default=DEFAULT_MERGE_GAP_SEC)
    parser.add_argument("--min_chunk_dbfs", type=float, default=DEFAULT_MIN_CHUNK_DBFS)
    parser.add_argument("--max_clip_dbfs", type=float, default=DEFAULT_MAX_CLIP_DBFS)
    parser.add_argument("--min_speech_ratio", type=float, default=DEFAULT_MIN_SPEECH_RATIO)
    parser.add_argument("--legacy_split", action="store_true", help="Use silence split before transcription")

    parser.add_argument("--no_bake_avatar", action="store_true", help="Skip avatar cache baking for lip-sync")
    parser.add_argument("--avatar_fps", type=float, default=DEFAULT_AVATAR_FPS)
    parser.add_argument("--avatar_start_sec", type=float, default=5.0)
    parser.add_argument("--avatar_loop_sec", type=float, default=10.0)
    parser.add_argument("--avatar_loop_fade_sec", type=float, default=0.0)
    parser.add_argument("--avatar_resize_factor", type=int, default=1)
    parser.add_argument("--avatar_pads", type=str, default=DEFAULT_AVATAR_PADS)
    parser.add_argument("--avatar_batch_size", type=int, default=16)
    parser.add_argument("--avatar_nosmooth", action="store_true", default=DEFAULT_AVATAR_NOSMOOTH)
    parser.add_argument("--avatar_no_blur_background", action="store_true")
    parser.add_argument("--avatar_blur_kernel", type=int, default=75)
    parser.add_argument("--avatar_device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", help="Reduce preprocessing logs")

    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    meta_path = process_video(
        video_path=args.video,
        audio_path=args.audio,
        speaker_name=args.name,
        profile_type=PROFILE_TYPE_AVATAR,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        vad_filter=not args.no_vad,
        min_avg_logprob=args.min_avg_logprob,
        max_no_speech_prob=args.max_no_speech_prob,
        min_words=args.min_words,
        merge_gap_sec=args.merge_gap_sec,
        min_chunk_dbfs=args.min_chunk_dbfs,
        max_clip_dbfs=args.max_clip_dbfs,
        min_speech_ratio=args.min_speech_ratio,
        legacy_split=args.legacy_split,
        bake_avatar=not args.no_bake_avatar,
        avatar_fps=args.avatar_fps,
        avatar_start_sec=args.avatar_start_sec,
        avatar_loop_sec=args.avatar_loop_sec,
        avatar_loop_fade_sec=args.avatar_loop_fade_sec,
        avatar_resize_factor=args.avatar_resize_factor,
        avatar_pads=args.avatar_pads,
        avatar_batch_size=args.avatar_batch_size,
        avatar_nosmooth=args.avatar_nosmooth,
        avatar_blur_background=not args.avatar_no_blur_background,
        avatar_blur_kernel=args.avatar_blur_kernel,
        avatar_device=args.avatar_device,
        quiet=args.quiet,
    )
    print(f"Metadata written to {meta_path}", flush=True)


if __name__ == "__main__":
    main()
