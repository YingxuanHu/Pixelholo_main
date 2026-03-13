import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    LIP_SYNCING_DIR,
    PROFILE_TYPE_AVATAR,
    PROFILE_TYPE_VOICE,
    inference_audio_dir,
    inference_video_dir,
    raw_videos_dir,
    resolve_dataset_root,
)


def _find_latest_video(profile: str, profile_type: str) -> Path | None:
    raw_dir = raw_videos_dir(profile, profile_type)
    if not raw_dir.exists():
        return None
    candidates = [p for p in raw_dir.iterdir() if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate speech with StyleTTS2, then lip-sync it to a profile video."
    )
    parser.add_argument("--profile", required=True, help="Profile name.")
    parser.add_argument(
        "--profile_type",
        choices=(PROFILE_TYPE_VOICE, PROFILE_TYPE_AVATAR),
        default=PROFILE_TYPE_AVATAR,
        help="Profile type to load data from.",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--video", type=Path, default=None, help="Override source video path.")
    parser.add_argument("--audio_out", type=Path, default=None, help="Override output wav path.")
    parser.add_argument("--video_out", type=Path, default=None, help="Override output mp4 path.")

    parser.add_argument(
        "--lipsync_mode",
        choices=("chunked", "lib", "cached"),
        default="cached",
        help="Chunked (stream), lib (full video), cached (fast, uses avatar cache).",
    )
    parser.add_argument(
        "--lipsync_dir",
        type=Path,
        default=LIP_SYNCING_DIR,
        help="Path to the lip_syncing repo.",
    )
    parser.add_argument(
        "--wav2lip_dir",
        type=Path,
        default=None,
        help="Override Wav2Lip repo path (defaults to <lip_syncing>/lib/Wav2Lip).",
    )
    parser.add_argument(
        "--wav2lip_checkpoint",
        type=Path,
        default=None,
        help="Override Wav2Lip checkpoint path (defaults to <lip_syncing>/models/wav2lip_gan.pth).",
    )
    parser.add_argument(
        "--lipsync_python",
        type=Path,
        default=None,
        help="Python executable for lip_syncing (optional).",
    )

    # Common chunked knobs
    parser.add_argument("--chunk_sec", type=float, default=1.0)
    parser.add_argument("--loop_sec", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--resize_factor", type=int, default=2)
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument("--face_det_batch_size", type=int, default=16)
    parser.add_argument("--wav2lip_batch_size", type=int, default=128)
    parser.add_argument("--pads", type=str, default="0 10 0 0")
    parser.add_argument("--concat_mode", choices=("copy", "reencode"), default="copy")

    # Pass-through args to speak.py (e.g., --pitch_shift, --phonemizer_lang)
    args, extra = parser.parse_known_args()

    profile_dir = resolve_dataset_root(args.profile, args.profile_type)
    if not profile_dir.exists():
        raise FileNotFoundError(f"Profile not found: {profile_dir}")

    video_path = args.video or _find_latest_video(args.profile, args.profile_type)
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(
            "No source video found. Provide --video or run preprocess.py with a video."
        )

    audio_dir = inference_audio_dir(args.profile, args.profile_type)
    video_dir = inference_video_dir(args.profile, PROFILE_TYPE_AVATAR)
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    audio_out = args.audio_out or (audio_dir / "speak.wav")
    video_out = args.video_out or (video_dir / "speak_lipsync.mp4")

    speak_py = Path(__file__).with_name("speak.py")
    speak_cmd = [
        sys.executable,
        str(speak_py),
        "--profile",
        args.profile,
        "--profile_type",
        args.profile_type,
        "--text",
        args.text,
        "--out",
        str(audio_out),
    ]
    speak_cmd.extend(extra)

    print(f"[voice] Generating audio -> {audio_out}")
    t0 = time.perf_counter()
    _run(speak_cmd)
    print(f"[voice] Done in {time.perf_counter() - t0:.2f}s")

    lipsync_dir = args.lipsync_dir.resolve()
    if not lipsync_dir.exists():
        raise FileNotFoundError(f"lip_syncing repo not found: {lipsync_dir}")
    wav2lip_dir = (args.wav2lip_dir or (lipsync_dir / "lib" / "Wav2Lip")).resolve()
    wav2lip_ckpt = (args.wav2lip_checkpoint or (lipsync_dir / "models" / "wav2lip_gan.pth")).resolve()

    python_exec = args.lipsync_python or sys.executable
    if args.lipsync_mode == "chunked":
        runner = lipsync_dir / "src" / "chunked_lipsync.py"
        cmd = [
            str(python_exec),
            str(runner),
            "--video",
            str(video_path),
            "--audio",
            str(audio_out),
            "--output",
            str(video_out),
            "--chunk_sec",
            str(args.chunk_sec),
            "--loop_sec",
            str(args.loop_sec),
            "--fps",
            str(args.fps),
            "--resize_factor",
            str(args.resize_factor),
            "--fourcc",
            args.fourcc,
            "--face_det_batch_size",
            str(args.face_det_batch_size),
            "--wav2lip_batch_size",
            str(args.wav2lip_batch_size),
            "--pads",
            args.pads,
            "--concat_mode",
            args.concat_mode,
            "--wav2lip_dir",
            str(wav2lip_dir),
            "--checkpoint",
            str(wav2lip_ckpt),
        ]
    elif args.lipsync_mode == "cached":
        runner = Path(__file__).with_name("run_lipsync_fast.py")
        if args.profile_type != PROFILE_TYPE_AVATAR:
            raise ValueError("cached lipsync mode requires an avatar profile.")
        cmd = [
            str(python_exec),
            str(runner),
            "--profile",
            args.profile,
            "--profile_type",
            args.profile_type,
            "--audio",
            str(audio_out),
            "--output",
            str(video_out),
            "--checkpoint",
            str(wav2lip_ckpt),
            "--wav2lip_batch_size",
            str(args.wav2lip_batch_size),
            "--fourcc",
            args.fourcc,
        ]
    else:
        runner = lipsync_dir / "src" / "run_lipsync_lib.py"
        cmd = [
            str(python_exec),
            str(runner),
            "--video",
            str(video_path),
            "--audio",
            str(audio_out),
            "--output",
            str(video_out),
            "--checkpoint",
            str(wav2lip_ckpt),
            "--wav2lip_dir",
            str(wav2lip_dir),
            "--resize_factor",
            str(args.resize_factor),
            "--fourcc",
            args.fourcc,
            "--face_det_batch_size",
            str(args.face_det_batch_size),
            "--wav2lip_batch_size",
            str(args.wav2lip_batch_size),
            "--pads",
            args.pads,
            "--cache_dir",
            str(lipsync_dir / "outputs" / "cache"),
            "--save_cache",
            "--nosmooth",
        ]

    print(f"[lipsync] Starting -> {video_out}")
    t1 = time.perf_counter()
    _run(cmd)
    print(f"[lipsync] Done in {time.perf_counter() - t1:.2f}s")


if __name__ == "__main__":
    main()
