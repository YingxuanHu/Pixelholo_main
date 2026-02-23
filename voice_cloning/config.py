from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LIP_SYNCING_DIR = BASE_DIR.parent / "lip_syncing"

INFERENCE_AUDIO_DIRNAME = "audio"
INFERENCE_VIDEO_DIRNAME = "video"

RAW_VIDEOS_DIRNAME = "raw_videos"
RAW_AUDIO_DIRNAME = "raw_audio"
PROCESSED_WAVS_DIRNAME = "processed_wavs"
METADATA_FILENAME = "metadata.csv"
AVATAR_CACHE_DIRNAME = "avatar_cache"
VOICE_PROFILE_DIRNAME = "voice_profiles"
AVATAR_PROFILE_DIRNAME = "avatar_profiles"
TRAINING_DIRNAME = "training"
PROFILE_TYPE_VOICE = "voice"
PROFILE_TYPE_AVATAR = "avatar"

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_F_MAX = 8000
TARGET_LUFS = -23.0

MIN_CHUNK_SEC = 2.0
MAX_CHUNK_SEC = 10.0
SILENCE_MIN_LEN_MS = 500
SILENCE_THRESH_DB = -40
KEEP_SILENCE_MS = 200

DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_LANGUAGE = "en"
DEFAULT_VAD_FILTER = True
DEFAULT_MIN_AVG_LOGPROB = -0.5
DEFAULT_MAX_NO_SPEECH_PROB = 0.4
DEFAULT_MIN_WORDS = 4
DEFAULT_MERGE_GAP_SEC = 0.2
DEFAULT_DENOISE = False
DEFAULT_MIN_CHUNK_DBFS = -40.0
DEFAULT_MAX_CLIP_DBFS = None
DEFAULT_MIN_SPEECH_RATIO = 0.6

# Avatar bake defaults (low-latency + stable lip-sync)
DEFAULT_AVATAR_FPS = 25.0
DEFAULT_AVATAR_PADS = "0 10 0 0"
DEFAULT_AVATAR_NOSMOOTH = False

DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_LEN = 400
DEFAULT_FP16 = True
DEFAULT_EPOCHS = 15

STYLE_TTS2_DIR = BASE_DIR / "lib" / "StyleTTS2"


def _normalize_profile_type(profile_type: str | None) -> str:
    if profile_type == PROFILE_TYPE_AVATAR:
        return PROFILE_TYPE_AVATAR
    return PROFILE_TYPE_VOICE


def dataset_root(speaker_name: str, profile_type: str | None = None) -> Path:
    normalized = _normalize_profile_type(profile_type)
    if normalized == PROFILE_TYPE_AVATAR:
        return DATA_DIR / AVATAR_PROFILE_DIRNAME / speaker_name
    return DATA_DIR / VOICE_PROFILE_DIRNAME / speaker_name


def resolve_dataset_root(speaker_name: str, profile_type: str | None = None) -> Path:
    if profile_type:
        return dataset_root(speaker_name, profile_type)
    for base in (DATA_DIR / VOICE_PROFILE_DIRNAME, DATA_DIR / AVATAR_PROFILE_DIRNAME, DATA_DIR):
        candidate = base / speaker_name
        if candidate.exists():
            return candidate
    return dataset_root(speaker_name, PROFILE_TYPE_VOICE)


def profile_data_root(profile_type: str | None = None) -> Path:
    normalized = _normalize_profile_type(profile_type)
    if normalized == PROFILE_TYPE_AVATAR:
        return DATA_DIR / AVATAR_PROFILE_DIRNAME
    return DATA_DIR / VOICE_PROFILE_DIRNAME


def raw_videos_dir(speaker_name: str, profile_type: str | None = None) -> Path:
    return dataset_root(speaker_name, profile_type) / RAW_VIDEOS_DIRNAME


def raw_audio_dir(speaker_name: str, profile_type: str | None = None) -> Path:
    return dataset_root(speaker_name, profile_type) / RAW_AUDIO_DIRNAME


def processed_wavs_dir(speaker_name: str, profile_type: str | None = None) -> Path:
    return dataset_root(speaker_name, profile_type) / PROCESSED_WAVS_DIRNAME


def metadata_path(speaker_name: str, profile_type: str | None = None) -> Path:
    return dataset_root(speaker_name, profile_type) / METADATA_FILENAME


def avatar_cache_dir(speaker_name: str, profile_type: str | None = None) -> Path:
    return dataset_root(speaker_name, profile_type) / AVATAR_CACHE_DIRNAME


def training_root(profile_type: str | None = None) -> Path:
    normalized = _normalize_profile_type(profile_type)
    return OUTPUTS_DIR / TRAINING_DIRNAME / normalized


def training_dir(profile: str, profile_type: str | None = None) -> Path:
    return training_root(profile_type) / profile


def resolve_training_dir(profile: str, profile_type: str | None = None) -> Path:
    if profile_type:
        return training_dir(profile, profile_type)
    for base in (training_root(PROFILE_TYPE_VOICE), training_root(PROFILE_TYPE_AVATAR), OUTPUTS_DIR / TRAINING_DIRNAME):
        candidate = base / profile
        if candidate.exists():
            return candidate
    return training_dir(profile, PROFILE_TYPE_VOICE)


def inference_audio_dir(profile: str, profile_type: str | None = None) -> Path:
    normalized = _normalize_profile_type(profile_type)
    return OUTPUTS_DIR / INFERENCE_AUDIO_DIRNAME / normalized / profile


def inference_video_dir(profile: str, profile_type: str | None = None) -> Path:
    return OUTPUTS_DIR / INFERENCE_VIDEO_DIRNAME / PROFILE_TYPE_AVATAR / profile
