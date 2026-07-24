import React, { useCallback, useEffect, useRef, useState } from 'react';
import ControlPanel from './components/ControlPanel';
import Header from './components/Header';
import StepCard from './components/StepCard';
import LogPanel from './components/LogPanel';
import VoiceControlsPanel from './components/VoiceControlsPanel';
import { useSpeechToText } from './hooks/useSpeechToText';
import {
  Profile,
  StepStatus,
  LogEntry,
  PreprocessStats,
  TrainStats,
  InferenceChunk,
  ProfileInfo,
  ProfileType,
  TTSBackend,
  VoiceControlValues,
  VoiceEmotion,
} from './types';

type TrainFlags = {
  autoSelectEpoch: boolean;
  autoTuneProfile: boolean;
  autoBuildLexicon: boolean;
  earlyStop: boolean;
};

type TrainParams = {
  batchSize: number;
  epochs: number;
  maxLen: number;
};

type LLMMode = 'legacy_fast' | 'live_search' | 'gemini_search' | 'auto';
type MuseTalkPreset = 'realistic' | 'low_latency' | 'balanced' | 'stable';
type VoiceControlBackendDefaults = {
  pitchShift: number;
  f0Scale: number;
  embeddingScale: number;
  paceScale: number;
  volumeGain: number;
  ttsExaggeration: number;
  ttsTemperature: number;
  ttsCfgWeight: number;
  ttsRepetitionPenalty: number;
  avatarEmotion: VoiceEmotion;
  avatarEmotionIntensity: number;
};
type ProfileRuntimeSettings = {
  voice_controls?: Partial<VoiceControlValues>;
};
type BinaryStreamPacketMetadata = {
  event?: string;
  detail?: string;
  inference_ms?: number;
  chunk_index?: number;
  sample_rate?: number;
  fps?: number;
  duration_sec?: number;
  audio_bytes_len?: number;
  audio_format?: 'wav' | 'pcm_s16le';
  audio_channels?: number;
  audio_samples?: number;
  frame_lengths?: number[];
};
type BinaryStreamPacket = {
  metadata: BinaryStreamPacketMetadata;
  payload: Uint8Array;
};
type VideoFrameSource = string | {
  url: string;
  bitmap?: ImageBitmap;
  bitmapPromise?: Promise<ImageBitmap | null>;
  bitmapState?: 'pending' | 'ready' | 'failed';
  drawn?: boolean;
};
type QueuedVideoFrame = {
  frame: VideoFrameSource;
  t: number;
};
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
// A raw upload is only a source asset.  Inference needs either a trained
// checkpoint (the legacy path) or the processed clips produced by the
// zero-shot preprocessing pipeline.  The avatar preprocessor writes its
// extracted full-track WAV at the profile root, so raw_audio_files is not a
// reliable readiness signal for video-only uploads.
const profileIsInferenceReady = (item?: ProfileInfo | null) => Boolean(
  item?.has_profile
  || (item?.processed_wavs ?? 0) > 0,
);
const env = (import.meta as any).env ?? {};
const IS_PRODUCTION_BUILD = env.PROD === true || env.MODE === 'production';
const envNumber = (key: string, fallback: number, min?: number, max?: number) => {
  const raw = (env[key] as string | undefined)?.trim();
  const parsed = raw ? Number(raw) : fallback;
  const value = Number.isFinite(parsed) ? parsed : fallback;
  return clamp(value, min ?? Number.NEGATIVE_INFINITY, max ?? Number.POSITIVE_INFINITY);
};
// Production is served through tools/web_proxy.py, which exposes FastAPI at the
// same-origin /api route. Pointing a public browser at 127.0.0.1 would instead
// target that visitor's own computer and make the engine appear offline.
const DEFAULT_API_BASE = (env.VITE_PIXELHOLO_API_BASE as string | undefined)?.trim()
  || (IS_PRODUCTION_BUILD ? '/api' : 'http://127.0.0.1:8000');
const BINARY_STREAM_MEDIA_TYPE = 'application/vnd.pixelholo.stream-v1';
const BINARY_STREAM_MAGIC = [80, 72, 83, 49]; // PHS1
const BINARY_STREAM_HEADER_BYTES = 12;
const BINARY_STREAM_ENABLED = (env.VITE_PIXELHOLO_BINARY_STREAM as string | undefined) !== '0';
const BINARY_PCM_AUDIO_ENABLED = (env.VITE_PIXELHOLO_BINARY_PCM_AUDIO as string | undefined) !== '0';
const LOCAL_STORAGE_API_BASE_KEY = 'voxclone_api_base';
const LOCAL_STORAGE_WORKSPACE_KEY = 'pixelholo_workspace_id';
const WORKSPACE_HEADER = 'X-PixelHolo-Workspace';

const createAnonymousWorkspaceId = (): string => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }
  throw new Error('This browser cannot create an anonymous workspace.');
};

const getOrCreateAnonymousWorkspaceId = (): string => {
  const cached = localStorage.getItem(LOCAL_STORAGE_WORKSPACE_KEY);
  if (cached && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(cached)) {
    return cached;
  }
  const workspaceId = createAnonymousWorkspaceId();
  localStorage.setItem(LOCAL_STORAGE_WORKSPACE_KEY, workspaceId);
  return workspaceId;
};
// The web product is avatar-first: one short video becomes the user's voice
// reference plus the baked face frames used by MuseTalk.  The legacy voice-only
// and training paths remain in the backend, but are intentionally not exposed in
// this experience.
const DEFAULT_PROFILE_TYPE: 'voice' | 'avatar' = 'avatar';
const DEFAULT_OUTPUT_MODE: 'voice' | 'avatar' = 'avatar';
const DEFAULT_TTS_BACKEND: TTSBackend = 'chatterbox';
const DEFAULT_LLM_MODE: LLMMode = 'legacy_fast';
const DEFAULT_MUSETALK_PRESET: MuseTalkPreset = 'realistic';
const LLM_MODE_OPTIONS: { value: LLMMode; label: string }[] = [
  { value: 'legacy_fast', label: 'Llama 3.1 8B Instant (Cutoff)' },
  { value: 'live_search', label: 'GPT-4o Mini (Live Search)' },
  { value: 'gemini_search', label: 'Gemini 2.5 Flash Lite (Live Search)' },
  { value: 'auto', label: 'Auto (Llama/GPT/Gemini)' },
];
const DEFAULT_AVATAR_START_SEC = 0;
const DEFAULT_AVATAR_LOOP_SEC = 20;
const DEFAULT_AVATAR_LOOP_FADE_SEC = 0.15;
const CAMERA_RECORDING_SECONDS = 25;
const CAMERA_PROMPT = 'Hi, this is my PixelHolo avatar. I am speaking clearly and naturally in my everyday voice. Today is a bright day, and this sample shows my pronunciation, rhythm, and tone. I am calm and confident, speaking at a steady pace just as I would in a normal conversation. Please listen to how I normally sound while I speak naturally and comfortably.';
const normalizeProfileName = (value: string) => value.trim().toLocaleLowerCase();
const BLUR_KERNEL_BY_LEVEL = { low: 60, medium: 75, high: 90 } as const;
const DEFAULT_AVATAR_BLUR_LEVEL: keyof typeof BLUR_KERNEL_BY_LEVEL = 'medium';
const DEFAULT_VIDEO_FPS = 25;
const DEFAULT_WEB_AVATAR_MAX_FRAME_EDGE = Number(
  (env.VITE_PIXELHOLO_AVATAR_MAX_FRAME_EDGE as string | undefined)?.trim() || '1080',
);
const DEFAULT_AUDIO_START_DELAY_SEC = envNumber('VITE_PIXELHOLO_AUDIO_START_DELAY_SEC', 0.04, 0.02, 0.25);
const AVATAR_AUDIO_START_DELAY_SEC = envNumber('VITE_PIXELHOLO_AVATAR_AUDIO_START_DELAY_SEC', 0.34, 0.12, 1.0);
const AVATAR_AUDIO_CHUNK_LEAD_SEC = envNumber('VITE_PIXELHOLO_AVATAR_AUDIO_CHUNK_LEAD_SEC', 0.08, 0.02, 0.5);
const VIDEO_FRAME_DECODE_PREWAIT_MS = envNumber('VITE_PIXELHOLO_VIDEO_PREDECODE_PREWAIT_MS', 45, 0, 200);
const WARMUP_CACHE_MAX_AGE_MS = 120_000;
const normalizeTtsBackend = (_value: unknown): TTSBackend => 'chatterbox';
const normalizeVoiceEmotion = (value: unknown): VoiceEmotion => {
  if (value === 'happy' || value === 'sad' || value === 'angry' || value === 'scared' || value === 'disgust') {
    return value;
  }
  return 'neutral';
};
const normalizeVoiceControls = (controls: Partial<VoiceControlValues>): VoiceControlValues => ({
  pitch: Number(clamp(Number(controls.pitch ?? 0), -4, 4).toFixed(1)),
  pace: Math.round(clamp(Number(controls.pace ?? 0), -100, 100)),
  tone: Math.round(clamp(Number(controls.tone ?? 0), -100, 100)),
  volume: Math.round(clamp(Number(controls.volume ?? 0), -100, 100)),
  expressiveness: Number(clamp(Number(controls.expressiveness ?? 0.5), 0.25, 1).toFixed(2)),
  variation: Number(clamp(Number(controls.variation ?? 0.8), 0.1, 1.2).toFixed(2)),
  guidance: Number(clamp(Number(controls.guidance ?? 0.5), 0, 1).toFixed(2)),
  repetition: Number(clamp(Number(controls.repetition ?? 1.2), 0.9, 2).toFixed(2)),
  emotion: normalizeVoiceEmotion(controls.emotion),
  emotionIntensity: Number(clamp(Number(controls.emotionIntensity ?? 0.5), 0, 1).toFixed(2)),
});
const normalizeRuntimeSettings = (settings: any): ProfileRuntimeSettings => {
  const normalized: ProfileRuntimeSettings = {};
  if (settings?.voice_controls && typeof settings.voice_controls === 'object') {
    normalized.voice_controls = normalizeVoiceControls({
      pitch: Number(settings.voice_controls.pitch ?? 0),
      pace: Number(settings.voice_controls.pace ?? 0),
      tone: Number(settings.voice_controls.tone ?? 0),
      volume: Number(settings.voice_controls.volume ?? 0),
      expressiveness: Number(settings.voice_controls.expressiveness ?? 0.5),
      variation: Number(settings.voice_controls.variation ?? 0.8),
      guidance: Number(settings.voice_controls.guidance ?? 0.5),
      repetition: Number(settings.voice_controls.repetition ?? 1.2),
      emotion: normalizeVoiceEmotion(settings.voice_controls.emotion),
      emotionIntensity: Number(settings.voice_controls.emotion_intensity ?? settings.voice_controls.emotionIntensity ?? 0.5),
    });
  }
  return normalized;
};
const DEFAULT_VOICE_CONTROL_BACKEND_DEFAULTS: VoiceControlBackendDefaults = {
  pitchShift: 0,
  f0Scale: 1,
  embeddingScale: 1.2,
  paceScale: 1,
  volumeGain: 1,
  // Chatterbox is most stable for identity when sampling is conservative.
  // Keep these defaults below the expressive/creative preset so an uploaded
  // voice is less likely to acquire a different accent between utterances.
  ttsExaggeration: 0.35,
  ttsTemperature: 0.5,
  ttsCfgWeight: 0.8,
  ttsRepetitionPenalty: 1.2,
  avatarEmotion: 'neutral',
  avatarEmotionIntensity: 0.5,
};
const DEFAULT_VOICE_CONTROL_VALUES = normalizeVoiceControls({
  pitch: 0,
  pace: 0,
  tone: 0,
  volume: 0,
  expressiveness: 0.35,
  variation: 0.5,
  guidance: 0.8,
  repetition: 1.2,
  emotion: 'neutral',
  emotionIntensity: 0.5,
});
const formatBytes = (value: number) => {
  if (!Number.isFinite(value) || value <= 0) return '0 MB';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let idx = 0;
  let size = value;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(idx <= 1 ? 0 : 1)} ${units[idx]}`;
};
const concatBytes = (left: Uint8Array, right: Uint8Array) => {
  if (!left.length) return right;
  if (!right.length) return left;
  const combined = new Uint8Array(left.length + right.length);
  combined.set(left, 0);
  combined.set(right, left.length);
  return combined;
};
const readUInt32BE = (bytes: Uint8Array, offset: number) => (
  ((bytes[offset] << 24) >>> 0) +
  ((bytes[offset + 1] << 16) >>> 0) +
  ((bytes[offset + 2] << 8) >>> 0) +
  (bytes[offset + 3] >>> 0)
);
const isBinaryStreamResponse = (response: Response) =>
  response.headers.get('content-type')?.includes(BINARY_STREAM_MEDIA_TYPE) ?? false;
const uint8ToArrayBuffer = (bytes: Uint8Array) =>
  bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const decodeBinaryStream = async (
  response: Response,
  onPacket: (packet: BinaryStreamPacket) => Promise<void> | void,
  signal?: AbortSignal,
) => {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = new Uint8Array(0);
  while (true) {
    if (signal?.aborted) break;
    const { value, done } = await reader.read();
    if (signal?.aborted) break;
    if (done) break;
    if (value) pending = concatBytes(pending, value);
    while (pending.length >= BINARY_STREAM_HEADER_BYTES) {
      if (!BINARY_STREAM_MAGIC.every((byte, index) => pending[index] === byte)) {
        throw new Error('Invalid PixelHolo binary stream packet.');
      }
      const metadataLength = readUInt32BE(pending, 4);
      const payloadLength = readUInt32BE(pending, 8);
      const packetLength = BINARY_STREAM_HEADER_BYTES + metadataLength + payloadLength;
      if (pending.length < packetLength) break;
      const metadataStart = BINARY_STREAM_HEADER_BYTES;
      const payloadStart = metadataStart + metadataLength;
      const metadataBytes = pending.slice(metadataStart, payloadStart);
      const payload = pending.slice(payloadStart, packetLength);
      const metadata = JSON.parse(decoder.decode(metadataBytes)) as BinaryStreamPacketMetadata;
      pending = pending.slice(packetLength);
      await onPacket({ metadata, payload });
      if (signal?.aborted) break;
    }
  }
};
const DEFAULT_STEP_STATUSES: Record<string, StepStatus> = {
  upload: 'idle',
  preprocess: 'idle',
  train: 'idle',
  inference: 'idle',
};

const createLog = (message: string, level: LogEntry['level'] = 'info'): LogEntry => ({
  id: Math.random().toString(36).slice(2, 10),
  timestamp: new Date().toLocaleTimeString([], {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }),
  level,
  message,
});

const defaultFlags: TrainFlags = {
  autoSelectEpoch: true,
  autoTuneProfile: true,
  autoBuildLexicon: true,
  earlyStop: true,
};

const defaultTrainParams: TrainParams = {
  batchSize: 2,
  epochs: 15,
  maxLen: 400,
};

const trainPreset: TrainParams = { batchSize: 2, epochs: 15, maxLen: 400 };

const App: React.FC = () => {
  const [activeStep, setActiveStep] = useState(1);
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [workspaceId] = useState(getOrCreateAnonymousWorkspaceId);
  const [apiStatus, setApiStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [profileType, setProfileType] = useState<'voice' | 'avatar'>(DEFAULT_PROFILE_TYPE);
  const [profile, setProfile] = useState<Profile>({ name: '', lastUploadedFile: null, fileSize: null });
  const [isCreatingProfile, setIsCreatingProfile] = useState(true);
  const [showWelcome, setShowWelcome] = useState(true);
  const [autoPrepareAfterUpload, setAutoPrepareAfterUpload] = useState(false);
  const [lastUploadedFilename, setLastUploadedFilename] = useState<string | null>(null);
  const [lastUploadedAudioFilename, setLastUploadedAudioFilename] = useState<string | null>(null);
  const [sourceMode, setSourceMode] = useState<'upload' | 'camera'>('camera');
  const [cameraState, setCameraState] = useState<'idle' | 'requesting' | 'ready' | 'recording' | 'recorded' | 'error'>('idle');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraElapsed, setCameraElapsed] = useState(0);
  const [cameraPreviewUrl, setCameraPreviewUrl] = useState<string | null>(null);
  const [capturedCameraFile, setCapturedCameraFile] = useState<File | null>(null);
  const [profileNameRequired, setProfileNameRequired] = useState(false);
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [profilesStatus, setProfilesStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [profileMenuKey, setProfileMenuKey] = useState<string | null>(null);
  const [isWarmingUp, setIsWarmingUp] = useState(false);
  const [warmupTargetName, setWarmupTargetName] = useState<string | null>(null);
  const [uiNotice, setUiNotice] = useState<string | null>(null);
  const [stepStatuses, setStepStatuses] = useState<Record<string, StepStatus>>(DEFAULT_STEP_STATUSES);
  const [preprocessLogs, setPreprocessLogs] = useState<LogEntry[]>([]);
  const [trainLogs, setTrainLogs] = useState<LogEntry[]>([]);
  const [preprocessStats, setPreprocessStats] = useState<PreprocessStats | null>(null);
  const [trainStats, setTrainStats] = useState<TrainStats | null>(null);
  const [preprocessProgress, setPreprocessProgress] = useState<number | null>(null);
  const [preprocessStageIndex, setPreprocessStageIndex] = useState<number | null>(null);
  const [trainStageIndex, setTrainStageIndex] = useState<number | null>(null);
  const [inferenceStageIndex, setInferenceStageIndex] = useState<number | null>(null);
  const [trainFlags, setTrainFlags] = useState<TrainFlags>(defaultFlags);
  const [trainParams, setTrainParams] = useState<TrainParams>(defaultTrainParams);
  const [showAdvancedTrain, setShowAdvancedTrain] = useState(false);
  const [avatarStartSec, setAvatarStartSec] = useState(DEFAULT_AVATAR_START_SEC);
  const [avatarBlurLevel, setAvatarBlurLevel] = useState<keyof typeof BLUR_KERNEL_BY_LEVEL>(
    DEFAULT_AVATAR_BLUR_LEVEL,
  );
  const [uploadPhaseVideo, setUploadPhaseVideo] = useState<'idle' | 'uploading' | 'error'>('idle');
  const [uploadPhaseAudio, setUploadPhaseAudio] = useState<'idle' | 'uploading' | 'error'>('idle');
  const [uploadProgressVideo, setUploadProgressVideo] = useState(0);
  const [uploadProgressAudio, setUploadProgressAudio] = useState(0);
  const [uploadBytesVideo, setUploadBytesVideo] = useState<{ loaded: number; total: number }>({ loaded: 0, total: 0 });
  const [uploadBytesAudio, setUploadBytesAudio] = useState<{ loaded: number; total: number }>({ loaded: 0, total: 0 });
  const uploadVideoLastPctRef = useRef(0);
  const uploadAudioLastPctRef = useRef(0);
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null);
  const minimalNameInputRef = useRef<HTMLInputElement | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const cameraPreviewUrlRef = useRef<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const cameraChunksRef = useRef<Blob[]>([]);
  const cameraTimerRef = useRef<number | null>(null);
  const [inferenceText, setInferenceText] = useState('');
  const [composerMode, setComposerMode] = useState<'say' | 'chat'>('chat');
  const [showVoiceSettings, setShowVoiceSettings] = useState(false);
  const [inferenceChunks, setInferenceChunks] = useState<InferenceChunk[]>([]);
  const [latency, setLatency] = useState<{ ttfa: number; total: number } | null>(null);
  const [modelOverride, setModelOverride] = useState('');
  const [refOverride, setRefOverride] = useState('');
  const [avatarBackend, setAvatarBackend] = useState<'wav2lip' | 'musetalk'>('musetalk');
  const [ttsBackend, setTtsBackend] = useState<TTSBackend>(DEFAULT_TTS_BACKEND);
  const [museTalkPreset, setMuseTalkPreset] = useState<MuseTalkPreset>(DEFAULT_MUSETALK_PRESET);
  const [outputMode, setOutputMode] = useState<'voice' | 'avatar'>(DEFAULT_OUTPUT_MODE);
  const [llmMode, setLlmMode] = useState<LLMMode>(DEFAULT_LLM_MODE);
  const [runtimeSettingsStatus, setRuntimeSettingsStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [runtimeSettingsError, setRuntimeSettingsError] = useState<string | null>(null);
  const [voiceControlBackendDefaults, setVoiceControlBackendDefaults] = useState<VoiceControlBackendDefaults>(
    DEFAULT_VOICE_CONTROL_BACKEND_DEFAULTS,
  );
  const [voiceControlDefaults, setVoiceControlDefaults] = useState<VoiceControlValues>(DEFAULT_VOICE_CONTROL_VALUES);
  const [voiceControlValues, setVoiceControlValues] = useState<VoiceControlValues>(DEFAULT_VOICE_CONTROL_VALUES);
  const [voiceControlsStatus, setVoiceControlsStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('ready');
  const [voiceControlsError, setVoiceControlsError] = useState<string | null>(null);
  const [voiceControlsDirty, setVoiceControlsDirty] = useState(false);
  const [videoState, setVideoState] = useState<'idle' | 'buffering' | 'playing'>('idle');
  const [isPlaybackActive, setIsPlaybackActive] = useState(false);
  const [videoFps, setVideoFps] = useState(DEFAULT_VIDEO_FPS);
  const [videoQueue, setVideoQueue] = useState(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const audioEndTimeRef = useRef<number>(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const audioUnlockedRef = useRef<boolean>(false);
  const streamAbortRef = useRef<AbortController | null>(null);
  const streamRunningRef = useRef(false);
  const streamSessionRef = useRef<number>(0);
  const isBusy = Object.values(stepStatuses).some(status => status === 'running');
  const isProfileSwitchBlocked = Object.entries(stepStatuses).some(
    ([step, status]) => step !== 'inference' && status === 'running',
  );
  const warmedProfilesRef = useRef<Map<string, number>>(new Map());
  const warmupInFlightRef = useRef<Map<string, Promise<void>>>(new Map());
  const profileSelectionRef = useRef(0);
  const backendRuntimeIdRef = useRef<string | null>(null);
  const activeWarmupKeyRef = useRef<string | null>(null);
  const videoCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoTimerRef = useRef<number | null>(null);
  const videoRafRef = useRef<number | null>(null);
  const videoStartTimeRef = useRef<number | null>(null);
  const videoNextFrameTimeRef = useRef<number | null>(null);
  const videoStateRef = useRef<'idle' | 'buffering' | 'playing'>('idle');
  const videoDrawSerialRef = useRef(0);
  const playbackSettleTimerRef = useRef<number | null>(null);
  const audioStartDelayRef = useRef<number>(DEFAULT_AUDIO_START_DELAY_SEC);
  const videoFpsRef = useRef<number>(DEFAULT_VIDEO_FPS);
  const frameQueueRef = useRef<QueuedVideoFrame[]>([]);
  const stopListeningRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    videoStateRef.current = videoState;
  }, [videoState]);

  const releaseFrameSource = useCallback((source?: VideoFrameSource | null) => {
    if (!source) return;
    if (typeof source === 'string') {
      if (source.startsWith('blob:')) URL.revokeObjectURL(source);
      return;
    }
    source.drawn = true;
    source.bitmap?.close();
    source.bitmap = undefined;
    if (source.url.startsWith('blob:')) {
      URL.revokeObjectURL(source.url);
    }
  }, []);

  useEffect(() => {
    setTrainParams(trainPreset);
  }, []);

  const apiFetch = useCallback((input: RequestInfo | URL, init: RequestInit = {}) => {
    const headers = new Headers(init.headers);
    headers.set(WORKSPACE_HEADER, workspaceId);
    return fetch(input, { ...init, headers });
  }, [workspaceId]);

  const uploadWithProgress = useCallback((url: string, form: FormData, onProgress: (loaded: number, total: number) => void) => {
    return new Promise<string>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url);
      xhr.setRequestHeader(WORKSPACE_HEADER, workspaceId);
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          onProgress(event.loaded, event.total);
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(xhr.responseText);
        } else {
          reject(new Error(xhr.responseText || `Upload failed (${xhr.status})`));
        }
      };
      xhr.onerror = () => reject(new Error('Upload failed: the PixelHolo API is unreachable. Start the backend or check the API endpoint in connection settings.'));
      xhr.send(form);
    });
  }, [workspaceId]);

  useEffect(() => {
    if (profileType === 'avatar') {
      setAvatarStartSec(DEFAULT_AVATAR_START_SEC);
      setAvatarBlurLevel(DEFAULT_AVATAR_BLUR_LEVEL);
    }
  }, [profileType]);

  useEffect(() => {
    if (IS_PRODUCTION_BUILD) {
      // Public visitors must always stay on the same-origin proxy. Do not let
      // a stale or manually saved development endpoint make the app offline.
      localStorage.removeItem(LOCAL_STORAGE_API_BASE_KEY);
      setApiBase('/api');
      return;
    }

    const cached = localStorage.getItem(LOCAL_STORAGE_API_BASE_KEY);
    if (!cached) return;
    setApiBase(cached);
  }, []);

  useEffect(() => {
    const handler = () => {
      if (document.visibilityState === 'visible') {
        unlockAudio();
      }
    };
    document.addEventListener('visibilitychange', handler);
    return () => document.removeEventListener('visibilitychange', handler);
  }, []);

  useEffect(() => {
    if (IS_PRODUCTION_BUILD) return;
    localStorage.setItem(LOCAL_STORAGE_API_BASE_KEY, apiBase);
  }, [apiBase]);

  useEffect(() => {
    if (profileType === 'voice') {
      setOutputMode('voice');
      return;
    }
    if (profileType === 'avatar' && outputMode === 'voice') {
      setOutputMode('avatar');
    }
  }, [profileType, outputMode]);

  useEffect(() => {
    videoFpsRef.current = videoFps;
  }, [videoFps]);

  const loadProfiles = useCallback(async () => {
    setProfilesStatus('loading');
    try {
      const res = await apiFetch(`${apiBase}/profiles?profile_type=${profileType}`, { cache: 'no-store' });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setProfiles(Array.isArray(data.profiles) ? data.profiles : []);
      setProfileMenuKey(null);
      setProfilesStatus('idle');
    } catch (err) {
      setProfilesStatus('error');
    }
  }, [apiBase, apiFetch, profileType]);

  const readErrorDetail = useCallback(async (res: Response) => {
    const raw = await res.text();
    if (!raw) return `HTTP ${res.status}`;
    try {
      const parsed = JSON.parse(raw);
      if (typeof parsed?.detail === 'string') return parsed.detail;
      return raw;
    } catch {
      return raw;
    }
  }, []);

  const invalidateWarmupEntriesForProfile = useCallback((profileName: string) => {
    const markers = [`|voice|${profileName}|`, `|avatar|${profileName}|`];

    const warmedKeysToDelete: string[] = [];
    for (const key of warmedProfilesRef.current.keys()) {
      if (markers.some((marker) => key.includes(marker))) {
        warmedKeysToDelete.push(key);
      }
    }
    for (const key of warmedKeysToDelete) {
      warmedProfilesRef.current.delete(key);
    }

    const inFlightKeysToDelete: string[] = [];
    for (const key of warmupInFlightRef.current.keys()) {
      if (markers.some((marker) => key.includes(marker))) {
        inFlightKeysToDelete.push(key);
      }
    }
    for (const key of inFlightKeysToDelete) {
      warmupInFlightRef.current.delete(key);
    }
  }, []);

  const handleRenameProfile = useCallback(
    async (item: ProfileInfo) => {
      if (isBusy) {
        setUiNotice('Stop the current job before renaming profiles.');
        return;
      }
      const currentType = (item.profile_type === 'avatar' ? 'avatar' : 'voice') as 'avatar' | 'voice';
      const nextNameRaw = window.prompt('Rename profile', item.name);
      if (nextNameRaw === null) return;
      const nextName = nextNameRaw.trim();
      if (!nextName || nextName === item.name) return;
      setProfileMenuKey(null);
      try {
        const res = await apiFetch(`${apiBase}/profiles/${encodeURIComponent(item.name)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ new_name: nextName, profile_type: currentType }),
        });
        if (!res.ok) {
          throw new Error(await readErrorDetail(res));
        }
        if (profile.name === item.name && profileType === currentType) {
          setProfile(prev => ({ ...prev, name: nextName }));
        }
        invalidateWarmupEntriesForProfile(item.name);
        invalidateWarmupEntriesForProfile(nextName);
        await loadProfiles();
      } catch (err) {
        setUiNotice(`Rename failed: ${String(err)}`);
      }
    },
    [apiBase, apiFetch, invalidateWarmupEntriesForProfile, isBusy, loadProfiles, profile.name, profileType, readErrorDetail],
  );

  const handleDeleteProfile = useCallback(
    async (item: ProfileInfo) => {
      if (isBusy) {
        setUiNotice('Stop the current job before deleting profiles.');
        return;
      }
      const currentType = (item.profile_type === 'avatar' ? 'avatar' : 'voice') as 'avatar' | 'voice';
      const confirmed = window.confirm(
        `Delete profile "${item.name}" and all related files (data, training checkpoints, inference cache)?`,
      );
      if (!confirmed) return;
      setProfileMenuKey(null);
      try {
        const params = new URLSearchParams({ profile_type: currentType });
        const res = await apiFetch(`${apiBase}/profiles/${encodeURIComponent(item.name)}?${params.toString()}`, {
          method: 'DELETE',
        });
        if (!res.ok) {
          throw new Error(await readErrorDetail(res));
        }
        if (profile.name === item.name && profileType === currentType) {
          setProfile(prev => ({ ...prev, name: '', lastUploadedFile: null, fileSize: null }));
          setIsCreatingProfile(true);
          setLastUploadedFilename(null);
          setLastUploadedAudioFilename(null);
          setActiveStep(1);
        }
        invalidateWarmupEntriesForProfile(item.name);
        await loadProfiles();
      } catch (err) {
        setUiNotice(`Delete failed: ${String(err)}`);
      }
    },
    [apiBase, apiFetch, invalidateWarmupEntriesForProfile, isBusy, loadProfiles, profile.name, profileType, readErrorDetail],
  );

  const warmupKeyFor = useCallback(
    (
      profileName: string,
      type: ProfileType,
      backend: 'wav2lip' | 'musetalk',
      tts: TTSBackend,
      mode: LLMMode,
      preset: MuseTalkPreset,
      runtimeId: string | null,
    ) =>
      `${runtimeId || 'unknown'}|${apiBase}|${type}|${profileName}|${tts}|${type === 'avatar' ? `${backend}:${preset}` : '-'}|${mode}`,
    [apiBase],
  );

  const hasFreshWarmup = useCallback((key: string) => {
    const warmedAt = warmedProfilesRef.current.get(key);
    if (!warmedAt) return false;
    if (Date.now() - warmedAt > WARMUP_CACHE_MAX_AGE_MS) {
      warmedProfilesRef.current.delete(key);
      return false;
    }
    return true;
  }, []);

  const isWarmupSatisfied = useCallback((data: any, type: ProfileType) => {
    const ttsReady = data?.tts_ready ?? (data?.tts_hot_before === true || data?.tts_warmed === true);
    const lipsyncReady = data?.lipsync_ready ?? (
      type !== 'avatar' || data?.lipsync_hot_before === true || data?.lipsync_warmed === true
    );
    const llmReady = data?.llm_ready ?? (data?.llm_hot_before === true || data?.llm_warmed === true);
    return ttsReady && lipsyncReady && llmReady;
  }, []);

  const warmupProfile = useCallback(async (profileName: string, type: ProfileType) => {
    if (!profileName) return;
    const selectionId = profileSelectionRef.current;
    const isCurrentSelection = () => selectionId === profileSelectionRef.current;
    const runtimeId = backendRuntimeIdRef.current;
    const key = warmupKeyFor(profileName, type, avatarBackend, ttsBackend, llmMode, museTalkPreset, runtimeId);
    if (hasFreshWarmup(key)) return;

    const inFlight = warmupInFlightRef.current.get(key);
    if (inFlight) {
      await inFlight;
      return;
    }

    const promise = (async () => {
      if (isCurrentSelection()) {
        activeWarmupKeyRef.current = key;
        setIsWarmingUp(true);
        setWarmupTargetName(profileName);
      }
      try {
        const res = await apiFetch(`${apiBase}/warmup`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            profile: profileName,
            profile_type: type,
            tts_backend: ttsBackend,
            lipsync_backend: type === 'avatar' ? avatarBackend : null,
            avatar_max_frame_edge:
              type === 'avatar' && Number.isFinite(DEFAULT_WEB_AVATAR_MAX_FRAME_EDGE) && DEFAULT_WEB_AVATAR_MAX_FRAME_EDGE > 0
                ? Math.round(DEFAULT_WEB_AVATAR_MAX_FRAME_EDGE)
                : null,
            musetalk_preset: type === 'avatar' && avatarBackend === 'musetalk' ? museTalkPreset : null,
            include_llm: true,
            llm_mode: llmMode,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        const resolvedRuntimeId = typeof data?.runtime_instance_id === 'string' ? data.runtime_instance_id : runtimeId;
        if (resolvedRuntimeId && backendRuntimeIdRef.current !== resolvedRuntimeId) {
          backendRuntimeIdRef.current = resolvedRuntimeId;
          warmedProfilesRef.current.clear();
          warmupInFlightRef.current.clear();
        }
        const resolvedBackend =
          type === 'avatar' && (data?.lipsync_backend === 'wav2lip' || data?.lipsync_backend === 'musetalk')
            ? data.lipsync_backend
            : avatarBackend;
        if (isWarmupSatisfied(data, type)) {
          warmedProfilesRef.current.set(
            warmupKeyFor(profileName, type, resolvedBackend, ttsBackend, llmMode, museTalkPreset, resolvedRuntimeId ?? backendRuntimeIdRef.current),
            Date.now(),
          );
        }
      } finally {
        warmupInFlightRef.current.delete(key);
        if (isCurrentSelection() && activeWarmupKeyRef.current === key) {
          activeWarmupKeyRef.current = null;
          setIsWarmingUp(false);
          setWarmupTargetName(null);
        }
      }
    })();

    warmupInFlightRef.current.set(key, promise);
    await promise;
  }, [apiBase, apiFetch, avatarBackend, hasFreshWarmup, isWarmupSatisfied, llmMode, museTalkPreset, ttsBackend, warmupKeyFor]);

  useEffect(() => {
    loadProfiles();
  }, [loadProfiles]);

  useEffect(() => {
    if (!profileMenuKey) return;
    const onDocClick = () => setProfileMenuKey(null);
    document.addEventListener('click', onDocClick);
    return () => document.removeEventListener('click', onDocClick);
  }, [profileMenuKey]);

  useEffect(() => {
    let cancelled = false;
    let controller: AbortController | null = null;

    const checkApi = async (initialCheck = false) => {
      controller?.abort();
      controller = new AbortController();
      if (initialCheck) setApiStatus('checking');
      try {
        const response = await apiFetch(`${apiBase}/docs`, {
          cache: 'no-store',
          signal: controller.signal,
        });
        if (!cancelled) setApiStatus(response.ok ? 'online' : 'offline');
      } catch (error) {
        if (!cancelled && !(error instanceof DOMException && error.name === 'AbortError')) {
          setApiStatus('offline');
        }
      }
    };

    void checkApi(true);
    // A deployment restart or short tunnel interruption should recover in the
    // UI by itself instead of leaving a visitor on a stale Offline state.
    const retryTimer = window.setInterval(() => void checkApi(), 8_000);
    return () => {
      cancelled = true;
      window.clearInterval(retryTimer);
      controller?.abort();
    };
  }, [apiBase, apiFetch]);

  useEffect(() => {
    if (apiStatus !== 'online') return;
    let cancelled = false;
    apiFetch(`${apiBase}/lipsync_backend`, { cache: 'no-store' })
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        const backend = data?.backend === 'wav2lip' ? 'wav2lip' : 'musetalk';
        const nextTtsBackend = normalizeTtsBackend(data?.tts_backend);
        const runtimeId = typeof data?.runtime_instance_id === 'string' ? data.runtime_instance_id : null;
        if (backendRuntimeIdRef.current && runtimeId && backendRuntimeIdRef.current !== runtimeId) {
          warmedProfilesRef.current.clear();
          warmupInFlightRef.current.clear();
        }
        backendRuntimeIdRef.current = runtimeId;
        setAvatarBackend(backend);
        setTtsBackend(nextTtsBackend);
      })
      .catch(() => {
        // Keep existing value on endpoint/read errors.
      });
    return () => {
      cancelled = true;
    };
  }, [apiBase, apiFetch, apiStatus]);

  const currentProfileInfo = profiles.find((item) => item.name === profile.name);
  const profileNameTaken = Boolean(
    isCreatingProfile
      && profile.name.trim()
      && profiles.some((item) => {
        const itemType = item.profile_type === 'voice' ? 'voice' : 'avatar';
        return itemType === profileType && normalizeProfileName(item.name) === normalizeProfileName(profile.name);
      }),
  );
  const hasTrainedProfile = Boolean(currentProfileInfo?.has_profile);
  const hasData = Boolean(currentProfileInfo?.has_data);
  const hasInferenceProfile = profileIsInferenceReady(currentProfileInfo);
  const resolvePaceScale = useCallback(
    (pace: number) => {
      const base = voiceControlBackendDefaults?.paceScale ?? 1;
      return Number(clamp(base + (pace / 100) * 0.15, 0.85, 1.15).toFixed(2));
    },
    [voiceControlBackendDefaults],
  );
  const resolveToneOverrides = useCallback(
    (tone: number) => {
      const baseF0 = voiceControlBackendDefaults?.f0Scale ?? 1;
      const baseEmbedding = voiceControlBackendDefaults?.embeddingScale ?? 1.2;
      const normalized = tone / 100;
      return {
        f0Scale: Number(clamp(baseF0 + normalized * 0.18, 0.75, 1.35).toFixed(2)),
        embeddingScale: Number(clamp(baseEmbedding + normalized * 0.55, 0.8, 2.2).toFixed(2)),
      };
    },
    [voiceControlBackendDefaults],
  );
  const resolveVolumeGain = useCallback(
    (volume: number) => {
      const base = voiceControlBackendDefaults?.volumeGain ?? 1;
      return Number(clamp(base + (volume / 100) * 0.45, 0.6, 1.45).toFixed(2));
    },
    [voiceControlBackendDefaults],
  );
  const handleVoiceControlsChange = useCallback((patch: Partial<VoiceControlValues>) => {
    setVoiceControlsDirty(true);
    setRuntimeSettingsStatus('idle');
    setVoiceControlValues((prev) => {
      const base = prev ?? voiceControlDefaults ?? DEFAULT_VOICE_CONTROL_VALUES;
      return normalizeVoiceControls({ ...base, ...patch });
    });
  }, [voiceControlDefaults]);
  const resetVoiceControls = useCallback(() => {
    setRuntimeSettingsStatus('idle');
    setVoiceControlValues(voiceControlDefaults ?? DEFAULT_VOICE_CONTROL_VALUES);
    setVoiceControlsDirty(false);
  }, [voiceControlDefaults]);
  const saveRuntimeSettings = useCallback(async () => {
    const profileName = profile.name.trim();
    if (apiStatus !== 'online') {
      setRuntimeSettingsStatus('error');
      setRuntimeSettingsError('Backend is offline. Start the backend, then save again.');
      return;
    }
    if (!profileName) {
      setRuntimeSettingsStatus('error');
      setRuntimeSettingsError('Select a profile before saving voice controls.');
      return;
    }
    setRuntimeSettingsStatus('saving');
    setRuntimeSettingsError(null);
    try {
      const res = await apiFetch(`${apiBase}/profiles/${encodeURIComponent(profileName)}/runtime-settings`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_type: profileType,
          voice_controls: voiceControlValues,
        }),
      });
      if (!res.ok) throw new Error(await readErrorDetail(res));
      setRuntimeSettingsStatus('saved');
      setVoiceControlsDirty(false);
    } catch (err) {
      setRuntimeSettingsStatus('error');
      setRuntimeSettingsError(err instanceof Error ? err.message : String(err));
    }
  }, [
    apiBase,
    apiFetch,
    apiStatus,
    profile.name,
    profileType,
    readErrorDetail,
    voiceControlValues,
  ]);

  useEffect(() => {
    if (apiStatus !== 'online' || !profile.name || !hasInferenceProfile) {
      setVoiceControlBackendDefaults(DEFAULT_VOICE_CONTROL_BACKEND_DEFAULTS);
      setVoiceControlDefaults(DEFAULT_VOICE_CONTROL_VALUES);
      setVoiceControlValues(DEFAULT_VOICE_CONTROL_VALUES);
      setVoiceControlsStatus('ready');
      setVoiceControlsError(null);
      setVoiceControlsDirty(false);
      setRuntimeSettingsStatus('idle');
      setRuntimeSettingsError(null);
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    setVoiceControlBackendDefaults(DEFAULT_VOICE_CONTROL_BACKEND_DEFAULTS);
    setVoiceControlDefaults(DEFAULT_VOICE_CONTROL_VALUES);
    setVoiceControlValues(DEFAULT_VOICE_CONTROL_VALUES);
    setVoiceControlsStatus('loading');
    setVoiceControlsError(null);
    setVoiceControlsDirty(false);

    apiFetch(
      `${apiBase}/profiles/${encodeURIComponent(profile.name)}/voice-controls?profile_type=${profileType}`,
      { cache: 'no-store', signal: controller.signal },
    )
      .then(async (res) => {
        if (!res.ok) throw new Error(await readErrorDetail(res));
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        const responseTtsBackend = normalizeTtsBackend(data?.tts_backend);
        const backendDefaults: VoiceControlBackendDefaults = {
          pitchShift: Number(data?.controls?.pitch_shift ?? 0),
          f0Scale: Number(data?.controls?.f0_scale ?? 1),
          embeddingScale: Number(data?.controls?.embedding_scale ?? 1.2),
          paceScale: Number(data?.controls?.pace_scale ?? 1),
          volumeGain: Number(data?.controls?.volume_gain ?? 1),
          ttsExaggeration: Number(data?.controls?.tts_exaggeration ?? 0.35),
          ttsTemperature: Number(data?.controls?.tts_temperature ?? 0.5),
          ttsCfgWeight: Number(data?.controls?.tts_cfg_weight ?? 0.8),
          ttsRepetitionPenalty: Number(data?.controls?.tts_repetition_penalty ?? 1.2),
          avatarEmotion: normalizeVoiceEmotion(data?.controls?.avatar_emotion),
          avatarEmotionIntensity: Number(data?.controls?.avatar_emotion_intensity ?? 0.5),
        };
        const controls = normalizeVoiceControls({
          pitch: backendDefaults.pitchShift,
          pace: 0,
          tone: 0,
          volume: 0,
          expressiveness: backendDefaults.ttsExaggeration,
          variation: backendDefaults.ttsTemperature,
          guidance: backendDefaults.ttsCfgWeight,
          repetition: backendDefaults.ttsRepetitionPenalty,
          emotion: backendDefaults.avatarEmotion,
          emotionIntensity: backendDefaults.avatarEmotionIntensity,
        });
        const runtimeSettings = normalizeRuntimeSettings(data?.runtime_settings);
        const selectedControls = runtimeSettings.voice_controls
          ? normalizeVoiceControls({
            pitch: runtimeSettings.voice_controls.pitch ?? controls.pitch,
            pace: runtimeSettings.voice_controls.pace ?? controls.pace,
            tone: runtimeSettings.voice_controls.tone ?? controls.tone,
            volume: runtimeSettings.voice_controls.volume ?? controls.volume,
            expressiveness: runtimeSettings.voice_controls.expressiveness ?? controls.expressiveness,
            variation: runtimeSettings.voice_controls.variation ?? controls.variation,
            guidance: runtimeSettings.voice_controls.guidance ?? controls.guidance,
            repetition: runtimeSettings.voice_controls.repetition ?? controls.repetition,
            emotion: runtimeSettings.voice_controls.emotion ?? controls.emotion,
            emotionIntensity: runtimeSettings.voice_controls.emotionIntensity ?? controls.emotionIntensity,
          })
          : controls;
        setTtsBackend(responseTtsBackend);
        setVoiceControlBackendDefaults(backendDefaults);
        setVoiceControlDefaults(controls);
        setVoiceControlValues(selectedControls);
        setVoiceControlsStatus('ready');
        setVoiceControlsDirty(false);
        setRuntimeSettingsStatus('idle');
        setRuntimeSettingsError(null);
      })
      .catch((err) => {
        if (cancelled || (err as Error)?.name === 'AbortError') return;
        setVoiceControlBackendDefaults(DEFAULT_VOICE_CONTROL_BACKEND_DEFAULTS);
        setVoiceControlDefaults(DEFAULT_VOICE_CONTROL_VALUES);
        setVoiceControlValues(DEFAULT_VOICE_CONTROL_VALUES);
        setVoiceControlsStatus('error');
        setVoiceControlsError(String(err));
        setVoiceControlsDirty(false);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [apiBase, apiFetch, apiStatus, hasInferenceProfile, profile.name, profileType, readErrorDetail]);

  const preprocessSteps = [
    ...(profileType === 'avatar' ? ['Bake avatar frames (Wav2Lip cache)'] : []),
    'Extract audio track',
    'Loudness normalize + filter',
    'Split on silence (2-10s)',
    'Transcribe with Whisper',
    'Write metadata.csv',
  ];
  const trainSteps = [
    'Patch config + load base model',
    'Train epochs & save checkpoints',
    'Auto-tune profile defaults',
    'Auto-select best epoch',
    'Build lexicon.json',
  ];
  const inferenceSteps =
    outputMode === 'avatar'
      ? [
          'Resolve profile + load model',
          'Chunk text for streaming',
          'Synthesize audio chunks',
          'Lip-sync video frames',
          'Stream frames to player',
        ]
      : [
          'Resolve profile + load model',
          'Chunk text for streaming',
          'Synthesize audio chunks',
          'Apply smoothing + post FX',
          'Stream audio output',
        ];

  const stageProgress = (index: number | null, total: number, cap: number) => {
    if (index === null || total <= 0) return 0;
    const raw = (index + 1) / total;
    return Math.min(cap, raw * cap);
  };

  const canProceedTo = (step: number) => {
    if (step === 1) return true;
    if (step === 2) return Boolean(profile.name);
    if (step === 3) return stepStatuses.preprocess === 'done' || hasData || hasTrainedProfile;
    if (step === 4) return stepStatuses.train === 'done' || hasInferenceProfile;
    return false;
  };

  useEffect(() => {
    if (apiStatus !== 'online') return;
    if (!hasInferenceProfile) return;
    if (!profile.name) return;
    void warmupProfile(profile.name, profileType);
  }, [apiStatus, hasInferenceProfile, profile.name, profileType, warmupProfile]);

  const streamResponseLines = async (
    response: Response,
    onLine: (line: string) => Promise<void> | void,
  ) => {
    if (!response.body) return;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index = buffer.indexOf('\n');
      while (index !== -1) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);
        if (line) await onLine(line);
        index = buffer.indexOf('\n');
      }
    }
    if (buffer.trim()) {
      await onLine(buffer.trim());
    }
  };

  const isErrorLine = (line: string) => {
    const normalized = line.toLowerCase();
    if (normalized.includes('[process exited')) {
      const match = normalized.match(/process exited\s+(\d+)/);
      if (match) {
        return match[1] !== '0';
      }
      return true;
    }
    return (
      normalized.includes('traceback') ||
      normalized.includes('exception') ||
      normalized.includes('runtimeerror') ||
      normalized.includes('error:') ||
      normalized.includes('failed') ||
      normalized.startsWith('error')
    );
  };

  const handleUpload = async (file: File, options: { autoPrepare?: boolean } = {}) => {
    if (!profile.name || !file) return;
    if (profileNameTaken) {
      setUiNotice('That profile name is already in use. Choose a different name, such as “Alvin 2”.');
      minimalNameInputRef.current?.focus();
      return;
    }
    setUiNotice(null);
    setStepStatuses(prev => ({ ...prev, upload: 'running' }));
    setUploadPhaseVideo('uploading');
    setUploadProgressVideo(0);
    setUploadBytesVideo({ loaded: 0, total: 0 });
    const form = new FormData();
    form.append('profile', profile.name);
    form.append('profile_type', profileType);
    form.append('file', file);
    try {
      const responseText = await uploadWithProgress(`${apiBase}/upload`, form, (loaded, total) => {
        const pct = Math.round((loaded / Math.max(total, 1)) * 100);
        if (pct !== uploadVideoLastPctRef.current || loaded === total) {
          uploadVideoLastPctRef.current = pct;
          setUploadProgressVideo(pct);
        }
        setUploadBytesVideo({ loaded, total });
      });
      const data = JSON.parse(responseText);
      setProfile(prev => ({
        ...prev,
        lastUploadedFile: file.name,
        fileSize: `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
      }));
      setLastUploadedFilename(data.filename);
      setCapturedCameraFile(null);
      setStepStatuses(prev => ({ ...prev, upload: 'done' }));
      setUploadPhaseVideo('idle');
      if (options.autoPrepare) {
        // The camera capture is already a complete source clip. Treat it as
        // the new profile immediately so the duplicate-name guard does not
        // mistake the profile we just created for an unrelated existing one.
        setIsCreatingProfile(false);
        setAutoPrepareAfterUpload(true);
      }
      loadProfiles();
    } catch (err) {
      setStepStatuses(prev => ({ ...prev, upload: 'error' }));
      setUploadPhaseVideo('error');
      setPreprocessLogs([createLog(`Upload failed: ${String(err)}`, 'error')]);
    }
  };

  const clearCameraPreview = useCallback(() => {
    if (cameraPreviewUrlRef.current) {
      URL.revokeObjectURL(cameraPreviewUrlRef.current);
      cameraPreviewUrlRef.current = null;
    }
    setCameraPreviewUrl(null);
  }, []);

  const setCameraPreview = useCallback((blob: Blob) => {
    clearCameraPreview();
    const url = URL.createObjectURL(blob);
    cameraPreviewUrlRef.current = url;
    setCameraPreviewUrl(url);
  }, [clearCameraPreview]);

  const stopCameraStream = useCallback(() => {
    if (cameraTimerRef.current !== null) {
      window.clearInterval(cameraTimerRef.current);
      cameraTimerRef.current = null;
    }
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.ondataavailable = null;
      recorder.onstop = null;
      recorder.onerror = null;
      try {
        recorder.stop();
      } catch {
        // The recorder may already be stopping as a permission/tab change occurs.
      }
    }
    mediaRecorderRef.current = null;
    cameraStreamRef.current?.getTracks().forEach(track => track.stop());
    cameraStreamRef.current = null;
    if (cameraVideoRef.current) {
      cameraVideoRef.current.pause();
      cameraVideoRef.current.srcObject = null;
      cameraVideoRef.current.removeAttribute('src');
      cameraVideoRef.current.load();
    }
    clearCameraPreview();
  }, [clearCameraPreview]);

  const openCamera = useCallback(async () => {
    if (profileNameTaken) {
      setUiNotice('That profile name is already in use. Choose a different name, such as “Alvin 2”.');
      minimalNameInputRef.current?.focus();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState('error');
      setCameraError('Camera capture is not supported by this browser.');
      return;
    }
    setCameraState('requesting');
    setCameraError(null);
    try {
      stopCameraStream();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 720 },
          height: { ideal: 960 },
        },
        audio: true,
      });
      cameraStreamRef.current = stream;
      setCameraElapsed(0);
      setCameraState('ready');
    } catch (error) {
      setCameraState('error');
      setCameraError(error instanceof DOMException && error.name === 'NotAllowedError'
        ? 'Camera permission was blocked. Allow camera and microphone access, then try again.'
        : `Could not open the camera: ${String(error)}`);
    }
  }, [profileNameTaken, stopCameraStream]);

  useEffect(() => {
    const video = cameraVideoRef.current;
    if (!video) return;
    if (cameraState === 'recorded' && cameraPreviewUrl) {
      video.srcObject = null;
      video.src = cameraPreviewUrl;
      video.loop = true;
      void video.play().catch(() => undefined);
      return;
    }
    video.loop = false;
    video.removeAttribute('src');
    const stream = cameraStreamRef.current;
    if (!stream) return;
    video.srcObject = stream;
    void video.play().catch(() => undefined);
  }, [cameraPreviewUrl, cameraState, sourceMode]);

  const stopCameraRecording = useCallback(() => {
    if (cameraTimerRef.current !== null) {
      window.clearInterval(cameraTimerRef.current);
      cameraTimerRef.current = null;
    }
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') recorder.stop();
  }, []);

  const startCameraRecording = useCallback(() => {
    const stream = cameraStreamRef.current;
    if (!stream || typeof MediaRecorder === 'undefined') {
      setCameraState('error');
      setCameraError('This browser cannot record video. Try Chrome or Safari with camera access enabled.');
      return;
    }
    const mimeType = [
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp8,opus',
      'video/webm',
      'video/mp4',
    ].find(type => MediaRecorder.isTypeSupported(type)) || '';
    try {
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      cameraChunksRef.current = [];
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = event => {
        if (event.data.size > 0) cameraChunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        setCameraState('error');
        setCameraError('Camera recording failed. Please try again.');
      };
      recorder.onstop = () => {
        const blobType = recorder.mimeType || mimeType || 'video/webm';
        const blob = new Blob(cameraChunksRef.current, { type: blobType });
        if (blob.size === 0) {
          setCameraState('error');
          setCameraError('No video was captured. Please try recording again.');
          return;
        }
        const extension = blobType.includes('mp4') ? 'mp4' : 'webm';
        const file = new File([blob], `pixelholo-camera-${Date.now()}.${extension}`, { type: blobType });
        setCameraPreview(blob);
        cameraStreamRef.current?.getTracks().forEach(track => track.stop());
        cameraStreamRef.current = null;
        mediaRecorderRef.current = null;
        setCameraState('recorded');
        setProfile(prev => ({
          ...prev,
          lastUploadedFile: file.name,
          fileSize: `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
        }));
        // Keep the capture in the browser until the user chooses Create
        // avatar. This lets someone record first and name the profile later.
        setCapturedCameraFile(file);
      };
      recorder.start(250);
      setCameraElapsed(0);
      setCameraState('recording');
      cameraTimerRef.current = window.setInterval(() => {
        setCameraElapsed(prev => {
          const next = prev + 1;
          if (next >= CAMERA_RECORDING_SECONDS) stopCameraRecording();
          return next;
        });
      }, 1000);
    } catch (error) {
      setCameraState('error');
      setCameraError(`Could not start recording: ${String(error)}`);
    }
  }, [setCameraPreview, stopCameraRecording]);

  useEffect(() => () => stopCameraStream(), [stopCameraStream]);

  const handleUploadAudio = async (file: File) => {
    if (!profile.name) return;
    setUiNotice(null);
    setStepStatuses(prev => ({ ...prev, upload: 'running' }));
    setUploadPhaseAudio('uploading');
    setUploadProgressAudio(0);
    setUploadBytesAudio({ loaded: 0, total: 0 });
    const form = new FormData();
    form.append('profile', profile.name);
    form.append('profile_type', profileType);
    form.append('file', file);
    try {
      const responseText = await uploadWithProgress(`${apiBase}/upload_audio`, form, (loaded, total) => {
        const pct = Math.round((loaded / Math.max(total, 1)) * 100);
        if (pct !== uploadAudioLastPctRef.current || loaded === total) {
          uploadAudioLastPctRef.current = pct;
          setUploadProgressAudio(pct);
        }
        setUploadBytesAudio({ loaded, total });
      });
      const data = JSON.parse(responseText);
      setLastUploadedAudioFilename(data.filename);
      setStepStatuses(prev => ({ ...prev, upload: 'done' }));
      setUploadPhaseAudio('idle');
    } catch (err) {
      setStepStatuses(prev => ({ ...prev, upload: 'error' }));
      setUploadPhaseAudio('error');
      setPreprocessLogs([createLog(`Audio upload failed: ${String(err)}`, 'error')]);
    }
  };

  const startPreprocess = async () => {
    if (!profile.name) return;
    if (profileNameTaken) {
      setUiNotice('That profile name is already in use. Choose a different name, such as “Alvin 2”.');
      minimalNameInputRef.current?.focus();
      return;
    }
    setUiNotice(null);
    setStepStatuses(prev => ({ ...prev, preprocess: 'running' }));
    setPreprocessLogs([createLog('Pipeline starting...', 'info')]);
    setPreprocessStats(null);
    setPreprocessProgress(0);
    setPreprocessStageIndex(preprocessSteps.length > 0 ? 0 : null);
    let sawError = false;
    const payload = {
      profile: profile.name,
      filename: lastUploadedFilename ?? null,
      audio_filename: lastUploadedAudioFilename ?? null,
      profile_type: profileType,
      bake_avatar: profileType === 'avatar',
      avatar_fps: profileType === 'avatar' ? 25 : null,
      avatar_start_sec: profileType === 'avatar' ? avatarStartSec : null,
      avatar_loop_sec: profileType === 'avatar' ? DEFAULT_AVATAR_LOOP_SEC : null,
      avatar_loop_fade_sec: profileType === 'avatar' ? DEFAULT_AVATAR_LOOP_FADE_SEC : null,
      avatar_resize_factor: profileType === 'avatar' ? 1 : null,
      avatar_pads: profileType === 'avatar' ? '0 10 0 0' : null,
      avatar_blur_background: profileType === 'avatar',
      avatar_blur_kernel: profileType === 'avatar' ? BLUR_KERNEL_BY_LEVEL[avatarBlurLevel] : null,
    };
    try {
      const res = await apiFetch(`${apiBase}/preprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      await streamResponseLines(res, line => {
        const errorLine = isErrorLine(line);
        if (errorLine) sawError = true;
        setPreprocessLogs(prev => [...prev, createLog(line, errorLine ? 'error' : 'info')]);
        const stageUpdate = (label: string) => {
          const idx = preprocessSteps.indexOf(label);
          if (idx >= 0) {
            setPreprocessStageIndex(prev => (prev === null || idx > prev ? idx : prev));
          }
        };
        if (line.includes('Baking avatar cache')) {
          stageUpdate('Bake avatar frames (Wav2Lip cache)');
        } else if (line.includes('Extracting audio')) {
          stageUpdate('Extract audio track');
        } else if (line.includes('Loaded audio')) {
          stageUpdate('Split on silence (2-10s)');
        } else if (line.includes('Transcribing full audio')) {
          stageUpdate('Transcribe with Whisper');
        } else if (line.includes('Segments: raw=')) {
          stageUpdate('Split on silence (2-10s)');
        } else if (line.includes('Exporting') || (line.includes('Wrote') && line.includes('.wav'))) {
          stageUpdate('Write metadata.csv');
        } else if (line.includes('Metadata written')) {
          stageUpdate('Write metadata.csv');
        }
        const match = line.match(/Segments: raw=(\d+) merged=(\d+) kept=(\d+)/);
        if (match) {
          const raw = Number(match[1]);
          const merged = Number(match[2]);
          const kept = Number(match[3]);
          setPreprocessStats({
            duration: '-',
            segmentsKept: kept,
            segmentsFiltered: merged - kept,
            avgClipLength: '-',
            sampleRate: '24 kHz',
          });
        }
        const wrote = line.match(/Wrote .* \((\d+)\/(\d+)\)/);
        if (wrote) {
          const current = Number(wrote[1]);
          const total = Number(wrote[2]);
          if (total > 0) {
            setPreprocessProgress(current / total);
          }
        }
      });
      if (sawError) {
        setStepStatuses(prev => ({ ...prev, preprocess: 'error' }));
        setPreprocessStageIndex(null);
        return;
      }
      setStepStatuses(prev => ({ ...prev, preprocess: 'done' }));
      setPreprocessProgress(1);
      setPreprocessStageIndex(preprocessSteps.length ? preprocessSteps.length - 1 : null);
      setIsCreatingProfile(false);
      loadProfiles();
    } catch (err) {
      setStepStatuses(prev => ({ ...prev, preprocess: 'error' }));
      setPreprocessLogs(prev => [...prev, createLog(`Preprocess failed: ${String(err)}`, 'error')]);
      setPreprocessStageIndex(null);
    }
  };

  useEffect(() => {
    if (
      !autoPrepareAfterUpload
      || !lastUploadedFilename
      || stepStatuses.upload !== 'done'
      || stepStatuses.preprocess !== 'idle'
    ) {
      return;
    }
    setAutoPrepareAfterUpload(false);
    void startPreprocess();
  }, [autoPrepareAfterUpload, lastUploadedFilename, startPreprocess, stepStatuses.preprocess, stepStatuses.upload]);

  const handleCreateAvatar = async () => {
    if (!profile.name.trim()) {
      setProfileNameRequired(true);
      setUiNotice('A profile name is required before you can create this avatar.');
      minimalNameInputRef.current?.focus();
      return;
    }
    if (profileNameTaken) {
      setUiNotice('That profile name is already in use. Choose a different name, such as “Alvin 2”.');
      minimalNameInputRef.current?.focus();
      return;
    }
    setProfileNameRequired(false);
    if (capturedCameraFile && !lastUploadedFilename) {
      await handleUpload(capturedCameraFile, { autoPrepare: true });
      return;
    }
    await startPreprocess();
  };

  const startTraining = async () => {
    if (!profile.name) return;
    setUiNotice(null);
    setStepStatuses(prev => ({ ...prev, train: 'running' }));
    setTrainLogs([createLog('Launching trainer...', 'info')]);
    setTrainStats(null);
    setTrainStageIndex(trainSteps.length > 0 ? 0 : null);
    let sawError = false;

    const payload = {
      profile: profile.name,
      profile_type: profileType,
      batch_size: trainParams.batchSize,
      epochs: trainParams.epochs,
      max_len: trainParams.maxLen,
      auto_select_epoch: trainFlags.autoSelectEpoch,
      auto_tune_profile: trainFlags.autoTuneProfile,
      auto_build_lexicon: trainFlags.autoBuildLexicon,
      early_stop: trainFlags.earlyStop,
    };

    try {
      const res = await apiFetch(`${apiBase}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      await streamResponseLines(res, line => {
        const errorLine = isErrorLine(line);
        if (errorLine) sawError = true;
        setTrainLogs(prev => [...prev, createLog(line, errorLine ? 'error' : 'info')]);
        const stageUpdate = (label: string) => {
          const idx = trainSteps.indexOf(label);
          if (idx >= 0) {
            setTrainStageIndex(prev => (prev === null || idx > prev ? idx : prev));
          }
        };
        if (line.includes('Base config') || line.includes('Patched config')) {
          stageUpdate('Patch config + load base model');
        }
        const epochMatch = line.match(/Epoch \[(\d+)\/(\d+)\], Step \[(\d+)\/(\d+)\]/);
        if (epochMatch) {
          stageUpdate('Train epochs & save checkpoints');
          const currentEpoch = Number(epochMatch[1]);
          const totalEpochs = Number(epochMatch[2]);
          const step = Number(epochMatch[3]);
          setTrainStats(prev => ({
            currentEpoch,
            totalEpochs,
            steps: prev?.steps ? prev.steps + step : step,
            eta: `${Math.max(0, totalEpochs - currentEpoch)} epochs`,
            gpuMemory: 'GPU active',
            bestCheckpoint: `outputs/training/${profileType}/${profile.name}`,
          }));
        }
        if (line.includes('Auto-tune') || line.includes('auto_tune')) {
          stageUpdate('Auto-tune profile defaults');
        }
        if (line.includes('Evaluating') || line.includes('Best checkpoint') || line.includes('Top checkpoints')) {
          stageUpdate('Auto-select best epoch');
        }
        if (line.toLowerCase().includes('lexicon.json')) {
          stageUpdate('Build lexicon.json');
        }
      });
      if (sawError) {
        setStepStatuses(prev => ({ ...prev, train: 'error' }));
        setTrainStageIndex(null);
        return;
      }
      setStepStatuses(prev => ({ ...prev, train: 'done' }));
      setTrainStageIndex(trainSteps.length ? trainSteps.length - 1 : null);
      loadProfiles();
    } catch (err) {
      setStepStatuses(prev => ({ ...prev, train: 'error' }));
      setTrainLogs(prev => [...prev, createLog(`Training failed: ${String(err)}`, 'error')]);
      setTrainStageIndex(null);
    }
  };

  const ensureAudioContext = async () => {
    const Ctx = window.AudioContext || (window as any).webkitAudioContext;
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
      audioContextRef.current = new Ctx();
    }
    if (audioContextRef.current.state !== 'running') {
      try {
        await audioContextRef.current.resume();
      } catch {
        // Safari may block without a user gesture.
      }
    }
  };

  const unlockAudio = async () => {
    await ensureAudioContext();
    const ctx = audioContextRef.current;
    if (!ctx || audioUnlockedRef.current) return;
    try {
      // Play a silent buffer to "unlock" Safari audio output.
      const buffer = ctx.createBuffer(1, 1, ctx.sampleRate);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.start(0);
      audioUnlockedRef.current = true;
    } catch {
      // ignore
    }
  };

  const interruptBackend = useCallback(async () => {
    try {
      await apiFetch(`${apiBase}/interrupt`, { method: 'POST', keepalive: true });
    } catch {
      // ignore best-effort interrupt failures
    }
  }, [apiBase, apiFetch]);

  const stopAllAudio = () => {
    for (const src of activeSourcesRef.current) {
      try {
        src.stop(0);
      } catch {}
    }
    activeSourcesRef.current.clear();
    nextStartTimeRef.current = 0;
    audioEndTimeRef.current = 0;
  };

  const resetAudio = async () => {
    stopAllAudio();
    if (audioContextRef.current?.state === 'running') {
      try {
        await audioContextRef.current.suspend();
      } catch {}
    }
  };

  const clearPlaybackSettleTimer = useCallback(() => {
    if (playbackSettleTimerRef.current !== null) {
      window.clearTimeout(playbackSettleTimerRef.current);
      playbackSettleTimerRef.current = null;
    }
  }, []);

  const settlePlayback = useCallback(() => {
    clearPlaybackSettleTimer();
    const check = () => {
      const audioContext = audioContextRef.current;
      const audioSettled = !audioContext || audioContext.currentTime + 0.05 >= audioEndTimeRef.current;
      const videoSettled = frameQueueRef.current.length === 0;
      if (!audioSettled || !videoSettled) {
        playbackSettleTimerRef.current = window.setTimeout(check, 100);
        return;
      }
      playbackSettleTimerRef.current = null;
      setIsPlaybackActive(false);
      setStepStatuses(prev => ({ ...prev, inference: 'done' }));
      videoStateRef.current = 'idle';
      setVideoState('idle');
      setVideoQueue(0);
    };
    check();
  }, [clearPlaybackSettleTimer]);

  const resetVideo = useCallback(() => {
    if (videoTimerRef.current !== null) {
      window.clearInterval(videoTimerRef.current);
      videoTimerRef.current = null;
    }
    if (videoRafRef.current !== null) {
      window.cancelAnimationFrame(videoRafRef.current);
      videoRafRef.current = null;
    }
    videoDrawSerialRef.current += 1;
    for (const item of frameQueueRef.current) {
      releaseFrameSource(item.frame);
    }
    frameQueueRef.current = [];
    setVideoQueue(0);
    videoStateRef.current = 'idle';
    setVideoState('idle');
    videoStartTimeRef.current = null;
    videoNextFrameTimeRef.current = null;
    // A profile switch must never leave the last avatar frame visible while
    // the next profile is warming up.  Clear the drawing surface completely;
    // the preview shell will show its neutral background until new frames
    // arrive.
    const canvas = videoCanvasRef.current;
    const context = canvas?.getContext('2d');
    if (canvas && context) {
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [releaseFrameSource]);

  const isFramePendingDecode = useCallback((frameSource: VideoFrameSource) => (
    typeof frameSource !== 'string'
    && !!frameSource.bitmapPromise
    && frameSource.bitmapState === 'pending'
  ), []);

  const waitForFramePredecode = useCallback(async (frames?: VideoFrameSource[]) => {
    const pending = (frames ?? [])
      .filter((frame): frame is Exclude<VideoFrameSource, string> => (
        typeof frame !== 'string'
        && !!frame.bitmapPromise
        && frame.bitmapState === 'pending'
      ))
      .slice(0, 4)
      .map(frame => frame.bitmapPromise!);
    if (!pending.length) return;
    await Promise.race([
      Promise.allSettled(pending),
      new Promise(resolve => window.setTimeout(resolve, VIDEO_FRAME_DECODE_PREWAIT_MS)),
    ]);
  }, []);

  const drawFrame = useCallback((frameSource: VideoFrameSource) => {
    const canvas = videoCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    // Keep the final portrait upscale smooth on both high-DPI monitors and
    // browsers that render the streamed frame below the canvas's CSS size.
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    const drawSerial = ++videoDrawSerialRef.current;
    const sourceUrl = typeof frameSource === 'string' ? frameSource : frameSource.url;
    const bitmap = typeof frameSource === 'string' ? undefined : frameSource.bitmap;
    const cleanup = () => {
      releaseFrameSource(frameSource);
    };
    const drawCover = (source: CanvasImageSource, sourceWidth: number, sourceHeight: number) => {
      if (!sourceWidth || !sourceHeight) return;
      const targetRatio = canvas.width / canvas.height;
      const sourceRatio = sourceWidth / sourceHeight;
      let sx = 0;
      let sy = 0;
      let sw = sourceWidth;
      let sh = sourceHeight;
      if (sourceRatio > targetRatio) {
        sw = sourceHeight * targetRatio;
        sx = (sourceWidth - sw) / 2;
      } else if (sourceRatio < targetRatio) {
        sh = sourceWidth / targetRatio;
        sy = (sourceHeight - sh) / 2;
      }
      ctx.drawImage(source, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    };
    if (bitmap) {
      if (drawSerial === videoDrawSerialRef.current) {
        drawCover(bitmap, bitmap.width, bitmap.height);
      }
      cleanup();
      return;
    }
    if (
      typeof frameSource !== 'string'
      && frameSource.bitmapPromise
      && frameSource.bitmapState === 'pending'
    ) {
      cleanup();
      return;
    }
    const img = new Image();
    img.onload = () => {
      if (drawSerial === videoDrawSerialRef.current) {
        drawCover(img, img.naturalWidth || img.width, img.naturalHeight || img.height);
      }
      cleanup();
    };
    img.onerror = () => {
      cleanup();
    };
    img.src = sourceUrl.startsWith('blob:') || sourceUrl.startsWith('data:')
      ? sourceUrl
      : `data:image/jpeg;base64,${sourceUrl}`;
  }, [releaseFrameSource]);

  const startVideoLoop = useCallback(() => {
    if (videoRafRef.current !== null) return;
    const tick = () => {
      const ctx = audioContextRef.current;
      if (!ctx) {
        videoRafRef.current = window.requestAnimationFrame(tick);
        return;
      }
      const startAt = videoStartTimeRef.current;
      if (startAt !== null) {
        const fps = Math.max(5, videoFpsRef.current || 25);
        if (videoNextFrameTimeRef.current === null) {
          videoNextFrameTimeRef.current = startAt;
        }
        const now = ctx.currentTime;
        let frameToDraw: QueuedVideoFrame | null = null;
        while (frameQueueRef.current.length > 0 && frameQueueRef.current[0].t <= now) {
          const candidate = frameQueueRef.current[0];
          if (isFramePendingDecode(candidate.frame)) {
            break;
          }
          const shifted = frameQueueRef.current.shift() || null;
          if (frameToDraw) {
            releaseFrameSource(frameToDraw.frame);
          }
          frameToDraw = shifted;
        }
        if (frameToDraw) {
          drawFrame(frameToDraw.frame);
        }
        if (!frameQueueRef.current.length && videoStateRef.current === 'playing') {
          videoStateRef.current = 'buffering';
          setVideoState('buffering');
        } else if (frameQueueRef.current.length && videoStateRef.current !== 'playing') {
          videoStateRef.current = 'playing';
          setVideoState('playing');
        }
        setVideoQueue(frameQueueRef.current.length);
      }
      videoRafRef.current = window.requestAnimationFrame(tick);
    };
    videoRafRef.current = window.requestAnimationFrame(tick);
  }, [drawFrame, isFramePendingDecode, releaseFrameSource]);

  const enqueueFrames = useCallback((frames: VideoFrameSource[], startAt: number, duration: number, fps?: number) => {
    if (!frames || frames.length === 0) return;
    if (fps && fps > 0) {
      setVideoFps(fps);
      videoFpsRef.current = fps;
    }
    const frameCount = frames.length;
    const frameDuration = frameCount > 0 ? duration / frameCount : 0;
    frames.forEach((frame, i) => {
      frameQueueRef.current.push({ frame, t: startAt + i * frameDuration });
    });
    setVideoQueue(frameQueueRef.current.length);
    if (videoStateRef.current === 'idle') {
      videoStateRef.current = 'buffering';
      setVideoState('buffering');
    }
    startVideoLoop();
  }, [startVideoLoop]);

  const scheduleBuffer = (buffer: AudioBuffer) => {
    if (!audioContextRef.current) return;
    const ctx = audioContextRef.current;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    activeSourcesRef.current.add(source);
    source.onended = () => activeSourcesRef.current.delete(source);

    // Backend already handles chunk-boundary stitching/crossfades.
    // Frontend should focus on stable scheduling with enough lead time.
    const desiredStart = nextStartTimeRef.current;
    const minLead = outputMode === 'avatar' ? AVATAR_AUDIO_CHUNK_LEAD_SEC : 0.03;
    const startAt = Math.max(ctx.currentTime + minLead, desiredStart);
    const endAt = startAt + buffer.duration;

    source.start(startAt);
    nextStartTimeRef.current = endAt;
    audioEndTimeRef.current = Math.max(audioEndTimeRef.current, endAt);
    return { startAt, endAt };
  };

  const runInference = useCallback(async (text: string, endpoint: string) => {
    if (!profile.name || !text) return;
    const selectionId = profileSelectionRef.current;
    const requestProfileName = profile.name;
    const requestProfileType = profileType;
    if (isWarmingUp) {
      setUiNotice(`Preparing ${requestProfileName} for its first response. Please wait a moment.`);
      return;
    }
    setUiNotice(null);
    clearPlaybackSettleTimer();
    setIsPlaybackActive(true);
    await unlockAudio();
    // Profile selection can happen while the browser is unlocking audio. Do
    // not let this stale invocation create a stream with the old voice.
    if (selectionId !== profileSelectionRef.current) return;
    if (audioContextRef.current?.state !== 'running') {
      clearPlaybackSettleTimer();
      setIsPlaybackActive(false);
      setUiNotice('Safari blocked audio. Click again to enable sound.');
      return;
    }
    // Startup/profile-switch warmup handles cache preparation. Starting a warmup
    // here can compete with the inference worker and make identical prompts slower.
    // End any existing stream immediately.
    streamSessionRef.current += 1;
    stopAllAudio();
    resetVideo();
    setStepStatuses(prev => ({ ...prev, inference: 'running' }));
    setInferenceChunks([]);
    setLatency(null);
    setInferenceStageIndex(inferenceSteps.length > 0 ? 0 : null);
    // Set lead first, then schedule against it.
    audioStartDelayRef.current = outputMode === 'avatar'
      ? AVATAR_AUDIO_START_DELAY_SEC
      : DEFAULT_AUDIO_START_DELAY_SEC;
    nextStartTimeRef.current = (audioContextRef.current?.currentTime || 0) + audioStartDelayRef.current;
    audioEndTimeRef.current = nextStartTimeRef.current;
    let sawError = false;

    if (streamAbortRef.current && streamRunningRef.current) {
      streamAbortRef.current.abort();
      await interruptBackend();
    }
    if (selectionId !== profileSelectionRef.current) return;
    const controller = new AbortController();
    streamAbortRef.current = controller;
    streamRunningRef.current = true;
    const sessionId = streamSessionRef.current;

    const startTime = performance.now();
    let firstChunk = true;

    const payload: Record<string, any> = {
      speaker: requestProfileName,
      profile_type: requestProfileType,
      tts_backend: ttsBackend,
      text,
      model_path: modelOverride || null,
      ref_wav_path: refOverride || null,
    };
    if (endpoint === '/chat') {
      payload.llm_mode = llmMode;
    }
    if ((voiceControlsStatus === 'ready' || voiceControlsDirty) && voiceControlValues) {
      payload.pace_scale = resolvePaceScale(voiceControlValues.pace);
      payload.volume_gain = resolveVolumeGain(voiceControlValues.volume);
      if (ttsBackend === 'chatterbox') {
        payload.tts_exaggeration = voiceControlValues.expressiveness;
        payload.tts_temperature = voiceControlValues.variation;
        payload.tts_cfg_weight = voiceControlValues.guidance;
        payload.tts_repetition_penalty = voiceControlValues.repetition;
        payload.avatar_emotion = voiceControlValues.emotion;
        payload.avatar_emotion_intensity = voiceControlValues.emotionIntensity;
      } else {
        payload.pitch_shift = voiceControlValues.pitch;
        const toneOverrides = resolveToneOverrides(voiceControlValues.tone);
        payload.f0_scale = toneOverrides.f0Scale;
        payload.embedding_scale = toneOverrides.embeddingScale;
      }
    }
    if (outputMode === 'avatar') {
      payload.avatar_profile = requestProfileName;
      payload.lipsync_backend = avatarBackend;
      if (Number.isFinite(DEFAULT_WEB_AVATAR_MAX_FRAME_EDGE) && DEFAULT_WEB_AVATAR_MAX_FRAME_EDGE > 0) {
        payload.avatar_max_frame_edge = Math.round(DEFAULT_WEB_AVATAR_MAX_FRAME_EDGE);
      }
      if (avatarBackend === 'musetalk') {
        payload.musetalk_preset = museTalkPreset;
      }
    }

    const finishStream = (inferenceMs?: number) => {
      if (sawError) return;
      setLatency(prev => prev ? { ...prev, total: inferenceMs ?? Math.round(performance.now() - startTime) } : null);
      setStepStatuses(prev => ({ ...prev, inference: 'done' }));
      setInferenceStageIndex(inferenceSteps.length ? inferenceSteps.length - 1 : null);
      settlePlayback();
      if (streamAbortRef.current === controller) {
        streamAbortRef.current = null;
      }
      streamRunningRef.current = false;
    };

    const failStream = () => {
      sawError = true;
      clearPlaybackSettleTimer();
      setIsPlaybackActive(false);
      setStepStatuses(prev => ({ ...prev, inference: 'error' }));
      setInferenceStageIndex(null);
    };

    const decodeBase64 = (value: string) => {
      const binary = atob(value);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      return bytes;
    };

    const revokeFrameSources = (sources?: VideoFrameSource[]) => {
      for (const source of sources ?? []) {
        releaseFrameSource(source);
      }
    };

    const isSessionActive = () => (
      sessionId === streamSessionRef.current
      && selectionId === profileSelectionRef.current
      && streamAbortRef.current === controller
      && !controller.signal.aborted
      && !sawError
    );

    const decodeAudioChunk = async (data: BinaryStreamPacketMetadata | any, bytes: Uint8Array) => {
      const audioContext = audioContextRef.current!;
      if (data.audio_format === 'pcm_s16le') {
        const sampleRate = Math.max(1, Math.round(data.sample_rate || audioContext.sampleRate));
        const channels = Math.max(1, Math.round(data.audio_channels || 1));
        const availableSamples = Math.floor(bytes.byteLength / (2 * channels));
        const sampleCount = Math.max(0, Math.min(Math.round(data.audio_samples || availableSamples), availableSamples));
        const buffer = audioContext.createBuffer(channels, sampleCount, sampleRate);
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        for (let channel = 0; channel < channels; channel += 1) {
          const output = buffer.getChannelData(channel);
          for (let i = 0; i < sampleCount; i += 1) {
            const offset = ((i * channels) + channel) * 2;
            output[i] = view.getInt16(offset, true) / 32768;
          }
        }
        return buffer;
      }
      return audioContext.decodeAudioData(uint8ToArrayBuffer(bytes));
    };

    const processChunk = async (data: any, audioBytes?: Uint8Array, frames?: VideoFrameSource[]) => {
        if (!isSessionActive()) {
          revokeFrameSources(frames);
          return;
        }
        if (data.event === 'error') {
          revokeFrameSources(frames);
          failStream();
          return;
        }
        if (data.event === 'done') {
          revokeFrameSources(frames);
          finishStream(data.inference_ms);
          return;
        }
        const bytes = audioBytes ?? (data.audio_base64 ? decodeBase64(data.audio_base64) : null);
        if (!bytes || bytes.byteLength === 0) {
          revokeFrameSources(frames);
          return;
        }
        if (firstChunk) {
          firstChunk = false;
          setLatency({ ttfa: Math.round(performance.now() - startTime), total: 0 });
          setInferenceStageIndex(Math.min(2, inferenceSteps.length - 1));
        }
        let buffer: AudioBuffer;
        try {
          buffer = await decodeAudioChunk(data, bytes);
        } catch {
          // Malformed or empty audio — skip this chunk rather than crashing the stream.
          revokeFrameSources(frames);
          return;
        }
        if (!isSessionActive()) {
          revokeFrameSources(frames);
          return;
        }
        // Zero-duration buffer would not advance nextStartTimeRef, causing all
        // subsequent chunks to pile up at the same start time and play as static.
        if (buffer.duration < 0.005) {
          revokeFrameSources(frames);
          return;
        }
        const frameSources = frames ?? data.frames_base64;
        const schedule = scheduleBuffer(buffer);
        if (outputMode === 'avatar' && schedule && videoStartTimeRef.current === null) {
          videoStartTimeRef.current = schedule.startAt;
          videoNextFrameTimeRef.current = schedule.startAt;
        }
        if (outputMode === 'avatar' && schedule && Array.isArray(frameSources)) {
          const duration = typeof data.duration_sec === 'number' ? data.duration_sec : buffer.duration;
          enqueueFrames(frameSources, schedule.startAt, duration, data.fps);
          void waitForFramePredecode(frameSources);
          setInferenceStageIndex(Math.min(3, inferenceSteps.length - 1));
        }
        if (outputMode === 'voice') {
          setInferenceStageIndex(Math.min(4, inferenceSteps.length - 1));
        }
        setInferenceChunks(prev => [...prev, { index: data.chunk_index, duration: buffer.duration, receivedAt: Date.now() }]);
    };

    try {
      const res = await apiFetch(`${apiBase}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(BINARY_STREAM_ENABLED
            ? {
                Accept: BINARY_STREAM_MEDIA_TYPE,
                'X-PixelHolo-Transport': 'binary',
                'X-PixelHolo-Client': 'web',
                ...(BINARY_PCM_AUDIO_ENABLED ? { 'X-PixelHolo-Audio-Format': 'pcm_s16le' } : {}),
              }
            : {}),
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(await res.text());
      if (BINARY_STREAM_ENABLED && isBinaryStreamResponse(res)) {
        await decodeBinaryStream(res, async packet => {
          if (!isSessionActive()) return;
          const audioLength = Math.max(0, packet.metadata.audio_bytes_len ?? 0);
          if (packet.payload.length < audioLength) {
            throw new Error('Invalid PixelHolo binary audio payload.');
          }
          const audioBytes = packet.payload.slice(0, audioLength);
          const framePayload = packet.payload.slice(audioLength);
          const frameSources: VideoFrameSource[] = [];
          let cursor = 0;
          for (const frameLength of packet.metadata.frame_lengths ?? []) {
            if (frameLength < 0 || cursor + frameLength > framePayload.length) {
              throw new Error('Invalid PixelHolo binary frame payload.');
            }
            const frameBytes = framePayload.slice(cursor, cursor + frameLength);
            cursor += frameLength;
            const blob = new Blob([frameBytes], { type: 'image/jpeg' });
            const frameSource: Exclude<VideoFrameSource, string> = {
              url: URL.createObjectURL(blob),
            };
            if ('createImageBitmap' in window) {
              frameSource.bitmapState = 'pending';
              frameSource.bitmapPromise = createImageBitmap(blob)
                .then(bitmap => {
                  if (frameSource.drawn) {
                    bitmap.close();
                    frameSource.bitmapState = 'failed';
                    return null;
                  }
                  frameSource.bitmap = bitmap;
                  frameSource.bitmapState = 'ready';
                  return bitmap;
                })
                .catch(() => {
                  frameSource.bitmapState = 'failed';
                  return null;
                });
            }
            frameSources.push(frameSource);
          }
          await processChunk(packet.metadata, audioBytes, frameSources);
        }, controller.signal);
      } else {
        await streamResponseLines(res, async line => {
          let data: any;
          try {
            data = JSON.parse(line);
          } catch {
            failStream();
            return;
          }
          await processChunk(data);
        });
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        clearPlaybackSettleTimer();
        setIsPlaybackActive(false);
        setStepStatuses(prev => ({ ...prev, inference: 'error' }));
        setInferenceStageIndex(null);
      }
    } finally {
      if (streamAbortRef.current === controller) {
        streamAbortRef.current = null;
      }
      streamRunningRef.current = false;
    }
  }, [
    apiBase,
    apiFetch,
    avatarBackend,
    clearPlaybackSettleTimer,
    enqueueFrames,
    interruptBackend,
    isWarmingUp,
    llmMode,
    museTalkPreset,
    modelOverride,
    outputMode,
    profile.name,
    profileType,
    refOverride,
    releaseFrameSource,
    resetVideo,
    settlePlayback,
    ttsBackend,
    unlockAudio,
    waitForFramePredecode,
    voiceControlValues,
    voiceControlsStatus,
    voiceControlsDirty,
    resolvePaceScale,
    resolveToneOverrides,
    resolveVolumeGain,
  ]);

  const startInference = useCallback(async () => {
    await runInference(inferenceText, '/speak');
  }, [inferenceText, runInference]);

  const stopInference = async () => {
    stopListeningRef.current?.();
    stopListening();
    if (streamAbortRef.current) streamAbortRef.current.abort();
    streamAbortRef.current = null;
    streamRunningRef.current = false;
    // Invalidate stale stream callbacks before waiting for the backend's
    // best-effort interrupt response.
    streamSessionRef.current += 1;
    clearPlaybackSettleTimer();
    setIsPlaybackActive(false);
    await interruptBackend();
    await resetAudio();
    resetVideo();
    setStepStatuses(prev => ({ ...prev, inference: 'idle' }));
    setInferenceStageIndex(null);
  };

  const trainingCommand = [
    'python src/train.py',
    `--dataset_path ./data/${profileType === 'avatar' ? 'avatar_profiles' : 'voice_profiles'}/${profile.name || '<profile>'}`,
    `--profile_type ${profileType}`,
    `--batch_size ${trainParams.batchSize}`,
    `--epochs ${trainParams.epochs}`,
    `--max_len ${trainParams.maxLen}`,
    trainFlags.autoSelectEpoch ? '--auto_select_epoch' : '',
    trainFlags.autoTuneProfile ? '--auto_tune_profile' : '',
    trainFlags.autoBuildLexicon ? '--auto_build_lexicon' : '',
    trainFlags.earlyStop ? '--early_stop' : '--no_early_stop',
  ]
    .filter(Boolean)
    .join(' ');

  const preprocessDisplayProgress =
    stepStatuses.preprocess === 'running'
      ? Math.max(
          preprocessProgress ?? 0,
          stageProgress(preprocessStageIndex, preprocessSteps.length, 0.6),
        )
      : stepStatuses.preprocess === 'done'
        ? 1
        : null;
  const trainEpochProgress =
    trainStats?.totalEpochs && trainStats.totalEpochs > 0
      ? Math.min(1, trainStats.currentEpoch / trainStats.totalEpochs)
      : 0;
  const trainDisplayProgress =
    stepStatuses.train === 'running'
      ? Math.max(trainEpochProgress, stageProgress(trainStageIndex, trainSteps.length, 0.2))
      : stepStatuses.train === 'done'
        ? 1
        : null;
  const trainingStatusCard = (
    <div className="bg-amber-50 border border-amber-200 rounded-2xl p-5 shadow-sm">
      <p className="text-[10px] font-bold uppercase tracking-widest text-amber-600">Training Progress</p>
      {trainStats ? (
        <div className="mt-4 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[10px] font-bold text-amber-700 uppercase">Epoch</p>
              <p className="text-3xl font-bold text-slate-900">{((trainStats.currentEpoch / trainStats.totalEpochs) * 100).toFixed(0)}%</p>
            </div>
            <div className="text-right">
              <p className="text-[10px] font-bold text-amber-700 uppercase">Time Remaining</p>
              <p className="text-lg font-bold text-slate-800">{trainStats.eta}</p>
            </div>
          </div>
          <div className="h-4 bg-amber-200 rounded-full overflow-hidden">
            <div className="h-full bg-amber-600 transition-all duration-700" style={{ width: `${(trainStats.currentEpoch / trainStats.totalEpochs) * 100}%` }}></div>
          </div>
          <div className="grid grid-cols-2 gap-4 text-xs font-bold">
            <div className="bg-white p-2 rounded border border-amber-100">GPU: {trainStats.gpuMemory}</div>
            <div className="bg-white p-2 rounded border border-amber-100">Steps: {trainStats.steps.toLocaleString()}</div>
          </div>
        </div>
      ) : (
        <div className="mt-4 text-xs text-amber-800">
          Initializing trainer and loading checkpoints...
        </div>
      )}
    </div>
  );

  const preprocessStatusCard = (
    <div className="bg-amber-50 border border-amber-200 rounded-2xl p-5 shadow-sm">
      <p className="text-[10px] font-bold uppercase tracking-widest text-amber-600">Preprocess Progress</p>
      <div className="mt-4 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] font-bold text-amber-700 uppercase">Segments</p>
            <p className="text-3xl font-bold text-slate-900">
              {preprocessStats?.segmentsKept ?? 0}
            </p>
          </div>
          <div className="text-right">
            <p className="text-[10px] font-bold text-amber-700 uppercase">Status</p>
            <p className="text-lg font-bold text-slate-800">
              {stepStatuses.preprocess === 'running' ? 'Processing' : 'Waiting'}
            </p>
          </div>
        </div>
        <div className="h-4 bg-amber-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-amber-600 transition-all duration-700"
            style={{ width: `${Math.round((preprocessDisplayProgress ?? 0) * 100)}%` }}
          ></div>
        </div>
        <div className="grid grid-cols-2 gap-4 text-xs font-bold">
          <div className="bg-white p-2 rounded border border-amber-100">
            Kept: {preprocessStats?.segmentsKept ?? '-'}
          </div>
          <div className="bg-white p-2 rounded border border-amber-100">
            Filtered: {preprocessStats?.segmentsFiltered ?? '-'}
          </div>
        </div>
      </div>
    </div>
  );

  // Once a zero-shot profile has been prepared, open the studio directly. The
  // old training step is intentionally left in this file for the legacy worker,
  // but it is not part of the public web flow.
  useEffect(() => {
    if (
      hasInferenceProfile
      && profile.name
      && activeStep === 1
      && (stepStatuses.preprocess === 'done' || stepStatuses.train === 'done')
    ) {
      setActiveStep(4);
    }
  }, [activeStep, hasInferenceProfile, profile.name, stepStatuses.preprocess, stepStatuses.train]);

  const publicSetupReady = Boolean(
    !profileNameTaken
      && profile.name
      && (capturedCameraFile || lastUploadedFilename || (currentProfileInfo?.raw_files ?? 0) > 0),
  );
  const publicSourceReady = Boolean(
    capturedCameraFile || lastUploadedFilename || (currentProfileInfo?.raw_files ?? 0) > 0,
  );
  const publicProfileReady = Boolean(!isCreatingProfile && hasInferenceProfile && profile.name);

  const sendComposerText = async () => {
    const text = inferenceText.trim();
    if (!text) return;
    await runInference(text, composerMode === 'chat' ? '/chat' : '/speak');
  };

  const applyVoiceInput = useCallback((spoken: string) => {
    setInferenceText(spoken);
    void runInference(spoken, '/chat');
  }, [runInference]);
  const {
    isListening,
    startListening,
    stopListening,
    hasSupport: hasSpeechSupport,
    transcript: speechTranscript,
  } = useSpeechToText(applyVoiceInput);
  const composerDisplayText = isListening ? speechTranscript : inferenceText;

  const selectProfile = async (item: ProfileInfo) => {
    if (isProfileSwitchBlocked) {
      setUiNotice('Finish the current profile preparation before switching avatars.');
      return;
    }
    const selectionId = ++profileSelectionRef.current;
    // Invalidate the old request synchronously, then update the visible
    // profile immediately. The backend interrupt can finish in the background;
    // no stale callback can pass the selection/session guards meanwhile.
    const stopPromise = stopInference();
    stopCameraStream();
    const selectedProfileType: ProfileType = item.profile_type === 'voice' ? 'voice' : 'avatar';
    const ready = profileIsInferenceReady(item);
    setShowWelcome(false);
    setProfileType(selectedProfileType);
    setProfile({ name: item.name, lastUploadedFile: null, fileSize: null });
    setIsCreatingProfile(false);
    setAutoPrepareAfterUpload(false);
    setLastUploadedFilename(null);
    setLastUploadedAudioFilename(null);
    setInferenceText('');
    setInferenceChunks([]);
    setLatency(null);
    setUiNotice(null);
    setActiveStep(ready ? 4 : 1);
    await stopPromise;
    if (selectionId !== profileSelectionRef.current) return;
    if (ready) {
      try {
        await warmupProfile(item.name, selectedProfileType);
        if (selectionId === profileSelectionRef.current) {
          setUiNotice(null);
        }
      } catch (error) {
        if (selectionId === profileSelectionRef.current) {
          setUiNotice(`Could not warm up ${item.name}: ${String(error)}`);
        }
      }
    }
  };

  const resetNewProfile = () => {
    if (isBusy || isPlaybackActive) {
      setUiNotice('Stop the current response before creating a new profile.');
      return;
    }
    profileSelectionRef.current += 1;
    streamSessionRef.current += 1;
    stopListeningRef.current?.();
    stopListening();
    stopAllAudio();
    resetVideo();
    stopCameraStream();
    setShowWelcome(false);
    setIsCreatingProfile(true);
    setAutoPrepareAfterUpload(false);
    setSourceMode('camera');
    setCameraState('idle');
    setCameraError(null);
    setCameraElapsed(0);
    setCapturedCameraFile(null);
    setProfileNameRequired(false);
    setProfile({ name: '', lastUploadedFile: null, fileSize: null });
    setLastUploadedFilename(null);
    setLastUploadedAudioFilename(null);
    setPreprocessStats(null);
    setPreprocessProgress(null);
    setStepStatuses(prev => ({ ...prev, upload: 'idle', preprocess: 'idle', inference: 'idle' }));
    setActiveStep(1);
    setUiNotice(null);
  };

  return (
    <div className="ph-minimal-app">
      <header className="ph-minimal-header">
        <button type="button" className="ph-minimal-brand" onClick={() => setShowWelcome(true)} aria-label="Go to PixelHolo home">
          <span className="ph-minimal-mark">✦</span>
        <span>PixelHolo</span>
      </button>
      <div className="ph-minimal-header-actions">
          <span className="ph-minimal-status"><span className={`ph-status-dot ${apiStatus}`} />{apiStatus === 'online' ? 'Ready' : apiStatus === 'checking' ? 'Connecting' : 'Offline'}</span>
          {!IS_PRODUCTION_BUILD && (
            <details className="ph-minimal-connection">
              <summary aria-label="Connection settings">•••</summary>
              <label>API endpoint<input value={apiBase} onChange={event => setApiBase(event.target.value)} /></label>
            </details>
          )}
        </div>
      </header>

      {showWelcome ? (
        <main className="ph-welcome-page">
          <section className="ph-welcome-hero">
            <div className="ph-welcome-copy">
              <span className="ph-minimal-eyebrow">One-shot avatar studio</span>
              <h1>Turn one video<br /><em>into a voice avatar.</em></h1>
              <p>Upload a 5–20 second talking-head video or record one with your camera. PixelHolo reuses the face and voice from that single clip, then streams Chatterbox speech with MuseTalk lip sync.</p>
              <button type="button" className="ph-welcome-cta" onClick={resetNewProfile}>Create a profile <span>→</span></button>
              <div className="ph-welcome-stack"><span className="ph-welcome-stack-mark">✦</span><span>One source clip</span><i>→</i><span>One speaking avatar</span></div>
            </div>
            <div className="ph-welcome-visual" aria-hidden="true">
              <div className="ph-welcome-orbit ph-welcome-orbit-one" />
              <div className="ph-welcome-orbit ph-welcome-orbit-two" />
              <div className="ph-welcome-glow" />
              <div className="ph-welcome-card">
                <div className="ph-welcome-card-top"><span>PROFILE PREVIEW</span><span><i /> Ready</span></div>
                <div className="ph-welcome-avatar-art"><span>✦</span><div className="ph-welcome-avatar-ring" /></div>
                <strong>One clip in. A speaking avatar out.</strong>
                <small>Chatterbox voice · MuseTalk lip sync</small>
                <div className="ph-welcome-wave"><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
              </div>
            </div>
          </section>
          <section className="ph-welcome-features">
            <div><span>01</span><strong>Bring a source clip</strong><p>Upload a talking video or record a guided 20-second sample. The same video supplies the face and voice.</p></div>
            <div><span>02</span><strong>Prepare the profile</strong><p>PixelHolo cleans the audio and bakes a reusable portrait cache. New profiles skip a separate training step.</p></div>
            <div><span>03</span><strong>Ask by text or voice</strong><p>Type a prompt or use your microphone. Chatterbox speech and MuseTalk frames stream back together.</p></div>
          </section>
          <section className="ph-welcome-bottom"><span>Anonymous device workspace · profiles stay with this browser</span><button type="button" onClick={resetNewProfile}>Start with a talking video <b>↗</b></button></section>
        </main>
      ) : (
      <div className="ph-minimal-body">
        <aside className="ph-minimal-sidebar" aria-label="Avatar profiles">
          <div className="ph-minimal-sidebar-heading">
            <div><span className="ph-minimal-eyebrow">Workspace</span><strong>Profiles</strong></div>
            <span className="ph-minimal-profile-count">{profiles.length}</span>
          </div>
          <button type="button" className={`ph-minimal-new-profile ${!profile.name ? 'is-current' : ''}`} onClick={resetNewProfile}>
            <span className="ph-minimal-new-profile-icon">＋</span>
            <span><strong>New profile</strong><small>Upload or record a talking video</small></span>
          </button>

          <div className="ph-minimal-sidebar-section-label">Existing profiles</div>
          <div className="ph-minimal-profile-list">
            {profilesStatus === 'loading' && <div className="ph-minimal-sidebar-empty">Loading profiles…</div>}
            {profilesStatus === 'error' && <div className="ph-minimal-sidebar-empty is-error">Could not load profiles.</div>}
            {profilesStatus === 'idle' && profiles.length === 0 && (
              <div className="ph-minimal-sidebar-empty">Your new profile will appear here after it is prepared.</div>
            )}
            {profiles.map(item => {
              const selected = item.name === profile.name;
              const ready = profileIsInferenceReady(item);
              const menuKey = `${item.profile_type || 'avatar'}:${item.name}`;
              return (
                <div key={menuKey} className={`ph-minimal-profile-item ${selected ? 'is-selected' : ''}`}>
                  <button type="button" className="ph-minimal-profile-select" onClick={() => void selectProfile(item)} disabled={isProfileSwitchBlocked}>
                    <span className="ph-minimal-profile-avatar">{item.name.slice(0, 1).toUpperCase()}</span>
                    <span className="ph-minimal-profile-copy"><strong>{item.name}</strong><small>{ready ? 'Ready to speak' : item.has_data ? 'Needs preparation' : 'Needs a source clip'}</small></span>
                    <span className={`ph-minimal-profile-dot ${ready ? 'is-ready' : ''}`} />
                  </button>
                  <div className="ph-minimal-profile-menu-wrap" onClick={event => event.stopPropagation()}>
                    <button type="button" className="ph-minimal-profile-options" disabled={isBusy} aria-label={`Profile options for ${item.name}`} onClick={() => setProfileMenuKey(prev => prev === menuKey ? null : menuKey)}>•••</button>
                    {profileMenuKey === menuKey && (
                      <div className="ph-minimal-profile-menu">
                        <button type="button" onClick={() => void handleRenameProfile(item)}>Rename</button>
                        <button type="button" className="is-danger" onClick={() => void handleDeleteProfile(item)}>Delete</button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="ph-minimal-sidebar-footer">
            <span className={`ph-status-dot ${apiStatus}`} />
            <div><strong>{apiStatus === 'online' ? 'Engine ready' : apiStatus === 'checking' ? 'Connecting…' : 'Engine offline'}</strong><small>Chatterbox · {avatarBackend === 'wav2lip' ? 'Wav2Lip' : 'MuseTalk'}</small></div>
          </div>
        </aside>

        <main className="ph-minimal-main">
        {!publicProfileReady && (
          <section className="ph-minimal-create">
            <div className="ph-minimal-intro">
              <span className="ph-minimal-eyebrow">New avatar</span>
              <h1>Make a voice avatar.</h1>
              <p>Upload a short video of yourself talking. PixelHolo uses the face and voice from that one clip, then you can type what you want your avatar to say.</p>
            </div>

            <div className="ph-minimal-create-card">
              {(() => {
                const sourceUploaded = Boolean(
                  profile.lastUploadedFile
                  || (currentProfileInfo?.raw_files ?? 0) > 0,
                );
                return (
                  <>
              <label className="ph-minimal-label" htmlFor="minimal-avatar-name">Name your avatar</label>
              <input
                id="minimal-avatar-name"
                className="ph-minimal-name-input"
                ref={minimalNameInputRef}
                value={profile.name}
                onChange={event => {
                  setProfile(prev => ({ ...prev, name: event.target.value }));
                  if (event.target.value.trim()) setProfileNameRequired(false);
                  setUiNotice(null);
                }}
                placeholder="Enter a profile name"
                autoComplete="off"
                spellCheck={false}
                aria-invalid={profileNameTaken || (profileNameRequired && !profile.name.trim())}
                disabled={isBusy}
              />
              {profileNameTaken && (
                <div className="ph-minimal-name-availability" role="alert">
                  Profile name already exists. Choose another name, such as <strong>“{profile.name.trim()} 2”</strong>.
                </div>
              )}

              <div className="ph-minimal-source-tabs" role="tablist" aria-label="Avatar source">
                <button
                  type="button"
                  role="tab"
                  aria-selected={sourceMode === 'camera'}
                  className={sourceMode === 'camera' ? 'is-selected' : ''}
                  disabled={isBusy}
                  onClick={() => {
                    if (profileNameTaken) {
                      setUiNotice('Choose a new profile name before recording.');
                      minimalNameInputRef.current?.focus();
                      return;
                    }
                    setSourceMode('camera');
                    void openCamera();
                  }}
                >
                  Record with camera
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={sourceMode === 'upload'}
                  className={sourceMode === 'upload' ? 'is-selected' : ''}
                  onClick={() => {
                    setSourceMode('upload');
                    stopCameraStream();
                    setCameraState('idle');
                    setCapturedCameraFile(null);
                    if (!lastUploadedFilename) {
                      setProfile(prev => ({ ...prev, lastUploadedFile: null, fileSize: null }));
                    }
                  }}
                >
                  Upload video
                </button>
              </div>

              {sourceMode === 'upload' ? (
                <div className="ph-minimal-file-list">
                  <label className={`ph-minimal-file-row ${sourceUploaded ? 'is-complete' : ''} ${!profile.name || profileNameTaken ? 'is-disabled' : ''}`}>
                    <input type="file" accept="video/*" onChange={event => event.target.files?.[0] && void handleUpload(event.target.files[0])} disabled={!profile.name || profileNameTaken || isBusy} />
                    <span className="ph-minimal-file-icon">▣</span>
                    <span className="ph-minimal-file-copy"><strong>Talking video</strong><small>{profile.lastUploadedFile || (sourceUploaded ? 'Video uploaded · ready to prepare' : 'Face + voice source · 5–20 seconds · 720p+ · front-lit')}</small></span>
                    <span className="ph-minimal-file-action">{uploadPhaseVideo === 'uploading' ? `${uploadProgressVideo}%` : sourceUploaded ? 'Replace' : 'Choose'}</span>
                  </label>
                </div>
              ) : (
                <div className="ph-camera-recorder">
                  <blockquote className="ph-camera-script">“{CAMERA_PROMPT}”</blockquote>
                  <div className="ph-camera-stage">
                    <video ref={cameraVideoRef} src={cameraPreviewUrl ?? undefined} autoPlay muted playsInline loop={cameraState === 'recorded'} aria-label="Camera preview" />
                    <div className="ph-camera-guide" aria-hidden="true">
                      <span className="ph-camera-head-frame" />
                      <small>Keep your face inside the frame</small>
                    </div>
                    {cameraState === 'requesting' && <div className="ph-camera-overlay">Opening camera…</div>}
                    {cameraState === 'idle' && <div className="ph-camera-overlay">Allow camera access to begin</div>}
                    {cameraState === 'recorded' && <div className="ph-camera-captured-badge">Recording captured · looping preview</div>}
                  </div>
                  <div className="ph-camera-meta"><span>{cameraState === 'recording' ? 'Recording…' : '20-second guided capture'}</span><strong>{cameraElapsed}s / {CAMERA_RECORDING_SECONDS}s</strong></div>
                  <p className="ph-camera-instructions">Face straight toward the camera, use bright even light from in front of you, keep your eyes and mouth visible, and speak naturally in a quiet room. Avoid a bright window behind you.</p>
                  {cameraError && <div className="ph-minimal-error">{cameraError}</div>}
                  <div className="ph-camera-actions">
                    {(cameraState === 'idle' || cameraState === 'error') && <button type="button" className="ph-minimal-secondary" onClick={() => void openCamera()} disabled={profileNameTaken || isBusy}>Allow camera</button>}
                    {cameraState === 'ready' && <button type="button" className="ph-minimal-primary" onClick={startCameraRecording} disabled={isBusy}>Start recording<span>●</span></button>}
                    {cameraState === 'recording' && <button type="button" className="ph-minimal-record-stop" onClick={stopCameraRecording}>Stop and use recording<span>■</span></button>}
                    {cameraState === 'recorded' && <button type="button" className="ph-minimal-secondary" onClick={() => void openCamera()} disabled={isBusy}>Record again</button>}
                    {uploadPhaseVideo === 'uploading' && <span className="ph-camera-upload-status">Uploading {uploadProgressVideo}%…</span>}
                  </div>
                </div>
              )}

              {stepStatuses.preprocess === 'running' && (
                <div className="ph-minimal-progress"><div><span>Preparing avatar</span><b>{Math.round((preprocessDisplayProgress ?? 0) * 100)}%</b></div><span className="ph-minimal-progress-track"><i style={{ width: `${Math.round((preprocessDisplayProgress ?? 0) * 100)}%` }} /></span></div>
              )}
              {stepStatuses.preprocess === 'error' && <div className="ph-minimal-error">{preprocessLogs[preprocessLogs.length - 1]?.message || 'Could not prepare this avatar.'}</div>}
              {uiNotice && <div className="ph-minimal-error">{uiNotice}</div>}

              <button type="button" className="ph-minimal-primary" onClick={() => void handleCreateAvatar()} disabled={profileNameTaken || autoPrepareAfterUpload || !publicSourceReady || stepStatuses.preprocess === 'running' || isBusy && stepStatuses.preprocess !== 'running'}>
                {autoPrepareAfterUpload || stepStatuses.preprocess === 'running' ? 'Preparing…' : 'Create avatar'}<span>→</span>
              </button>

              <details className="ph-minimal-help">
                <summary>What makes a good source?</summary>
                <p>Use a steady 720p or higher clip with bright, even front lighting. Keep your mouth visible, avoid a bright window behind you, and speak clearly without music or echo.</p>
              </details>
                  </>
                );
              })()}
            </div>
          </section>
        )}

        {publicProfileReady && (
          <section className="ph-minimal-studio">
            <div className="ph-minimal-studio-heading">
              <div><span className="ph-minimal-eyebrow">Avatar</span><h1>{profile.name}</h1></div>
              {isWarmingUp && <span className="ph-minimal-warmup-pill"><i /> Preparing this profile…</span>}
            </div>

            <div className="ph-minimal-workspace">
              <section className="ph-minimal-preview">
                <div className="ph-minimal-preview-heading"><span>Live preview</span><span><i className={`ph-mini-dot ${isWarmingUp ? 'is-preparing' : videoState === 'playing' ? 'is-speaking' : ''}`} />{isWarmingUp ? 'Preparing' : videoState === 'playing' ? 'Speaking' : 'Ready'}</span></div>
                <div className="ph-minimal-canvas-wrap">
                  <canvas ref={videoCanvasRef} width={810} height={1080} aria-label="Live avatar preview" />
                  {isWarmingUp && <div className="ph-minimal-buffering ph-minimal-preparing">Preparing avatar…</div>}
                  {videoState === 'buffering' && stepStatuses.inference === 'running' && <div className="ph-minimal-buffering">Buffering…</div>}
                </div>
                <div className="ph-minimal-preview-footer"><span>Chatterbox + {avatarBackend === 'wav2lip' ? 'Wav2Lip' : 'MuseTalk'}</span><span>{videoQueue ? `${videoQueue} frames` : `${videoFps} FPS`}</span></div>
              </section>

              <section className="ph-minimal-composer">
                <div className="ph-minimal-composer-heading">
                  <div><span className="ph-minimal-eyebrow">Ask your avatar</span><small>Type a question or use your voice.</small></div>
                </div>
                <div className="ph-minimal-input-shell">
                  <textarea
                    className="ph-minimal-composer-input"
                    value={composerDisplayText}
                    onChange={event => setInferenceText(event.target.value)}
                    onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void sendComposerText(); } }}
                    placeholder={isListening ? 'Listening…' : 'Ask your avatar something…'}
                    rows={5}
                  />
                  <button type="button" className={`ph-minimal-voice-button ${isListening ? 'is-listening' : ''}`} onClick={() => (isListening ? stopListening() : startListening())} disabled={!hasSpeechSupport || stepStatuses.inference === 'running' || isWarmingUp} title={hasSpeechSupport ? 'Speak; it will send automatically when you stop' : 'Voice input is not supported by this browser'} aria-label={hasSpeechSupport ? (isListening ? 'Stop listening' : 'Voice input') : 'Voice input unavailable'} aria-pressed={isListening}>
                    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Zm6-3a1 1 0 0 0-2 0 4 4 0 0 1-8 0 1 1 0 0 0-2 0 6 6 0 0 0 5 5.91V19H8a1 1 0 1 0 0 2h8a1 1 0 1 0 0-2h-3v-2.09A6 6 0 0 0 18 11Z" /></svg>
                    <span>{isListening ? 'Stop listening' : 'Voice'}</span>
                  </button>
                </div>
                <div className="ph-minimal-composer-row"><span>{inferenceText.length} characters</span><span>{isListening ? (speechTranscript ? 'Listening…' : 'Start speaking…') : 'Enter to send · voice sends automatically'}</span></div>

                <div className="ph-minimal-model-grid">
                  <label><span>Lip sync</span><select value={avatarBackend} onChange={event => setAvatarBackend(event.target.value as 'musetalk' | 'wav2lip')}><option value="musetalk">MuseTalk</option><option value="wav2lip">Wav2Lip</option></select></label>
                  <label><span>Assistant</span><select value={llmMode} onChange={event => setLlmMode(event.target.value as LLMMode)}>{LLM_MODE_OPTIONS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
                </div>

                <div className="ph-minimal-control-row"><label>Emotion<select value={voiceControlValues?.emotion || 'neutral'} onChange={event => handleVoiceControlsChange({ emotion: event.target.value as VoiceEmotion })}>{(['neutral', 'happy', 'sad', 'angry', 'scared', 'disgust'] as VoiceEmotion[]).map(emotion => <option key={emotion} value={emotion}>{emotion[0].toUpperCase() + emotion.slice(1)}</option>)}</select></label><button type="button" className="ph-minimal-advanced-button" onClick={() => setShowVoiceSettings(prev => !prev)}>{showVoiceSettings ? 'Hide options' : 'More options'}</button></div>

                {showVoiceSettings && (
                  <div className="ph-minimal-advanced-panel">
                    <label><span>Expressiveness <b>{Math.round((voiceControlValues?.expressiveness ?? 0.5) * 100)}%</b></span><input type="range" min="0.25" max="1" step="0.01" value={voiceControlValues?.expressiveness ?? 0.5} onChange={event => handleVoiceControlsChange({ expressiveness: Number(event.target.value) })} /></label>
                    <label><span>Variation / temperature <b>{(voiceControlValues?.variation ?? 0.8).toFixed(2)}</b></span><input type="range" min="0.1" max="1.2" step="0.01" value={voiceControlValues?.variation ?? 0.8} onChange={event => handleVoiceControlsChange({ variation: Number(event.target.value) })} /></label>
                    <label><span>Voice match / CFG <b>{Math.round((voiceControlValues?.guidance ?? 0.5) * 100)}%</b></span><input type="range" min="0" max="1" step="0.01" value={voiceControlValues?.guidance ?? 0.5} onChange={event => handleVoiceControlsChange({ guidance: Number(event.target.value) })} /></label>
                    <label><span>Repetition penalty <b>{(voiceControlValues?.repetition ?? 1.2).toFixed(2)}</b></span><input type="range" min="0.9" max="2" step="0.01" value={voiceControlValues?.repetition ?? 1.2} onChange={event => handleVoiceControlsChange({ repetition: Number(event.target.value) })} /></label>
                    <label><span>Emotion intensity <b>{Math.round((voiceControlValues?.emotionIntensity ?? 0.5) * 100)}%</b></span><input type="range" min="0" max="1" step="0.05" value={voiceControlValues?.emotionIntensity ?? 0.5} disabled={voiceControlValues?.emotion === 'neutral'} onChange={event => handleVoiceControlsChange({ emotionIntensity: Number(event.target.value) })} /></label>
                    <div className="ph-minimal-advanced-actions"><button type="button" onClick={resetVoiceControls}>Reset</button><button type="button" className="is-accent" onClick={() => void saveRuntimeSettings()} disabled={runtimeSettingsStatus === 'saving'}>{runtimeSettingsStatus === 'saved' ? 'Saved' : runtimeSettingsStatus === 'saving' ? 'Saving…' : 'Save defaults'}</button></div>
                  </div>
                )}

                {uiNotice && <div className="ph-minimal-error">{uiNotice}</div>}
                <div className="ph-minimal-composer-footer"><span><i className={`ph-status-dot ${isListening ? 'checking' : 'online'}`} /> {isListening ? 'Listening…' : stepStatuses.inference === 'running' ? 'Generating…' : 'Ready to stream'}</span><div className="ph-minimal-composer-actions"><button type="button" className={`ph-minimal-mic-button ${isListening ? 'is-listening' : ''}`} onClick={() => (isListening ? stopListening() : startListening())} disabled={!hasSpeechSupport || stepStatuses.inference === 'running' || isWarmingUp} title={hasSpeechSupport ? 'Speak; it will send automatically when you stop' : 'Voice input is not supported by this browser'} aria-label={hasSpeechSupport ? 'Voice input' : 'Voice input unavailable'} aria-pressed={isListening}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Zm6-3a1 1 0 0 0-2 0 4 4 0 0 1-8 0 1 1 0 0 0-2 0 6 6 0 0 0 5 5.91V19H8a1 1 0 1 0 0 2h8a1 1 0 1 0 0-2h-3v-2.09A6 6 0 0 0 18 11Z" /></svg></button>{stepStatuses.inference === 'running' && <button type="button" className="ph-minimal-stop is-active" onClick={() => void stopInference()}>Stop</button>}</div></div>
              </section>
            </div>

            <details className="ph-minimal-session-details"><summary>Session details</summary><div><span>Voice engine <b>Chatterbox</b></span><span>Lip sync <b>{avatarBackend === 'wav2lip' ? 'Wav2Lip' : 'MuseTalk'}</b></span><span>First audio <b>{latency ? `${latency.ttfa} ms` : '—'}</b></span><span>Chunks <b>{inferenceChunks.length || '—'}</b></span></div></details>
          </section>
        )}
        </main>
      </div>
      )}
    </div>
  );

  return (
    <div className="ph-app">
      <header className="ph-topbar">
        <div className="ph-brand-lockup">
          <span className="ph-brand-mark">✦</span>
          <div>
            <div className="ph-brand-name">PixelHolo</div>
            <div className="ph-brand-subtitle">Realtime avatar studio</div>
          </div>
          <span className="ph-dev-badge">DEV</span>
        </div>
        <div className="ph-topbar-actions">
          <div className="ph-connection-pill">
            <span className={`ph-status-dot ${apiStatus}`} />
            <span>{apiStatus === 'online' ? 'Engine online' : apiStatus === 'checking' ? 'Connecting' : 'Engine offline'}</span>
          </div>
          <label className="ph-api-control">
            <span>API</span>
            <input value={apiBase} onChange={(event) => setApiBase(event.target.value)} aria-label="API base URL" />
          </label>
        </div>
      </header>

      <div className="ph-shell">
        <aside className="ph-sidebar">
          <button type="button" className="ph-new-avatar" onClick={resetNewProfile}>
            <span>＋</span> New avatar
          </button>

          <div className="ph-sidebar-label">Your avatars</div>
          <div className="ph-profile-list">
            {profilesStatus === 'loading' && <div className="ph-sidebar-empty">Loading avatars…</div>}
            {profilesStatus === 'error' && <div className="ph-sidebar-empty ph-error-text">Could not load avatars.</div>}
            {profilesStatus === 'idle' && profiles.length === 0 && (
              <div className="ph-sidebar-empty">Your first avatar will appear here.</div>
            )}
            {profiles.map((item) => {
              const selected = item.name === profile.name;
              return (
                <div key={`${item.profile_type || 'avatar'}:${item.name}`} className={`ph-profile-row ${selected ? 'is-selected' : ''}`}>
                  <button type="button" className="ph-profile-select" onClick={() => selectProfile(item)}>
                    <span className="ph-profile-avatar">{item.name.slice(0, 1).toUpperCase()}</span>
                    <span className="ph-profile-copy">
                      <strong>{item.name}</strong>
                      <small>{item.has_data ? 'Ready to speak' : 'Needs a source clip'}</small>
                    </span>
                    <span className={`ph-ready-dot ${item.has_data ? 'ready' : ''}`} />
                  </button>
                  <div className="ph-profile-actions">
                    <button type="button" onClick={() => void handleRenameProfile(item)} aria-label={`Rename ${item.name}`}>Rename</button>
                    <button type="button" onClick={() => void handleDeleteProfile(item)} aria-label={`Delete ${item.name}`}>Delete</button>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="ph-sidebar-footer">
            <div className="ph-engine-stack">
              <span className="ph-stack-icon">◌</span>
              <div><strong>Chatterbox</strong><small>voice cloning</small></div>
              <span className="ph-stack-arrow">→</span>
              <span className="ph-stack-icon">◉</span>
              <div><strong>MuseTalk</strong><small>lip sync</small></div>
            </div>
            <p>Zero-shot setup. No training step required.</p>
          </div>
        </aside>

        <main className="ph-main">
          {!publicProfileReady && (
            <section className="ph-setup-view">
              <div className="ph-page-intro">
                <div>
                  <span className="ph-kicker">Create an avatar</span>
                  <h1>Give your voice a face.</h1>
                  <p>Upload a short talking-head video. PixelHolo extracts the voice and prepares the face reference once, then streams Chatterbox audio through MuseTalk in real time.</p>
                </div>
                <div className="ph-pipeline-card">
                  <span className="ph-pipeline-label">The fast path</span>
                  <div className="ph-pipeline-line"><span>Source clip</span><b>→</b><span>Chatterbox</span><b>→</b><span>MuseTalk</span></div>
                  <small>One setup. Unlimited conversations.</small>
                </div>
              </div>

              <div className="ph-setup-grid">
                <section className="ph-card ph-setup-card">
                  <div className="ph-card-heading">
                    <div><span className="ph-step-index">01</span><h2>Make a new avatar</h2></div>
                    <span className="ph-card-status">{stepStatuses.preprocess === 'running' ? 'Preparing…' : publicSetupReady ? 'Ready to prepare' : '2 files'}</span>
                  </div>
                  <label className="ph-field-label" htmlFor="avatar-name">Avatar name</label>
                  <input
                    id="avatar-name"
                    className="ph-text-input"
                    value={profile.name}
                    onChange={(event) => setProfile(prev => ({ ...prev, name: event.target.value }))}
                    placeholder="Enter a profile name"
                    disabled={isBusy}
                  />

                  <div className="ph-upload-grid">
                    <label className={`ph-upload-tile ${profile.lastUploadedFile ? 'is-complete' : ''} ${!profile.name ? 'is-disabled' : ''}`}>
                      <input type="file" accept="video/*" onChange={(event) => event.target.files?.[0] && void handleUpload(event.target.files[0])} disabled={!profile.name || isBusy} />
                      <span className="ph-upload-icon">↥</span>
                      <strong>{profile.lastUploadedFile ? 'Video added' : 'Add portrait video'}</strong>
                      <small>{profile.lastUploadedFile || '5–20 seconds · MP4 or MOV'}</small>
                      {uploadPhaseVideo === 'uploading' && <span className="ph-upload-progress">Uploading {uploadProgressVideo}%</span>}
                    </label>
                    <label className={`ph-upload-tile ${lastUploadedAudioFilename ? 'is-complete' : ''} ${!profile.name ? 'is-disabled' : ''}`}>
                      <input type="file" accept="audio/*" onChange={(event) => event.target.files?.[0] && void handleUploadAudio(event.target.files[0])} disabled={!profile.name || isBusy} />
                      <span className="ph-upload-icon">∿</span>
                      <strong>{lastUploadedAudioFilename ? 'Optional voice override added' : 'Optional voice override'}</strong>
                      <small>{lastUploadedAudioFilename || 'Clean WAV or MP3 · same speaker'}</small>
                      {uploadPhaseAudio === 'uploading' && <span className="ph-upload-progress">Uploading {uploadProgressAudio}%</span>}
                    </label>
                  </div>

                  <div className="ph-setup-note"><span>✦</span><p>Use a steady 720p+ video with bright, even light on your face. Keep the camera at eye level, avoid a bright window behind you, and keep your mouth clearly visible.</p></div>

                  {stepStatuses.preprocess === 'running' && (
                    <div className="ph-progress-block">
                      <div className="ph-progress-label"><span>Preparing your avatar</span><strong>{Math.round((preprocessDisplayProgress ?? 0) * 100)}%</strong></div>
                      <div className="ph-progress-track"><span style={{ width: `${Math.round((preprocessDisplayProgress ?? 0) * 100)}%` }} /></div>
                      <small>Extracting the voice, transcribing the track, and baking the face loop…</small>
                    </div>
                  )}
                  {stepStatuses.preprocess === 'error' && <div className="ph-inline-error">{preprocessLogs[preprocessLogs.length - 1]?.message || 'Could not prepare this avatar.'}</div>}
                  {uiNotice && <div className="ph-inline-error">{uiNotice}</div>}

                  <button
                    type="button"
                    className="ph-primary-button ph-wide-button"
                    onClick={() => void startPreprocess()}
                    disabled={!publicSetupReady || stepStatuses.preprocess === 'running' || isBusy && stepStatuses.preprocess !== 'running'}
                  >
                    {stepStatuses.preprocess === 'running' ? 'Preparing avatar…' : 'Prepare avatar'}<span>→</span>
                  </button>
                </section>

                <aside className="ph-card ph-guidance-card">
                  <span className="ph-kicker">A good source clip</span>
                  <h2>Make the first take count.</h2>
                  <div className="ph-guidance-list">
                    <div><span className="ph-guidance-number">01</span><p><strong>Use good light</strong><br />Choose bright, even front lighting and avoid backlight.</p></div>
                    <div><span className="ph-guidance-number">02</span><p><strong>Frame your face</strong><br />Use a steady 720p+ clip with your eyes and mouth visible.</p></div>
                    <div><span className="ph-guidance-number">03</span><p><strong>Use clean audio</strong><br />Avoid music, rooms with echo, and overlapping speakers.</p></div>
                    <div><span className="ph-guidance-number">04</span><p><strong>Speak naturally</strong><br />A steady, relaxed delivery helps Chatterbox capture your voice.</p></div>
                  </div>
                  <div className="ph-guidance-footer"><span className="ph-checkmark">✓</span><span>No training step is required for this flow.</span></div>
                </aside>
              </div>

              {preprocessStats && stepStatuses.preprocess === 'done' && (
                <div className="ph-success-banner"><span>✓</span><div><strong>Avatar ready.</strong><small>{preprocessStats.segmentsKept || 0} clean voice segments prepared. Opening the studio…</small></div></div>
              )}
            </section>
          )}

          {publicProfileReady && (
            <section className="ph-studio-view">
              <div className="ph-studio-heading">
                <div><span className="ph-kicker">Avatar studio</span><h1>{profile.name}</h1><p>Speak naturally. PixelHolo handles the voice and face in one stream.</p></div>
                <div className="ph-studio-heading-actions"><span className="ph-live-pill"><span className="ph-status-dot online" /> {isWarmingUp ? `Warming ${warmupTargetName || 'avatar'}…` : 'Ready'}</span><button type="button" className="ph-secondary-button" onClick={() => setActiveStep(1)}>Change avatar</button></div>
              </div>

              <div className="ph-studio-grid">
                <section className="ph-card ph-avatar-card">
                  <div className="ph-avatar-card-head"><div><span className="ph-kicker">Live preview</span><h2>{isWarmingUp ? 'Preparing avatar' : videoState === 'playing' ? 'Speaking now' : 'Your avatar'}</h2></div><span className="ph-engine-chip"><span>◌</span> MuseTalk</span></div>
                  <div className="ph-avatar-stage">
                    <canvas ref={videoCanvasRef} width={810} height={1080} aria-label="Live avatar preview" />
                    {isWarmingUp && <div className="ph-avatar-overlay">Preparing avatar…</div>}
                    {videoState === 'buffering' && stepStatuses.inference === 'running' && <div className="ph-avatar-overlay">Buffering face frames…</div>}
                  </div>
                  <div className="ph-avatar-card-foot"><span><i className={`ph-mini-dot ${videoState === 'playing' ? 'is-speaking' : ''}`} /> {videoState === 'playing' ? 'Streaming' : 'Waiting for a prompt'}</span><span>{videoFps} FPS · {videoQueue} frames queued</span></div>
                </section>

                <section className="ph-card ph-composer-card">
                  <div className="ph-composer-tabs"><button type="button" className={composerMode === 'say' ? 'is-active' : ''} onClick={() => setComposerMode('say')}>Say it</button><button type="button" className={composerMode === 'chat' ? 'is-active' : ''} onClick={() => setComposerMode('chat')}>Ask assistant</button></div>
                  <textarea
                    className="ph-composer-input"
                    value={inferenceText}
                    onChange={(event) => setInferenceText(event.target.value)}
                    onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void sendComposerText(); } }}
                    placeholder={composerMode === 'say' ? 'Type something for your avatar to say…' : 'Ask your avatar anything…'}
                    rows={5}
                  />
                  <div className="ph-composer-meta"><span>{inferenceText.length} characters</span><span>Enter to generate · Shift + Enter for a new line</span></div>
                  <button type="button" className="ph-primary-button ph-generate-button" onClick={() => void sendComposerText()} disabled={!inferenceText.trim() || stepStatuses.inference === 'running' || isWarmingUp}>{stepStatuses.inference === 'running' ? 'Generating…' : composerMode === 'say' ? 'Generate voice + lip sync' : 'Ask and animate'}<span>↗</span></button>

                  <div className="ph-emotion-row"><div><span className="ph-control-label">Emotion</span><small>Chatterbox expression</small></div><select value={voiceControlValues?.emotion || 'neutral'} onChange={(event) => handleVoiceControlsChange({ emotion: event.target.value as VoiceEmotion })}>{(['neutral', 'happy', 'sad', 'angry', 'scared', 'disgust'] as VoiceEmotion[]).map((emotion) => <option key={emotion} value={emotion}>{emotion[0].toUpperCase() + emotion.slice(1)}</option>)}</select></div>

                  <button type="button" className="ph-settings-toggle" onClick={() => setShowVoiceSettings(prev => !prev)}><span>Voice settings</span><span>{showVoiceSettings ? '−' : '+'}</span></button>
                  {showVoiceSettings && (
                    <div className="ph-voice-settings">
                      <label><span>Expressiveness <b>{Math.round((voiceControlValues?.expressiveness ?? 0.5) * 100)}%</b></span><input type="range" min="0.25" max="1" step="0.01" value={voiceControlValues?.expressiveness ?? 0.5} onChange={(event) => handleVoiceControlsChange({ expressiveness: Number(event.target.value) })} /></label>
                      <label><span>Voice match <b>{Math.round((voiceControlValues?.guidance ?? 0.5) * 100)}%</b></span><input type="range" min="0" max="1" step="0.01" value={voiceControlValues?.guidance ?? 0.5} onChange={(event) => handleVoiceControlsChange({ guidance: Number(event.target.value) })} /></label>
                      <label><span>Pace <b>{voiceControlValues?.pace ?? 0}%</b></span><input type="range" min="-100" max="100" step="1" value={voiceControlValues?.pace ?? 0} onChange={(event) => handleVoiceControlsChange({ pace: Number(event.target.value) })} /></label>
                      <div className="ph-voice-setting-actions"><button type="button" className="ph-text-button" onClick={resetVoiceControls}>Reset</button><button type="button" className="ph-text-button is-accent" onClick={() => void saveRuntimeSettings()} disabled={runtimeSettingsStatus === 'saving'}>{runtimeSettingsStatus === 'saved' ? 'Saved' : runtimeSettingsStatus === 'saving' ? 'Saving…' : 'Save defaults'}</button></div>
                    </div>
                  )}

                  {uiNotice && <div className="ph-inline-error">{uiNotice}</div>}
                  <div className="ph-composer-footer"><span><span className="ph-status-dot online" /> Chatterbox · {museTalkPreset.replace('_', ' ')}</span><button type="button" className="ph-stop-button" onClick={() => void stopInference()} disabled={stepStatuses.inference !== 'running'}>Stop</button></div>
                </section>
              </div>

              <div className="ph-metrics-row">
                <div className="ph-metric-card"><span className="ph-kicker">Time to first audio</span><strong>{latency ? `${latency.ttfa} ms` : '—'}</strong><small>Measured for the last prompt</small></div>
                <div className="ph-metric-card"><span className="ph-kicker">Stream</span><strong>{inferenceChunks.length ? `${inferenceChunks.length} chunks` : 'Ready'}</strong><small>{latency?.total ? `${latency.total} ms total` : 'Chunked playback keeps the avatar responsive'}</small></div>
                <div className="ph-metric-card ph-metric-card-wide"><span className="ph-kicker">Pipeline</span><div className="ph-pipeline-line"><span className="is-current">Chatterbox</span><b>→</b><span className="is-current">MuseTalk</span><b>→</b><span>Live browser playback</span></div><small>Voice identity is cached once per avatar session.</small></div>
              </div>
            </section>
          )}
        </main>
      </div>
      <footer className="ph-footer"><span>PixelHolo dev workspace</span><span>Local first · deploy target <strong>dev.pixelholo.com</strong></span></footer>
    </div>
  );

  return (
    <div className="min-h-screen pb-24 bg-[#FDFCF8]">
      <Header profile={profile} apiBase={apiBase} apiStatus={apiStatus} onApiChange={setApiBase} />

      <div className="max-w-7xl mx-auto px-6 pt-12">
        <div className="flex items-center justify-between relative mb-12">
          <div className="absolute top-1/2 left-0 w-full h-0.5 bg-slate-100 -translate-y-1/2 -z-10"></div>
          {[1, 2, 3, 4].map((step) => (
            <button
              key={step}
              onClick={() => {
                if (isBusy) {
                  setUiNotice('Stop the current job before changing steps.');
                  return;
                }
                if (canProceedTo(step)) setActiveStep(step);
              }}
              disabled={isBusy || !canProceedTo(step)}
              className={`
                relative flex items-center justify-center w-12 h-12 rounded-full border-2 font-bold transition-all duration-300
                ${activeStep === step ? 'bg-teal-600 text-white border-teal-600 scale-110 shadow-lg shadow-teal-600/20' :
                  canProceedTo(step) && !isBusy ? 'bg-white text-teal-600 border-teal-600 cursor-pointer' : 'bg-slate-50 text-slate-300 border-slate-100 cursor-not-allowed'}
              `}
            >
              {step}
              <span className={`absolute -bottom-7 left-1/2 -translate-x-1/2 text-[10px] uppercase tracking-widest font-bold whitespace-nowrap
                ${activeStep === step ? 'text-teal-600' : 'text-slate-400'}`}>
                {step === 1 ? 'Profile' : step === 2 ? 'Preprocess' : step === 3 ? 'Training' : 'Generation'}
              </span>
            </button>
          ))}
        </div>

        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          {activeStep === 1 && (
            <StepCard
              stepNumber={1}
              title="Profile & Identity Setup"
              description="Name your voice profile and upload clean audio or video."
              status={stepStatuses.upload}
              isActive={true}
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
                <div className="space-y-4">
                  <div className="bg-white p-4 rounded-xl border border-slate-100">
                    <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Workflow</label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          if (isBusy) {
                            setUiNotice('A job is running. Stop it before switching workflows.');
                            return;
                          }
                          setProfileType('voice');
                          setProfile(prev => ({ ...prev, lastUploadedFile: null, fileSize: null }));
                          setLastUploadedFilename(null);
                        }}
                        disabled={isBusy}
                        className={`px-3 py-2 rounded-lg text-xs font-bold transition-all ${profileType === 'voice' ? 'bg-teal-600 text-white shadow-lg shadow-teal-600/20' : 'bg-slate-100 text-slate-500'} ${isBusy ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        Voice Only
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          if (isBusy) {
                            setUiNotice('A job is running. Stop it before switching workflows.');
                            return;
                          }
                          setProfileType('avatar');
                          setProfile(prev => ({ ...prev, lastUploadedFile: null, fileSize: null }));
                          setLastUploadedFilename(null);
                        }}
                        disabled={isBusy}
                        className={`px-3 py-2 rounded-lg text-xs font-bold transition-all ${profileType === 'avatar' ? 'bg-slate-900 text-white shadow-lg' : 'bg-slate-100 text-slate-500'} ${isBusy ? 'opacity-50 cursor-not-allowed' : ''}`}
                      >
                        Voice + Lip Sync
                      </button>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-2 italic">
                      Voice-only profiles use audio datasets. Lip-sync profiles require video and an avatar cache.
                    </p>
                    {isBusy && (
                      <p className="text-[10px] text-amber-600 mt-2 font-semibold">
                        Active job running. Stop it before changing profile or workflow.
                      </p>
                    )}
                    {uiNotice && (
                      <p className="text-[10px] text-rose-600 mt-2 font-semibold">
                        {uiNotice}
                      </p>
                    )}
                  </div>
                  <div className="bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Voice Identity Name</label>
                    <input
                      type="text"
                      value={profile.name}
                      onChange={(e) => {
                        if (isBusy) {
                          setUiNotice('Stop the current job before changing the profile name.');
                          return;
                        }
                        setProfile(prev => ({ ...prev, name: e.target.value }));
                      }}
                      disabled={isBusy}
                      placeholder="Enter a profile name"
                      className={`w-full bg-white border border-slate-200 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-teal-600 outline-none transition-all font-semibold ${isBusy ? 'opacity-60 cursor-not-allowed' : ''}`}
                    />
                    <p className="text-[10px] text-slate-400 mt-2 italic">Used to organize your models and generated assets.</p>
                  </div>
                  <div className="bg-white border border-slate-100 rounded-xl p-4 text-xs text-slate-500">
                    <p className="uppercase tracking-widest text-[9px] font-bold text-slate-400">Paths</p>
                    <p>
                      Dataset: <span className="font-semibold">
                        data/{profileType === 'avatar' ? 'avatar_profiles' : 'voice_profiles'}/{profile.name || '<profile>'}
                      </span>
                    </p>
                    <p>
                      Outputs: <span className="font-semibold">
                        outputs/training/{profileType === 'avatar' ? 'avatar' : 'voice'}/{profile.name || '<profile>'}
                      </span>
                    </p>
                  </div>
                  <div className="bg-white border border-slate-100 rounded-xl p-4 text-xs text-slate-500 space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="uppercase tracking-widest text-[9px] font-bold text-slate-400">Existing Profiles</p>
                      <button
                        onClick={loadProfiles}
                        className="text-[10px] font-bold text-teal-600"
                        type="button"
                      >
                        Refresh
                      </button>
                    </div>
                    {profilesStatus === 'loading' && <p>Loading profiles...</p>}
                    {profilesStatus === 'error' && <p className="text-rose-500">Failed to load profiles.</p>}
                    {profilesStatus === 'idle' && profiles.length === 0 && (
                      <p className="italic text-slate-400">No profiles found yet.</p>
                    )}
                    {profilesStatus === 'idle' && profiles.length > 0 && (
                      <div className="space-y-2 max-h-40 overflow-auto pr-1">
                        {profiles.map((item) => (
                          <div
                            key={`${item.profile_type || profileType}:${item.name}`}
                            className={`w-full flex items-center gap-2 border rounded-lg px-3 py-2 ${
                              profile.name === item.name
                                ? 'border-teal-600 bg-teal-50 text-teal-800'
                                : 'border-slate-200 bg-white text-slate-600'
                            } ${isBusy ? 'opacity-60' : ''}`}
                          >
                            <button
                              type="button"
                              onClick={() => {
                                if (isBusy) {
                                  setUiNotice('Stop the current job before switching profiles.');
                                  return;
                                }
                                if (item.profile_type === 'avatar') {
                                  setProfileType('avatar');
                                } else if (item.profile_type === 'voice') {
                                  setProfileType('voice');
                                }
                                setProfile(prev => ({ ...prev, name: item.name }));
                                setIsCreatingProfile(false);
                                setShowWelcome(false);
                                const selectedType = item.profile_type === 'avatar' ? 'avatar' : 'voice';
                                void warmupProfile(item.name, selectedType);
                                setLastUploadedFilename(null);
                                setLastUploadedAudioFilename(null);
                                if (item.has_profile) {
                                  setActiveStep(4);
                                }
                                setProfileMenuKey(null);
                              }}
                              disabled={isBusy}
                              className="flex-1 min-w-0 flex items-center justify-between text-left disabled:cursor-not-allowed"
                            >
                              <div className="min-w-0">
                                <p className="text-xs font-bold truncate">{item.name}</p>
                                <p className="text-[10px] text-slate-400 truncate">
                                  {item.processed_wavs} clips | {(item.raw_audio_files ?? 0)} audio | {item.raw_files} video | {item.profile_type || profileType}
                                </p>
                              </div>
                              <div className="text-[10px] font-bold ml-2">
                                {item.has_profile ? 'ready' : 'needs training'}
                              </div>
                            </button>

                            <div className="relative" onClick={(e) => e.stopPropagation()}>
                              <button
                                type="button"
                                disabled={isBusy}
                                aria-label={`Profile options for ${item.name}`}
                                className="h-7 w-7 rounded-md border border-slate-200 bg-white text-slate-500 hover:text-slate-700 hover:border-slate-300 disabled:cursor-not-allowed disabled:opacity-50 text-lg leading-none"
                                onClick={() => {
                                  const key = `${item.profile_type || profileType}:${item.name}`;
                                  setProfileMenuKey(prev => (prev === key ? null : key));
                                }}
                              >
                                ...
                              </button>
                              {profileMenuKey === `${item.profile_type || profileType}:${item.name}` && (
                                <div className="absolute right-0 mt-1 min-w-[120px] rounded-lg border border-slate-200 bg-white shadow-lg z-20 py-1">
                                  <button
                                    type="button"
                                    className="w-full text-left px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
                                    onClick={() => handleRenameProfile(item)}
                                  >
                                    Rename
                                  </button>
                                  <button
                                    type="button"
                                    className="w-full text-left px-3 py-1.5 text-xs font-medium text-rose-600 hover:bg-rose-50"
                                    onClick={() => handleDeleteProfile(item)}
                                  >
                                    Delete
                                  </button>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-4">
                  <p className="text-xs text-slate-500 bg-slate-50 border border-slate-100 rounded-lg px-3 py-2">
                    Avatar setup uses one talking video by default. A separate audio override remains available for developer workflows.
                  </p>
                  <div className="relative group">
                    <input
                      type="file"
                      accept={profileType === 'voice' ? 'audio/*' : 'video/*'}
                      onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
                      disabled={!profile.name}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10 disabled:cursor-not-allowed"
                    />
                    <div
                      className={`border-2 border-dashed rounded-xl p-10 text-center transition-all ${
                        !profile.name
                          ? 'opacity-50 bg-slate-100 border-slate-200'
                          : profile.lastUploadedFile
                            ? 'bg-emerald-50 border-emerald-500'
                            : 'group-hover:border-teal-600 bg-white border-slate-200'
                      }`}
                    >
                      <svg
                        className={`w-10 h-10 mx-auto mb-3 transition-colors ${
                          !profile.name
                            ? 'text-slate-300'
                            : profile.lastUploadedFile
                              ? 'text-emerald-600'
                              : 'text-slate-400 group-hover:text-teal-600'
                        }`}
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                      </svg>
                      <p className={`text-sm font-bold ${profile.lastUploadedFile ? 'text-emerald-700' : 'text-slate-700'}`}>
                        {profile.name
                          ? profileType === 'voice'
                            ? 'Select High-Quality Audio'
                            : 'Select High-Quality Video'
                          : 'Enter Profile Name First'}
                      </p>
                      <p className={`text-xs mt-1 ${profile.lastUploadedFile ? 'text-emerald-600' : 'text-slate-400'}`}>
                        {profileType === 'voice'
                          ? 'Lossless formats preferred (.wav, .flac). MP4 works if it contains audio.'
                          : 'Portrait video preferred (.mp4, .mov)'}
                      </p>
                      {profile.lastUploadedFile && (
                        <span className="inline-flex mt-3 px-3 py-1 rounded-full bg-emerald-100 text-[10px] font-bold tracking-widest text-emerald-700">
                          Uploaded
                        </span>
                      )}
                      {uploadPhaseVideo !== 'idle' && profile.name && (
                        <div className="mt-4 flex flex-col items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                          <div className="flex items-center gap-2">
                            <span className="inline-flex h-3 w-3 rounded-full border-2 border-slate-300 border-t-emerald-500 animate-spin" />
                            {uploadPhaseVideo === 'uploading'
                              ? uploadProgressVideo >= 100
                                ? 'Processing file'
                                : 'Uploading file'
                              : 'Upload failed'}
                          </div>
                          {uploadPhaseVideo === 'uploading' && (
                            <span className="text-[11px] font-semibold text-slate-600">
                                {formatBytes(uploadBytesVideo.loaded)} / {formatBytes(uploadBytesVideo.total)} | {uploadProgressVideo}%
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                  {profileType === 'avatar' && (
                    <div className="relative group">
                      <input
                        type="file"
                        accept="audio/*,video/*"
                        onChange={(e) => e.target.files?.[0] && handleUploadAudio(e.target.files[0])}
                        disabled={!profile.name}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10 disabled:cursor-not-allowed"
                      />
                      <div
                        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                          !profile.name
                            ? 'opacity-50 bg-slate-100 border-slate-200'
                            : lastUploadedAudioFilename
                              ? 'bg-emerald-50 border-emerald-500'
                              : 'group-hover:border-teal-600 bg-white border-slate-200'
                        }`}
                      >
                        <svg
                          className={`w-8 h-8 mx-auto mb-3 transition-colors ${
                            !profile.name
                              ? 'text-slate-300'
                              : lastUploadedAudioFilename
                                ? 'text-emerald-600'
                                : 'text-slate-400 group-hover:text-teal-600'
                          }`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M9 19V6l12-2v13M9 19a2 2 0 11-4 0 2 2 0 014 0zm12-6a2 2 0 11-4 0 2 2 0 014 0z"
                          />
                        </svg>
                        <p className={`text-sm font-bold ${lastUploadedAudioFilename ? 'text-emerald-700' : 'text-slate-700'}`}>
                          {profile.name ? 'Upload Training Audio (Required)' : 'Enter Profile Name First'}
                        </p>
                        <p className={`text-xs mt-1 ${lastUploadedAudioFilename ? 'text-emerald-600' : 'text-slate-400'}`}>
                          Audio files and videos with audio (.mp4, .mov) are both accepted.
                        </p>
                        {lastUploadedAudioFilename && (
                          <span className="inline-flex mt-3 px-3 py-1 rounded-full bg-emerald-100 text-[10px] font-bold tracking-widest text-emerald-700">
                            Uploaded
                          </span>
                        )}
                        {uploadPhaseAudio !== 'idle' && profile.name && (
                          <div className="mt-4 flex flex-col items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                            <div className="flex items-center gap-2">
                              <span className="inline-flex h-3 w-3 rounded-full border-2 border-slate-300 border-t-emerald-500 animate-spin" />
                              {uploadPhaseAudio === 'uploading'
                                ? uploadProgressAudio >= 100
                                  ? 'Processing file'
                                  : 'Uploading file'
                                : 'Upload failed'}
                            </div>
                            {uploadPhaseAudio === 'uploading' && (
                              <span className="text-[11px] font-semibold text-slate-600">
                                {formatBytes(uploadBytesAudio.loaded)} / {formatBytes(uploadBytesAudio.total)} | {uploadProgressAudio}%
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  {profile.lastUploadedFile && (
                    <div className="bg-teal-50 border border-teal-100 p-3 rounded-lg flex items-center justify-between">
                      <span className="text-xs font-bold text-teal-800 truncate">{profile.lastUploadedFile}</span>
                      <span className="text-[10px] font-bold text-teal-600 bg-white px-2 py-1 rounded shadow-sm">{profile.fileSize}</span>
                    </div>
                  )}
                  {profileType === 'avatar' && lastUploadedAudioFilename && (
                    <div className="bg-slate-50 border border-slate-100 p-3 rounded-lg flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-700 truncate">{lastUploadedAudioFilename}</span>
                      <span className="text-[10px] font-bold text-slate-500 bg-white px-2 py-1 rounded shadow-sm">Audio</span>
                    </div>
                  )}
                </div>
              </div>
            </StepCard>
          )}

          {activeStep === 2 && (
            <StepCard
              stepNumber={2}
              title="Data Analysis & Preprocessing"
              description="Segmenting audio, removing silence, and transcribing with Whisper."
              status={stepStatuses.preprocess}
              isActive={true}
              statusSteps={preprocessSteps}
              statusNote={
                profileType === 'avatar'
                  ? 'Avatar baking runs once per profile and speeds up live lip-sync.'
                  : 'Audio-only pipeline; no avatar baking required.'
              }
              progress={preprocessDisplayProgress}
              activeStepIndex={stepStatuses.preprocess === 'running' ? preprocessStageIndex : null}
              statusContent={preprocessStatusCard}
              showStepsWithContent={true}
            >
              <div className="space-y-6">
                <div className="bg-slate-50 border border-slate-100 rounded-xl p-4 text-xs text-slate-600">
                  <p className="uppercase tracking-widest text-[9px] font-bold text-slate-400">Input</p>
                  <p>Profile: <span className="font-semibold">{profile.name || '-'}</span></p>
                  <p>File: <span className="font-semibold">{profile.lastUploadedFile || 'Upload a file first'}</span></p>
                </div>
                {profileType === 'avatar' && (
                  <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-2">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
                      Avatar Cache Window
                    </p>
                    <div className="flex items-center justify-between gap-4 text-sm">
                      <label className="text-slate-600 font-semibold">Start at (seconds)</label>
                      <input
                        type="number"
                        min={0}
                        step={0.1}
                        value={avatarStartSec}
                        onChange={(e) => setAvatarStartSec(Number(e.target.value))}
                        className="w-24 rounded-lg border border-slate-200 px-2 py-1 text-right text-slate-700"
                      />
                    </div>
                    <div className="flex items-center justify-between gap-4 text-sm">
                      <label className="text-slate-600 font-semibold">Background blur</label>
                      <div className="flex items-center gap-2">
                        {(['low', 'medium', 'high'] as const).map((level) => (
                          <button
                            key={level}
                            type="button"
                            onClick={() => setAvatarBlurLevel(level)}
                            className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest transition-all ${
                              avatarBlurLevel === level
                                ? 'bg-teal-600 text-white'
                                : 'bg-slate-100 text-slate-500'
                            }`}
                          >
                            {level}
                          </button>
                        ))}
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-500">
                      Cache length is fixed at 10 seconds. Default start is 5s. Low=60, Medium=75, High=90.
                    </p>
                  </div>
                )}
                {preprocessStats ? (
                  <div className="grid grid-cols-4 gap-4 animate-in fade-in zoom-in-95">
                    {[
                      { l: 'Raw Duration', v: preprocessStats.duration },
                      { l: 'Kept Segs', v: preprocessStats.segmentsKept },
                      { l: 'Filtered', v: preprocessStats.segmentsFiltered },
                      { l: 'Sample Rate', v: preprocessStats.sampleRate },
                    ].map((s, i) => (
                      <div key={i} className="bg-teal-600 text-white p-4 rounded-xl shadow-lg shadow-teal-600/10">
                        <p className="text-[9px] font-bold opacity-70 uppercase tracking-widest">{s.l}</p>
                        <p className="text-lg font-bold">{s.v}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <button
                    onClick={startPreprocess}
                    disabled={
                      stepStatuses.preprocess === 'running' ||
                      !profile.name ||
                      (!lastUploadedFilename && !(currentProfileInfo?.raw_files && currentProfileInfo.raw_files > 0)) ||
                      (profileType === 'avatar' &&
                        !lastUploadedAudioFilename &&
                        !(currentProfileInfo?.raw_audio_files && currentProfileInfo.raw_audio_files > 0))
                    }
                    className="w-full py-4 bg-teal-600 text-white font-bold rounded-xl hover:bg-teal-700 transition-all flex items-center justify-center gap-3 shadow-xl shadow-teal-600/20"
                  >
                    {stepStatuses.preprocess === 'running' ? 'Processing Pipeline...' : 'Begin Preprocessing'}
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" /></svg>
                  </button>
                )}
                <LogPanel logs={preprocessLogs} />
              </div>
            </StepCard>
          )}

          {activeStep === 3 && (
            <StepCard
              stepNumber={3}
              title={ttsBackend !== 'styletts2' ? 'StyleTTS2 Backup Training' : 'Voice Model Training'}
              description={
                ttsBackend !== 'styletts2'
                  ? 'Optional backup training. This TTS backend can use the processed profile audio directly.'
                  : 'Fine-tune StyleTTS2 with your settings and flags.'
              }
              status={stepStatuses.train}
              isActive={true}
              statusSteps={trainSteps}
              statusNote={
                ttsBackend !== 'styletts2'
                  ? 'Skip this step unless you also want a StyleTTS2 backup checkpoint.'
                  : 'Training runs on GPU. Early-stop finishes once the sweet spot is detected.'
              }
              progress={trainDisplayProgress}
              activeStepIndex={stepStatuses.train === 'running' ? trainStageIndex : null}
              statusContent={trainingStatusCard}
              showStepsWithContent={true}
            >
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-1 bg-slate-50 p-6 rounded-2xl border border-slate-100 space-y-6">
                  <div className="space-y-3">
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Training Profile</p>
                    <p className="text-[11px] text-slate-500">
                      Unified training profile (25 epochs, max_len 400). The fast/quality toggle is removed.
                    </p>
                  </div>

                  <div>
                    <button
                      type="button"
                      onClick={() => setShowAdvancedTrain(prev => !prev)}
                      className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2"
                    >
                      Advanced Parameters
                      <span className="text-[9px] text-slate-500">{showAdvancedTrain ? 'Hide' : 'Show'}</span>
                    </button>
                    {showAdvancedTrain && (
                      <div className="mt-4 space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-xs font-bold text-slate-600">Batch Size</span>
                          <input
                            type="number"
                            min={1}
                            value={trainParams.batchSize}
                            onChange={(e) => setTrainParams(prev => ({ ...prev, batchSize: Number(e.target.value) }))}
                            className="text-xs font-mono font-bold bg-white px-2 py-1 rounded w-20 text-right"
                          />
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-xs font-bold text-slate-600">Epochs</span>
                          <input
                            type="number"
                            min={1}
                            value={trainParams.epochs}
                            onChange={(e) => setTrainParams(prev => ({ ...prev, epochs: Number(e.target.value) }))}
                            className="text-xs font-mono font-bold bg-white px-2 py-1 rounded w-20 text-right"
                          />
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-xs font-bold text-slate-600">Max Len</span>
                          <input
                            type="number"
                            min={1}
                            value={trainParams.maxLen}
                            onChange={(e) => setTrainParams(prev => ({ ...prev, maxLen: Number(e.target.value) }))}
                            className="text-xs font-mono font-bold bg-white px-2 py-1 rounded w-20 text-right"
                          />
                        </div>
                        <div className="space-y-2 text-xs font-semibold text-slate-600">
                          {[
                            ['Auto-select epoch', 'autoSelectEpoch'],
                            ['Auto-tune profile', 'autoTuneProfile'],
                            ['Build lexicon', 'autoBuildLexicon'],
                            ['Early stop', 'earlyStop'],
                          ].map(([label, key]) => (
                            <label key={key} className="flex items-center justify-between">
                              <span>{label}</span>
                              <input
                                type="checkbox"
                                checked={(trainFlags as any)[key]}
                                onChange={(e) => setTrainFlags(prev => ({ ...prev, [key]: e.target.checked }))}
                                className="accent-teal-600"
                              />
                            </label>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <button
                    onClick={startTraining}
                    disabled={stepStatuses.train === 'running'}
                    className="w-full py-3 bg-slate-900 text-white font-bold rounded-xl hover:bg-black transition-all shadow-lg"
                  >
                    {stepStatuses.train === 'running' ? 'Training...' : 'Launch Trainer'}
                  </button>
                  <div className="text-[10px] font-mono bg-white border border-slate-200 rounded-lg p-3 text-slate-500">
                    {trainingCommand}
                  </div>
                </div>

                <div className="md:col-span-2 space-y-4">
                  <LogPanel logs={trainLogs} title="StyleTTS2 Local Worker Output" />
                </div>
              </div>
            </StepCard>
          )}

          {activeStep === 4 && (
            <StepCard
              stepNumber={4}
              title="Real-time Generation"
              description="Stream voice-only or voice + lip sync with chunked playback."
              status={stepStatuses.inference}
              isActive={true}
            >
                <div className="space-y-6">
                {isWarmingUp && (
                  <div className="bg-amber-50 border border-amber-200 text-amber-800 text-xs font-semibold px-4 py-3 rounded-xl">
                    Warming up {warmupTargetName || 'selected profile'}...
                  </div>
                )}
                <div className="bg-slate-50 border border-slate-100 rounded-xl px-4 py-3 text-sm text-slate-700">
                  <span className="font-semibold">Profile:</span> {profile.name || '-'}
                </div>
                <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
                  <div className={`xl:col-span-7 bg-slate-950 border border-slate-900 rounded-2xl p-4 flex flex-col ${outputMode === 'avatar' ? '' : 'opacity-40'}`}>
                    <div className="flex items-center justify-between text-xs text-slate-300">
                      <span className="uppercase tracking-widest text-[9px] font-bold text-slate-400">Avatar Preview</span>
                      <span className="text-[10px] font-bold text-teal-300">{outputMode === 'avatar' ? `${videoFps} FPS | ${videoQueue} queued` : 'disabled'}</span>
                    </div>
                    <div className="mt-3 bg-black rounded-xl overflow-hidden border border-slate-800 w-full max-w-[720px] mx-auto min-h-[720px]">
                      <canvas ref={videoCanvasRef} width={810} height={1080} className="w-full h-full" />
                    </div>
                    <div className="mt-3 text-[11px] text-slate-400 flex items-center justify-between">
                      <span>Status: <span className="font-semibold text-slate-200">{videoState}</span></span>
                      <span>Queue: <span className="font-semibold text-slate-200">{videoQueue}</span></span>
                    </div>
                  </div>

                  <div className="xl:col-span-5 space-y-4">
                    <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-3">
                      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                        Lip Sync Model
                      </p>
                      <div className="grid grid-cols-1 gap-2">
                        <div className="px-3 py-2 rounded-lg text-sm font-semibold bg-teal-600 text-white text-center">
                          {avatarBackend === 'wav2lip' ? 'Wav2Lip' : 'MuseTalk'}
                        </div>
                      </div>
                    </div>

                    <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-3">
                      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                        LLM Mode
                      </p>
                      <select
                        value={llmMode}
                        onChange={(event) => setLlmMode(event.target.value as LLMMode)}
                        className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700"
                      >
                        {LLM_MODE_OPTIONS.map(option => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <VoiceControlsPanel
                      values={voiceControlValues}
                      defaults={voiceControlDefaults}
                      status={voiceControlsStatus}
                      error={voiceControlsError}
                      saveStatus={runtimeSettingsStatus}
                      saveError={runtimeSettingsError}
                      canSave={apiStatus === 'online' && Boolean(profile.name.trim())}
                      ttsBackend={ttsBackend}
                      onChange={handleVoiceControlsChange}
                      onReset={resetVoiceControls}
                      onSave={saveRuntimeSettings}
                    />

                    <div className="bg-slate-950 border border-slate-900 rounded-2xl p-4">
                      <ControlPanel
                        variant="embedded"
                        onInterrupt={stopInference}
                        stopListeningRef={stopListeningRef}
                        onSendChat={async (text) => runInference(text, '/chat')}
                        onSendDirect={async (text) => {
                          setInferenceText(text);
                          await runInference(text, '/speak');
                        }}
                      />
                    </div>
                    <button
                      onClick={stopInference}
                      className="w-full px-4 py-2 bg-slate-900 text-white text-sm font-bold rounded-lg"
                    >
                      Stop
                    </button>
                  </div>
                </div>

                {latency && (
                  <div className="grid grid-cols-2 gap-4 animate-in slide-in-from-top-4">
                    <div className="bg-slate-900 p-6 rounded-2xl flex items-center justify-between">
                      <div>
                        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Time to First Audio</p>
                        <p className="text-2xl font-bold text-teal-400">{latency.ttfa}ms</p>
                      </div>
                    </div>
                    <div className="bg-slate-900 p-6 rounded-2xl flex items-center justify-between">
                      <div>
                        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Inference Chunks</p>
                        <p className="text-2xl font-bold text-slate-200">{inferenceChunks.length}</p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="bg-slate-50 border border-slate-100 rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-4">
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Live Playback Buffers</p>
                    <span className="text-[10px] font-bold text-teal-600 bg-teal-50 px-2 py-1 rounded">{outputMode === 'avatar' ? 'Audio + Video' : 'Audio'}</span>
                  </div>
                  <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
                    {inferenceChunks.length === 0 ? (
                      <div className="w-full h-12 flex items-center justify-center border border-dashed border-slate-200 rounded-lg text-[11px] text-slate-400 font-bold italic">
                        Stream pending...
                      </div>
                    ) : (
                      [...inferenceChunks]
                        .sort((a, b) => a.index - b.index)
                        .map(c => (
                        <div key={c.index} className="flex-shrink-0 w-24 bg-white border border-slate-200 p-2 rounded-lg flex flex-col animate-in scale-in">
                          <span className="text-[9px] font-bold text-slate-400">CHUNK {c.index + 1}</span>
                          <span className="text-xs font-bold text-teal-600">{c.duration.toFixed(2)}s</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </StepCard>
          )}
        </div>
      </div>

      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex gap-4">
        {activeStep > 1 && (
          <button
            onClick={() => {
              if (isBusy) {
                setUiNotice('Stop the current job before changing steps.');
                return;
              }
              setActiveStep(prev => prev - 1);
            }}
            disabled={isBusy}
            className={`bg-white border-2 border-slate-100 text-slate-600 font-bold px-6 py-3 rounded-2xl shadow-xl hover:bg-slate-50 transition-all flex items-center gap-2 ${isBusy ? 'opacity-60 cursor-not-allowed' : ''}`}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7" /></svg>
            Previous
          </button>
        )}
        {canProceedTo(activeStep + 1) && activeStep < 4 && (
          <button
            onClick={() => {
              if (isBusy) {
                setUiNotice('Stop the current job before changing steps.');
                return;
              }
              setActiveStep(prev => prev + 1);
            }}
            disabled={isBusy}
            className={`bg-teal-600 text-white font-bold px-8 py-3 rounded-2xl shadow-xl shadow-teal-600/20 hover:bg-teal-700 transition-all flex items-center gap-2 ${isBusy ? 'opacity-60 cursor-not-allowed' : 'animate-bounce-short'}`}
          >
            Next Stage
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7" /></svg>
          </button>
        )}
      </div>
    </div>
  );
};

export default App;
