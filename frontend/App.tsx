import React, { useCallback, useEffect, useRef, useState } from 'react';
import ControlPanel from './components/ControlPanel';
import Header from './components/Header';
import StepCard from './components/StepCard';
import LogPanel from './components/LogPanel';
import {
  Profile,
  StepStatus,
  LogEntry,
  PreprocessStats,
  TrainStats,
  InferenceChunk,
  ProfileInfo,
  ProfileType,
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

const DEFAULT_API_BASE = 'http://127.0.0.1:8000';
const LOCAL_STORAGE_API_BASE_KEY = 'voxclone_api_base';
const DEFAULT_PROFILE_TYPE: 'voice' | 'avatar' = 'voice';
const DEFAULT_OUTPUT_MODE: 'voice' | 'avatar' = 'voice';
const DEFAULT_AVATAR_START_SEC = 5;
const BLUR_KERNEL_BY_LEVEL = { low: 60, medium: 75, high: 90 } as const;
const DEFAULT_AVATAR_BLUR_LEVEL: keyof typeof BLUR_KERNEL_BY_LEVEL = 'medium';
const DEFAULT_VIDEO_FPS = 25;
const DEFAULT_AUDIO_START_DELAY_SEC = 0.05;
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
  const [apiStatus, setApiStatus] = useState<'online' | 'offline' | 'checking'>('checking');
  const [profileType, setProfileType] = useState<'voice' | 'avatar'>(DEFAULT_PROFILE_TYPE);
  const [profile, setProfile] = useState<Profile>({ name: '', lastUploadedFile: null, fileSize: null });
  const [lastUploadedFilename, setLastUploadedFilename] = useState<string | null>(null);
  const [lastUploadedAudioFilename, setLastUploadedAudioFilename] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [profilesStatus, setProfilesStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const warmupTimerRef = useRef<number | null>(null);
  const warmupNoticeTimerRef = useRef<number | null>(null);
  const [isWarmingUp, setIsWarmingUp] = useState(false);
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
  const [inferenceText, setInferenceText] = useState('');
  const [inferenceChunks, setInferenceChunks] = useState<InferenceChunk[]>([]);
  const [latency, setLatency] = useState<{ ttfa: number; total: number } | null>(null);
  const [modelOverride, setModelOverride] = useState('');
  const [refOverride, setRefOverride] = useState('');
  const [outputMode, setOutputMode] = useState<'voice' | 'avatar'>(DEFAULT_OUTPUT_MODE);
  const [videoState, setVideoState] = useState<'idle' | 'buffering' | 'playing'>('idle');
  const [videoFps, setVideoFps] = useState(DEFAULT_VIDEO_FPS);
  const [videoQueue, setVideoQueue] = useState(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const audioEndTimeRef = useRef<number>(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const audioUnlockedRef = useRef<boolean>(false);
  const streamAbortRef = useRef<AbortController | null>(null);
  const streamSessionRef = useRef<number>(0);
  const isBusy = Object.values(stepStatuses).some(status => status === 'running');
  const warmedProfilesRef = useRef<Set<string>>(new Set());
  const videoCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoTimerRef = useRef<number | null>(null);
  const videoRafRef = useRef<number | null>(null);
  const videoStartTimeRef = useRef<number | null>(null);
  const videoNextFrameTimeRef = useRef<number | null>(null);
  const audioStartDelayRef = useRef<number>(DEFAULT_AUDIO_START_DELAY_SEC);
  const videoFpsRef = useRef<number>(DEFAULT_VIDEO_FPS);
  const frameQueueRef = useRef<{ img: string; t: number }[]>([]);

  useEffect(() => {
    setTrainParams(trainPreset);
  }, []);

  const uploadWithProgress = useCallback((url: string, form: FormData, onProgress: (loaded: number, total: number) => void) => {
    return new Promise<string>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url);
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
      xhr.onerror = () => reject(new Error('Upload failed'));
      xhr.send(form);
    });
  }, []);

  useEffect(() => {
    if (profileType === 'avatar') {
      setAvatarStartSec(DEFAULT_AVATAR_START_SEC);
      setAvatarBlurLevel(DEFAULT_AVATAR_BLUR_LEVEL);
    }
  }, [profileType]);

  useEffect(() => {
    const cached = localStorage.getItem(LOCAL_STORAGE_API_BASE_KEY);
    if (cached) setApiBase(cached);
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
      const res = await fetch(`${apiBase}/profiles?profile_type=${profileType}`, { cache: 'no-store' });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setProfiles(Array.isArray(data.profiles) ? data.profiles : []);
      setProfilesStatus('idle');
    } catch (err) {
      setProfilesStatus('error');
    }
  }, [apiBase, profileType]);

  const triggerWarmup = useCallback(
    (profileName: string, type: ProfileType) => {
      if (!profileName) return;
      if (warmupTimerRef.current) {
        window.clearTimeout(warmupTimerRef.current);
      }
      warmupTimerRef.current = window.setTimeout(() => {
        setIsWarmingUp(true);
        if (warmupNoticeTimerRef.current) {
          window.clearTimeout(warmupNoticeTimerRef.current);
        }
        warmupNoticeTimerRef.current = window.setTimeout(() => {
          setIsWarmingUp(false);
        }, 8000);
        fetch(`${apiBase}/warmup`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ profile: profileName, profile_type: type }),
        }).catch(() => {});
      }, 300);
    },
    [apiBase],
  );

  useEffect(() => {
    loadProfiles();
  }, [loadProfiles]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setApiStatus('checking');
    fetch(`${apiBase}/docs`, { signal: controller.signal })
      .then((res) => {
        if (!cancelled) setApiStatus(res.ok ? 'online' : 'offline');
      })
      .catch(() => {
        if (!cancelled) setApiStatus('offline');
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [apiBase]);

  const currentProfileInfo = profiles.find((item) => item.name === profile.name);
  const hasTrainedProfile = Boolean(currentProfileInfo?.has_profile);
  const hasData = Boolean(currentProfileInfo?.has_data);

  const preprocessSteps = [
    ...(profileType === 'avatar' ? ['Bake avatar frames (Wav2Lip cache)'] : []),
    'Extract audio track',
    'Loudness normalize + filter',
    'Split on silence (2–10s)',
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

  const warmupProfile = useCallback(async (profileName: string) => {
    if (!profileName) return;
    if (warmedProfilesRef.current.has(profileName)) return;
    warmedProfilesRef.current.add(profileName);
    try {
      const res = await fetch(`${apiBase}/warmup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile: profileName,
          profile_type: profileType,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      await res.json();
    } catch (err) {
      warmedProfilesRef.current.delete(profileName);
    }
  }, [apiBase, profileType]);

  const canProceedTo = (step: number) => {
    if (step === 1) return true;
    if (step === 2) return Boolean(profile.name);
    if (step === 3) return stepStatuses.preprocess === 'done' || hasData || hasTrainedProfile;
    if (step === 4) return stepStatuses.train === 'done' || hasTrainedProfile;
    return false;
  };

  useEffect(() => {
    if (apiStatus !== 'online') return;
    if (!hasTrainedProfile) return;
    if (!profile.name) return;
    warmupProfile(profile.name);
  }, [apiStatus, hasTrainedProfile, profile.name, warmupProfile]);

  const streamResponseLines = async (response: Response, onLine: (line: string) => void) => {
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
        if (line) onLine(line);
        index = buffer.indexOf('\n');
      }
    }
    if (buffer.trim()) {
      onLine(buffer.trim());
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

  const handleUpload = async (file: File) => {
    if (!profile.name || !file) return;
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
      setStepStatuses(prev => ({ ...prev, upload: 'done' }));
      setUploadPhaseVideo('idle');
      loadProfiles();
    } catch (err) {
      setStepStatuses(prev => ({ ...prev, upload: 'error' }));
      setUploadPhaseVideo('error');
      setPreprocessLogs([createLog(`Upload failed: ${String(err)}`, 'error')]);
    }
  };

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
      avatar_start_sec: profileType === 'avatar' ? avatarStartSec : null,
      avatar_loop_sec: profileType === 'avatar' ? 10 : null,
      avatar_blur_background: profileType === 'avatar',
      avatar_blur_kernel: profileType === 'avatar' ? BLUR_KERNEL_BY_LEVEL[avatarBlurLevel] : null,
    };
    try {
      const res = await fetch(`${apiBase}/preprocess`, {
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
          stageUpdate('Split on silence (2–10s)');
        } else if (line.includes('Transcribing full audio')) {
          stageUpdate('Transcribe with Whisper');
        } else if (line.includes('Segments: raw=')) {
          stageUpdate('Split on silence (2–10s)');
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
            duration: '—',
            segmentsKept: kept,
            segmentsFiltered: merged - kept,
            avgClipLength: '—',
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
      loadProfiles();
    } catch (err) {
      setStepStatuses(prev => ({ ...prev, preprocess: 'error' }));
      setPreprocessLogs(prev => [...prev, createLog(`Preprocess failed: ${String(err)}`, 'error')]);
      setPreprocessStageIndex(null);
    }
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
      const res = await fetch(`${apiBase}/train`, {
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

  const resetVideo = useCallback(() => {
    if (videoTimerRef.current !== null) {
      window.clearInterval(videoTimerRef.current);
      videoTimerRef.current = null;
    }
    if (videoRafRef.current !== null) {
      window.cancelAnimationFrame(videoRafRef.current);
      videoRafRef.current = null;
    }
    frameQueueRef.current = [];
    setVideoQueue(0);
    setVideoState('idle');
    videoStartTimeRef.current = null;
    videoNextFrameTimeRef.current = null;
    const canvas = videoCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  const drawFrame = useCallback((frameBase64: string) => {
    const canvas = videoCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = `data:image/jpeg;base64,${frameBase64}`;
  }, []);

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
        const frameInterval = 1 / fps;
        if (videoNextFrameTimeRef.current === null) {
          videoNextFrameTimeRef.current = startAt;
        }
        const now = ctx.currentTime;
        let frameToDraw: { img: string; t: number } | null = null;
        while (frameQueueRef.current.length > 0 && frameQueueRef.current[0].t <= now) {
          frameToDraw = frameQueueRef.current.shift() || null;
        }
        if (frameToDraw) {
          drawFrame(frameToDraw.img);
        }
        if (!frameQueueRef.current.length && videoState === 'playing') {
          setVideoState('buffering');
        } else if (frameQueueRef.current.length && videoState !== 'playing') {
          setVideoState('playing');
        }
        setVideoQueue(frameQueueRef.current.length);
      }
      videoRafRef.current = window.requestAnimationFrame(tick);
    };
    videoRafRef.current = window.requestAnimationFrame(tick);
  }, [drawFrame, videoState]);

  const enqueueFrames = useCallback((frames: string[], startAt: number, duration: number, fps?: number) => {
    if (!frames || frames.length === 0) return;
    if (fps && fps > 0) {
      setVideoFps(fps);
      videoFpsRef.current = fps;
    }
    const frameCount = frames.length;
    const frameDuration = frameCount > 0 ? duration / frameCount : 0;
    frames.forEach((img, i) => {
      frameQueueRef.current.push({ img, t: startAt + i * frameDuration });
    });
    setVideoQueue(frameQueueRef.current.length);
    if (videoState === 'idle') setVideoState('buffering');
    startVideoLoop();
  }, [startVideoLoop, videoState]);

  const scheduleBuffer = (buffer: AudioBuffer) => {
    if (!audioContextRef.current) return;
    const ctx = audioContextRef.current;
    const source = ctx.createBufferSource();
    const gain = ctx.createGain();
    source.buffer = buffer;
    source.connect(gain);
    gain.connect(ctx.destination);
    activeSourcesRef.current.add(source);
    source.onended = () => activeSourcesRef.current.delete(source);

    const fadeMs = 12;
    const fadeTime = Math.min(fadeMs / 1000, Math.max(0.003, buffer.duration / 4));
    // Slight overlap to hide tiny inter-chunk gaps.
    const overlap = Math.min(0.04, fadeTime); // up to 40ms overlap
    const desiredStart = nextStartTimeRef.current - overlap;
    const startAt = Math.max(ctx.currentTime + 0.05, desiredStart);
    const endAt = startAt + buffer.duration;

    const fadeSamples = Math.max(32, Math.floor(ctx.sampleRate * fadeTime));
    const fadeIn = new Float32Array(fadeSamples);
    const fadeOut = new Float32Array(fadeSamples);
    for (let i = 0; i < fadeSamples; i += 1) {
      const t = i / (fadeSamples - 1);
      fadeIn[i] = Math.sin(t * Math.PI * 0.5);
      fadeOut[i] = Math.cos(t * Math.PI * 0.5);
    }

    gain.gain.setValueAtTime(0, startAt);
    gain.gain.setValueCurveAtTime(fadeIn, startAt, fadeTime);
    gain.gain.setValueAtTime(1, Math.max(startAt + fadeTime, endAt - fadeTime));
    gain.gain.setValueCurveAtTime(fadeOut, Math.max(startAt, endAt - fadeTime), fadeTime);

    source.start(startAt);
    nextStartTimeRef.current = endAt;
    audioEndTimeRef.current = Math.max(audioEndTimeRef.current, endAt);
    return { startAt, endAt };
  };

  const runInference = useCallback(async (text: string, endpoint: string) => {
    if (!profile.name || !text) return;
    setUiNotice(null);
    await unlockAudio();
    if (audioContextRef.current?.state !== 'running') {
      setUiNotice('Safari blocked audio. Click again to enable sound.');
      return;
    }
    await warmupProfile(profile.name);
    // End any existing stream immediately.
    streamSessionRef.current += 1;
    stopAllAudio();
    resetVideo();
    setStepStatuses(prev => ({ ...prev, inference: 'running' }));
    setInferenceChunks([]);
    setLatency(null);
    setInferenceStageIndex(inferenceSteps.length > 0 ? 0 : null);
    nextStartTimeRef.current = audioContextRef.current?.currentTime || 0;
    audioEndTimeRef.current = nextStartTimeRef.current;
    audioStartDelayRef.current = outputMode === 'avatar' ? 0.35 : 0.05;
    let sawError = false;

    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
    }
    const controller = new AbortController();
    streamAbortRef.current = controller;
    const sessionId = streamSessionRef.current;

    const startTime = performance.now();
    let firstChunk = true;

    const payload: Record<string, any> = {
      speaker: profile.name,
      profile_type: profileType,
      text,
      model_path: modelOverride || null,
      ref_wav_path: refOverride || null,
    };
    if (outputMode === 'avatar') {
      payload.avatar_profile = profile.name;
    }

    try {
      const res = await fetch(`${apiBase}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(await res.text());
      await streamResponseLines(res, async line => {
        if (sessionId !== streamSessionRef.current) {
          return;
        }
        if (sawError) return;
        let data: any;
        try {
          data = JSON.parse(line);
        } catch (parseErr) {
          sawError = true;
          setStepStatuses(prev => ({ ...prev, inference: 'error' }));
          setInferenceStageIndex(null);
          return;
        }
        if (data.event === 'error') {
          sawError = true;
          setStepStatuses(prev => ({ ...prev, inference: 'error' }));
          setInferenceStageIndex(null);
          return;
        }
        if (data.event === 'done') {
          if (sawError) return;
          setLatency(prev => prev ? { ...prev, total: data.inference_ms ?? Math.round(performance.now() - startTime) } : null);
          setStepStatuses(prev => ({ ...prev, inference: 'done' }));
          setInferenceStageIndex(inferenceSteps.length ? inferenceSteps.length - 1 : null);
          return;
        }
        if (!data.audio_base64) return;
        if (firstChunk) {
          firstChunk = false;
          setLatency({ ttfa: Math.round(performance.now() - startTime), total: 0 });
          setInferenceStageIndex(Math.min(2, inferenceSteps.length - 1));
        }
        const binary = atob(data.audio_base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
          bytes[i] = binary.charCodeAt(i);
        }
        const buffer = await audioContextRef.current!.decodeAudioData(bytes.buffer);
        const schedule = scheduleBuffer(buffer);
        if (outputMode === 'avatar' && schedule && videoStartTimeRef.current === null) {
          videoStartTimeRef.current = schedule.startAt;
          videoNextFrameTimeRef.current = schedule.startAt;
        }
        if (outputMode === 'avatar' && schedule && Array.isArray(data.frames_base64)) {
          const duration = typeof data.duration_sec === 'number' ? data.duration_sec : buffer.duration;
          enqueueFrames(data.frames_base64, schedule.startAt, duration, data.fps);
          setInferenceStageIndex(Math.min(3, inferenceSteps.length - 1));
        }
        if (outputMode === 'voice') {
          setInferenceStageIndex(Math.min(4, inferenceSteps.length - 1));
        }
        setInferenceChunks(prev => [...prev, { index: data.chunk_index, duration: buffer.duration, receivedAt: Date.now() }]);
      });
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setStepStatuses(prev => ({ ...prev, inference: 'error' }));
        setInferenceStageIndex(null);
      }
    }
  }, [apiBase, enqueueFrames, modelOverride, outputMode, profile.name, profileType, refOverride, resetVideo, unlockAudio]);

  const startInference = useCallback(async () => {
    await runInference(inferenceText, '/speak');
  }, [inferenceText, runInference]);

  const stopInference = async () => {
    if (streamAbortRef.current) streamAbortRef.current.abort();
    streamSessionRef.current += 1;
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
  const inferenceDisplayProgress =
    stepStatuses.inference === 'running'
      ? Math.max(
          Math.min(1, inferenceChunks.length / 20),
          stageProgress(inferenceStageIndex, inferenceSteps.length, 0.2),
        )
      : stepStatuses.inference === 'done'
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
            Kept: {preprocessStats?.segmentsKept ?? '—'}
          </div>
          <div className="bg-white p-2 rounded border border-amber-100">
            Filtered: {preprocessStats?.segmentsFiltered ?? '—'}
          </div>
        </div>
      </div>
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
                      placeholder="e.g. Alvin Studio Master"
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
                          <button
                            key={item.name}
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
                              triggerWarmup(item.name, item.profile_type);
                              setLastUploadedFilename(null);
                              setLastUploadedAudioFilename(null);
                              // Jump to generation if this profile is ready.
                              if (item.has_profile) {
                                setActiveStep(4);
                              }
                            }}
                            disabled={isBusy}
                            className={`w-full flex items-center justify-between border rounded-lg px-3 py-2 text-left ${
                              profile.name === item.name
                                ? 'border-teal-600 bg-teal-50 text-teal-800'
                                : 'border-slate-200 bg-white text-slate-600'
                            } ${isBusy ? 'opacity-60 cursor-not-allowed' : ''}`}
                          >
                            <div>
                              <p className="text-xs font-bold">{item.name}</p>
                              <p className="text-[10px] text-slate-400">
                                {item.processed_wavs} clips · {(item.raw_audio_files ?? 0)} audio · {item.raw_files} video · {item.profile_type || profileType}
                              </p>
                            </div>
                            <div className="text-[10px] font-bold">
                              {item.has_profile ? 'ready' : 'needs training'}
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-4">
                  <p className="text-xs text-slate-500 bg-slate-50 border border-slate-100 rounded-lg px-3 py-2">
                    Avatar setup uses two files: a video for the face loop and a separate audio file for voice training.
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
                              {formatBytes(uploadBytesVideo.loaded)} / {formatBytes(uploadBytesVideo.total)} · {uploadProgressVideo}%
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
                                {formatBytes(uploadBytesAudio.loaded)} / {formatBytes(uploadBytesAudio.total)} · {uploadProgressAudio}%
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
                  <p>Profile: <span className="font-semibold">{profile.name || '—'}</span></p>
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
              title="Voice Model Training"
              description="Fine-tune StyleTTS2 with your settings and flags."
              status={stepStatuses.train}
              isActive={true}
              statusSteps={trainSteps}
              statusNote="Training runs on GPU. Early-stop finishes once the sweet spot is detected."
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
              statusSteps={inferenceSteps}
              statusNote="First request warms the model. Subsequent requests are near-instant."
              progress={inferenceDisplayProgress}
              activeStepIndex={stepStatuses.inference === 'running' ? inferenceStageIndex : null}
            >
                <div className="space-y-6">
                {isWarmingUp && (
                  <div className="bg-amber-50 border border-amber-200 text-amber-800 text-xs font-semibold px-4 py-3 rounded-xl">
                    Warming up the selected profile for faster first response…
                  </div>
                )}
                <div className="grid grid-cols-1 gap-6 items-stretch">
                  <div className="flex flex-col gap-4 w-full max-w-[740px] mx-auto items-center">
                    <div className="w-full bg-slate-50 border border-slate-100 rounded-xl p-4 text-xs text-slate-600">
                      <p className="uppercase tracking-widest text-[9px] font-bold text-slate-400">Defaults</p>
                      <p>Profile: <span className="font-semibold">{profile.name || '—'}</span></p>
                      <p>Model/Ref: <span className="font-semibold">profile.json unless overridden</span></p>
                    </div>
                    <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-4">
                      <input
                        value={modelOverride}
                        onChange={(e) => setModelOverride(e.target.value)}
                        placeholder="Model path override (optional)"
                        className="w-full bg-white border-2 border-slate-100 rounded-xl px-4 py-3 text-sm"
                      />
                      <input
                        value={refOverride}
                        onChange={(e) => setRefOverride(e.target.value)}
                        placeholder="Reference wav override (optional)"
                        className="w-full bg-white border-2 border-slate-100 rounded-xl px-4 py-3 text-sm"
                      />
                    </div>
                    <div className={`w-full bg-slate-950 border border-slate-900 rounded-2xl p-4 flex flex-col ${outputMode === 'avatar' ? '' : 'opacity-40'}`}>
                      <div className="flex items-center justify-between text-xs text-slate-300">
                        <span className="uppercase tracking-widest text-[9px] font-bold text-slate-400">Avatar Preview</span>
                        <span className="text-[10px] font-bold text-teal-300">{outputMode === 'avatar' ? `${videoFps} FPS · ${videoQueue} queued` : 'disabled'}</span>
                      </div>
                      <div className="mt-3 bg-black rounded-xl overflow-hidden border border-slate-800 flex-1 min-h-[720px] w-full max-w-[640px] mx-auto">
                        <canvas ref={videoCanvasRef} width={640} height={853} className="w-full h-full" />
                      </div>
                      <div className="mt-3 text-[11px] text-slate-400 flex items-center justify-between">
                        <span>Status: <span className="font-semibold text-slate-200">{videoState}</span></span>
                        <span>Queue: <span className="font-semibold text-slate-200">{videoQueue}</span></span>
                      </div>
                      <button
                        onClick={stopInference}
                        className="mt-3 w-full px-4 py-2 bg-slate-900 text-white text-xs font-bold rounded-lg"
                      >
                        Stop
                      </button>
                      <div className="mt-4">
                        <ControlPanel
                          variant="embedded"
                          onInterrupt={stopInference}
                          onSendChat={async (text) => runInference(text, '/chat')}
                          onSendDirect={async (text) => {
                            setInferenceText(text);
                            await runInference(text, '/speak');
                          }}
                        />
                      </div>
                    </div>
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
