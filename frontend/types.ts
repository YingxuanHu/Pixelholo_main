
export type StepStatus = 'idle' | 'running' | 'done' | 'error';
export type ProfileType = 'voice' | 'avatar';

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'warn' | 'error' | 'success';
  message: string;
}

export interface Profile {
  name: string;
  lastUploadedFile: string | null;
  fileSize: string | null;
}

export interface PreprocessStats {
  duration: string;
  segmentsKept: number;
  segmentsFiltered: number;
  avgClipLength: string;
  sampleRate: string;
}

export interface TrainStats {
  currentEpoch: number;
  totalEpochs: number;
  steps: number;
  eta: string;
  gpuMemory: string;
  bestCheckpoint: string;
}

export interface InferenceChunk {
  index: number;
  duration: number;
  receivedAt: number;
}

export interface ProfileInfo {
  name: string;
  profile_type?: string;
  has_data: boolean;
  raw_files: number;
  raw_audio_files?: number;
  processed_wavs: number;
  has_profile: boolean;
  best_checkpoint: string | null;
  latest_checkpoint: string | null;
}
