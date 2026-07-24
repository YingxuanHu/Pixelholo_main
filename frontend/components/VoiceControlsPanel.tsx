import React from 'react';
import { TTSBackend, VoiceControlValues, VoiceEmotion } from '../types';

type VoiceControlsPanelProps = {
  values: VoiceControlValues | null;
  defaults: VoiceControlValues | null;
  status: 'idle' | 'loading' | 'ready' | 'error';
  error?: string | null;
  saveStatus?: 'idle' | 'saving' | 'saved' | 'error';
  saveError?: string | null;
  canSave?: boolean;
  ttsBackend?: TTSBackend;
  onChange: (patch: Partial<VoiceControlValues>) => void;
  onReset: () => void;
  onSave: () => void;
};

type SliderConfig = {
  key: Extract<keyof VoiceControlValues, 'pitch' | 'pace' | 'tone' | 'volume' | 'expressiveness' | 'variation' | 'guidance' | 'repetition'>;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  minLabel: string;
  maxLabel: string;
  describe: (value: number) => string;
};

const describeRelative = (value: number, softer: string, stronger: string) => {
  const amount = Math.abs(value);
  if (amount < 5) return 'Default';
  if (amount < 35) return value < 0 ? `Slightly ${softer}` : `Slightly ${stronger}`;
  if (amount < 70) return value < 0 ? softer : stronger;
  return value < 0 ? `Much ${softer}` : `Much ${stronger}`;
};

const describePitch = (value: number) => {
  const amount = Math.abs(value);
  if (amount < 0.05) return 'Default';
  if (amount < 1.0) return value < 0 ? 'Slightly deeper' : 'Slightly higher';
  if (amount < 2.5) return value < 0 ? 'Deeper' : 'Higher';
  return value < 0 ? 'Much deeper' : 'Much higher';
};

const describeDefault = (value: string) => (value === 'Default' ? 'Profile default' : `Default ${value.toLowerCase()}`);

const STYLE_TTS2_SLIDERS: SliderConfig[] = [
  {
    key: 'pitch',
    label: 'Pitch',
    hint: 'Make the voice deeper or higher.',
    min: -4,
    max: 4,
    step: 0.1,
    minLabel: 'Deeper',
    maxLabel: 'Higher',
    describe: describePitch,
  },
  {
    key: 'pace',
    label: 'Pace',
    hint: 'Slow the delivery down or speed it up.',
    min: -100,
    max: 100,
    step: 1,
    minLabel: 'Slower',
    maxLabel: 'Faster',
    describe: (value) => describeRelative(value, 'slower', 'faster'),
  },
  {
    key: 'tone',
    label: 'Tone',
    hint: 'Move the delivery toward calmer or more expressive.',
    min: -100,
    max: 100,
    step: 1,
    minLabel: 'Calmer',
    maxLabel: 'More expressive',
    describe: (value) => describeRelative(value, 'calmer', 'more expressive'),
  },
  {
    key: 'volume',
    label: 'Volume',
    hint: 'Make the voice softer or stronger.',
    min: -100,
    max: 100,
    step: 1,
    minLabel: 'Softer',
    maxLabel: 'Stronger',
    describe: (value) => describeRelative(value, 'softer', 'stronger'),
  },
];

const FALLBACK_VALUES: VoiceControlValues = {
  pitch: 0,
  pace: 0,
  tone: 0,
  volume: 0,
  expressiveness: 0.5,
  variation: 0.8,
  guidance: 0.5,
  repetition: 1.2,
  emotion: 'neutral',
  emotionIntensity: 0.5,
};

const describeCentered = (
  value: number,
  center: number,
  low: string,
  high: string,
  tolerance = 0.03,
) => {
  const delta = value - center;
  if (Math.abs(delta) <= tolerance) return 'Default';
  return delta < 0 ? low : high;
};

const CHATTERBOX_SLIDERS: SliderConfig[] = [
  {
    key: 'expressiveness',
    label: 'Expressiveness',
    hint: 'Control Chatterbox delivery energy.',
    min: 0.25,
    max: 1,
    step: 0.01,
    minLabel: 'Subtle',
    maxLabel: 'Expressive',
    describe: (value) => describeCentered(value, 0.5, 'More subtle', 'More expressive'),
  },
  {
    key: 'guidance',
    label: 'Voice Match',
    hint: 'How strongly generation follows the voice prompt.',
    min: 0,
    max: 1,
    step: 0.01,
    minLabel: 'Looser',
    maxLabel: 'Closer',
    describe: (value) => describeCentered(value, 0.5, 'Looser match', 'Closer match'),
  },
  {
    key: 'pace',
    label: 'Pace',
    hint: 'Slow the delivery down or speed it up after TTS.',
    min: -100,
    max: 100,
    step: 1,
    minLabel: 'Slower',
    maxLabel: 'Faster',
    describe: (value) => describeRelative(value, 'slower', 'faster'),
  },
  {
    key: 'volume',
    label: 'Volume',
    hint: 'Make the voice softer or stronger after TTS.',
    min: -100,
    max: 100,
    step: 1,
    minLabel: 'Softer',
    maxLabel: 'Stronger',
    describe: (value) => describeRelative(value, 'softer', 'stronger'),
  },
];

const EMOTION_OPTIONS: { value: VoiceEmotion; label: string }[] = [
  { value: 'neutral', label: 'Neutral' },
  { value: 'happy', label: 'Happy' },
  { value: 'sad', label: 'Sad' },
  { value: 'angry', label: 'Angry' },
  { value: 'scared', label: 'Scared' },
  { value: 'disgust', label: 'Disgust' },
];

const EMOTION_VALUE_PRESETS: Record<VoiceEmotion, Pick<VoiceControlValues, 'expressiveness' | 'variation' | 'guidance'>> = {
  neutral: { expressiveness: 0.5, variation: 0.8, guidance: 0.5 },
  happy: { expressiveness: 0.62, variation: 0.9, guidance: 0.45 },
  sad: { expressiveness: 0.42, variation: 0.7, guidance: 0.58 },
  angry: { expressiveness: 0.7, variation: 0.82, guidance: 0.5 },
  scared: { expressiveness: 0.66, variation: 0.95, guidance: 0.48 },
  disgust: { expressiveness: 0.58, variation: 0.82, guidance: 0.52 },
};

const VoiceControlsPanel: React.FC<VoiceControlsPanelProps> = ({
  values,
  defaults,
  status,
  error,
  saveStatus = 'idle',
  saveError = null,
  canSave = true,
  ttsBackend = 'chatterbox',
  onChange,
  onReset,
  onSave,
}) => {
  const effectiveValues = values ?? defaults ?? FALLBACK_VALUES;
  const effectiveDefaults = defaults ?? FALLBACK_VALUES;
  const sliders = ttsBackend === 'chatterbox'
    ? CHATTERBOX_SLIDERS
    : STYLE_TTS2_SLIDERS;
  const title = ttsBackend === 'chatterbox'
    ? 'Chatterbox Voice Controls'
    : 'StyleTTS2 Voice Controls';

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{title}</p>
          <span
            className="group relative inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-slate-50 text-[11px] font-bold text-slate-500"
            aria-label="Voice control changes apply to the next generated response. Use Save to keep them as this profile's default voice controls."
          >
            ?
            <span className="pointer-events-none absolute left-0 top-6 z-20 hidden w-64 rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-[11px] font-semibold normal-case tracking-normal text-slate-600 shadow-lg group-hover:block">
              <span className="block">Changes apply to the next generated response.</span>
              <span className="mt-1 block">
                Click{' '}
                <span className="inline-flex rounded-md border border-teal-200 bg-teal-50 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-widest text-teal-700">
                  Save
                </span>{' '}
                to make these voice controls the default for this profile.
              </span>
            </span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onReset}
            disabled={!defaults}
            className="rounded-lg border border-slate-200 px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-slate-600 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Reset
          </button>
          <button
            type="button"
            onClick={onSave}
            disabled={!canSave || saveStatus === 'saving'}
            className="rounded-lg border border-teal-200 bg-teal-50 px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-teal-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {saveStatus === 'saving' ? 'Saving' : 'Save'}
          </button>
        </div>
      </div>

      {saveStatus === 'saved' && (
        <p className="text-[11px] font-semibold text-teal-700">Voice controls saved for this profile.</p>
      )}
      {saveStatus === 'error' && (
        <p className="text-[11px] font-semibold text-rose-600">
          {saveError || 'Failed to save voice controls.'}
        </p>
      )}

      {status === 'loading' && (
        <div className="rounded-xl border border-slate-100 bg-slate-50 px-4 py-5 text-xs font-semibold text-slate-500">
          Loading profile defaults...
        </div>
      )}

      {status === 'error' && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-5 text-xs font-semibold text-rose-700">
          {error || 'Failed to load voice controls for this profile.'}
        </div>
      )}

      <div className="space-y-4">
        {ttsBackend === 'chatterbox' && (
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="mb-2 flex items-center justify-between gap-3">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">Emotion</span>
              <span className="text-xs font-semibold capitalize text-teal-700">
                {effectiveValues.emotion === 'neutral' ? 'Default' : effectiveValues.emotion}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
              {EMOTION_OPTIONS.map((option) => {
                const selected = effectiveValues.emotion === option.value;
                return (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => onChange({ emotion: option.value, ...EMOTION_VALUE_PRESETS[option.value] })}
                    className={`h-8 rounded-md border text-xs font-bold transition ${
                      selected
                        ? 'border-teal-600 bg-teal-600 text-white'
                        : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300'
                    }`}
                    aria-pressed={selected}
                  >
                    {option.label}
                  </button>
                );
              })}
            </div>
            <div className={`mt-3 ${effectiveValues.emotion === 'neutral' ? 'opacity-45' : ''}`}>
              <div className="mb-1 flex items-center justify-between text-[11px] font-semibold text-slate-500">
                <span>Intensity</span>
                <span>{Math.round(effectiveValues.emotionIntensity * 100)}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={effectiveValues.emotionIntensity}
                disabled={effectiveValues.emotion === 'neutral'}
                onChange={(event) => onChange({ emotionIntensity: Number(event.target.value) })}
                className="w-full accent-teal-600"
              />
            </div>
          </div>
        )}
        {sliders.map((slider) => {
          const value = effectiveValues[slider.key];
          const defaultValue = effectiveDefaults[slider.key];
          return (
            <div key={slider.key} className="space-y-2">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-slate-800">{slider.label}</p>
                  <p className="text-[11px] text-slate-500">{slider.hint}</p>
                </div>
                <div className="min-w-[170px] shrink-0 text-right">
                  <p className="whitespace-nowrap text-sm font-bold text-teal-700">{slider.describe(value)}</p>
                  <p className="whitespace-nowrap text-[10px] text-slate-400">{describeDefault(slider.describe(defaultValue))}</p>
                </div>
              </div>
              <input
                type="range"
                min={slider.min}
                max={slider.max}
                step={slider.step}
                value={value}
                onChange={(event) =>
                  onChange({ [slider.key]: Number(event.target.value) } as Partial<VoiceControlValues>)
                }
                className="w-full accent-teal-600"
              />
              <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                <span>{slider.minLabel}</span>
                <span>{slider.maxLabel}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default VoiceControlsPanel;
