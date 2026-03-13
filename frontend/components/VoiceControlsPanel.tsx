import React from 'react';
import { VoiceControlValues } from '../types';

type VoiceControlsPanelProps = {
  values: VoiceControlValues | null;
  defaults: VoiceControlValues | null;
  status: 'idle' | 'loading' | 'ready' | 'error';
  error?: string | null;
  onChange: (patch: Partial<VoiceControlValues>) => void;
  onReset: () => void;
};

type SliderConfig = {
  key: keyof VoiceControlValues;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
};

const SLIDERS: SliderConfig[] = [
  {
    key: 'pitchShift',
    label: 'Pitch',
    hint: 'Shift the final voice higher or lower.',
    min: -4,
    max: 4,
    step: 0.1,
    format: (value) => `${value > 0 ? '+' : ''}${value.toFixed(1)} st`,
  },
  {
    key: 'f0Scale',
    label: 'Pitch Range',
    hint: 'Control how much pitch movement the voice keeps.',
    min: 0.75,
    max: 1.35,
    step: 0.01,
    format: (value) => `${value.toFixed(2)}x`,
  },
  {
    key: 'embeddingScale',
    label: 'Style Strength',
    hint: 'Blend more or less of the trained style into output.',
    min: 0.8,
    max: 2.2,
    step: 0.05,
    format: (value) => `${value.toFixed(2)}x`,
  },
  {
    key: 'diffusionSteps',
    label: 'Diffusion Steps',
    hint: 'More steps can sound cleaner, but will add latency.',
    min: 6,
    max: 20,
    step: 1,
    format: (value) => `${Math.round(value)} steps`,
  },
  {
    key: 'brightness',
    label: 'Brightness',
    hint: 'Softens or brightens the top end around the profile default.',
    min: -100,
    max: 100,
    step: 1,
    format: (value) => `${value > 0 ? '+' : ''}${Math.round(value)}`,
  },
];

const VoiceControlsPanel: React.FC<VoiceControlsPanelProps> = ({
  values,
  defaults,
  status,
  error,
  onChange,
  onReset,
}) => {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Voice Controls</p>
          <p className="mt-1 text-xs text-slate-500">
            Applies to the next response you generate. These controls do not change saved profile defaults.
          </p>
        </div>
        <button
          type="button"
          onClick={onReset}
          disabled={!defaults || status !== 'ready'}
          className="rounded-lg border border-slate-200 px-3 py-2 text-[10px] font-bold uppercase tracking-widest text-slate-600 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Reset
        </button>
      </div>

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

      {status !== 'error' && values && defaults && (
        <div className="space-y-4">
          {SLIDERS.map((slider) => {
            const value = values[slider.key];
            const defaultValue = defaults[slider.key];
            return (
              <div key={slider.key} className="space-y-2">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-sm font-semibold text-slate-800">{slider.label}</p>
                    <p className="text-[11px] text-slate-500">{slider.hint}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-bold text-teal-700">{slider.format(value)}</p>
                    <p className="text-[10px] text-slate-400">Default {slider.format(defaultValue)}</p>
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
                  disabled={status !== 'ready'}
                />
                <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-widest text-slate-400">
                  <span>{slider.format(slider.min)}</span>
                  <span>{slider.format(slider.max)}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default VoiceControlsPanel;
