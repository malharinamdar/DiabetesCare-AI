"use client";

import { useState } from "react";

export function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-3">
      <span className="text-[15px]">{label}</span>
      <div role="radiogroup" aria-label={label} className="flex rounded-full bg-wash p-0.5">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={o.value === value}
            onClick={() => onChange(o.value)}
            className={`rounded-full px-3.5 py-1 text-[13px] transition-all ${
              o.value === value ? "bg-card text-ink shadow-[0_1px_2px_rgba(0,0,0,0.08)]" : "text-muted hover:text-ink"
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Slider({
  id,
  label,
  note,
  unit,
  min,
  max,
  step,
  value,
  onChange,
}: {
  id: string;
  label: string;
  note?: React.ReactNode;
  unit: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  const fill = `${(((value - min) / (max - min)) * 100).toFixed(1)}%`;
  const decimals = step < 1 ? 1 : 0;
  // Free typing; clamp only on blur so partial values like "1" (on the way to "165") aren't snapped.
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <div className="py-3">
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-[15px]">
          {label}
          {note && <span className="ml-2 text-xs text-faint">{note}</span>}
        </label>
        <div className="flex items-baseline gap-1.5">
          <input
            type="number"
            inputMode="decimal"
            aria-label={`${label} value`}
            className="num w-16 rounded-md border border-transparent bg-transparent px-1.5 py-0.5 text-right text-[15px] outline-none transition-colors hover:border-line focus:border-ink focus:bg-card"
            min={min}
            max={max}
            step={step}
            value={draft ?? Number(value.toFixed(decimals))}
            onChange={(e) => {
              setDraft(e.target.value);
              const v = Number(e.target.value);
              if (e.target.value !== "" && v >= min && v <= max) onChange(v);
            }}
            onBlur={() => {
              const v = Number(draft);
              if (draft && Number.isFinite(v)) onChange(Math.min(max, Math.max(min, v)));
              setDraft(null);
            }}
          />
          <span className="num w-12 text-xs text-faint">{unit}</span>
        </div>
      </div>
      <input
        id={id}
        type="range"
        className="range mt-1.5"
        style={{ ["--fill" as string]: fill }}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
