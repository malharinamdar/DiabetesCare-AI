"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import cohort from "@/lib/model/cohort.json";
import { saveEntry } from "@/lib/history";
import { streamPost } from "@/lib/stream";
import { BAND_LABEL, bmiCategory, type AssessInput, type Assessment } from "@/lib/types";
import { Segmented, Slider } from "./Controls";
import Distribution from "./Distribution";
import Markdown from "./Markdown";

const START: AssessInput = {
  sex: "female",
  age: 45,
  height: 163,
  weight: 70,
  hba1c: 5.8,
  glucose: 140,
  hypertension: false,
  heartDisease: false,
  smoking: "never",
};

const BAND_STYLE = {
  low: "bg-low/10 text-low",
  elevated: "bg-mid/12 text-mid",
  high: "bg-high/10 text-high",
} as const;

const YES_NO = [
  { value: "no", label: "No" },
  { value: "yes", label: "Yes" },
] as const;

export default function AssessTool() {
  const [input, setInput] = useState<AssessInput>(START);
  const [result, setResult] = useState<Assessment | null>(null);
  const [assessed, setAssessed] = useState<AssessInput | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const set = <K extends keyof AssessInput>(k: K) => (v: AssessInput[K]) => setInput((s) => ({ ...s, [k]: v }));
  const bmi = input.weight / (input.height / 100) ** 2;
  const stale = result && assessed && JSON.stringify(assessed) !== JSON.stringify(input);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/assess", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(input),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Request failed");
      setResult(data);
      setAssessed(input);
      saveEntry({ id: crypto.randomUUID(), at: new Date().toISOString(), input, result: data });
      if (window.innerWidth < 1024) resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="grid gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-14">
      <form onSubmit={submit}>
        <Group title="About you">
          <Segmented
            label="Sex"
            options={[
              { value: "female", label: "Female" },
              { value: "male", label: "Male" },
            ]}
            value={input.sex}
            onChange={set("sex")}
          />
          <Slider id="age" label="Age" unit="years" min={1} max={100} step={1} value={input.age} onChange={set("age")} />
          <Slider id="height" label="Height" unit="cm" min={120} max={210} step={1} value={input.height} onChange={set("height")} />
          <Slider id="weight" label="Weight" unit="kg" min={30} max={180} step={1} value={input.weight} onChange={set("weight")} />
          <p className="pb-3 text-[13px] text-muted">
            BMI <span className="num text-ink">{bmi.toFixed(1)}</span> <span className="text-faint">·</span>{" "}
            {bmiCategory(bmi)}
          </p>
        </Group>

        <Group title="Blood work">
          <Slider
            id="hba1c"
            label="HbA1c"
            note="3-month average"
            unit="%"
            min={3.5}
            max={10}
            step={0.1}
            value={input.hba1c}
            onChange={set("hba1c")}
          />
          <Slider
            id="glucose"
            label="Blood glucose"
            unit="mg/dL"
            min={60}
            max={320}
            step={1}
            value={input.glucose}
            onChange={set("glucose")}
          />
        </Group>

        <Group title="History">
          <Segmented
            label="High blood pressure"
            options={[...YES_NO]}
            value={input.hypertension ? "yes" : "no"}
            onChange={(v) => set("hypertension")(v === "yes")}
          />
          <Segmented
            label="Heart disease"
            options={[...YES_NO]}
            value={input.heartDisease ? "yes" : "no"}
            onChange={(v) => set("heartDisease")(v === "yes")}
          />
          <Segmented
            label="Smoking"
            options={[
              { value: "never", label: "Never" },
              { value: "former", label: "Former" },
              { value: "current", label: "Current" },
            ]}
            value={input.smoking}
            onChange={set("smoking")}
          />
        </Group>

        <div className="mt-8 flex items-center gap-4">
          <button
            type="submit"
            disabled={loading}
            className="rounded-full bg-ink px-6 py-3 text-[15px] text-bg transition-transform active:scale-[0.98] disabled:opacity-60"
          >
            {loading ? "Estimating…" : result ? "Update estimate" : "Estimate risk"}
          </button>
          {stale && <p className="text-[13px] text-mid">Inputs changed since the last estimate.</p>}
        </div>
        {error && <p className="mt-4 text-sm text-high">{error}</p>}
      </form>

      <aside ref={resultRef} className="scroll-mt-6">
        <div className={`rounded-2xl border border-line bg-card p-6 transition-opacity sm:p-8 ${stale ? "opacity-60" : ""}`}>
          {result && assessed ? <Result result={result} input={assessed} /> : <Empty />}
        </div>
      </aside>
    </section>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <fieldset className="mt-8 first:mt-0">
      <legend className="num mb-1 w-full border-b border-line pb-2 text-[11px] uppercase tracking-[0.16em] text-muted">
        {title}
      </legend>
      <div className="divide-y divide-line/70">{children}</div>
    </fieldset>
  );
}

function Empty() {
  return (
    <div>
      <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">Estimated risk</p>
      <p className="mt-3 font-display text-[84px] leading-none text-line">—%</p>
      <p className="mt-4 max-w-sm text-[15px] leading-relaxed text-muted">
        Fill in your latest numbers and run the estimate. You&apos;ll see the risk, the answers that moved it most, and
        where you sit among {cohort.n.toLocaleString("en-IN")} people in the training data.
      </p>
    </div>
  );
}

function Result({ result, input }: { result: Assessment; input: AssessInput }) {
  const pct = result.probability * 100;
  return (
    <div className="rise">
      <div className="flex items-center justify-between">
        <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">Estimated risk</p>
        <span className={`rounded-full px-3 py-1 text-[13px] font-medium ${BAND_STYLE[result.band]}`}>
          {BAND_LABEL[result.band]}
        </span>
      </div>
      <p className="mt-3 font-display text-[84px] leading-none tracking-[-0.03em]">
        {pct < 1 ? "<1" : pct.toFixed(0)}
        <span className="text-[44px] text-muted">%</span>
      </p>
      <p className="mt-3 max-w-md text-[15px] leading-relaxed text-muted">
        Estimated probability of diabetes for someone with this profile.{" "}
        {result.band === "high"
          ? "This is above the screen-positive line. Confirmatory testing is worth booking."
          : result.band === "elevated"
            ? "Below the screen-positive line, but close enough to keep an eye on."
            : "Comfortably below the screen-positive line."}
      </p>
      <Scale probability={result.probability} threshold={result.threshold} />

      <Section title="What moved the estimate" note="Change in the forest's score from each answer, summed over the decision paths of all 100 trees.">
        <Contributions result={result} />
      </Section>

      <Section title="Where you sit" note="Shaded grey: people without diabetes. Red outline: people with diabetes. Line: you.">
        <div className="grid gap-x-8 gap-y-7 sm:grid-cols-2">
          <Distribution
            title="HbA1c"
            unit="%"
            {...cohort.features.HbA1c_level}
            value={input.hba1c}
            refs={[
              { at: 5.7, label: "5.7" },
              { at: 6.5, label: "6.5" },
            ]}
          />
          <Distribution
            title="Blood glucose"
            unit="mg/dL"
            {...cohort.features.blood_glucose_level}
            value={input.glucose}
            refs={[
              { at: 140, label: "140" },
              { at: 200, label: "200" },
            ]}
          />
          <Distribution
            title="BMI"
            unit=""
            {...cohort.features.bmi}
            value={Number(result.bmi.toFixed(1))}
            refs={[
              { at: 25, label: "25" },
              { at: 30, label: "30" },
            ]}
          />
          <Distribution title="Age" unit="yrs" {...cohort.features.age} value={input.age} />
        </div>
      </Section>

      <Guidance input={input} />
    </div>
  );
}

function Scale({ probability, threshold }: { probability: number; threshold: number }) {
  const at = (p: number) => `${Math.min(100, p * 100)}%`;
  return (
    <div className="mt-6">
      <div className="relative h-2 rounded-full bg-gradient-to-r from-low/25 via-mid/30 to-high/40">
        <div className="absolute -top-1.5 h-5 w-px bg-ink/40" style={{ left: at(threshold) }} />
        <div
          className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-card bg-ink shadow transition-[left] duration-700"
          style={{ left: at(probability) }}
        />
      </div>
      <div className="num relative mt-2 h-4 text-[10.5px] text-faint">
        <span className="absolute whitespace-nowrap" style={{ left: at(threshold) }}>
          ↑ screen-positive above {(threshold * 100).toFixed(0)}%
        </span>
        <span className="absolute right-0">100%</span>
      </div>
    </div>
  );
}

function Section({ title, note, children }: { title: string; note: string; children: React.ReactNode }) {
  return (
    <div className="mt-9 border-t border-line pt-6">
      <h3 className="font-display text-[22px]">{title}</h3>
      <p className="mt-1 text-[13px] text-faint">{note}</p>
      <div className="mt-5">{children}</div>
    </div>
  );
}

function Contributions({ result }: { result: Assessment }) {
  const max = Math.max(...result.contributions.map((c) => Math.abs(c.delta)), 0.01);
  return (
    <ul className="space-y-2">
      {result.contributions.map((c) => {
        const w = `${(Math.abs(c.delta) / max) * 50}%`;
        const up = c.delta >= 0;
        return (
          <li key={c.key} className="grid grid-cols-[minmax(0,168px)_1fr_48px] items-center gap-3 text-[13.5px]">
            <span className="truncate">
              {c.label} <span className="text-faint">{c.value}</span>
            </span>
            <span className="relative h-3">
              <span className="absolute inset-y-0 left-1/2 w-px bg-line" />
              <span
                className={`absolute inset-y-0.5 rounded-sm ${up ? "left-1/2 bg-high/75" : "right-1/2 bg-low/70"}`}
                style={{ width: w }}
              />
            </span>
            <span className={`num text-right text-xs ${Math.abs(c.delta) < 0.005 ? "text-faint" : up ? "text-high" : "text-low"}`}>
              {up ? "+" : "−"}
              {Math.abs(c.delta * 100).toFixed(1)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function Guidance({ input }: { input: AssessInput }) {
  const [text, setText] = useState("");
  const [state, setState] = useState<"idle" | "streaming" | "done" | "error">("idle");
  const [error, setError] = useState("");

  async function run() {
    setState("streaming");
    setText("");
    try {
      await streamPost("/api/guidance", input, setText);
      setState("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setState("error");
    }
  }

  return (
    <div className="mt-9 border-t border-line pt-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="font-display text-[22px]">Personal guidance</h3>
          <p className="mt-1 text-[13px] text-faint">Food, movement and tests, written for these numbers by GPT-4.1 mini.</p>
        </div>
        {state !== "streaming" && (
          <button
            type="button"
            onClick={run}
            className="rounded-full border border-ink px-4 py-2 text-[14px] transition-colors hover:bg-ink hover:text-bg"
          >
            {state === "idle" || state === "error" ? "Write my guidance" : "Rewrite"}
          </button>
        )}
      </div>
      {text && (
        <div className="mt-5 rounded-xl bg-bg/70 p-5">
          <Markdown text={text} streaming={state === "streaming"} />
        </div>
      )}
      {state === "streaming" && !text && <p className="mt-5 text-sm text-muted">Reading your numbers…</p>}
      {state === "error" && <p className="mt-4 text-sm text-high">{error}</p>}
      {state === "done" && (
        <p className="mt-4 text-[13px] text-muted">
          Follow-up questions?{" "}
          <Link href="/assistant" className="text-ink underline decoration-line underline-offset-4 hover:decoration-ink">
            Ask the assistant
          </Link>
          , it can see this assessment.
        </p>
      )}
    </div>
  );
}
