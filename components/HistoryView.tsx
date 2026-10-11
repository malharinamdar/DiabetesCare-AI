"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { clearHistory, readHistory } from "@/lib/history";
import { BAND_LABEL, type HistoryEntry } from "@/lib/types";

const DOT = { low: "bg-low", elevated: "bg-mid", high: "bg-high" } as const;

export default function HistoryView() {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);
  useEffect(() => setEntries(readHistory()), []);

  return (
    <section className="pt-8 sm:pt-14">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">History</p>
          <h1 className="mt-5 font-display text-[40px] leading-[1.05] tracking-[-0.02em] sm:text-[56px]">Past assessments</h1>
          <p className="mt-3 text-[15px] text-muted">Stored only in this browser. Nothing is sent to a server.</p>
        </div>
        {entries && entries.length > 0 && (
          <button
            onClick={() => {
              if (confirm("Delete all saved assessments from this browser?")) {
                clearHistory();
                setEntries([]);
              }
            }}
            className="rounded-full border border-line px-4 py-2 text-[13px] text-muted hover:border-high hover:text-high"
          >
            Clear history
          </button>
        )}
      </div>

      {entries && entries.length === 0 && (
        <p className="mt-12 text-[15px] text-muted">
          Nothing yet.{" "}
          <Link href="/" className="text-ink underline decoration-line underline-offset-4">
            Run your first assessment
          </Link>
          .
        </p>
      )}

      {entries && entries.length > 1 && <Trend entries={entries} />}

      {entries && entries.length > 0 && (
        <div className="mt-10 overflow-x-auto">
          <table className="w-full min-w-[640px] text-[14px]">
            <thead>
              <tr className="num border-b border-line text-left text-[11px] uppercase tracking-[0.12em] text-faint">
                <th className="py-3 font-normal">Date</th>
                <th className="py-3 font-normal">Risk</th>
                <th className="py-3 text-right font-normal">Probability</th>
                <th className="py-3 text-right font-normal">HbA1c</th>
                <th className="py-3 text-right font-normal">Glucose</th>
                <th className="py-3 text-right font-normal">BMI</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {entries.map((e) => (
                <tr key={e.id}>
                  <td className="py-3">
                    {new Date(e.at).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}
                  </td>
                  <td className="py-3">
                    <span className={`mr-2 inline-block size-2 rounded-full ${DOT[e.result.band]}`} />
                    {BAND_LABEL[e.result.band]}
                  </td>
                  <td className="num py-3 text-right">{(e.result.probability * 100).toFixed(0)}%</td>
                  <td className="num py-3 text-right">{e.input.hba1c.toFixed(1)}%</td>
                  <td className="num py-3 text-right">{Math.round(e.input.glucose)}</td>
                  <td className="num py-3 text-right">{e.result.bmi.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Trend({ entries }: { entries: HistoryEntry[] }) {
  const pts = [...entries].reverse().slice(-20);
  const W = 600;
  const H = 120;
  const max = Math.max(0.25, ...pts.map((p) => p.result.probability));
  const x = (i: number) => (pts.length === 1 ? W / 2 : (i / (pts.length - 1)) * W);
  const y = (p: number) => H - (p / max) * (H - 10);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${x(i)} ${y(p.result.probability)}`).join(" ");
  return (
    <div className="mt-10 rounded-2xl border border-line bg-card p-6">
      <p className="text-[13px] text-muted">Estimated probability over your last {pts.length} assessments</p>
      <svg viewBox={`0 -6 ${W} ${H + 12}`} className="mt-4 w-full overflow-visible" role="img" aria-label="Risk trend">
        <line x1="0" x2={W} y1={H} y2={H} stroke="var(--color-line)" />
        <path d={d} fill="none" stroke="var(--color-ink)" strokeWidth="1.5" />
        {pts.map((p, i) => (
          <circle key={p.id} cx={x(i)} cy={y(p.result.probability)} r="3.5" fill="var(--color-card)" stroke="var(--color-ink)" strokeWidth="1.5" />
        ))}
      </svg>
    </div>
  );
}
