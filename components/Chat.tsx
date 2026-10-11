"use client";

import { useEffect, useRef, useState } from "react";
import { readHistory } from "@/lib/history";
import { streamPost } from "@/lib/stream";
import { BAND_LABEL, type HistoryEntry } from "@/lib/types";
import Markdown from "./Markdown";

type Msg = { role: "user" | "assistant"; content: string };

const SUGGESTIONS = [
  "What does an HbA1c of 6.1% actually mean?",
  "Is brown rice really better than white rice for blood sugar?",
  "How is prediabetes different from type 2 diabetes?",
  "What should a diabetes-friendly Indian breakfast look like?",
];

export default function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latest, setLatest] = useState<HistoryEntry | null>(null);
  const [share, setShare] = useState(true);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => setLatest(readHistory()[0] ?? null), []);
  useEffect(() => endRef.current?.scrollIntoView({ block: "end" }), [messages]);

  async function send(text: string) {
    const content = text.trim();
    if (!content || busy) return;
    const next: Msg[] = [...messages, { role: "user", content }];
    setMessages([...next, { role: "assistant", content: "" }]);
    setDraft("");
    setBusy(true);
    setError(null);
    abortRef.current = new AbortController();
    try {
      await streamPost(
        "/api/chat",
        { messages: next, profile: share && latest ? latest.input : undefined },
        (full) => setMessages([...next, { role: "assistant", content: full }]),
        abortRef.current.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setError(err instanceof Error ? err.message : "Something went wrong.");
        setMessages(next);
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="mx-auto flex min-h-[70dvh] max-w-3xl flex-col pt-6 sm:pt-10">
      {messages.length === 0 ? (
        <div className="pb-10">
          <p className="num text-[11px] uppercase tracking-[0.16em] text-muted">Assistant</p>
          <h1 className="mt-5 font-display text-[40px] leading-[1.05] tracking-[-0.02em] sm:text-[56px]">
            Ask about blood sugar, food, tests or <em>medication.</em>
          </h1>
          <div className="mt-10 grid gap-2 sm:grid-cols-2">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="rounded-xl border border-line bg-card px-4 py-3 text-left text-[14px] text-muted transition-colors hover:border-ink hover:text-ink"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex-1 space-y-6 pb-8">
          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="flex justify-end">
                <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-ink px-4 py-2.5 text-[15px] text-bg">
                  {m.content}
                </p>
              </div>
            ) : (
              <div key={i} className="max-w-[92%]">
                {m.content ? (
                  <Markdown text={m.content} streaming={busy && i === messages.length - 1} />
                ) : (
                  <p className="text-sm text-faint">Thinking…</p>
                )}
              </div>
            ),
          )}
          <div ref={endRef} />
        </div>
      )}

      <div className="sticky bottom-4 mt-auto">
        {error && <p className="mb-2 text-sm text-high">{error}</p>}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(draft);
          }}
          className="rounded-2xl border border-line bg-card p-2 shadow-[0_8px_30px_rgba(0,0,0,0.05)] focus-within:border-ink/40"
        >
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(draft);
              }
            }}
            rows={2}
            maxLength={4000}
            placeholder="Ask a question…"
            aria-label="Message"
            className="w-full resize-none bg-transparent px-3 py-2 text-[15px] outline-none placeholder:text-faint"
          />
          <div className="flex items-center justify-between gap-3 px-2 pb-1">
            {latest ? (
              <label className="flex cursor-pointer items-center gap-2 text-[12.5px] text-muted">
                <input type="checkbox" checked={share} onChange={(e) => setShare(e.target.checked)} className="accent-ink" />
                Use my latest assessment ({BAND_LABEL[latest.result.band].toLowerCase()} risk,{" "}
                {new Date(latest.at).toLocaleDateString("en-IN", { day: "numeric", month: "short" })})
              </label>
            ) : (
              <span className="text-[12.5px] text-faint">Run an assessment first to get answers about your numbers.</span>
            )}
            {busy ? (
              <button type="button" onClick={() => abortRef.current?.abort()} className="rounded-full border border-line px-4 py-1.5 text-[13px]">
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!draft.trim()}
                className="rounded-full bg-ink px-4 py-1.5 text-[13px] text-bg disabled:opacity-30"
              >
                Send
              </button>
            )}
          </div>
        </form>
      </div>
    </section>
  );
}
