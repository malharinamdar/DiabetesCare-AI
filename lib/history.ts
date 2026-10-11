import type { HistoryEntry } from "./types";

const KEY = "diabetescare.history.v1";

export function readHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function saveEntry(entry: HistoryEntry) {
  try {
    localStorage.setItem(KEY, JSON.stringify([entry, ...readHistory()].slice(0, 50)));
  } catch {
    // Storage blocked (private mode); history is a convenience only.
  }
}

export function clearHistory() {
  try {
    localStorage.removeItem(KEY);
  } catch {}
}
