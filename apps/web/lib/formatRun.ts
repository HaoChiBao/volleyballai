import type { AnalysisRunInfo } from "@volleyballai/types";

/** Format a run timestamp for UI (local timezone, exact to the second). */
export function formatRunDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  // Fixed locale so SSR (Node) and the browser emit identical strings
  // (e.g. "AM" vs "a.m." hydration mismatches with locale `undefined`).
  return d.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
}

/** @deprecated use formatRunDateTime */
export function formatRunStartedAt(iso: string | null | undefined): string {
  return formatRunDateTime(iso);
}

/** Wall-clock duration between two ISO timestamps (or precomputed seconds). */
export function formatRunDuration(
  startedAt: string | null | undefined,
  finishedAt?: string | null,
  durationS?: number | null,
): string {
  if (durationS != null && Number.isFinite(durationS) && durationS >= 0) {
    return formatSeconds(durationS);
  }
  if (!startedAt || !finishedAt) return "—";
  const a = Date.parse(startedAt);
  const b = Date.parse(finishedAt);
  if (Number.isNaN(a) || Number.isNaN(b) || b < a) return "—";
  return formatSeconds((b - a) / 1000);
}

function formatSeconds(totalSec: number): string {
  const sec = Math.round(totalSec);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) return s === 0 ? `${m}m` : `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  if (rm === 0 && s === 0) return `${h}h`;
  if (s === 0) return `${h}h ${rm}m`;
  return `${h}h ${rm}m ${s}s`;
}

/** "Started … · Finished … · Duration …" (or in-progress). */
export function formatRunTiming(run: AnalysisRunInfo | null | undefined): string {
  if (!run?.started_at) return "—";
  const start = formatRunDateTime(run.started_at);
  if (!run.finished_at) {
    return `Started ${start} · in progress`;
  }
  const finish = formatRunDateTime(run.finished_at);
  const dur = formatRunDuration(run.started_at, run.finished_at, run.duration_s);
  return `Started ${start} · Finished ${finish} · ${dur}`;
}

export function formatRunModels(run: AnalysisRunInfo | null | undefined): string {
  if (!run?.models) return "—";
  const m = run.models;
  const players =
    m.players_fps != null
      ? `${m.players} @ ${m.players_fps}fps`
      : m.players;
  const ballParts = [m.ball];
  if (m.ball_infer_mode) ballParts.push(m.ball_infer_mode);
  if (m.ball_model_key) ballParts.push(m.ball_model_key);
  const court =
    m.court != null
      ? m.court_detections != null
        ? `${m.court} (${m.court_detections} hits)`
        : m.court
      : null;
  const parts = [
    `players: ${players}`,
    `ball: ${ballParts.join(" / ")}`,
  ];
  if (court) parts.push(`court: ${court}`);
  return parts.join(" · ");
}

export function runLabel(run: AnalysisRunInfo | null | undefined): string {
  if (!run?.started_at) return "";
  return `${formatRunTiming(run)} · ${formatRunModels(run)}`;
}
