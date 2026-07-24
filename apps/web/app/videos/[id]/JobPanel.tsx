"use client";

import { useCallback, useEffect, useState } from "react";
import type { Job } from "@volleyballai/types";

export function JobPanel({
  videoId,
  initialJobs,
  onJobSettled,
}: {
  videoId: string;
  initialJobs: Job[];
  onJobSettled?: () => void;
}) {
  const [jobs, setJobs] = useState<Job[]>(initialJobs);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [wasActive, setWasActive] = useState(false);

  const refresh = useCallback(async () => {
    const res = await fetch(`/api/jobs?video_id=${encodeURIComponent(videoId)}`, {
      cache: "no-store",
    });
    if (!res.ok) {
      setError(await res.text());
      return;
    }
    const data = (await res.json()) as { jobs: Job[] };
    setJobs(data.jobs);
    setError(null);
  }, [videoId]);

  useEffect(() => {
    const latest = jobs[0];
    const active =
      latest && (latest.status === "queued" || latest.status === "running");
    if (active) {
      setWasActive(true);
    } else if (
      wasActive &&
      latest &&
      (latest.status === "completed" ||
        latest.status === "failed" ||
        latest.status === "needs_calibration")
    ) {
      setWasActive(false);
      onJobSettled?.();
    }
    if (!active) return;
    const id = window.setInterval(() => {
      void refresh();
    }, 1000);
    return () => window.clearInterval(id);
  }, [jobs, refresh, wasActive, onJobSettled]);

  async function startJob() {
    setCreating(true);
    setError(null);
    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: videoId }),
      });
      if (!res.ok) throw new Error(await res.text());
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  }

  const latest = jobs[0];

  return (
    <div className="card stack">
      <div className="row between">
        <h2>Analysis job</h2>
        <button
          type="button"
          className="secondary"
          onClick={() => void refresh()}
        >
          Refresh
        </button>
      </div>

      {!latest ? (
        <p className="muted">No jobs yet for this video.</p>
      ) : (
        <div className="stack">
          <div className="row">
            <span className={`badge ${latest.status}`}>{latest.status}</span>
            <span className="meta-line">
              Stage <code>{latest.stage}</code>
            </span>
            <span className="meta-line">
              {(latest.progress * 100).toFixed(0)}%
            </span>
          </div>
          <div className="progress">
            <span
              style={{
                width: `${Math.min(100, Math.max(0, latest.progress * 100))}%`,
              }}
            />
          </div>
          {latest.error ? <div className="error">{latest.error}</div> : null}
          <p className="hint">
            Job <code>{latest.id}</code> · pipeline {latest.pipeline_version}
          </p>
        </div>
      )}

      {error ? <div className="error">{error}</div> : null}

      <div className="row">
        <button type="button" disabled={creating} onClick={() => void startJob()}>
          {creating ? "Queuing…" : "Queue new analysis"}
        </button>
      </div>

      {latest?.status === "needs_calibration" ? (
        <p className="hint">
          Tracking finished. Click the four court corners below (near left → near
          right → far right → far left) so players and ball map onto the 18×9m
          court.
        </p>
      ) : (
        <p className="hint">
          Worker required (<code>npm run worker</code>). Pipeline: normalize →
          Modal SAM 3.1 players → ball → 3D. Calibrate corners to place everyone
          on the court.
        </p>
      )}
    </div>
  );
}
