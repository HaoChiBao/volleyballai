"use client";

import { useCallback, useEffect, useState } from "react";
import type { Job } from "@volleyballai/types";
import {
  formatRunDateTime,
  formatRunDuration,
  formatRunModels,
  formatRunTiming,
} from "@/lib/formatRun";

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
  const prior = jobs.slice(1, 6);

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
          {latest.run?.started_at ? (
            <div className="stack" style={{ gap: "0.25rem" }}>
              <p className="meta-line" style={{ margin: 0 }}>
                Started{" "}
                <strong>{formatRunDateTime(latest.run.started_at)}</strong>
              </p>
              <p className="meta-line" style={{ margin: 0 }}>
                {latest.run.finished_at ? (
                  <>
                    Finished{" "}
                    <strong>
                      {formatRunDateTime(latest.run.finished_at)}
                    </strong>
                    {" · "}
                    Duration{" "}
                    <strong>
                      {formatRunDuration(
                        latest.run.started_at,
                        latest.run.finished_at,
                        latest.run.duration_s,
                      )}
                    </strong>
                  </>
                ) : (
                  <>Finished — · in progress</>
                )}
              </p>
              {latest.run.run_id ? (
                <p className="meta-line" style={{ margin: 0 }}>
                  Run folder <code>runs/{latest.run.run_id}</code>
                </p>
              ) : null}
              <p className="meta-line" style={{ margin: 0 }}>
                Models {formatRunModels(latest.run)}
              </p>
            </div>
          ) : null}
          <p className="hint">
            Job <code>{latest.id}</code> · pipeline {latest.pipeline_version}
          </p>
        </div>
      )}

      {prior.length > 0 ? (
        <div className="stack" style={{ gap: "0.35rem" }}>
          <p className="meta-line" style={{ margin: 0 }}>
            Previous runs
          </p>
          {prior.map((job) => (
            <p key={job.id} className="hint" style={{ margin: 0 }}>
              <span className={`badge ${job.status}`}>{job.status}</span>{" "}
              {job.run?.started_at
                ? formatRunTiming(job.run)
                : formatRunDateTime(job.created_at)}
              {job.run?.run_id ? (
                <>
                  {" "}
                  · <code>{job.run.run_id}</code>
                </>
              ) : null}
            </p>
          ))}
        </div>
      ) : null}

      {error ? <div className="error">{error}</div> : null}

      <div className="row">
        <button type="button" disabled={creating} onClick={() => void startJob()}>
          {creating ? "Queuing…" : "Queue new analysis"}
        </button>
      </div>

      {latest?.status === "needs_calibration" ? (
        <p className="hint">
          Tracks ready. Open the page to apply YOLO court keypoints for 3D, or
          draw lines below to override manually.
        </p>
      ) : (
        <p className="hint">
          Worker required (<code>npm run worker</code>). Each run saves under{" "}
          <code>runs/&lt;date_time&gt;/</code>. Court keypoints auto-drive the
          3D camera; manual lines override.
        </p>
      )}
    </div>
  );
}
