"use client";

import { useMemo, useRef, useState } from "react";
import type { Calibration, Point2 } from "@volleyballai/types";
import { DEFAULT_COURT } from "@volleyballai/types";
import {
  courtLinesForSize,
  type CourtLineId,
} from "@volleyballai/court-math";

type DrawnLine = {
  id: CourtLineId;
  /** Image endpoints in click order (A then B of the court line def) */
  image: [Point2, Point2];
};

function linesToCorrespondences(
  lines: DrawnLine[],
  length_m: number,
  width_m: number,
): {
  image_points: Point2[];
  court_points_m: Point2[];
} {
  const defs = courtLinesForSize(length_m, width_m);
  const image_points: Point2[] = [];
  const court_points_m: Point2[] = [];
  for (const line of lines) {
    const def = defs.find((d) => d.id === line.id);
    if (!def) continue;
    image_points.push(line.image[0], line.image[1]);
    court_points_m.push(def.a, def.b);
  }
  return { image_points, court_points_m };
}

function initialLines(cal: Calibration | null): DrawnLine[] {
  const kf = cal?.keyframes?.[0];
  if (!kf || kf.image_points.length < 4) return [];
  // Legacy 4-corner save → boundary lines
  if (kf.image_points.length === 4) {
    const [nl, nr, fr, fl] = kf.image_points;
    return [
      { id: "near", image: [nl, nr] },
      { id: "right", image: [nr, fr] },
      { id: "far", image: [fr, fl] },
      { id: "left", image: [fl, nl] },
    ];
  }
  return [];
}

function parsePositive(raw: string, fallback: number): number {
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return n;
}

export function CalibrationEditor({
  videoId,
  mediaUrl,
  initial,
  onSaved,
}: {
  videoId: string;
  mediaUrl: string;
  initial: Calibration | null;
  onSaved?: (cal: Calibration) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [lines, setLines] = useState<DrawnLine[]>(() => initialLines(initial));
  const [lengthM, setLengthM] = useState(
    initial?.court?.length_m ?? DEFAULT_COURT.length_m,
  );
  const [widthM, setWidthM] = useState(
    initial?.court?.width_m ?? DEFAULT_COURT.width_m,
  );
  const [activeId, setActiveId] = useState<CourtLineId>("near");
  const [pending, setPending] = useState<Point2 | null>(null);
  const [size, setSize] = useState({ w: 1280, h: 720 });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const courtLines = useMemo(
    () => courtLinesForSize(lengthM, widthM),
    [lengthM, widthM],
  );
  const drawnIds = useMemo(() => new Set(lines.map((l) => l.id)), [lines]);
  const correspondences = useMemo(
    () => linesToCorrespondences(lines, lengthM, widthM),
    [lines, lengthM, widthM],
  );
  const canSave = correspondences.image_points.length >= 4;

  const activeDef = courtLines.find((d) => d.id === activeId)!;

  function pointerToVideo(e: React.MouseEvent<HTMLVideoElement>): Point2 | null {
    const video = e.currentTarget;
    if (!video.videoWidth) return null;
    const rect = video.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const scaleX = video.videoWidth / rect.width;
    const scaleY = video.videoHeight / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  function onVideoClick(e: React.MouseEvent<HTMLVideoElement>) {
    // Ignore clicks on the native control bar (bottom ~40px of the element box).
    const rect = e.currentTarget.getBoundingClientRect();
    if (e.clientY > rect.bottom - 44) return;

    const p = pointerToVideo(e);
    if (!p) return;
    setMessage(null);
    setError(null);

    if (!pending) {
      setPending(p);
      return;
    }

    const image: [Point2, Point2] = [pending, p];
    const finishedId = activeId;
    setLines((prev) => {
      const rest = prev.filter((l) => l.id !== finishedId);
      const nextLines = [...rest, { id: finishedId, image }];
      const ids = new Set(nextLines.map((l) => l.id));
      const next = courtLines.find(
        (d) => d.id !== finishedId && !ids.has(d.id),
      );
      if (next) setActiveId(next.id);
      return nextLines;
    });
    setPending(null);
  }

  async function save() {
    if (!canSave) {
      setError("Draw at least 2 court lines (4 endpoints).");
      return;
    }
    if (lengthM <= 0 || widthM <= 0) {
      setError("Court length and width must be positive (meters).");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const { image_points, court_points_m } = linesToCorrespondences(
        lines,
        lengthM,
        widthM,
      );
      const res = await fetch(`/api/videos/${videoId}/calibration`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_width: size.w,
          image_height: size.h,
          court: { length_m: lengthM, width_m: widthM },
          keyframes: [
            {
              t: videoRef.current?.currentTime ?? 0,
              image_points,
              court_points_m,
            },
          ],
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = (await res.json()) as { calibration: Calibration };
      setMessage(
        `Calibration saved (${lengthM}×${widthM} m). Tracks reprojected for 3D.`,
      );
      onSaved?.(data.calibration);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="card stack">
      <div>
        <h2>Court calibration</h2>
        <p className="hint">
          Set the real court size (default FIVB indoor{" "}
          <strong>{DEFAULT_COURT.length_m}×{DEFAULT_COURT.width_m} m</strong>),
          then draw visible lines. Homography uses those meters so camera angle
          and player/ball positions stay accurate.
        </p>
      </div>

      <div className="row court-dims" style={{ flexWrap: "wrap", gap: "1rem" }}>
        <label className="stack" style={{ gap: "0.25rem" }}>
          <span className="meta-line">Length (m)</span>
          <input
            type="number"
            min={1}
            step={0.1}
            value={lengthM}
            onChange={(e) =>
              setLengthM(parsePositive(e.target.value, DEFAULT_COURT.length_m))
            }
            style={{ width: "7rem" }}
          />
        </label>
        <label className="stack" style={{ gap: "0.25rem" }}>
          <span className="meta-line">Width (m)</span>
          <input
            type="number"
            min={1}
            step={0.1}
            value={widthM}
            onChange={(e) =>
              setWidthM(parsePositive(e.target.value, DEFAULT_COURT.width_m))
            }
            style={{ width: "7rem" }}
          />
        </label>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setLengthM(DEFAULT_COURT.length_m);
            setWidthM(DEFAULT_COURT.width_m);
          }}
        >
          Reset to FIVB 18×9
        </button>
      </div>

      <div className="line-picker row" style={{ flexWrap: "wrap", gap: "0.4rem" }}>
        {courtLines.map((def) => {
          const done = drawnIds.has(def.id);
          const active = activeId === def.id;
          return (
            <button
              key={def.id}
              type="button"
              className={`toggle-chip${active ? " active" : ""}${done ? " done" : ""}`}
              onClick={() => {
                setActiveId(def.id);
                setPending(null);
              }}
            >
              {done ? "✓ " : ""}
              {def.label}
            </button>
          );
        })}
      </div>

      <p className="meta-line">
        Court <strong>{lengthM}×{widthM} m</strong> · Drawing:{" "}
        <strong>{activeDef.label}</strong>
        {pending
          ? " — click the other end"
          : drawnIds.has(activeId)
            ? " — click to redraw (two clicks)"
            : " — click first end, then second"}
        {" · "}
        {lines.length} line{lines.length === 1 ? "" : "s"}
      </p>

      <div className="video-shell calib-draw" style={{ position: "relative" }}>
        <video
          ref={videoRef}
          src={mediaUrl}
          controls
          onClick={onVideoClick}
          onLoadedMetadata={(e) => {
            const el = e.currentTarget;
            setSize({
              w: el.videoWidth || 1280,
              h: el.videoHeight || 720,
            });
          }}
          style={{ cursor: "crosshair" }}
        />
        <svg
          viewBox={`0 0 ${size.w} ${size.h}`}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
          }}
        >
          {lines.map((line) => {
            const [a, b] = line.image;
            return (
              <g key={line.id}>
                <line
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke="#ffffff"
                  strokeWidth="4"
                />
                <line
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke="#0a0a0a"
                  strokeWidth="1.5"
                />
                <circle cx={a.x} cy={a.y} r={6} fill="#fff" stroke="#0a0a0a" strokeWidth="2" />
                <circle cx={b.x} cy={b.y} r={6} fill="#fff" stroke="#0a0a0a" strokeWidth="2" />
              </g>
            );
          })}
          {pending ? (
            <circle
              cx={pending.x}
              cy={pending.y}
              r={7}
              fill="#fff"
              stroke="#0a0a0a"
              strokeWidth="3"
            />
          ) : null}
        </svg>
      </div>

      <div className="row">
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setLines((prev) => prev.filter((l) => l.id !== activeId));
            setPending(null);
            setMessage(null);
          }}
          disabled={!drawnIds.has(activeId) && !pending}
        >
          Clear current line
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setLines([]);
            setPending(null);
            setMessage(null);
          }}
        >
          Clear all
        </button>
        <button
          type="button"
          disabled={saving || !canSave}
          onClick={() => void save()}
        >
          {saving ? "Saving…" : "Save calibration"}
        </button>
      </div>
      {error ? <div className="error">{error}</div> : null}
      {message ? <p className="hint">{message}</p> : null}
    </div>
  );
}
