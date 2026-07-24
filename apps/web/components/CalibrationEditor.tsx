"use client";

import { useRef, useState } from "react";
import type { Calibration, Point2 } from "@volleyballai/types";
import { DEFAULT_COURT_CORNERS } from "@volleyballai/court-math";

const LABELS = ["Near left", "Near right", "Far right", "Far left"];

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
  const [points, setPoints] = useState<Point2[]>(
    initial?.keyframes[0]?.image_points?.slice(0, 4) ?? [],
  );
  const [size, setSize] = useState({ w: 1280, h: 720 });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const nextLabel = LABELS[points.length] ?? null;

  function onClick(e: React.MouseEvent<HTMLVideoElement>) {
    if (points.length >= 4) return;
    const el = e.currentTarget;
    const rect = el.getBoundingClientRect();
    if (!el.videoWidth || !rect.width) return;
    const scaleX = el.videoWidth / rect.width;
    const scaleY = el.videoHeight / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    setPoints((prev) => [...prev, { x, y }]);
    setMessage(null);
  }

  async function save() {
    if (points.length < 4) {
      setError("Click all 4 court corners first.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const res = await fetch(`/api/videos/${videoId}/calibration`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_width: size.w,
          image_height: size.h,
          keyframes: [
            {
              t: videoRef.current?.currentTime ?? 0,
              image_points: points,
              court_points_m: DEFAULT_COURT_CORNERS,
            },
          ],
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = (await res.json()) as { calibration: Calibration };
      setMessage("Calibration saved. Tracks reprojected for 3D.");
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
          Pause on a clear frame, then click corners in order:{" "}
          <strong>near left → near right → far right → far left</strong>
          {nextLabel ? (
            <>
              . Next: <strong>{nextLabel}</strong>
            </>
          ) : (
            ". Ready to save."
          )}
        </p>
      </div>

      <div className="video-shell" style={{ position: "relative" }}>
        <video
          ref={videoRef}
          src={mediaUrl}
          controls
          onClick={onClick}
          onLoadedMetadata={(e) => {
            const el = e.currentTarget;
            setSize({
              w: el.videoWidth || 1280,
              h: el.videoHeight || 720,
            });
          }}
          style={{ cursor: points.length < 4 ? "crosshair" : "default" }}
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
          {points.map((p, i) => (
            <g key={i}>
              <circle cx={p.x} cy={p.y} r={8} fill="#fff" stroke="#0a0a0a" strokeWidth="3" />
              <text
                x={p.x + 14}
                y={p.y - 10}
                fill="#ffffff"
                stroke="#0a0a0a"
                strokeWidth="3"
                paintOrder="stroke"
                fontSize="28"
                fontFamily="monospace"
                fontWeight="700"
              >
                {i + 1}
              </text>
            </g>
          ))}
          {points.length === 4 ? (
            <polygon
              points={points.map((p) => `${p.x},${p.y}`).join(" ")}
              fill="rgba(255,255,255,0.12)"
              stroke="#ffffff"
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
            setPoints([]);
            setMessage(null);
          }}
        >
          Clear points
        </button>
        <button
          type="button"
          disabled={saving || points.length < 4}
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
