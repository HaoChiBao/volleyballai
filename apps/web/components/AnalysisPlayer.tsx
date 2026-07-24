"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  applyHomography,
  invertHomography,
} from "@volleyballai/court-math";
import type {
  BallTracksFile,
  Calibration,
  PlayersTracksFile,
  Point2,
} from "@volleyballai/types";
import type { Court3dFile } from "./Court3D";

const COLORS = ["#0b6e4f", "#1d4ed8", "#b45309", "#be123c", "#7c3aed", "#0f766e"];

function formatTime(s: number) {
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function nearestPlayerOverlays(tracks: PlayersTracksFile | null, t: number) {
  if (!tracks) return [];
  return tracks.players
    .map((p, idx) => {
      if (!p.frames.length) return null;
      const frame = p.frames.reduce((a, b) =>
        Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b,
      );
      if (Math.abs(frame.t - t) > 0.35) return null;
      return {
        track_id: p.track_id,
        bbox: frame.bbox,
        outline: frame.outline,
        court_xy: frame.court_xy,
        color: COLORS[idx % COLORS.length],
      };
    })
    .filter(Boolean) as {
    track_id: number;
    bbox: [number, number, number, number];
    outline?: [number, number][];
    court_xy?: [number, number];
    color: string;
  }[];
}

function hexToRgba(hex: string, alpha: number) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

/** Fallback body-ish polygon when SAM outline is missing (bbox-only tracks). */
function outlineFromBbox(
  bbox: [number, number, number, number],
  n = 24,
): [number, number][] {
  const [x, y, w, h] = bbox;
  const cx = x + w / 2;
  const cy = y + h / 2;
  const rx = w * 0.42;
  const ry = h * 0.48;
  const pts: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const a = (Math.PI * 2 * i) / n;
    const waist = 0.85 + 0.15 * Math.abs(Math.sin(a));
    const scale = Math.abs(Math.sin(a)) < 0.35 ? 0.75 : waist;
    pts.push([cx + Math.cos(a) * rx * scale, cy + Math.sin(a) * ry]);
  }
  return pts;
}

function nearestBall(ball: BallTracksFile | null, t: number) {
  if (!ball?.frames.length) return null;
  const frame = ball.frames.reduce((a, b) =>
    Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b,
  );
  if (Math.abs(frame.t - t) > 0.35) return null;
  return frame;
}

function courtLinesImage(Hinv: number[] | null): Point2[][] {
  if (!Hinv) return [];
  const map = (x: number, y: number) => applyHomography(Hinv, { x, y });
  const boundary = [
    map(0, 0),
    map(18, 0),
    map(18, 9),
    map(0, 9),
  ];
  const center = [map(9, 0), map(9, 9)];
  const attackA = [map(6, 0), map(6, 9)];
  const attackB = [map(12, 0), map(12, 9)];
  const net = [map(9, 0), map(9, 9)];
  // Height hints for net posts (image-space uplift)
  const netTop = [
    { x: net[0].x, y: net[0].y - 40 },
    { x: net[1].x, y: net[1].y - 40 },
  ];
  return [boundary, center, attackA, attackB, net, netTop];
}

export function AnalysisPlayer({
  mediaUrl,
  posterUrl,
  calibration,
  tracks,
  ball,
  court3d,
  onTime,
}: {
  mediaUrl: string;
  posterUrl?: string;
  calibration: Calibration | null;
  tracks: PlayersTracksFile | null;
  ball: BallTracksFile | null;
  court3d: Court3dFile | null;
  onTime?: (t: number) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(1);
  const [size, setSize] = useState({ w: 1280, h: 720 });
  const [showOutlines, setShowOutlines] = useState(true);
  const [showBoxes, setShowBoxes] = useState(false);
  const [showCourt3dOverlay, setShowCourt3dOverlay] = useState(true);
  const [showBall, setShowBall] = useState(true);

  const Hinv = useMemo(() => {
    if (!calibration?.H || calibration.H.length !== 9) return null;
    try {
      return invertHomography(calibration.H);
    } catch {
      return null;
    }
  }, [calibration]);

  const lines = useMemo(() => courtLinesImage(Hinv), [Hinv]);
  const playerOverlays = useMemo(
    () => nearestPlayerOverlays(tracks, t),
    [tracks, t],
  );
  const ballFrame = useMemo(() => nearestBall(ball, t), [ball, t]);
  const sample3d = useMemo(() => {
    if (!court3d?.samples?.length) return null;
    return court3d.samples.reduce((a, b) =>
      Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b,
    );
  }, [court3d, t]);

  useEffect(() => {
    onTime?.(t);
  }, [t, onTime]);

  function togglePlay() {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  }

  function seek(next: number) {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = Math.max(0, Math.min(el.duration || next, next));
    setT(el.currentTime);
  }

  function setPlaybackRate(next: number) {
    const el = videoRef.current;
    if (!el) return;
    el.playbackRate = next;
    setRate(next);
  }

  return (
    <div className="card stack analysis-player">
      <div className="row between">
        <h2>Analysis player</h2>
        <span className="meta-line">
          {tracks?.source ? `players:${tracks.source}` : "no tracks"}
          {ball?.source ? ` · ball:${ball.source}` : ""}
        </span>
      </div>

      <div className="overlay-toggles row">
        <button
          type="button"
          className={`toggle-chip${showOutlines ? " active" : ""}`}
          onClick={() => setShowOutlines((v) => !v)}
        >
          Body outlines {showOutlines ? "on" : "off"}
        </button>
        <button
          type="button"
          className={`toggle-chip${showBoxes ? " active" : ""}`}
          onClick={() => setShowBoxes((v) => !v)}
        >
          Boxes {showBoxes ? "on" : "off"}
        </button>
        <button
          type="button"
          className={`toggle-chip${showCourt3dOverlay ? " active" : ""}`}
          onClick={() => setShowCourt3dOverlay((v) => !v)}
          disabled={!Hinv}
          title={!Hinv ? "Calibrate court corners first" : undefined}
        >
          3D court overlay {showCourt3dOverlay && Hinv ? "on" : "off"}
        </button>
        <button
          type="button"
          className={`toggle-chip${showBall ? " active" : ""}`}
          onClick={() => setShowBall((v) => !v)}
        >
          Ball {showBall ? "on" : "off"}
        </button>
      </div>

      <div className="video-shell analysis-shell">
        <video
          ref={videoRef}
          src={mediaUrl}
          poster={posterUrl}
          playsInline
          onClick={togglePlay}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onTimeUpdate={(e) => setT(e.currentTarget.currentTime)}
          onSeeked={(e) => setT(e.currentTarget.currentTime)}
          onLoadedMetadata={(e) => {
            const el = e.currentTarget;
            setSize({
              w: el.videoWidth || 1280,
              h: el.videoHeight || 720,
            });
            setDuration(el.duration || 0);
          }}
        />

        <svg
          className="analysis-overlay"
          viewBox={`0 0 ${size.w} ${size.h}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {showCourt3dOverlay && lines.length > 0 ? (
            <g className="court-overlay">
              <polygon
                points={lines[0].map((p) => `${p.x},${p.y}`).join(" ")}
                fill="rgba(11,110,79,0.12)"
                stroke="#0b6e4f"
                strokeWidth={Math.max(2, size.w / 500)}
              />
              {lines.slice(1).map((seg, i) => (
                <polyline
                  key={i}
                  points={seg.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="none"
                  stroke={i >= 3 ? "#222" : "#f5f5f5"}
                  strokeWidth={Math.max(2, size.w / 550)}
                  opacity={0.9}
                />
              ))}
              {/* Projected 3D player feet for proportion check */}
              {sample3d?.players.map((p, i) => {
                if (!Hinv) return null;
                const img = applyHomography(Hinv, { x: p.x, y: p.y });
                return (
                  <circle
                    key={`p3d-${p.track_id}`}
                    cx={img.x}
                    cy={img.y}
                    r={Math.max(5, size.w / 180)}
                    fill={COLORS[i % COLORS.length]}
                    opacity={0.85}
                  />
                );
              })}
              {showBall && sample3d?.ball && Hinv
                ? (() => {
                    const b = sample3d.ball!;
                    const img = applyHomography(Hinv, { x: b.x, y: b.y });
                    const lift = b.z * (size.h * 0.035);
                    return (
                      <circle
                        cx={img.x}
                        cy={img.y - lift}
                        r={Math.max(6, size.w / 160)}
                        fill="#f59e0b"
                        stroke="#fff"
                        strokeWidth={2}
                      />
                    );
                  })()
                : null}
            </g>
          ) : null}

          {playerOverlays.map((p) => {
            const outline =
              p.outline && p.outline.length >= 3
                ? p.outline
                : outlineFromBbox(p.bbox);
            const labelX = outline[0]?.[0] ?? p.bbox[0];
            const labelY = outline[0]?.[1] ?? p.bbox[1];
            return (
              <g key={p.track_id}>
                {showOutlines ? (
                  <polygon
                    points={outline.map(([x, y]) => `${x},${y}`).join(" ")}
                    fill={hexToRgba(p.color, 0.28)}
                    stroke={p.color}
                    strokeWidth={Math.max(2.5, size.w / 380)}
                    strokeLinejoin="round"
                  />
                ) : null}
                {showBoxes ? (
                  <rect
                    x={p.bbox[0]}
                    y={p.bbox[1]}
                    width={p.bbox[2]}
                    height={p.bbox[3]}
                    fill="none"
                    stroke={p.color}
                    strokeWidth={Math.max(2, size.w / 400)}
                    strokeDasharray={showOutlines ? "6 4" : undefined}
                    opacity={0.85}
                  />
                ) : null}
                {(showOutlines || showBoxes) ? (
                  <text
                    x={labelX + 4}
                    y={labelY - 6}
                    fill={p.color}
                    fontSize={Math.max(14, size.w / 60)}
                    fontFamily="sans-serif"
                    fontWeight="700"
                  >
                    #{p.track_id}
                  </text>
                ) : null}
              </g>
            );
          })}

          {showBall && ballFrame?.xy ? (
            <circle
              cx={ballFrame.xy[0]}
              cy={ballFrame.xy[1]}
              r={ballFrame.r ?? 8}
              fill="none"
              stroke="#f59e0b"
              strokeWidth={Math.max(2, size.w / 400)}
            />
          ) : null}
        </svg>
      </div>

      <div className="player-controls">
        <div className="row">
          <button type="button" className="secondary control-btn" onClick={togglePlay}>
            {playing ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            className="secondary control-btn"
            onClick={() => seek(t - 1)}
          >
            −1s
          </button>
          <button
            type="button"
            className="secondary control-btn"
            onClick={() => seek(t + 1)}
          >
            +1s
          </button>
          <span className="meta-line time-readout">
            {formatTime(t)} / {formatTime(duration)}
          </span>
          <label className="rate-label meta-line">
            Speed
            <select
              value={rate}
              onChange={(e) => setPlaybackRate(Number(e.target.value))}
            >
              {[0.25, 0.5, 1, 1.5, 2].map((r) => (
                <option key={r} value={r}>
                  {r}×
                </option>
              ))}
            </select>
          </label>
        </div>
        <input
          className="scrubber"
          type="range"
          min={0}
          max={duration || 0}
          step={0.01}
          value={Math.min(t, duration || 0)}
          onChange={(e) => seek(Number(e.target.value))}
        />
      </div>

      <p className="hint">
        Body outlines are SAM mask contours (or mock silhouettes). Boxes are
        optional. Court overlay projects the FIVB court through calibration.
      </p>
    </div>
  );
}
