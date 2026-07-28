"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  applyHomography,
  invertHomography,
} from "@volleyballai/court-math";
import type {
  BallFrame,
  BallTracksFile,
  Calibration,
  CameraMotionFile,
  NetTracksFile,
  PlayersTracksFile,
  Point2,
  TrackFrame,
} from "@volleyballai/types";
import {
  formatRunDateTime,
  formatRunDuration,
  formatRunModels,
} from "@/lib/formatRun";
import { netFrameAtTime } from "@/lib/netTracks";
import {
  bracketFrames,
  lerp,
  lerpBbox,
  lerpOutline,
} from "@/lib/trackInterp";
import type { Court3dFile } from "./Court3D";

/* High-contrast B/W + mid grays so tracks stay readable on video */
const COLORS = ["#ffffff", "#d4d4d4", "#a3a3a3", "#737373", "#f5f5f5", "#e5e5e5"];

/** SAM samples ~5fps → allow lerp across ~1 sample gap (+slack). */
const PLAYER_MAX_GAP_S = 0.45;
/** Ball is ~30fps but may drop frames during occlusion. */
const BALL_MAX_GAP_S = 0.2;

function formatTime(s: number) {
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function interpolatePlayerOverlays(tracks: PlayersTracksFile | null, t: number) {
  if (!tracks) return [];
  return tracks.players
    .map((p, idx) => {
      if (!p.frames.length) return null;
      const br = bracketFrames(p.frames, t, PLAYER_MAX_GAP_S);
      if (!br) return null;

      let bbox: [number, number, number, number];
      let outline: [number, number][] | undefined;
      let court_xy: [number, number] | undefined;

      if (br.kind === "lerp") {
        const a = br.a as TrackFrame;
        const b = br.b as TrackFrame;
        bbox = lerpBbox(a.bbox, b.bbox, br.u);
        outline = lerpOutline(a.outline, b.outline, br.u);
        if (a.court_xy && b.court_xy) {
          court_xy = [
            lerp(a.court_xy[0], b.court_xy[0], br.u),
            lerp(a.court_xy[1], b.court_xy[1], br.u),
          ];
        } else {
          court_xy = a.court_xy ?? b.court_xy;
        }
      } else {
        const f = br.frame as TrackFrame;
        bbox = f.bbox;
        outline = f.outline;
        court_xy = f.court_xy;
      }

      return {
        track_id: p.track_id,
        bbox,
        outline,
        court_xy,
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

function interpolateBall(ball: BallTracksFile | null, t: number): BallFrame | null {
  if (!ball?.frames.length) return null;
  // Only interpolate detections that have image positions.
  const frames = ball.frames.filter((f) => f.xy != null) as Array<
    BallFrame & { xy: [number, number] }
  >;
  if (!frames.length) return null;
  const br = bracketFrames(frames, t, BALL_MAX_GAP_S);
  if (!br) return null;
  if (br.kind === "lerp") {
    const a = br.a;
    const b = br.b;
    const xy: [number, number] = [
      lerp(a.xy[0], b.xy[0], br.u),
      lerp(a.xy[1], b.xy[1], br.u),
    ];
    const r =
      a.r != null && b.r != null ? lerp(a.r, b.r, br.u) : (a.r ?? b.r);
    let court_xyz = a.court_xyz ?? b.court_xyz;
    if (a.court_xyz && b.court_xyz) {
      court_xyz = [
        lerp(a.court_xyz[0], b.court_xyz[0], br.u),
        lerp(a.court_xyz[1], b.court_xyz[1], br.u),
        lerp(a.court_xyz[2], b.court_xyz[2], br.u),
      ];
    }
    return { t, xy, r, court_xyz };
  }
  return br.frame;
}

function courtLinesImage(
  Hinv: number[] | null,
  length_m = 18,
  width_m = 9,
): Point2[][] {
  if (!Hinv) return [];
  const L = length_m;
  const W = width_m;
  const mid = L / 2;
  const attackA = L / 3;
  const attackB = (2 * L) / 3;
  const map = (x: number, y: number) => applyHomography(Hinv, { x, y });
  const boundary = [map(0, 0), map(L, 0), map(L, W), map(0, W)];
  const center = [map(mid, 0), map(mid, W)];
  const attackLineA = [map(attackA, 0), map(attackA, W)];
  const attackLineB = [map(attackB, 0), map(attackB, W)];
  const net = [map(mid, 0), map(mid, W)];
  // Height hints for net posts (image-space uplift)
  const netTop = [
    { x: net[0].x, y: net[0].y - 40 },
    { x: net[1].x, y: net[1].y - 40 },
  ];
  return [boundary, center, attackLineA, attackLineB, net, netTop];
}

export function AnalysisPlayer({
  mediaUrl,
  posterUrl,
  calibration,
  tracks,
  ball,
  court3d,
  cameraMotion,
  netTracks,
  onTime,
  compact,
}: {
  mediaUrl: string;
  posterUrl?: string;
  calibration: Calibration | null;
  tracks: PlayersTracksFile | null;
  ball: BallTracksFile | null;
  court3d: Court3dFile | null;
  /** Optional camera-motion ticks on the scrubber (test / pipeline artifact). */
  cameraMotion?: CameraMotionFile | null;
  /** Settle → net corners (+ optional ground projection overlay). */
  netTracks?: NetTracksFile | null;
  onTime?: (t: number) => void;
  /** Side-by-side pane: fill height, tighter video */
  compact?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const onTimeRef = useRef(onTime);
  onTimeRef.current = onTime;
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [duration, setDuration] = useState(0);
  const [rate, setRate] = useState(1);
  const [size, setSize] = useState({ w: 1280, h: 720 });
  const [showOutlines, setShowOutlines] = useState(true);
  const [showBoxes, setShowBoxes] = useState(false);
  const [showCourt3dOverlay, setShowCourt3dOverlay] = useState(false);
  const [showBall, setShowBall] = useState(true);
  const [showMotionTicks, setShowMotionTicks] = useState(true);
  const [showNet, setShowNet] = useState(true);

  /** Push clock to React state + parent without a t→useEffect cascade (that trips React's max-update-depth guard when overlays are expensive). */
  const publishTime = useCallback((next: number) => {
    setT((prev) => (Math.abs(prev - next) < 1e-4 ? prev : next));
    onTimeRef.current?.(next);
  }, []);

  const activeNetFrame = useMemo(
    () => netFrameAtTime(netTracks, t),
    [netTracks, t],
  );

  const Hinv = useMemo(() => {
    // Court overlay uses calibration H only — net-settle does not drive court drawing.
    if (!calibration?.H || calibration.H.length !== 9) return null;
    try {
      return invertHomography(calibration.H);
    } catch {
      return null;
    }
  }, [calibration]);

  const lines = useMemo(
    () =>
      courtLinesImage(
        Hinv,
        calibration?.court?.length_m ?? 18,
        calibration?.court?.width_m ?? 9,
      ),
    [Hinv, calibration?.court?.length_m, calibration?.court?.width_m],
  );
  const playerOverlays = useMemo(
    () => interpolatePlayerOverlays(tracks, t),
    [tracks, t],
  );
  const ballFrame = useMemo(() => interpolateBall(ball, t), [ball, t]);
  const sample3d = useMemo(() => {
    if (!court3d?.samples?.length) return null;
    return court3d.samples.reduce((a, b) =>
      Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b,
    );
  }, [court3d, t]);

  const motionSpan = Math.max(
    duration || 0,
    cameraMotion?.duration_s || 0,
    1e-6,
  );
  const motionSegments = cameraMotion?.segments ?? [];
  const settlePoints = useMemo(() => {
    if (cameraMotion?.settle_points?.length) {
      return cameraMotion.settle_points;
    }
    // Back-compat: derive from motion_end events
    return (cameraMotion?.events ?? [])
      .filter((e) => e.type === "motion_end")
      .map((e) => ({
        t: e.t,
        frame_index: e.frame_index,
        kind: "motion_settled" as const,
        use_for_net_detect: true,
      }));
  }, [cameraMotion]);
  const mergeGapS = cameraMotion?.thresholds?.merge_gap_s;
  const startsUnsettled = Boolean(cameraMotion?.settle_policy?.starts_unsettled);
  const prefixSettleT = cameraMotion?.settle_policy?.prefix_use_settle_t ?? null;
  const inMotion = useMemo(() => {
    return motionSegments.some((s) => t >= s.start_t && t <= s.end_t);
  }, [motionSegments, t]);
  const activeSettle = useMemo(() => {
    if (!settlePoints.length) return null;
    // Unsettled opening: hold the first settle pose until that settle time.
    if (startsUnsettled && prefixSettleT != null && t < prefixSettleT) {
      return settlePoints[0] ?? null;
    }
    let best: (typeof settlePoints)[number] | null = null;
    for (const sp of settlePoints) {
      if (sp.t <= t + 1e-6) best = sp;
      else break;
    }
    return best;
  }, [settlePoints, t, startsUnsettled, prefixSettleT]);

  // Sync overlays to the video clock every animation frame while playing
  // (timeupdate alone is ~4–10Hz and makes tracks look lagged).
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    let raf = 0;
    const tick = () => {
      publishTime(el.currentTime);
      if (!el.paused && !el.ended) {
        raf = requestAnimationFrame(tick);
      }
    };
    const onPlay = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
      setPlaying(true);
    };
    const onPause = () => {
      cancelAnimationFrame(raf);
      publishTime(el.currentTime);
      setPlaying(false);
    };
    const onSeekOrTimeUpdate = () => {
      // While playing, RAF owns the clock; timeupdate is only a fallback when paused.
      if (el.paused) publishTime(el.currentTime);
    };
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    el.addEventListener("ended", onPause);
    el.addEventListener("seeked", onSeekOrTimeUpdate);
    el.addEventListener("timeupdate", onSeekOrTimeUpdate);
    if (!el.paused) onPlay();
    return () => {
      cancelAnimationFrame(raf);
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
      el.removeEventListener("ended", onPause);
      el.removeEventListener("seeked", onSeekOrTimeUpdate);
      el.removeEventListener("timeupdate", onSeekOrTimeUpdate);
    };
  }, [mediaUrl, publishTime]);

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
    publishTime(el.currentTime);
  }

  function setPlaybackRate(next: number) {
    const el = videoRef.current;
    if (!el) return;
    el.playbackRate = next;
    setRate(next);
  }

  return (
    <div
      className={
        compact
          ? "stack analysis-player analysis-player-pane"
          : "card stack analysis-player"
      }
    >
      <div className="row between">
        <h2>{compact ? "Video" : "Analysis player"}</h2>
        <span className="meta-line">
          {tracks?.run?.started_at
            ? `run ${formatRunDateTime(tracks.run.started_at)}${
                tracks.run.finished_at
                  ? ` → ${formatRunDateTime(tracks.run.finished_at)} (${formatRunDuration(
                      tracks.run.started_at,
                      tracks.run.finished_at,
                      tracks.run.duration_s,
                    )})`
                  : ""
              } · ${formatRunModels(tracks.run)}`
            : tracks?.source
              ? `players:${tracks.source}${ball?.source ? ` · ball:${ball.source}` : ""}`
              : "no tracks"}
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
        {cameraMotion ? (
          <button
            type="button"
            className={`toggle-chip${showMotionTicks ? " active" : ""}`}
            onClick={() => setShowMotionTicks((v) => !v)}
            title={`${settlePoints.length} settle points · merge_gap=${mergeGapS ?? "—"}s · ${cameraMotion.method}`}
          >
            Cam settle {showMotionTicks ? "on" : "off"}
            {inMotion ? " · moving" : ""}
          </button>
        ) : null}
        {netTracks?.frames?.length ? (
          <button
            type="button"
            className={`toggle-chip${showNet ? " active" : ""}`}
            onClick={() => setShowNet((v) => !v)}
            title={`Net tracks @ ${netTracks.frames.length} samples`}
          >
            Net {showNet ? "on" : "off"}
          </button>
        ) : null}
      </div>

      <div className="video-shell analysis-shell">
        <div
          className="analysis-video-box"
          style={
            size.w > 0 && size.h > 0
              ? { aspectRatio: `${size.w} / ${size.h}` }
              : undefined
          }
        >
          <video
            ref={videoRef}
            src={mediaUrl}
            poster={posterUrl}
            playsInline
            onClick={togglePlay}
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
            preserveAspectRatio="none"
          >
          {showCourt3dOverlay && lines.length > 0 ? (
            <g className="court-overlay">
              <polygon
                points={lines[0].map((p) => `${p.x},${p.y}`).join(" ")}
                fill="rgba(255,255,255,0.1)"
                stroke="#ffffff"
                strokeWidth={Math.max(2, size.w / 500)}
              />
              {lines.slice(1).map((seg, i) => (
                <polyline
                  key={i}
                  points={seg.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="none"
                  stroke={i >= 3 ? "#0a0a0a" : "#ffffff"}
                  strokeWidth={Math.max(2, size.w / 550)}
                  opacity={0.95}
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
                        fill="#ffffff"
                        stroke="#0a0a0a"
                        strokeWidth={2}
                      />
                    );
                  })()
                : null}
            </g>
          ) : null}

          {showNet && activeNetFrame?.net ? (
            <g className="net-overlay">
              <polygon
                points={[
                  activeNetFrame.net.top_left,
                  activeNetFrame.net.top_right,
                  activeNetFrame.net.bottom_right,
                  activeNetFrame.net.bottom_left,
                ]
                  .map((p) => `${p.x},${p.y}`)
                  .join(" ")}
                fill="rgba(0, 200, 255, 0.18)"
                stroke="#00c8ff"
                strokeWidth={Math.max(2.5, size.w / 420)}
              />
              {(
                [
                  activeNetFrame.net.top_left,
                  activeNetFrame.net.top_right,
                  activeNetFrame.net.bottom_right,
                  activeNetFrame.net.bottom_left,
                ] as Point2[]
              ).map((p, i) => (
                <circle
                  key={`net-c-${i}`}
                  cx={p.x}
                  cy={p.y}
                  r={Math.max(4, size.w / 200)}
                  fill="#00c8ff"
                />
              ))}
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
              stroke="#ffffff"
              strokeWidth={Math.max(2.5, size.w / 400)}
            />
          ) : null}
        </svg>
        </div>
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
            {inMotion && showMotionTicks ? (
              <span className="motion-live-badge"> cam moving</span>
            ) : null}
            {!inMotion &&
            showMotionTicks &&
            startsUnsettled &&
            prefixSettleT != null &&
            t < prefixSettleT ? (
              <span className="motion-live-badge"> using next settle</span>
            ) : null}
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
        <div className="scrubber-wrap">
          {showMotionTicks && cameraMotion ? (
            <div className="scrubber-motion" aria-hidden>
              {startsUnsettled &&
              prefixSettleT != null &&
              prefixSettleT > 0 ? (
                <span
                  className="scrubber-motion-band scrubber-unsettled-prefix"
                  style={{
                    left: "0%",
                    width: `${(prefixSettleT / motionSpan) * 100}%`,
                  }}
                  title={`Unsettled prefix — use settle @ ${prefixSettleT.toFixed(1)}s`}
                />
              ) : null}
              {motionSegments.map((seg, i) => {
                const left = (seg.start_t / motionSpan) * 100;
                const width = Math.max(
                  0.15,
                  ((seg.end_t - seg.start_t) / motionSpan) * 100,
                );
                return (
                  <span
                    key={`seg-${i}`}
                    className="scrubber-motion-band"
                    style={{ left: `${left}%`, width: `${width}%` }}
                    title={`Moving ${seg.start_t.toFixed(1)}s–${seg.end_t.toFixed(1)}s`}
                  />
                );
              })}
              {settlePoints.map((sp, i) => {
                const left = (sp.t / motionSpan) * 100;
                const isActive =
                  activeSettle != null &&
                  Math.abs((activeSettle.t ?? -1) - sp.t) < 1e-3;
                return (
                  <button
                    key={`settle-${sp.t}-${i}`}
                    type="button"
                    className={`scrubber-motion-tick tick-settle${isActive ? " active" : ""}`}
                    style={{ left: `${left}%` }}
                    title={`Camera settled @ ${sp.t.toFixed(1)}s — set pose / net redetect`}
                    onClick={() => seek(sp.t)}
                  />
                );
              })}
            </div>
          ) : null}
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
        {showMotionTicks && cameraMotion ? (
          <div className="motion-legend meta-line">
            <span className="motion-legend-swatch band" /> Moving
            {startsUnsettled ? (
              <>
                <span className="motion-legend-swatch prefix" /> Unsettled→next
              </>
            ) : null}
            <span className="motion-legend-swatch end" /> Settle / set
            <span>
              {settlePoints.length} settles ·{" "}
              {cameraMotion.summary?.num_segments ?? motionSegments.length} segs
              {mergeGapS != null ? ` · merge ≤${mergeGapS}s` : ""} ·{" "}
              {cameraMotion.method}
            </span>
          </div>
        ) : null}
      </div>

      <p className="hint">
        Body outlines come from SAM masks.
        {cameraMotion
          ? " Settle ticks mark when the camera stops (net/pose refresh)."
          : ""}
        {netTracks?.frames?.length
          ? " Net overlay shows detected net corners only (no court fill yet)."
          : " Court overlay requires calibration."}
      </p>
    </div>
  );
}
