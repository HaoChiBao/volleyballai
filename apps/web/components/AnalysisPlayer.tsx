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
  buildParabolicTrail,
  stabilizeBallAtTime,
  type BallSample,
} from "@/lib/ballPhysics";
import {
  bracketFrames,
  lerp,
  lerpBbox,
  lerpOutline,
  smoothOutlineToward,
} from "@/lib/trackInterp";
import {
  playerTrackScale,
  scaleBbox,
  scaleOutline,
} from "@/lib/trackScale";
import { PLAYER_COLORS, playerColorAt } from "@/lib/playerStyle";
import type { Court3dFile } from "./Court3D";

/** Distinct track colors for overlay + playlist (stable by player index). */
const COLORS = PLAYER_COLORS;

/** SAM samples ~5fps → allow lerp across a couple of sample gaps. */
const PLAYER_MAX_GAP_S = 1.25;
/**
 * Ball detections can drop during occlusion. Interpolate / hold across gaps
 * so the overlay dot does not blink out every miss (was 0.25s — too tight).
 */
const BALL_MAX_GAP_S = 1.5;
/** Keep showing the last smoothed ball if samples go silent briefly. */
const BALL_HOLD_S = 1.5;
/** Default / clamp range for the on-screen ball trail (seconds of history). */
const BALL_TRAIL_MIN_S = 0.15;
const BALL_TRAIL_MAX_S = 2.5;
const BALL_TRAIL_DEFAULT_S = 0.75;
/** Dense trail resampling along the fitted parabola. */
const BALL_TRAIL_SAMPLE_HZ = 60;

/** Overlay colors for the two ball trackers. */
const BALL_VBALLNET_STROKE = "rgba(255, 214, 10, 0.8)";
const BALL_VBALLNET_TRAIL = "rgba(255, 214, 10, 0.28)";
const BALL_VBALLNET_FILL = "rgba(255, 214, 10, 0.55)";
const BALL_YOLO_STROKE = "rgba(0, 200, 255, 0.85)";
const BALL_YOLO_TRAIL = "rgba(0, 200, 255, 0.3)";
const BALL_YOLO_FILL = "rgba(0, 200, 255, 0.55)";
const BALL_WASB_STROKE = "rgba(255, 90, 160, 0.9)";
const BALL_WASB_TRAIL = "rgba(255, 90, 160, 0.3)";
const BALL_WASB_FILL = "rgba(255, 90, 160, 0.55)";

/** FIVB ball diameter — used to turn camera distance into trail stroke px. */
const BALL_DIAMETER_M = 0.21;
/** EMA toward new outline each paint (~smoother than raw sample morph). */
const OUTLINE_SMOOTH_ALPHA = 0.38;
const BALL_SMOOTH_ALPHA = 0.45;

function formatTime(s: number) {
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export type PlayerOverlayState = {
  track_id: number;
  /** Display label #1, #2, … (not jersey / not raw tracker id). */
  label: number;
  bbox: [number, number, number, number];
  outline?: [number, number][];
  court_xy?: [number, number];
  color: string;
};

export type PlayerPlaylistEntry = {
  track_id: number;
  label: number;
  color: string;
  visible: boolean;
};

function playerColor(idx: number) {
  return playerColorAt(idx);
}

function interpolatePlayerOverlays(
  tracks: PlayersTracksFile | null,
  t: number,
  videoW: number,
  videoH: number,
): PlayerOverlayState[] {
  if (!tracks) return [];
  const scale = playerTrackScale(tracks, videoW, videoH);
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
        bbox = scaleBbox(lerpBbox(a.bbox, b.bbox, br.u), scale);
        outline = scaleOutline(lerpOutline(a.outline, b.outline, br.u), scale);
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
        bbox = scaleBbox(f.bbox, scale);
        outline = scaleOutline(f.outline, scale);
        court_xy = f.court_xy;
      }

      return {
        track_id: p.track_id,
        label: idx + 1,
        bbox,
        outline,
        court_xy,
        color: playerColor(idx),
      };
    })
    .filter(Boolean) as PlayerOverlayState[];
}

function buildPlayerPlaylist(
  tracks: PlayersTracksFile | null,
  visibleIds: Set<number>,
): PlayerPlaylistEntry[] {
  if (!tracks) return [];
  return tracks.players.map((p, idx) => ({
    track_id: p.track_id,
    label: idx + 1,
    color: playerColor(idx),
    visible: visibleIds.has(p.track_id),
  }));
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

type BallTrailPoint = {
  xy: [number, number];
  t: number;
  r?: number;
  court_xyz?: [number, number, number];
};

function ballDetections(ball: BallTracksFile | null): BallSample[] {
  if (!ball?.frames.length) return [];
  const out: BallSample[] = [];
  for (const f of ball.frames) {
    if (!f.xy) continue;
    out.push({
      t: f.t,
      xy: [f.xy[0], f.xy[1]],
      r: f.r,
      court_xyz: f.court_xyz,
    });
  }
  return out;
}

/**
 * Parabolic flight trail: split on teleports, fit x(t)/y(t) quadratics,
 * resample densely. Never draws a chord across a detector jump.
 */
function ballTrailSegments(
  ball: BallTracksFile | null,
  t: number,
  lengthS: number,
  tipXy?: [number, number] | null,
): BallTrailPoint[][] {
  const detections = ballDetections(ball);
  if (!detections.length || lengthS <= 0) return [];
  const trails = buildParabolicTrail(detections, t, lengthS, {
    sampleHz: BALL_TRAIL_SAMPLE_HZ,
  }).map((seg) =>
    seg.map((p) => ({
      xy: p.xy,
      t: p.t,
      r: p.r,
      court_xyz: p.court_xyz,
    })),
  );
  // Staple playhead tip onto the newest segment so the arc meets the marker.
  if (tipXy && trails.length) {
    const last = trails[trails.length - 1];
    const end = last[last.length - 1];
    if (
      !end ||
      Math.hypot(end.xy[0] - tipXy[0], end.xy[1] - tipXy[1]) > 0.5 ||
      Math.abs(end.t - t) > 1e-3
    ) {
      last.push({ xy: tipXy, t });
    }
  }
  return trails;
}

/** Catmull-Rom → cubic Bezier SVG path (smooth arcs through samples). */
function trailPointsToSmoothPath(pts: BallTrailPoint[]): string {
  if (pts.length < 2) return "";
  if (pts.length === 2) {
    return `M ${pts[0].xy[0]} ${pts[0].xy[1]} L ${pts[1].xy[0]} ${pts[1].xy[1]}`;
  }
  let d = `M ${pts[0].xy[0]} ${pts[0].xy[1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)].xy;
    const p1 = pts[i].xy;
    const p2 = pts[i + 1].xy;
    const p3 = pts[Math.min(pts.length - 1, i + 2)].xy;
    const c1x = p1[0] + (p2[0] - p0[0]) / 6;
    const c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6;
    const c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C ${c1x} ${c1y}, ${c2x} ${c2y}, ${p2[0]} ${p2[1]}`;
  }
  return d;
}

/** Euclidean camera→ball distance in court meters, when both are known. */
function ballCameraDistanceM(
  courtXyz: [number, number, number] | undefined,
  camPos: [number, number, number] | undefined,
): number | null {
  if (!courtXyz || !camPos) return null;
  const d = Math.hypot(
    courtXyz[0] - camPos[0],
    courtXyz[1] - camPos[1],
    courtXyz[2] - camPos[2],
  );
  return d > 0.25 ? d : null;
}

/**
 * Perspective trail thickness from tip distance (stable — not per-noisy-sample).
 */
function trailStrokeWidthPx(
  tip: BallTrailPoint | undefined,
  cam: Calibration["camera"] | null | undefined,
  imageW: number,
): number {
  if (tip) {
    const dist = ballCameraDistanceM(tip.court_xyz, cam?.position);
    if (dist != null && cam?.fx) {
      const apparentDiam = (cam.fx * BALL_DIAMETER_M) / dist;
      return Math.max(1.5, apparentDiam * 0.4);
    }
    if (tip.r != null && tip.r > 1) {
      return Math.max(1.5, tip.r * 0.85);
    }
  }
  return Math.max(2, imageW / 520);
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
  ballYolo = null,
  ballWasb = null,
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
  /** SetOptics YOLO comparison tracks (optional). */
  ballYolo?: BallTracksFile | null;
  /** WASB HRNet raw comparison tracks (optional). */
  ballWasb?: BallTracksFile | null;
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
  // Prefer calibration / net frame size so overlays aren't stuck at the
  // 1280×720 default when cached video skips the React onLoadedMetadata handler.
  const [size, setSize] = useState(() => {
    const cam = calibration?.camera;
    if (cam?.image_width && cam?.image_height) {
      return { w: cam.image_width, h: cam.image_height };
    }
    return { w: 1280, h: 720 };
  });
  const [showOutlines, setShowOutlines] = useState(true);
  const [showBoxes, setShowBoxes] = useState(false);
  const [showCourt3dOverlay, setShowCourt3dOverlay] = useState(false);
  const [showBallVballnet, setShowBallVballnet] = useState(true);
  const [showBallYolo, setShowBallYolo] = useState(true);
  const [showBallWasb, setShowBallWasb] = useState(true);
  const [showBallTrail, setShowBallTrail] = useState(true);
  const [trailLengthS, setTrailLengthS] = useState(BALL_TRAIL_DEFAULT_S);
  const [showMotionTicks, setShowMotionTicks] = useState(true);
  const [showNet, setShowNet] = useState(true);
  /** Temporally smoothed silhouettes keyed by track_id. */
  const smoothOutlinesRef = useRef<Map<number, [number, number][]>>(new Map());
  const smoothBallRef = useRef<{ xy: [number, number]; r: number } | null>(null);
  const smoothBallYoloRef = useRef<{ xy: [number, number]; r: number } | null>(
    null,
  );
  const smoothBallWasbRef = useRef<{ xy: [number, number]; r: number } | null>(
    null,
  );
  const lastBallGoodTRef = useRef<number | null>(null);
  const lastBallYoloGoodTRef = useRef<number | null>(null);
  const lastBallWasbGoodTRef = useRef<number | null>(null);
  const lastSmoothTRef = useRef<number | null>(null);

  /** Push clock to React state + parent without a t→useEffect cascade (that trips React's max-update-depth guard when overlays are expensive). */
  const publishTime = useCallback((next: number) => {
    setT((prev) => (Math.abs(prev - next) < 1e-4 ? prev : next));
    onTimeRef.current?.(next);
  }, []);

  // Upgrade placeholder viewBox when calibration arrives before video metadata.
  useEffect(() => {
    const cam = calibration?.camera;
    if (!cam?.image_width || !cam?.image_height) return;
    setSize((prev) =>
      prev.w === 1280 && prev.h === 720
        ? { w: cam.image_width, h: cam.image_height }
        : prev,
    );
  }, [calibration?.camera?.image_width, calibration?.camera?.image_height]);

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
  const playerOverlaysRaw = useMemo(
    () => interpolatePlayerOverlays(tracks, t, size.w, size.h),
    [tracks, t, size.w, size.h],
  );
  const ballFrameRaw = useMemo(() => interpolateBall(ball, t), [ball, t]);
  const ballYoloFrameRaw = useMemo(
    () => interpolateBall(ballYolo ?? null, t),
    [ballYolo, t],
  );
  const ballWasbFrameRaw = useMemo(
    () => interpolateBall(ballWasb ?? null, t),
    [ballWasb, t],
  );

  // Temporal EMA so silhouettes / ball ease between SAM samples instead of stepping.
  // Advance at most once per distinct `t` (avoids Strict Mode double-apply).
  // While paused (scrubbing), snap fully so outlines don't lag halfway.
  const smoothStep = lastSmoothTRef.current !== t;
  if (smoothStep) lastSmoothTRef.current = t;
  const outlineAlpha = playing ? OUTLINE_SMOOTH_ALPHA : 1;
  const ballAlpha = playing ? BALL_SMOOTH_ALPHA : 1;

  const playerOverlays = (() => {
    const seen = new Set<number>();
    const next = playerOverlaysRaw.map((p) => {
      seen.add(p.track_id);
      const target =
        p.outline && p.outline.length >= 3 ? p.outline : outlineFromBbox(p.bbox);
      const prev = smoothOutlinesRef.current.get(p.track_id);
      const outline = smoothStep
        ? smoothOutlineToward(prev, target, outlineAlpha)
        : prev && prev.length === target.length
          ? prev
          : target;
      if (smoothStep) smoothOutlinesRef.current.set(p.track_id, outline);
      return { ...p, outline };
    });
    if (smoothStep) {
      for (const id of [...smoothOutlinesRef.current.keys()]) {
        if (!seen.has(id)) smoothOutlinesRef.current.delete(id);
      }
    }
    return next;
  })();

  const ballPhysics = useMemo(() => {
    const detections = ballDetections(ball);
    if (!detections.length) return null;
    return stabilizeBallAtTime(detections, t);
  }, [ball, t]);

  const ballYoloPhysics = useMemo(() => {
    const detections = ballDetections(ballYolo ?? null);
    if (!detections.length) return null;
    return stabilizeBallAtTime(detections, t);
  }, [ballYolo, t]);

  const ballWasbPhysics = useMemo(() => {
    const detections = ballDetections(ballWasb ?? null);
    if (!detections.length) return null;
    return stabilizeBallAtTime(detections, t);
  }, [ballWasb, t]);

  const ballFrame = (() => {
    // Prefer parabolic prediction when the detector teleported or went silent.
    const rawXy = ballFrameRaw?.xy;
    const stabilized = ballPhysics;
    const preferPhysics = Boolean(
      stabilized?.xy && (stabilized.teleported || !rawXy),
    );
    const sourceXy: [number, number] | null = preferPhysics
      ? stabilized!.xy
      : rawXy
        ? [rawXy[0], rawXy[1]]
        : stabilized?.xy ?? null;

    if (!sourceXy) {
      const held = smoothBallRef.current;
      const lastGood = lastBallGoodTRef.current;
      if (
        held &&
        lastGood != null &&
        t - lastGood <= BALL_HOLD_S &&
        lastGood - t <= BALL_HOLD_S
      ) {
        return { t, xy: held.xy, r: held.r };
      }
      if (smoothStep) {
        smoothBallRef.current = null;
        lastBallGoodTRef.current = null;
      }
      return null;
    }
    const targetR = ballFrameRaw?.r ?? 8;
    if (!smoothStep && smoothBallRef.current) {
      return {
        ...(ballFrameRaw ?? { t }),
        xy: smoothBallRef.current.xy,
        r: smoothBallRef.current.r,
      };
    }
    const prev = smoothBallRef.current;
    // Hard snap on teleport correction; otherwise EMA.
    if (!prev || preferPhysics || ballAlpha >= 1) {
      smoothBallRef.current = { xy: sourceXy, r: targetR };
    } else {
      const jump = Math.hypot(
        prev.xy[0] - sourceXy[0],
        prev.xy[1] - sourceXy[1],
      );
      if (jump > 80) {
        smoothBallRef.current = { xy: sourceXy, r: targetR };
      } else {
        smoothBallRef.current = {
          xy: [
            lerp(prev.xy[0], sourceXy[0], ballAlpha),
            lerp(prev.xy[1], sourceXy[1], ballAlpha),
          ],
          r: lerp(prev.r, targetR, ballAlpha),
        };
      }
    }
    lastBallGoodTRef.current = t;
    return {
      ...(ballFrameRaw ?? { t }),
      xy: smoothBallRef.current.xy,
      r: smoothBallRef.current.r,
    };
  })();

  const ballYoloFrame = (() => {
    const rawXy = ballYoloFrameRaw?.xy;
    const stabilized = ballYoloPhysics;
    const preferPhysics = Boolean(
      stabilized?.xy && (stabilized.teleported || !rawXy),
    );
    const sourceXy: [number, number] | null = preferPhysics
      ? stabilized!.xy
      : rawXy
        ? [rawXy[0], rawXy[1]]
        : stabilized?.xy ?? null;

    if (!sourceXy) {
      const held = smoothBallYoloRef.current;
      const lastGood = lastBallYoloGoodTRef.current;
      if (
        held &&
        lastGood != null &&
        t - lastGood <= BALL_HOLD_S &&
        lastGood - t <= BALL_HOLD_S
      ) {
        return { t, xy: held.xy, r: held.r };
      }
      if (smoothStep) {
        smoothBallYoloRef.current = null;
        lastBallYoloGoodTRef.current = null;
      }
      return null;
    }
    const targetR = ballYoloFrameRaw?.r ?? 8;
    if (!smoothStep && smoothBallYoloRef.current) {
      return {
        ...(ballYoloFrameRaw ?? { t }),
        xy: smoothBallYoloRef.current.xy,
        r: smoothBallYoloRef.current.r,
      };
    }
    const prev = smoothBallYoloRef.current;
    if (!prev || preferPhysics || ballAlpha >= 1) {
      smoothBallYoloRef.current = { xy: sourceXy, r: targetR };
    } else {
      const jump = Math.hypot(
        prev.xy[0] - sourceXy[0],
        prev.xy[1] - sourceXy[1],
      );
      if (jump > 80) {
        smoothBallYoloRef.current = { xy: sourceXy, r: targetR };
      } else {
        smoothBallYoloRef.current = {
          xy: [
            lerp(prev.xy[0], sourceXy[0], ballAlpha),
            lerp(prev.xy[1], sourceXy[1], ballAlpha),
          ],
          r: lerp(prev.r, targetR, ballAlpha),
        };
      }
    }
    lastBallYoloGoodTRef.current = t;
    return {
      ...(ballYoloFrameRaw ?? { t }),
      xy: smoothBallYoloRef.current.xy,
      r: smoothBallYoloRef.current.r,
    };
  })();

  const ballWasbFrame = (() => {
    const rawXy = ballWasbFrameRaw?.xy;
    const stabilized = ballWasbPhysics;
    const preferPhysics = Boolean(
      stabilized?.xy && (stabilized.teleported || !rawXy),
    );
    const sourceXy: [number, number] | null = preferPhysics
      ? stabilized!.xy
      : rawXy
        ? [rawXy[0], rawXy[1]]
        : stabilized?.xy ?? null;

    if (!sourceXy) {
      const held = smoothBallWasbRef.current;
      const lastGood = lastBallWasbGoodTRef.current;
      if (
        held &&
        lastGood != null &&
        t - lastGood <= BALL_HOLD_S &&
        lastGood - t <= BALL_HOLD_S
      ) {
        return { t, xy: held.xy, r: held.r };
      }
      if (smoothStep) {
        smoothBallWasbRef.current = null;
        lastBallWasbGoodTRef.current = null;
      }
      return null;
    }
    const targetR = ballWasbFrameRaw?.r ?? 8;
    if (!smoothStep && smoothBallWasbRef.current) {
      return {
        ...(ballWasbFrameRaw ?? { t }),
        xy: smoothBallWasbRef.current.xy,
        r: smoothBallWasbRef.current.r,
      };
    }
    const prev = smoothBallWasbRef.current;
    if (!prev || preferPhysics || ballAlpha >= 1) {
      smoothBallWasbRef.current = { xy: sourceXy, r: targetR };
    } else {
      const jump = Math.hypot(
        prev.xy[0] - sourceXy[0],
        prev.xy[1] - sourceXy[1],
      );
      if (jump > 80) {
        smoothBallWasbRef.current = { xy: sourceXy, r: targetR };
      } else {
        smoothBallWasbRef.current = {
          xy: [
            lerp(prev.xy[0], sourceXy[0], ballAlpha),
            lerp(prev.xy[1], sourceXy[1], ballAlpha),
          ],
          r: lerp(prev.r, targetR, ballAlpha),
        };
      }
    }
    lastBallWasbGoodTRef.current = t;
    return {
      ...(ballWasbFrameRaw ?? { t }),
      xy: smoothBallWasbRef.current.xy,
      r: smoothBallWasbRef.current.r,
    };
  })();

  const anyBallVisible = showBallVballnet || showBallYolo || showBallWasb;
  const hasYoloBall = Boolean(ballYolo?.frames?.length);
  const hasWasbBall = Boolean(ballWasb?.frames?.length);

  const ballTrail = useMemo(() => {
    if (!showBallVballnet || !showBallTrail) return [];
    return ballTrailSegments(
      ball,
      t,
      trailLengthS,
      ballFrame?.xy ?? null,
    );
  }, [ball, t, trailLengthS, showBallVballnet, showBallTrail, ballFrame?.xy]);

  const ballYoloTrail = useMemo(() => {
    if (!showBallYolo || !showBallTrail || !hasYoloBall) return [];
    return ballTrailSegments(
      ballYolo ?? null,
      t,
      trailLengthS,
      ballYoloFrame?.xy ?? null,
    );
  }, [
    ballYolo,
    t,
    trailLengthS,
    showBallYolo,
    showBallTrail,
    hasYoloBall,
    ballYoloFrame?.xy,
  ]);

  const ballWasbTrail = useMemo(() => {
    if (!showBallWasb || !showBallTrail || !hasWasbBall) return [];
    return ballTrailSegments(
      ballWasb ?? null,
      t,
      trailLengthS,
      ballWasbFrame?.xy ?? null,
    );
  }, [
    ballWasb,
    t,
    trailLengthS,
    showBallWasb,
    showBallTrail,
    hasWasbBall,
    ballWasbFrame?.xy,
  ]);

  const playerPlaylist = useMemo(() => {
    const visible = new Set(playerOverlaysRaw.map((p) => p.track_id));
    return buildPlayerPlaylist(tracks, visible);
  }, [tracks, playerOverlaysRaw]);

  // Reset smoothing when media / track set changes.
  useEffect(() => {
    smoothOutlinesRef.current.clear();
    smoothBallRef.current = null;
    smoothBallYoloRef.current = null;
    smoothBallWasbRef.current = null;
    lastSmoothTRef.current = null;
  }, [mediaUrl, tracks, ballYolo, ballWasb]);

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

  /** Frames actually used (or planned) for net detect — colored scrubber marks. */
  const netSamplePoints = useMemo(() => {
    if (netTracks?.frames?.length) {
      return netTracks.frames.map((fr) => ({
        t: fr.t,
        frame_index: fr.frame_index,
        kind: fr.kind || (fr.trigger === "static_refresh" ? "static_refresh" : "motion_settled"),
      }));
    }
    if (cameraMotion?.net_sample_points?.length) {
      return cameraMotion.net_sample_points;
    }
    return [];
  }, [netTracks, cameraMotion]);

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

  // Keep SVG viewBox in native video pixels. Cached media often fires
  // loadedmetadata before React attaches the JSX handler, leaving the
  // default 1280×720 viewBox and shrinking 640×360 tracks into the corner.
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    const syncSize = () => {
      if (el.videoWidth > 0 && el.videoHeight > 0) {
        setSize({ w: el.videoWidth, h: el.videoHeight });
      }
      if (el.duration && Number.isFinite(el.duration)) {
        setDuration(el.duration);
      }
    };
    syncSize();
    el.addEventListener("loadedmetadata", syncSize);
    el.addEventListener("loadeddata", syncSize);
    el.addEventListener("resize", syncSize);
    return () => {
      el.removeEventListener("loadedmetadata", syncSize);
      el.removeEventListener("loadeddata", syncSize);
      el.removeEventListener("resize", syncSize);
    };
  }, [mediaUrl]);

  // True while the user is dragging the scrubber — skip RAF/timeupdate so
  // async seeks don't snap the controlled range back to the old clock.
  const scrubbingRef = useRef(false);

  // Sync overlays to the video clock every animation frame while playing
  // (timeupdate alone is ~4–10Hz and makes tracks look lagged).
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    let raf = 0;
    const tick = () => {
      if (!scrubbingRef.current) publishTime(el.currentTime);
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
      if (!scrubbingRef.current) publishTime(el.currentTime);
      setPlaying(false);
    };
    const onSeekOrTimeUpdate = () => {
      // While playing, RAF owns the clock; timeupdate is only a fallback when paused.
      // Ignore while scrubbing — currentTime lags until the range request completes.
      if (el.paused && !scrubbingRef.current) publishTime(el.currentTime);
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
    // Prefer React state over el.currentTime — browsers keep the old clock
    // until the async seek (byte-range fetch) finishes.
    const max =
      Number.isFinite(el.duration) && el.duration > 0
        ? el.duration
        : Number.isFinite(duration) && duration > 0
          ? duration
          : next;
    const clamped = Math.max(0, Math.min(max, next));
    el.currentTime = clamped;
    publishTime(clamped);
  }

  function onScrubPointerDown() {
    scrubbingRef.current = true;
  }

  function onScrubPointerUp() {
    scrubbingRef.current = false;
    const el = videoRef.current;
    if (el) publishTime(el.currentTime);
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
          Body highlights {showOutlines ? "on" : "off"}
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
          className={`toggle-chip ball-vballnet${showBallVballnet ? " active" : ""}`}
          onClick={() => setShowBallVballnet((v) => !v)}
          title="VballNet heatmap tracker (yellow)"
          disabled={!ball?.frames?.length}
        >
          VballNet {showBallVballnet ? "on" : "off"}
        </button>
        <button
          type="button"
          className={`toggle-chip ball-yolo${showBallYolo ? " active" : ""}`}
          onClick={() => setShowBallYolo((v) => !v)}
          title={
            hasYoloBall
              ? "SetOptics YOLO + BoT-SORT tracker (cyan)"
              : "Re-run analysis after deploying track_ball_yolo"
          }
          disabled={!hasYoloBall}
        >
          YOLO {showBallYolo && hasYoloBall ? "on" : "off"}
        </button>
        <button
          type="button"
          className={`toggle-chip ball-wasb${showBallWasb ? " active" : ""}`}
          onClick={() => setShowBallWasb((v) => !v)}
          title={
            hasWasbBall
              ? "WASB HRNet volleyball tracker (magenta) — raw, no fusion"
              : "Re-run analysis after deploying track_ball_wasb"
          }
          disabled={!hasWasbBall}
        >
          WASB {showBallWasb && hasWasbBall ? "on" : "off"}
        </button>
        {ball?.frames?.length || hasYoloBall || hasWasbBall ? (
          <>
            <button
              type="button"
              className={`toggle-chip${anyBallVisible && showBallTrail ? " active" : ""}`}
              onClick={() => setShowBallTrail((v) => !v)}
              disabled={!anyBallVisible}
              title="Parabolic flight trail (splits on detector teleports)"
            >
              Trail {anyBallVisible && showBallTrail ? "on" : "off"}
            </button>
            {anyBallVisible && showBallTrail ? (
              <label className="trail-length-control" title="How far back the trail looks">
                <span>Length</span>
                <input
                  type="range"
                  min={BALL_TRAIL_MIN_S}
                  max={BALL_TRAIL_MAX_S}
                  step={0.05}
                  value={trailLengthS}
                  onChange={(e) => setTrailLengthS(Number(e.target.value))}
                />
                <span className="trail-length-value">{trailLengthS.toFixed(2)}s</span>
              </label>
            ) : null}
          </>
        ) : null}
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
            {netSamplePoints.length
              ? ` · ${netSamplePoints.length} marks`
              : ""}
          </button>
        ) : null}
      </div>

      <div className="video-shell analysis-shell">
        <div className="analysis-video-box">
          <video
            ref={videoRef}
            src={mediaUrl}
            poster={posterUrl}
            playsInline
            onClick={togglePlay}
          />
          <svg
            className="analysis-overlay"
            width="100%"
            height="100%"
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
              {showBallVballnet && sample3d?.ball && Hinv
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
            const top = outline.reduce(
              (best, pt) => (pt[1] < best[1] ? pt : best),
              outline[0] ?? [p.bbox[0], p.bbox[1]],
            );
            const labelX = top[0];
            const labelY = top[1];
            return (
              <g key={p.track_id}>
                {showOutlines ? (
                  <polygon
                    points={outline.map(([x, y]) => `${x},${y}`).join(" ")}
                    fill={hexToRgba(p.color, 0.42)}
                    stroke="none"
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
                {showOutlines || showBoxes ? (
                  <text
                    x={labelX}
                    y={labelY - 8}
                    fill={p.color}
                    stroke="rgba(0,0,0,0.55)"
                    strokeWidth={Math.max(2, size.w / 500)}
                    paintOrder="stroke"
                    fontSize={Math.max(15, size.w / 55)}
                    fontFamily="sans-serif"
                    fontWeight="700"
                    textAnchor="middle"
                  >
                    #{p.label}
                  </text>
                ) : null}
              </g>
            );
          })}

          {showBallVballnet && showBallTrail
            ? ballTrail.map((seg, segIdx) => {
                const cam = calibration?.camera;
                const tip = seg[seg.length - 1];
                const baseW = trailStrokeWidthPx(tip, cam, size.w);
                const fullPath = trailPointsToSmoothPath(seg);
                const headStart = Math.max(0, Math.floor(seg.length * 0.45) - 1);
                const headPath = trailPointsToSmoothPath(seg.slice(headStart));
                return (
                  <g key={`ball-trail-vb-${segIdx}`}>
                    <path
                      d={fullPath}
                      fill="none"
                      stroke={BALL_VBALLNET_TRAIL}
                      strokeWidth={baseW * 0.75}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    {headPath ? (
                      <path
                        d={headPath}
                        fill="none"
                        stroke={BALL_VBALLNET_STROKE}
                        strokeWidth={baseW}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    ) : null}
                  </g>
                );
              })
            : null}

          {showBallYolo && showBallTrail
            ? ballYoloTrail.map((seg, segIdx) => {
                const cam = calibration?.camera;
                const tip = seg[seg.length - 1];
                const baseW = trailStrokeWidthPx(tip, cam, size.w);
                const fullPath = trailPointsToSmoothPath(seg);
                const headStart = Math.max(0, Math.floor(seg.length * 0.45) - 1);
                const headPath = trailPointsToSmoothPath(seg.slice(headStart));
                return (
                  <g key={`ball-trail-yolo-${segIdx}`}>
                    <path
                      d={fullPath}
                      fill="none"
                      stroke={BALL_YOLO_TRAIL}
                      strokeWidth={baseW * 0.75}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    {headPath ? (
                      <path
                        d={headPath}
                        fill="none"
                        stroke={BALL_YOLO_STROKE}
                        strokeWidth={baseW}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    ) : null}
                  </g>
                );
              })
            : null}

          {showBallVballnet && ballFrame?.xy ? (
            <circle
              cx={ballFrame.xy[0]}
              cy={ballFrame.xy[1]}
              r={ballFrame.r ?? 8}
              fill={
                ballPhysics?.teleported
                  ? "rgba(255, 120, 40, 0.7)"
                  : BALL_VBALLNET_FILL
              }
              stroke={ballPhysics?.teleported ? "rgba(255,80,0,0.9)" : "none"}
              strokeWidth={ballPhysics?.teleported ? 2 : 0}
            />
          ) : null}

          {showBallYolo && ballYoloFrame?.xy ? (
            <circle
              cx={ballYoloFrame.xy[0]}
              cy={ballYoloFrame.xy[1]}
              r={ballYoloFrame.r ?? 8}
              fill={
                ballYoloPhysics?.teleported
                  ? "rgba(255, 120, 40, 0.7)"
                  : BALL_YOLO_FILL
              }
              stroke={
                ballYoloPhysics?.teleported
                  ? "rgba(255,80,0,0.9)"
                  : BALL_YOLO_STROKE
              }
              strokeWidth={ballYoloPhysics?.teleported ? 2 : 1.5}
            />
          ) : null}

          {showBallWasb && showBallTrail
            ? ballWasbTrail.map((seg, segIdx) => {
                const cam = calibration?.camera;
                const tip = seg[seg.length - 1];
                const baseW = trailStrokeWidthPx(tip, cam, size.w);
                const fullPath = trailPointsToSmoothPath(seg);
                const headStart = Math.max(0, Math.floor(seg.length * 0.45) - 1);
                const headPath = trailPointsToSmoothPath(seg.slice(headStart));
                return (
                  <g key={`ball-trail-wasb-${segIdx}`}>
                    <path
                      d={fullPath}
                      fill="none"
                      stroke={BALL_WASB_TRAIL}
                      strokeWidth={baseW * 0.75}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    {headPath ? (
                      <path
                        d={headPath}
                        fill="none"
                        stroke={BALL_WASB_STROKE}
                        strokeWidth={baseW}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    ) : null}
                  </g>
                );
              })
            : null}

          {showBallWasb && ballWasbFrame?.xy ? (
            <circle
              cx={ballWasbFrame.xy[0]}
              cy={ballWasbFrame.xy[1]}
              r={ballWasbFrame.r ?? 8}
              fill={
                ballWasbPhysics?.teleported
                  ? "rgba(255, 120, 40, 0.7)"
                  : BALL_WASB_FILL
              }
              stroke={
                ballWasbPhysics?.teleported
                  ? "rgba(255,80,0,0.9)"
                  : BALL_WASB_STROKE
              }
              strokeWidth={ballWasbPhysics?.teleported ? 2 : 1.5}
            />
          ) : null}
        </svg>
        </div>
      </div>

      {playerPlaylist.length > 0 ? (
        <div className="player-playlist" aria-label="Player playlist">
          <div className="player-playlist-head row between">
            <span className="meta-line">Players</span>
            <span className="meta-line">
              {playerOverlays.length}/{playerPlaylist.length} on screen
            </span>
          </div>
          <ul className="player-playlist-list">
            {playerPlaylist.map((entry) => (
              <li
                key={entry.track_id}
                className={`player-playlist-item${entry.visible ? " visible" : ""}`}
              >
                <span
                  className="player-playlist-swatch"
                  style={{ background: hexToRgba(entry.color, 0.55) }}
                  aria-hidden
                />
                <span className="player-playlist-label" style={{ color: entry.color }}>
                  #{entry.label}
                </span>
                <span className="player-playlist-status meta-line">
                  {entry.visible ? "in frame" : "off"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

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
          {(showMotionTicks && cameraMotion) || netSamplePoints.length > 0 ? (
            <div className="scrubber-motion" aria-hidden>
              {showMotionTicks &&
              startsUnsettled &&
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
              {showMotionTicks
                ? motionSegments.map((seg, i) => {
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
                  })
                : null}
              {showMotionTicks
                ? settlePoints.map((sp, i) => {
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
                        title={`Camera settled @ ${sp.t.toFixed(1)}s`}
                        onClick={() => seek(sp.t)}
                      />
                    );
                  })
                : null}
              {netSamplePoints.map((sp, i) => {
                const left = (sp.t / motionSpan) * 100;
                const isRefresh = sp.kind === "static_refresh";
                const isActive =
                  activeNetFrame != null &&
                  Math.abs(activeNetFrame.t - sp.t) < 1e-3;
                return (
                  <button
                    key={`net-${sp.t}-${i}`}
                    type="button"
                    className={`scrubber-motion-tick tick-net${isRefresh ? " tick-net-refresh" : " tick-net-settle"}${isActive ? " active" : ""}`}
                    style={{ left: `${left}%` }}
                    title={
                      isRefresh
                        ? `Net static refresh @ ${sp.t.toFixed(1)}s`
                        : `Net settle sample @ ${sp.t.toFixed(1)}s`
                    }
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
            onPointerDown={onScrubPointerDown}
            onPointerUp={onScrubPointerUp}
            onPointerCancel={onScrubPointerUp}
            onBlur={onScrubPointerUp}
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
            {netSamplePoints.length > 0 ? (
              <>
                <span className="motion-legend-swatch net-settle" /> Net settle
                <span className="motion-legend-swatch net-refresh" /> Net refresh
              </>
            ) : null}
            <span>
              {settlePoints.length} settles
              {netSamplePoints.length
                ? ` · ${netSamplePoints.length} net samples`
                : ""}{" "}
              · {cameraMotion.summary?.num_segments ?? motionSegments.length} segs
              {mergeGapS != null ? ` · merge ≤${mergeGapS}s` : ""} ·{" "}
              {cameraMotion.method}
            </span>
          </div>
        ) : netSamplePoints.length > 0 ? (
          <div className="motion-legend meta-line">
            <span className="motion-legend-swatch net-settle" /> Net settle
            <span className="motion-legend-swatch net-refresh" /> Net refresh
            <span>{netSamplePoints.length} net samples</span>
          </div>
        ) : null}
      </div>

      <p className="hint">
        Body highlights are SAM silhouettes (translucent fill, no border).
        Labels are playlist numbers (#1, #2…), not jersey numbers.
        Ball trail fits a parabola per flight and breaks on detector teleports
        (orange ball = corrected jump).
        {cameraMotion
          ? " Settle ticks mark when the camera stops."
          : ""}
        {netSamplePoints.length
          ? " Cyan / amber marks are frames used for net analysis (settle vs static refresh)."
          : ""}
        {netTracks?.frames?.length
          ? " Net overlay shows detected net corners only (no court fill yet)."
          : " Court overlay requires calibration."}
      </p>
    </div>
  );
}
