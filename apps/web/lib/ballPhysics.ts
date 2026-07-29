/**
 * Ball flight model for trail rendering.
 *
 * Between contacts the ball is approximately ballistic. In image space over a
 * short window that is well modeled by independent quadratics:
 *   x(t) = ax + bx·τ + cx·τ²
 *   y(t) = ay + by·τ + cy·τ²
 * with τ = t − t₀.
 *
 * Sudden teleports (detector ID swaps / false peaks) violate that model — we
 * split segments and drop outliers instead of drawing a jagged polyline.
 */

export type BallSample = {
  t: number;
  xy: [number, number];
  r?: number;
  court_xyz?: [number, number, number];
};

export type QuadCoeffs = { a: number; b: number; c: number };

export type FlightFit = {
  t0: number;
  x: QuadCoeffs;
  y: QuadCoeffs;
  /** RMS residual in pixels after outlier rejection. */
  rms: number;
  /** Samples that survived the fit. */
  inliers: BallSample[];
};

/** Image-space speed above this ⇒ treat as teleport / new flight. */
export const BALL_TELEPORT_PX_S = 1600;
/** Points farther than this from the parabola are outliers. */
export const BALL_FIT_OUTLIER_PX = 28;
/** Minimum inliers to trust a quadratic flight fit. */
export const BALL_FIT_MIN_POINTS = 4;

function hypot2(dx: number, dy: number): number {
  return Math.hypot(dx, dy);
}

/**
 * Drop single-sample spikes: A→B is insane but A→C (skipping B) is fine.
 * Keeps true flight breaks (A→B and A→C both huge).
 */
export function despikeSamples(
  samples: BallSample[],
  maxSpeedPxS: number = BALL_TELEPORT_PX_S,
): BallSample[] {
  if (samples.length < 3) return samples;
  const ordered = [...samples].sort((a, b) => a.t - b.t);
  const keep: BallSample[] = [ordered[0]];
  for (let i = 1; i < ordered.length - 1; i++) {
    const prev = keep[keep.length - 1];
    const cur = ordered[i];
    const next = ordered[i + 1];
    const dt0 = cur.t - prev.t;
    const dt1 = next.t - cur.t;
    const dtAC = next.t - prev.t;
    if (dt0 <= 1e-6 || dt1 <= 1e-6 || dtAC <= 1e-6) {
      keep.push(cur);
      continue;
    }
    const speedIn =
      hypot2(cur.xy[0] - prev.xy[0], cur.xy[1] - prev.xy[1]) / dt0;
    const speedOut =
      hypot2(next.xy[0] - cur.xy[0], next.xy[1] - cur.xy[1]) / dt1;
    const speedSkip =
      hypot2(next.xy[0] - prev.xy[0], next.xy[1] - prev.xy[1]) / dtAC;
    if (
      speedIn > maxSpeedPxS &&
      speedOut > maxSpeedPxS &&
      speedSkip < maxSpeedPxS * 0.65
    ) {
      continue; // spike
    }
    keep.push(cur);
  }
  keep.push(ordered[ordered.length - 1]);
  return keep;
}

/** Split detections whenever consecutive motion looks like a teleport. */
export function splitOnTeleports(
  samples: BallSample[],
  maxSpeedPxS: number = BALL_TELEPORT_PX_S,
): BallSample[][] {
  if (!samples.length) return [];
  const ordered = despikeSamples(samples, maxSpeedPxS);
  const segments: BallSample[][] = [];
  let cur: BallSample[] = [ordered[0]];

  for (let i = 1; i < ordered.length; i++) {
    const prev = cur[cur.length - 1];
    const next = ordered[i];
    const dt = next.t - prev.t;
    if (dt <= 1e-6) {
      cur[cur.length - 1] = next;
      continue;
    }
    const speed =
      hypot2(next.xy[0] - prev.xy[0], next.xy[1] - prev.xy[1]) / dt;
    // Also break on long temporal holes — different rallies.
    if (speed > maxSpeedPxS || dt > 0.55) {
      if (cur.length >= 2) segments.push(cur);
      cur = [next];
    } else {
      cur.push(next);
    }
  }
  if (cur.length >= 2) segments.push(cur);
  return segments;
}

/**
 * Least-squares fit of v(τ) = a + b·τ + c·τ².
 * Returns null if the design is degenerate.
 */
export function fitQuadratic(
  times: number[],
  values: number[],
  t0: number,
): QuadCoeffs | null {
  const n = times.length;
  if (n < 3) return null;

  // Normal equations for [a,b,c] via 3×3 Gram matrix.
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  let s4 = 0;
  let sy0 = 0;
  let sy1 = 0;
  let sy2 = 0;
  for (let i = 0; i < n; i++) {
    const tau = times[i] - t0;
    const tau2 = tau * tau;
    const tau3 = tau2 * tau;
    const tau4 = tau2 * tau2;
    const v = values[i];
    s0 += 1;
    s1 += tau;
    s2 += tau2;
    s3 += tau3;
    s4 += tau4;
    sy0 += v;
    sy1 += v * tau;
    sy2 += v * tau2;
  }

  // Solve M · [a,b,c]^T = rhs with M = [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]]
  const det =
    s0 * (s2 * s4 - s3 * s3) -
    s1 * (s1 * s4 - s3 * s2) +
    s2 * (s1 * s3 - s2 * s2);
  if (Math.abs(det) < 1e-9) {
    // Fall back to linear a + b·τ
    const den = s0 * s2 - s1 * s1;
    if (Math.abs(den) < 1e-9) return null;
    const a = (sy0 * s2 - sy1 * s1) / den;
    const b = (s0 * sy1 - s1 * sy0) / den;
    return { a, b, c: 0 };
  }

  const a =
    (sy0 * (s2 * s4 - s3 * s3) -
      s1 * (sy1 * s4 - s3 * sy2) +
      s2 * (sy1 * s3 - s2 * sy2)) /
    det;
  const b =
    (s0 * (sy1 * s4 - s3 * sy2) -
      sy0 * (s1 * s4 - s3 * s2) +
      s2 * (s1 * sy2 - sy1 * s2)) /
    det;
  const c =
    (s0 * (s2 * sy2 - s3 * sy1) -
      s1 * (s1 * sy2 - s3 * sy0) +
      sy0 * (s1 * s3 - s2 * s2)) /
    det;
  return { a, b, c };
}

export function evalQuad(q: QuadCoeffs, t: number, t0: number): number {
  const tau = t - t0;
  return q.a + q.b * tau + q.c * tau * tau;
}

function residualPx(fit: { t0: number; x: QuadCoeffs; y: QuadCoeffs }, s: BallSample): number {
  const px = evalQuad(fit.x, s.t, fit.t0);
  const py = evalQuad(fit.y, s.t, fit.t0);
  return hypot2(px - s.xy[0], py - s.xy[1]);
}

function rmsOf(
  fit: { t0: number; x: QuadCoeffs; y: QuadCoeffs },
  samples: BallSample[],
): number {
  if (!samples.length) return Infinity;
  let acc = 0;
  for (const s of samples) {
    const r = residualPx(fit, s);
    acc += r * r;
  }
  return Math.sqrt(acc / samples.length);
}

/**
 * Fit a parabolic flight to a segment; reject outliers once and refit.
 */
export function fitFlightParabola(
  samples: BallSample[],
  outlierPx: number = BALL_FIT_OUTLIER_PX,
): FlightFit | null {
  if (samples.length < BALL_FIT_MIN_POINTS) return null;
  const ordered = [...samples].sort((a, b) => a.t - b.t);
  const t0 = ordered[0].t;

  const fitOnce = (pts: BallSample[]): FlightFit | null => {
    if (pts.length < BALL_FIT_MIN_POINTS) return null;
    const times = pts.map((p) => p.t);
    const xs = pts.map((p) => p.xy[0]);
    const ys = pts.map((p) => p.xy[1]);
    const x = fitQuadratic(times, xs, t0);
    const y = fitQuadratic(times, ys, t0);
    if (!x || !y) return null;
    const draft = { t0, x, y, rms: 0, inliers: pts };
    draft.rms = rmsOf(draft, pts);
    return draft;
  };

  let fit = fitOnce(ordered);
  if (!fit) return null;

  const inliers = ordered.filter((s) => residualPx(fit!, s) <= outlierPx);
  if (inliers.length >= BALL_FIT_MIN_POINTS && inliers.length < ordered.length) {
    const refit = fitOnce(inliers);
    if (refit) fit = refit;
  } else {
    fit = { ...fit, inliers: ordered };
  }
  return fit;
}

/** Dense samples along a fitted parabola (inclusive end). */
export function sampleFlightFit(
  fit: FlightFit,
  tStart: number,
  tEnd: number,
  hz: number = 60,
): BallSample[] {
  if (tEnd < tStart) return [];
  const dt = 1 / Math.max(hz, 1);
  const out: BallSample[] = [];
  for (let t = tStart; t <= tEnd + 1e-9; t += dt) {
    const tt = Math.min(t, tEnd);
    out.push({
      t: tt,
      xy: [evalQuad(fit.x, tt, fit.t0), evalQuad(fit.y, tt, fit.t0)],
    });
    if (tt >= tEnd) break;
  }
  // Ensure exact end.
  const last = out[out.length - 1];
  if (!last || Math.abs(last.t - tEnd) > 1e-4) {
    out.push({
      t: tEnd,
      xy: [evalQuad(fit.x, tEnd, fit.t0), evalQuad(fit.y, tEnd, fit.t0)],
    });
  }
  return out;
}

/**
 * Build trail polylines for [t − lengthS, t]:
 * teleport-split → parabola fit per flight → dense resample.
 * Falls back to raw (non-teleport) polyline when a segment is too short to fit.
 */
export function buildParabolicTrail(
  detections: BallSample[],
  t: number,
  lengthS: number,
  opts?: {
    teleportPxS?: number;
    outlierPx?: number;
    sampleHz?: number;
  },
): BallSample[][] {
  if (!detections.length || lengthS <= 0) return [];
  const t0 = Math.max(0, t - lengthS);
  const window = detections.filter((d) => d.t >= t0 - 1e-3 && d.t <= t + 1e-3);
  if (window.length < 2) return [];

  const teleportPxS = opts?.teleportPxS ?? BALL_TELEPORT_PX_S;
  const outlierPx = opts?.outlierPx ?? BALL_FIT_OUTLIER_PX;
  const sampleHz = opts?.sampleHz ?? 60;

  const flights = splitOnTeleports(window, teleportPxS);
  const trails: BallSample[][] = [];

  for (const flight of flights) {
    // Clip to visible window.
    const clipped = flight.filter((d) => d.t >= t0 - 1e-3 && d.t <= t + 1e-3);
    if (clipped.length < 2) continue;

    const fit = fitFlightParabola(clipped, outlierPx);
    if (fit && fit.inliers.length >= BALL_FIT_MIN_POINTS) {
      const start = Math.max(t0, fit.inliers[0].t);
      const end = Math.min(t, fit.inliers[fit.inliers.length - 1].t);
      if (end - start < 1e-3) continue;
      const dense = sampleFlightFit(fit, start, end, sampleHz);
      if (dense.length >= 2) trails.push(dense);
    } else {
      // Short / sparse: draw straight segments but never across teleports.
      trails.push(clipped.map((d) => ({ t: d.t, xy: d.xy as [number, number] })));
    }
  }
  return trails;
}

/**
 * If the current detection teleported off the active flight parabola,
 * return the model prediction instead (and mark as corrected).
 */
export function stabilizeBallAtTime(
  detections: BallSample[],
  t: number,
  lookbackS: number = 1.25,
): { xy: [number, number]; teleported: boolean; residual: number } | null {
  const t0 = Math.max(0, t - lookbackS);
  // Allow a short lookahead so scrubbing mid-gap still gets a flight fit.
  const window = detections.filter((d) => d.t >= t0 && d.t <= t + 0.35);
  if (window.length < BALL_FIT_MIN_POINTS) return null;

  const flights = splitOnTeleports(window);
  // Prefer the flight that covers / is nearest to t.
  let flight = flights[flights.length - 1];
  for (const f of flights) {
    if (!f.length) continue;
    if (f[0].t - 1e-3 <= t && t <= f[f.length - 1].t + 1e-3) {
      flight = f;
      break;
    }
  }
  if (!flight || flight.length < BALL_FIT_MIN_POINTS) return null;

  const fit = fitFlightParabola(flight);
  if (!fit) return null;

  const pred: [number, number] = [
    evalQuad(fit.x, t, fit.t0),
    evalQuad(fit.y, t, fit.t0),
  ];

  // Latest raw sample at/near t
  let nearest: BallSample | null = null;
  for (const s of flight) {
    if (nearest == null || Math.abs(s.t - t) < Math.abs(nearest.t - t)) {
      nearest = s;
    }
  }
  // If we are between detections, trust the parabola so the overlay stays up.
  if (!nearest || Math.abs(nearest.t - t) > 0.05) {
    return { xy: pred, teleported: true, residual: 0 };
  }
  const residual = hypot2(nearest.xy[0] - pred[0], nearest.xy[1] - pred[1]);
  if (residual > BALL_FIT_OUTLIER_PX * 1.5) {
    return { xy: pred, teleported: true, residual };
  }
  return { xy: nearest.xy, teleported: false, residual };
}
