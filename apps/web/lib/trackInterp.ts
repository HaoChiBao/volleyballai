/**
 * Smooth track sampling for playback overlays.
 * Models may sample below video FPS (e.g. SAM @ 5fps); interpolate between
 * bracketing detections so outlines/ball don't step/jump.
 */

export type Timed = { t: number };

/** Binary search: largest index with frames[i].t <= t (or -1). */
export function lowerBoundTime<T extends Timed>(frames: T[], t: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t <= t) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

export function lerp(a: number, b: number, u: number): number {
  return a + (b - a) * u;
}

export function lerpBbox(
  a: [number, number, number, number],
  b: [number, number, number, number],
  u: number,
): [number, number, number, number] {
  return [
    lerp(a[0], b[0], u),
    lerp(a[1], b[1], u),
    lerp(a[2], b[2], u),
    lerp(a[3], b[3], u),
  ];
}

function perimeter(pts: [number, number][]): number {
  let len = 0;
  for (let i = 0; i < pts.length; i++) {
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % pts.length];
    len += Math.hypot(x1 - x0, y1 - y0);
  }
  return len;
}

/** Resample a closed polygon to `n` evenly spaced points along the perimeter. */
export function resamplePolygon(
  pts: [number, number][],
  n: number,
): [number, number][] {
  if (pts.length === 0 || n <= 0) return [];
  if (pts.length === 1) return Array.from({ length: n }, () => [...pts[0]] as [number, number]);
  const total = perimeter(pts);
  if (total < 1e-6) {
    return Array.from({ length: n }, () => [...pts[0]] as [number, number]);
  }
  const out: [number, number][] = [];
  for (let k = 0; k < n; k++) {
    let target = (total * k) / n;
    let i = 0;
    while (i < pts.length) {
      const [x0, y0] = pts[i];
      const [x1, y1] = pts[(i + 1) % pts.length];
      const seg = Math.hypot(x1 - x0, y1 - y0);
      if (target <= seg || i === pts.length - 1) {
        const u = seg > 1e-6 ? target / seg : 0;
        out.push([lerp(x0, x1, u), lerp(y0, y1, u)]);
        break;
      }
      target -= seg;
      i++;
    }
  }
  return out;
}

/** Squared distance sum between corresponding polygon vertices. */
function outlineAlignCost(
  a: [number, number][],
  b: [number, number][],
  shift: number,
  reverse: boolean,
): number {
  const n = a.length;
  let cost = 0;
  for (let i = 0; i < n; i++) {
    const j = reverse
      ? (shift - i + n * 2) % n
      : (i + shift) % n;
    const dx = a[i][0] - b[j][0];
    const dy = a[i][1] - b[j][1];
    cost += dx * dx + dy * dy;
  }
  return cost;
}

/**
 * Rotate / optionally reverse `b` so vertices best match `a`.
 * SAM mask contours often start at different points (or flip winding),
 * which makes naive lerp look like the blob is flickering / jumping.
 */
export function alignOutline(
  a: [number, number][],
  b: [number, number][],
): [number, number][] {
  const n = a.length;
  if (n === 0 || b.length !== n) return b;
  let bestShift = 0;
  let bestReverse = false;
  let bestCost = Infinity;
  for (const reverse of [false, true]) {
    for (let shift = 0; shift < n; shift++) {
      const cost = outlineAlignCost(a, b, shift, reverse);
      if (cost < bestCost) {
        bestCost = cost;
        bestShift = shift;
        bestReverse = reverse;
      }
    }
  }
  const out: [number, number][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const j = bestReverse
      ? (bestShift - i + n * 2) % n
      : (i + bestShift) % n;
    out[i] = b[j];
  }
  return out;
}

export function lerpOutline(
  a: [number, number][] | undefined,
  b: [number, number][] | undefined,
  u: number,
  n = 64,
): [number, number][] | undefined {
  if ((!a || a.length < 3) && (!b || b.length < 3)) return undefined;
  const ra = resamplePolygon(a && a.length >= 3 ? a : b!, n);
  const rbRaw = resamplePolygon(b && b.length >= 3 ? b : a!, n);
  const rb = alignOutline(ra, rbRaw);
  // Ease endpoints so morph settles instead of hard-stepping at sample hits.
  const s = u * u * (3 - 2 * u);
  return ra.map((p, i) => [
    lerp(p[0], rb[i][0], s),
    lerp(p[1], rb[i][1], s),
  ]);
}

/** Temporal EMA toward a target outline (same vertex count). */
export function smoothOutlineToward(
  prev: [number, number][] | undefined,
  target: [number, number][],
  alpha: number,
  /** If previous centroid is farther than this (px), snap instead of easing. */
  snapDistance = 80,
): [number, number][] {
  if (!prev || prev.length !== target.length) {
    return target.map((p) => [p[0], p[1]] as [number, number]);
  }
  const aligned = alignOutline(prev, target);
  let pcx = 0;
  let pcy = 0;
  let tcx = 0;
  let tcy = 0;
  for (let i = 0; i < prev.length; i++) {
    pcx += prev[i][0];
    pcy += prev[i][1];
    tcx += aligned[i][0];
    tcy += aligned[i][1];
  }
  const n = prev.length;
  const dist = Math.hypot(pcx / n - tcx / n, pcy / n - tcy / n);
  if (dist > snapDistance) {
    return aligned.map((p) => [p[0], p[1]] as [number, number]);
  }
  const a = Math.min(1, Math.max(0, alpha));
  return prev.map((p, i) => [
    lerp(p[0], aligned[i][0], a),
    lerp(p[1], aligned[i][1], a),
  ]);
}

export type Bracket<T extends Timed> =
  | { kind: "exact"; frame: T }
  | { kind: "lerp"; a: T; b: T; u: number }
  | { kind: "hold"; frame: T }
  | null;

/**
 * Find bracketing samples for time `t`.
 * - lerp when both neighbors exist and span <= maxGap
 * - hold nearest if within holdRadius (edge / short gap)
 */
export function bracketFrames<T extends Timed>(
  frames: T[],
  t: number,
  maxGap: number,
  holdRadius = maxGap,
): Bracket<T> {
  if (!frames.length) return null;
  const i = lowerBoundTime(frames, t);
  if (i >= 0 && Math.abs(frames[i].t - t) < 1e-4) {
    return { kind: "exact", frame: frames[i] };
  }
  if (i >= 0 && i + 1 < frames.length) {
    const a = frames[i];
    const b = frames[i + 1];
    const span = b.t - a.t;
    if (span > 0 && span <= maxGap && t >= a.t && t <= b.t) {
      return { kind: "lerp", a, b, u: (t - a.t) / span };
    }
  }
  // Fall back to nearest hold within radius.
  let best = frames[0];
  let bestD = Math.abs(best.t - t);
  if (i >= 0) {
    const d = Math.abs(frames[i].t - t);
    if (d < bestD) {
      best = frames[i];
      bestD = d;
    }
  }
  if (i + 1 < frames.length) {
    const d = Math.abs(frames[i + 1].t - t);
    if (d < bestD) {
      best = frames[i + 1];
      bestD = d;
    }
  }
  if (bestD > holdRadius) return null;
  return { kind: "hold", frame: best };
}
