import type { Point2 } from "@volleyballai/types";

/** Solve Ah = b for 8 DOF homography (h33 = 1) via Gaussian elimination. */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) {
      throw new Error("Singular homography system");
    }
    [M[col], M[pivot]] = [M[pivot], M[col]];
    const div = M[col][col];
    for (let c = col; c <= n; c++) M[col][c] /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row) => row[n]);
}

/** Image points → court meters. Returns row-major 3x3 H. */
export function computeHomography(
  imagePoints: Point2[],
  courtPoints: Point2[],
): number[] {
  if (imagePoints.length < 4 || courtPoints.length < 4) {
    throw new Error("Need at least 4 point pairs");
  }
  const n = Math.min(imagePoints.length, courtPoints.length);
  const A: number[][] = [];
  const b: number[] = [];

  for (let i = 0; i < n; i++) {
    const { x, y } = imagePoints[i];
    const X = courtPoints[i].x;
    const Y = courtPoints[i].y;
    A.push([x, y, 1, 0, 0, 0, -X * x, -X * y]);
    b.push(X);
    A.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]);
    b.push(Y);
  }

  const h = solveLinearSystem(A, b);
  return [...h, 1];
}

export function applyHomography(H: number[], p: Point2): Point2 {
  const denom = H[6] * p.x + H[7] * p.y + H[8];
  if (Math.abs(denom) < 1e-9) return { x: p.x, y: p.y };
  return {
    x: (H[0] * p.x + H[1] * p.y + H[2]) / denom,
    y: (H[3] * p.x + H[4] * p.y + H[5]) / denom,
  };
}

/** Invert 3x3 row-major matrix. */
export function invertHomography(H: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = H;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const Hh = -(a * f - c * d);
  const I = a * e - b * d;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) throw new Error("Non-invertible H");
  return [A, D, G, B, E, Hh, C, F, I].map((v) => v / det);
}

/** Default FIVB indoor court size (meters). */
export const DEFAULT_COURT_LENGTH_M = 18;
export const DEFAULT_COURT_WIDTH_M = 9;

/** Default FIVB court corners in meters (order: BL, BR, TR, TL). */
export const DEFAULT_COURT_CORNERS: Point2[] = [
  { x: 0, y: 0 },
  { x: DEFAULT_COURT_LENGTH_M, y: 0 },
  { x: DEFAULT_COURT_LENGTH_M, y: DEFAULT_COURT_WIDTH_M },
  { x: 0, y: DEFAULT_COURT_WIDTH_M },
];

export function courtCorners(
  length_m = DEFAULT_COURT_LENGTH_M,
  width_m = DEFAULT_COURT_WIDTH_M,
): Point2[] {
  return [
    { x: 0, y: 0 },
    { x: length_m, y: 0 },
    { x: length_m, y: width_m },
    { x: 0, y: width_m },
  ];
}

/** Named court lines for manual line-drawing calibration (meters). */
export type CourtLineId =
  | "near"
  | "right"
  | "far"
  | "left"
  | "net"
  | "attack_a"
  | "attack_b";

export interface CourtLineDef {
  id: CourtLineId;
  label: string;
  /** Endpoint A → B in court meters */
  a: Point2;
  b: Point2;
}

/**
 * Court line template in meters for a given size.
 * Attack lines sit at 1/3 and 2/3 of length (FIVB: 6m / 12m on an 18m court).
 */
export function courtLinesForSize(
  length_m = DEFAULT_COURT_LENGTH_M,
  width_m = DEFAULT_COURT_WIDTH_M,
): CourtLineDef[] {
  const L = length_m;
  const W = width_m;
  const mid = L / 2;
  const attackA = L / 3;
  const attackB = (2 * L) / 3;
  return [
    { id: "near", label: "Near sideline", a: { x: 0, y: 0 }, b: { x: L, y: 0 } },
    { id: "right", label: "Right endline", a: { x: L, y: 0 }, b: { x: L, y: W } },
    { id: "far", label: "Far sideline", a: { x: L, y: W }, b: { x: 0, y: W } },
    { id: "left", label: "Left endline", a: { x: 0, y: W }, b: { x: 0, y: 0 } },
    { id: "net", label: "Center / net line", a: { x: mid, y: 0 }, b: { x: mid, y: W } },
    {
      id: "attack_a",
      label: "Attack line (near half)",
      a: { x: attackA, y: 0 },
      b: { x: attackA, y: W },
    },
    {
      id: "attack_b",
      label: "Attack line (far half)",
      a: { x: attackB, y: 0 },
      b: { x: attackB, y: W },
    },
  ];
}

/** @deprecated Use courtLinesForSize(18, 9) — kept for imports. */
export const FIVB_COURT_LINES: CourtLineDef[] = courtLinesForSize(
  DEFAULT_COURT_LENGTH_M,
  DEFAULT_COURT_WIDTH_M,
);
