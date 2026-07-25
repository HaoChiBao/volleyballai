import type { Point2 } from "@volleyballai/types";
import { applyHomography, invertHomography } from "./homography";

/** Camera pose in court world coords: X=length, Y=width, Z=up (meters). */
export interface CameraPose {
  /** Camera center in world meters */
  position: [number, number, number];
  /**
   * World→camera rotation, row-major 3×3 (OpenCV-style:
   * camera X right, Y down, Z forward).
   */
  R: number[];
  /** Translation t where x_cam = R X + t */
  t: number[];
  /** Intrinsics */
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  image_width: number;
  image_height: number;
  /** Vertical FOV degrees for Three.js PerspectiveCamera */
  fov_y_deg: number;
}

function matMul3(A: number[], B: number[]): number[] {
  const out = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] =
        A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
    }
  }
  return out;
}

function matVec3(M: number[], v: [number, number, number]): [number, number, number] {
  return [
    M[0] * v[0] + M[1] * v[1] + M[2] * v[2],
    M[3] * v[0] + M[4] * v[1] + M[5] * v[2],
    M[6] * v[0] + M[7] * v[1] + M[8] * v[2],
  ];
}

function norm3(v: [number, number, number]): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize3(v: [number, number, number]): [number, number, number] {
  const n = norm3(v) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function transpose3(M: number[]): number[] {
  return [M[0], M[3], M[6], M[1], M[4], M[7], M[2], M[5], M[8]];
}

/** Approximate pinhole K from frame size (≈55° horizontal FOV). */
export function guessIntrinsics(width: number, height: number) {
  const fx = width * 0.95;
  const fy = fx;
  const cx = width / 2;
  const cy = height / 2;
  const fov_y_deg = (2 * Math.atan(height / (2 * fy)) * 180) / Math.PI;
  return { fx, fy, cx, cy, fov_y_deg };
}

/**
 * Recover camera pose from image→court homography H and frame size.
 * Court plane is Z=0; corners in meters.
 */
export function estimateCameraPoseFromHomography(
  H_image_to_court: number[],
  imageWidth: number,
  imageHeight: number,
  courtLengthM = 18,
  courtWidthM = 9,
): CameraPose {
  const H_court_to_image = invertHomography(H_image_to_court);
  const { fx, fy, cx, cy, fov_y_deg } = guessIntrinsics(imageWidth, imageHeight);

  // K^{-1}
  const Kinv = [1 / fx, 0, -cx / fx, 0, 1 / fy, -cy / fy, 0, 0, 1];
  const M = matMul3(Kinv, H_court_to_image);

  const m1: [number, number, number] = [M[0], M[3], M[6]];
  const m2: [number, number, number] = [M[1], M[4], M[7]];
  const m3: [number, number, number] = [M[2], M[5], M[8]];

  const lam = 1 / (norm3(m1) || 1e-9);
  let r1 = normalize3([m1[0] * lam, m1[1] * lam, m1[2] * lam]);
  let r2 = normalize3([m2[0] * lam, m2[1] * lam, m2[2] * lam]);
  let r3 = normalize3(cross(r1, r2));
  // Re-orthogonalize r2
  r2 = normalize3(cross(r3, r1));

  let R = [
    r1[0],
    r2[0],
    r3[0],
    r1[1],
    r2[1],
    r3[1],
    r1[2],
    r2[2],
    r3[2],
  ];
  let t: [number, number, number] = [m3[0] * lam, m3[1] * lam, m3[2] * lam];

  // Ensure camera looks toward court center (positive depth)
  const centerCourt: [number, number, number] = [
    courtLengthM / 2,
    courtWidthM / 2,
    0,
  ];
  const inCam = matVec3(R, centerCourt);
  const zCam = inCam[2] + t[2];
  if (zCam < 0) {
    // Flip pose
    R = R.map((v) => -v);
    t = [-t[0], -t[1], -t[2]];
  }

  // C = -R^T t
  const Rt = transpose3(R);
  const C = matVec3(Rt, [-t[0], -t[1], -t[2]]);

  return {
    position: [C[0], C[1], C[2]],
    R,
    t: [t[0], t[1], t[2]],
    fx,
    fy,
    cx,
    cy,
    image_width: imageWidth,
    image_height: imageHeight,
    fov_y_deg,
  };
}

/** Unit ray direction in world coords through pixel (u,v). */
export function rayFromPixel(
  pose: CameraPose,
  u: number,
  v: number,
): { origin: [number, number, number]; dir: [number, number, number] } {
  const x = (u - pose.cx) / pose.fx;
  const y = (v - pose.cy) / pose.fy;
  const dirCam: [number, number, number] = normalize3([x, y, 1]);
  const Rt = transpose3(pose.R);
  const dirWorld = normalize3(matVec3(Rt, dirCam));
  return { origin: pose.position, dir: dirWorld };
}

/**
 * Place ball in 3D: along camera ray using apparent size when possible,
 * else intersect a horizontal plane at fallback height.
 */
export function estimateBallWorldPosition(
  pose: CameraPose,
  imageXy: [number, number],
  radiusPx: number | undefined,
  H_image_to_court: number[] | null,
): [number, number, number] {
  const BALL_DIAMETER_M = 0.21;
  const { origin, dir } = rayFromPixel(pose, imageXy[0], imageXy[1]);

  let distance: number | null = null;
  if (radiusPx && radiusPx > 1) {
    distance = (pose.fx * BALL_DIAMETER_M) / (2 * radiusPx);
    distance = Math.min(Math.max(distance, 1.5), 45);
  }

  if (distance != null) {
    const p: [number, number, number] = [
      origin[0] + dir[0] * distance,
      origin[1] + dir[1] * distance,
      origin[2] + dir[2] * distance,
    ];
    // Clamp onto/near court volume
    p[0] = Math.min(Math.max(p[0], -2), 20);
    p[1] = Math.min(Math.max(p[1], -2), 11);
    p[2] = Math.min(Math.max(p[2], 0.05), 6);
    return p;
  }

  // Fallback: ground-plane homography for XY + ray plane at z≈1.5m
  let xy: Point2 | null = null;
  if (H_image_to_court) {
    xy = applyHomography(H_image_to_court, { x: imageXy[0], y: imageXy[1] });
  }
  const zPlane = 1.5;
  const denom = dir[2];
  if (Math.abs(denom) > 1e-6) {
    const tHit = (zPlane - origin[2]) / denom;
    if (tHit > 0) {
      return [
        origin[0] + dir[0] * tHit,
        origin[1] + dir[1] * tHit,
        zPlane,
      ];
    }
  }
  if (xy) return [xy.x, xy.y, zPlane];
  return [9, 4.5, zPlane];
}

/**
 * Three.js camera placement: world (X,Y,Z_up) → three (X, Z_up, Y).
 * Returns position + lookAt target in Three.js coords.
 */
export function poseToThreeCamera(pose: CameraPose): {
  position: [number, number, number];
  lookAt: [number, number, number];
  fov: number;
} {
  const [cx, cy, cz] = pose.position;
  // Look toward court center in front of camera
  const { dir } = rayFromPixel(pose, pose.cx, pose.cy);
  const look: [number, number, number] = [
    cx + dir[0] * 12,
    cy + dir[1] * 12,
    Math.max(0, cz + dir[2] * 12),
  ];
  return {
    position: [cx, cz, cy],
    lookAt: [look[0], look[2], look[1]],
    fov: pose.fov_y_deg,
  };
}
