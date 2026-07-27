import {
  computeHomography,
  estimateCameraPoseFromHomography,
} from "@volleyballai/court-math";
import type {
  Calibration,
  CourtKeypoint,
  CourtKeypointsFile,
  CourtKeypointsFrame,
  Point2,
} from "@volleyballai/types";
import { DEFAULT_COURT, PIPELINE_VERSION } from "@volleyballai/types";

/** Prefer unique geometry landmarks (skip duplicate net/center labels). */
const PREFERRED_NAMES = [
  "corner_top_left",
  "corner_top_right",
  "corner_bottom_right",
  "corner_bottom_left",
  "attack_top_left",
  "attack_top_right",
  "attack_bottom_right",
  "attack_bottom_left",
  "net_left",
  "net_right",
  "midline_left",
  "midline_right",
] as const;

const CORNER_NAMES = [
  "corner_top_left",
  "corner_top_right",
  "corner_bottom_right",
  "corner_bottom_left",
] as const;

function visibleMap(
  keypoints: CourtKeypoint[],
  minConf: number,
): Map<string, CourtKeypoint> {
  const map = new Map<string, CourtKeypoint>();
  for (const kp of keypoints) {
    if (!kp.visible || !kp.xy || kp.conf < minConf) continue;
    // Keep highest-conf instance per name
    const prev = map.get(kp.name);
    if (!prev || kp.conf > prev.conf) map.set(kp.name, kp);
  }
  return map;
}

function scoreFrame(frame: CourtKeypointsFrame, minConf: number): number {
  const map = visibleMap(frame.keypoints, minConf);
  let score = (frame.box_conf ?? 0) * 2;
  for (const name of PREFERRED_NAMES) {
    const kp = map.get(name);
    if (kp) score += kp.conf + (name.startsWith("corner_") ? 0.5 : 0);
  }
  // Strong bonus when all 4 corners are present
  if (CORNER_NAMES.every((n) => map.has(n))) score += 3;
  return score;
}

function pickCorrespondences(
  map: Map<string, CourtKeypoint>,
): { image: Point2[]; court: Point2[]; names: string[] } | null {
  const image: Point2[] = [];
  const court: Point2[] = [];
  const names: string[] = [];

  const tryAdd = (name: string) => {
    const kp = map.get(name);
    if (!kp?.xy || !kp.court_m) return;
    if (names.includes(name)) return;
    image.push({ x: kp.xy[0], y: kp.xy[1] });
    court.push({ x: kp.court_m.x, y: kp.court_m.y });
    names.push(name);
  };

  // Prefer the four outer corners (clean rectangle → stable H).
  for (const name of CORNER_NAMES) tryAdd(name);

  if (image.length < 4) {
    for (const name of PREFERRED_NAMES) {
      if (image.length >= 4) break;
      tryAdd(name);
    }
  }

  if (image.length < 4) return null;
  // Homography solver uses the first 4 pairs (8 DOF).
  return {
    image: image.slice(0, 4),
    court: court.slice(0, 4),
    names: names.slice(0, 4),
  };
}

export function selectBestKeypointFrame(
  file: CourtKeypointsFile,
  minConf = 0.25,
): CourtKeypointsFrame | null {
  const frames = file.frames ?? [];
  if (!frames.length) return null;
  let best: CourtKeypointsFrame | null = null;
  let bestScore = -Infinity;
  for (const fr of frames) {
    const s = scoreFrame(fr, minConf);
    if (s > bestScore) {
      bestScore = s;
      best = fr;
    }
  }
  return best;
}

/**
 * Build a Calibration (+ H + camera) from Modal court.keypoints.json.
 * Returns null when fewer than 4 usable landmarks are available.
 */
export function calibrationFromCourtKeypoints(
  file: CourtKeypointsFile,
  opts: {
    videoId: string;
    length_m?: number;
    width_m?: number;
    minConf?: number;
    imageWidth?: number;
    imageHeight?: number;
    fromRunId?: string | null;
  },
): Calibration | null {
  const minConf = opts.minConf ?? 0.25;
  const frame = selectBestKeypointFrame(file, minConf);
  if (!frame) return null;

  const map = visibleMap(frame.keypoints, minConf);
  const pairs = pickCorrespondences(map);
  if (!pairs) return null;

  const length_m = opts.length_m ?? DEFAULT_COURT.length_m;
  const width_m = opts.width_m ?? DEFAULT_COURT.width_m;
  // Scale court_m if court size differs from the baked 18×9 template.
  const sx = length_m / 18;
  const sy = width_m / 9;
  const courtScaled = pairs.court.map((p) => ({ x: p.x * sx, y: p.y * sy }));

  let H: number[];
  try {
    H = computeHomography(pairs.image, courtScaled);
  } catch {
    return null;
  }

  const imageWidth =
    opts.imageWidth ?? file.image_size?.width ?? 1280;
  const imageHeight =
    opts.imageHeight ?? file.image_size?.height ?? 720;

  let camera = null;
  try {
    camera = estimateCameraPoseFromHomography(
      H,
      imageWidth,
      imageHeight,
      length_m,
      width_m,
    );
  } catch {
    camera = null;
  }

  return {
    video_id: opts.videoId,
    pipeline_version: PIPELINE_VERSION,
    court: { length_m, width_m },
    source: "auto_keypoints",
    from_run_id: opts.fromRunId ?? file.run?.run_id ?? null,
    keyframes: [
      {
        t: frame.t,
        image_points: pairs.image,
        court_points_m: courtScaled,
      },
    ],
    H,
    camera,
  };
}
