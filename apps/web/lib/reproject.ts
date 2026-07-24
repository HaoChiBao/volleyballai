import { promises as fs } from "fs";
import path from "path";
import {
  applyHomography,
  computeHomography,
  DEFAULT_COURT_CORNERS,
  estimateBallWorldPosition,
  estimateCameraPoseFromHomography,
} from "@volleyballai/court-math";
import type {
  BallTracksFile,
  Calibration,
  PlayersTracksFile,
  Point2,
} from "@volleyballai/types";
import { PIPELINE_VERSION } from "@volleyballai/types";
import { videoDir } from "./paths";

export function ensureHomography(cal: Calibration): Calibration {
  const kf = cal.keyframes[0];
  if (!kf || kf.image_points.length < 4) return cal;
  const court =
    kf.court_points_m.length >= 4
      ? kf.court_points_m
      : DEFAULT_COURT_CORNERS;
  const H = computeHomography(kf.image_points, court);
  return { ...cal, H };
}

export function ensureCamera(
  cal: Calibration,
  imageWidth?: number,
  imageHeight?: number,
): Calibration {
  const withH = ensureHomography(cal);
  if (!withH.H) return withH;
  const w =
    imageWidth ??
    withH.camera?.image_width ??
    1280;
  const h =
    imageHeight ??
    withH.camera?.image_height ??
    720;
  try {
    const camera = estimateCameraPoseFromHomography(withH.H, w, h);
    return { ...withH, camera };
  } catch {
    return withH;
  }
}

export async function reprojectArtifacts(
  videoId: string,
  cal: Calibration,
  imageSize?: { width: number; height: number },
): Promise<void> {
  let withCam = ensureCamera(
    cal,
    imageSize?.width,
    imageSize?.height,
  );
  if (!withCam.H) return;

  // Prefer stored camera image size from calibration if present
  if (!imageSize && withCam.camera) {
    withCam = ensureCamera(
      withCam,
      withCam.camera.image_width,
      withCam.camera.image_height,
    );
  }

  const dir = videoDir(videoId);
  const tracksPath = path.join(dir, "players.tracks.json");
  const ballPath = path.join(dir, "ball.tracks.json");
  const court3dPath = path.join(dir, "court3d.json");
  const calPath = path.join(dir, "calibration.json");

  // Persist camera onto calibration
  await fs.writeFile(calPath, JSON.stringify(withCam, null, 2) + "\n");

  let tracks: PlayersTracksFile | null = null;
  try {
    tracks = JSON.parse(await fs.readFile(tracksPath, "utf8")) as PlayersTracksFile;
  } catch {
    return;
  }

  const H = withCam.H!;
  const players = tracks.players.map((p) => ({
    ...p,
    frames: p.frames.map((f) => {
      const [x, y, w, h] = f.bbox;
      const foot: Point2 = { x: x + w / 2, y: y + h };
      const court = applyHomography(H, foot);
      return {
        ...f,
        court_xy: [court.x, court.y] as [number, number],
      };
    }),
  }));

  const nextTracks: PlayersTracksFile = {
    ...tracks,
    players,
    pipeline_version: PIPELINE_VERSION,
  };
  await fs.writeFile(tracksPath, JSON.stringify(nextTracks, null, 2) + "\n");

  let ball: BallTracksFile | null = null;
  try {
    ball = JSON.parse(await fs.readFile(ballPath, "utf8")) as BallTracksFile;
  } catch {
    ball = null;
  }

  if (ball) {
    ball = {
      ...ball,
      pipeline_version: PIPELINE_VERSION,
      frames: ball.frames.map((f) => {
        if (!f.xy) return f;
        if (withCam.camera) {
          const [X, Y, Z] = estimateBallWorldPosition(
            withCam.camera,
            f.xy,
            f.r,
            H,
          );
          return {
            ...f,
            court_xyz: [X, Y, Z] as [number, number, number],
          };
        }
        const court = applyHomography(H, { x: f.xy[0], y: f.xy[1] });
        const z = f.court_xyz?.[2] ?? 1.5;
        return {
          ...f,
          court_xyz: [court.x, court.y, z] as [number, number, number],
        };
      }),
    };
    await fs.writeFile(ballPath, JSON.stringify(ball, null, 2) + "\n");
  }

  const samples: {
    t: number;
    players: { track_id: number; x: number; y: number; z: number }[];
    ball: { x: number; y: number; z: number } | null;
  }[] = [];
  const times = new Set<number>();
  for (const p of players) {
    for (const f of p.frames) times.add(f.t);
  }
  for (const f of ball?.frames ?? []) times.add(f.t);

  for (const t of [...times].sort((a, b) => a - b).filter((_, i) => i % 2 === 0)) {
    const markers = [];
    for (const p of players) {
      const best = p.frames.reduce((a, b) =>
        Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b,
      );
      if (!best.court_xy || Math.abs(best.t - t) > 0.25) continue;
      markers.push({
        track_id: p.track_id,
        x: best.court_xy[0],
        y: best.court_xy[1],
        z: 0,
      });
    }

    let ballPos: { x: number; y: number; z: number } | null = null;
    if (ball?.frames.length) {
      const bestB = ball.frames.reduce((a, b) =>
        Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b,
      );
      if (bestB.court_xyz && Math.abs(bestB.t - t) <= 0.25) {
        ballPos = {
          x: bestB.court_xyz[0],
          y: bestB.court_xyz[1],
          z: bestB.court_xyz[2],
        };
      }
    }

    samples.push({ t, players: markers, ball: ballPos });
  }

  await fs.writeFile(
    court3dPath,
    JSON.stringify(
      {
        video_id: videoId,
        pipeline_version: PIPELINE_VERSION,
        court: withCam.court,
        camera: withCam.camera ?? null,
        samples,
      },
      null,
      2,
    ) + "\n",
  );
}
