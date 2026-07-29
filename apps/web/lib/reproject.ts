import { promises as fs } from "fs";
import path from "path";
import {
  applyHomography,
  computeHomography,
  courtCorners,
  estimateBallWorldPosition,
  estimateCameraPoseFromHomography,
} from "@volleyballai/court-math";
import type {
  BallTracksFile,
  Calibration,
  PlayersTracksFile,
  Point2,
} from "@volleyballai/types";
import { DEFAULT_COURT, PIPELINE_VERSION } from "@volleyballai/types";
import { videoDir } from "./paths";
import { resolveArtifactDir } from "./store";

export function ensureHomography(cal: Calibration): Calibration {
  const kf = cal.keyframes[0];
  if (!kf || kf.image_points.length < 4) return cal;
  const length = cal.court?.length_m ?? DEFAULT_COURT.length_m;
  const width = cal.court?.width_m ?? DEFAULT_COURT.width_m;
  const court =
    kf.court_points_m.length >= 4
      ? kf.court_points_m
      : courtCorners(length, width);
  const H = computeHomography(kf.image_points, court);
  return {
    ...cal,
    court: { length_m: length, width_m: width },
    H,
  };
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
    const length = withH.court?.length_m ?? DEFAULT_COURT.length_m;
    const width = withH.court?.width_m ?? DEFAULT_COURT.width_m;
    const camera = estimateCameraPoseFromHomography(
      withH.H,
      w,
      h,
      length,
      width,
    );
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

  const root = videoDir(videoId);
  const artifactDir = await resolveArtifactDir(videoId);
  const tracksPath = path.join(artifactDir, "players.tracks.json");
  const ballPath = path.join(artifactDir, "ball.tracks.json");
  const ballYoloPath = path.join(artifactDir, "ball.tracks.yolo.json");
  const ballWasbPath = path.join(artifactDir, "ball.tracks.wasb.json");
  const court3dPath = path.join(artifactDir, "court3d.json");
  const calPath = path.join(root, "calibration.json");

  // Persist camera onto calibration (video-scoped)
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
  const tracksJson = JSON.stringify(nextTracks, null, 2) + "\n";
  await fs.writeFile(tracksPath, tracksJson);
  if (artifactDir !== root) {
    await fs.writeFile(path.join(root, "players.tracks.json"), tracksJson);
  }

  async function projectBallFile(
    filePath: string,
    mirrorName: string,
  ): Promise<BallTracksFile | null> {
    let ball: BallTracksFile | null = null;
    try {
      ball = JSON.parse(await fs.readFile(filePath, "utf8")) as BallTracksFile;
    } catch {
      return null;
    }
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
    const ballJson = JSON.stringify(ball, null, 2) + "\n";
    await fs.writeFile(filePath, ballJson);
    if (artifactDir !== root) {
      await fs.writeFile(path.join(root, mirrorName), ballJson);
    }
    return ball;
  }

  const ball = await projectBallFile(ballPath, "ball.tracks.json");
  await projectBallFile(ballYoloPath, "ball.tracks.yolo.json");
  await projectBallFile(ballWasbPath, "ball.tracks.wasb.json");

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

  const court3dJson =
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
    ) + "\n";
  await fs.writeFile(court3dPath, court3dJson);
  if (artifactDir !== root) {
    await fs.writeFile(path.join(root, "court3d.json"), court3dJson);
  }
}
