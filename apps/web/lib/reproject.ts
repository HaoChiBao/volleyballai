import { promises as fs } from "fs";
import path from "path";
import {
  applyHomography,
  computeHomography,
  DEFAULT_COURT_CORNERS,
} from "@volleyballai/court-math";
import type { Calibration, PlayersTracksFile, Point2 } from "@volleyballai/types";
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

export async function reprojectArtifacts(
  videoId: string,
  cal: Calibration,
): Promise<void> {
  const withH = ensureHomography(cal);
  if (!withH.H) return;

  const dir = videoDir(videoId);
  const tracksPath = path.join(dir, "players.tracks.json");
  const court3dPath = path.join(dir, "court3d.json");

  let tracks: PlayersTracksFile | null = null;
  try {
    tracks = JSON.parse(await fs.readFile(tracksPath, "utf8")) as PlayersTracksFile;
  } catch {
    return;
  }

  const H = withH.H;
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

  const samples: { t: number; players: { track_id: number; x: number; y: number; z: number }[] }[] =
    [];
  const times = new Set<number>();
  for (const p of players) {
    for (const f of p.frames) times.add(f.t);
  }
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
    samples.push({ t, players: markers });
  }

  await fs.writeFile(
    court3dPath,
    JSON.stringify(
      {
        video_id: videoId,
        pipeline_version: PIPELINE_VERSION,
        court: withH.court,
        samples,
      },
      null,
      2,
    ) + "\n",
  );
}
