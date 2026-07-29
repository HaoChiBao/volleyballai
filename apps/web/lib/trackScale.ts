import type { PlayersTracksFile } from "@volleyballai/types";

/** SAM Modal path resamples with ffmpeg max_width=640 before inference. */
const SAM_MAX_WIDTH = 640;

export type TrackScale = { sx: number; sy: number };

function samResampleSize(nativeW: number, nativeH: number): {
  w: number;
  h: number;
} {
  if (nativeW <= SAM_MAX_WIDTH) {
    return { w: nativeW, h: nativeH - (nativeH % 2) };
  }
  const w = SAM_MAX_WIDTH;
  let h = Math.round(nativeH * (w / nativeW));
  h = Math.max(2, h - (h % 2));
  return { w, h };
}

function bboxExtent(tracks: PlayersTracksFile): { maxR: number; maxB: number } {
  let maxR = 0;
  let maxB = 0;
  for (const p of tracks.players) {
    for (const f of p.frames) {
      const [x, y, w, h] = f.bbox;
      maxR = Math.max(maxR, x + w);
      maxB = Math.max(maxB, y + h);
      for (const pt of f.outline ?? []) {
        maxR = Math.max(maxR, pt[0]);
        maxB = Math.max(maxB, pt[1]);
      }
    }
  }
  return { maxR, maxB };
}

/**
 * Scale factor from player-track pixel space → native video / SVG viewBox.
 * Returns 1×1 when tracks are already native.
 */
export function playerTrackScale(
  tracks: PlayersTracksFile | null | undefined,
  videoW: number,
  videoH: number,
): TrackScale {
  if (!tracks?.players?.length || videoW <= 0 || videoH <= 0) {
    return { sx: 1, sy: 1 };
  }

  const taggedW = tracks.image_width;
  const taggedH = tracks.image_height;
  if (taggedW && taggedH && taggedW > 0 && taggedH > 0) {
    if (taggedW === videoW && taggedH === videoH) return { sx: 1, sy: 1 };
    return { sx: videoW / taggedW, sy: videoH / taggedH };
  }

  const samW = tracks.sam_width;
  const samH = tracks.sam_height;
  if (samW && samH && samW > 0 && samH > 0) {
    if (samW === videoW && samH === videoH) return { sx: 1, sy: 1 };
    return { sx: videoW / samW, sy: videoH / samH };
  }

  const { maxR, maxB } = bboxExtent(tracks);
  // Already spans most of the frame → treat as native.
  if (maxR >= videoW * 0.65 && maxB >= videoH * 0.55) {
    return { sx: 1, sy: 1 };
  }

  const sam = samResampleSize(videoW, videoH);
  if (sam.w === videoW && sam.h === videoH) return { sx: 1, sy: 1 };
  // Only apply when extents fit the expected SAM canvas.
  if (maxR <= sam.w * 1.05 && maxB <= sam.h * 1.05) {
    return { sx: videoW / sam.w, sy: videoH / sam.h };
  }
  return { sx: 1, sy: 1 };
}

export function scaleBbox(
  bbox: [number, number, number, number],
  scale: TrackScale,
): [number, number, number, number] {
  if (scale.sx === 1 && scale.sy === 1) return bbox;
  return [
    bbox[0] * scale.sx,
    bbox[1] * scale.sy,
    bbox[2] * scale.sx,
    bbox[3] * scale.sy,
  ];
}

export function scaleOutline(
  outline: [number, number][] | undefined,
  scale: TrackScale,
): [number, number][] | undefined {
  if (!outline || (scale.sx === 1 && scale.sy === 1)) return outline;
  return outline.map(([x, y]) => [x * scale.sx, y * scale.sy]);
}
