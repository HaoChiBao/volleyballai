import type {
  CalibrationCamera,
  NetTrackFrame,
  NetTracksFile,
} from "@volleyballai/types";

/** Active net-settle frame for time t (unsettled prefix → first settle). */
export function netFrameAtTime(
  netTracks: NetTracksFile | null | undefined,
  t: number,
): NetTrackFrame | null {
  const frames = netTracks?.frames;
  if (!frames?.length) return null;
  const startsUnsettled = Boolean(netTracks?.settle_policy?.starts_unsettled);
  const first = frames[0];
  if (startsUnsettled && t < first.t) return first;

  let best: NetTrackFrame | null = null;
  for (const fr of frames) {
    if (fr.t <= t + 1e-6) best = fr;
    else break;
  }
  // Before first settle when not unsettled: still use first once we have it as preview
  return best ?? first;
}

export function netCameraAtTime(
  netTracks: NetTracksFile | null | undefined,
  t: number,
): CalibrationCamera | null {
  const fr = netFrameAtTime(netTracks, t);
  return fr?.camera ?? null;
}
