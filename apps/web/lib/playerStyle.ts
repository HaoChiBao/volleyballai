/**
 * Stable playlist colors / labels shared by the 2D overlay and 3D court.
 * Index is position in `tracks.players` (display #1 = index 0).
 */

export const PLAYER_COLORS = [
  "#ff5a36",
  "#2dd4bf",
  "#3b82f6",
  "#f59e0b",
  "#a855f7",
  "#22c55e",
  "#ec4899",
  "#06b6d4",
  "#eab308",
  "#f97316",
  "#6366f1",
  "#14b8a6",
] as const;

export function playerColorAt(idx: number): string {
  return PLAYER_COLORS[idx % PLAYER_COLORS.length];
}

/** Display number (#1…) for a track_id from the players list order. */
export function playerLabelForTrackId(
  players: { track_id: number }[] | undefined,
  trackId: number,
): number | null {
  if (!players?.length) return null;
  const idx = players.findIndex((p) => p.track_id === trackId);
  return idx >= 0 ? idx + 1 : null;
}

export function playerColorForTrackId(
  players: { track_id: number }[] | undefined,
  trackId: number,
): string {
  if (!players?.length) return PLAYER_COLORS[0];
  const idx = players.findIndex((p) => p.track_id === trackId);
  return playerColorAt(idx >= 0 ? idx : trackId);
}
