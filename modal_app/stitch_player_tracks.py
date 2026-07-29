"""
Stitch fragmented player tracklets into stable identities.

SAM 3 is run in time chunks; each chunk remaps object IDs, so the same person
becomes many track_ids. Occlusions also birth new IDs mid-chunk. This module
links tracklets that are close in time and space (bbox IoU + foot position +
simple velocity prediction) and, when court_xy is available, adaptive dwell
occupancy on the court (where a player has been spending time).

Pure geometry / occupancy — no neural re-ID (see docs/AI_POLICY.md).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MAX_GAP_S = 5.0
DEFAULT_MIN_IOU = 0.05
DEFAULT_MAX_FOOT_DIST_PX = 150.0
# Slightly looser than 3.0 when dwell can confirm identity across a hop.
DEFAULT_MAX_COURT_DIST_M = 3.5
DEFAULT_MIN_FRAMES = 6
DEFAULT_MIN_DURATION_S = 0.75
# Drop tracklets that almost never project onto the FIVB court (+ margin).
DEFAULT_ON_COURT_MARGIN_M = 1.5
DEFAULT_MIN_ON_COURT_FRAC = 0.15

# Adaptive court-area dwell (rolling occupancy while tracked).
DEFAULT_DWELL_CELL_M = 1.0
DEFAULT_DWELL_WINDOW_S = 4.0
DEFAULT_DWELL_MIN_S = 1.0
DEFAULT_DWELL_WEIGHT = 2.0
DEFAULT_DWELL_MARGIN_M = 1.5
# Foot px threshold was tuned at SAM max-width 640; scale with image width.
FOOT_DIST_REF_WIDTH = 640.0


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _foot(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + 0.5 * w, y + h)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def scale_foot_dist_px(
    base_px: float,
    image_width: int | float | None,
) -> float:
    """Scale legacy max_foot_dist_px from the 640-wide SAM reference."""
    if not image_width or image_width <= 0:
        return float(base_px)
    return float(base_px) * (float(image_width) / FOOT_DIST_REF_WIDTH)


@dataclass
class _DwellGrid:
    """Soft time-weighted occupancy on a court meter grid."""

    cell_m: float
    margin_m: float
    nx: int
    ny: int
    # Flat row-major hist: index = iy * nx + ix
    mass: list[float] = field(default_factory=list)

    @classmethod
    def empty(
        cls,
        *,
        cell_m: float = DEFAULT_DWELL_CELL_M,
        margin_m: float = DEFAULT_DWELL_MARGIN_M,
    ) -> _DwellGrid:
        cell_m = max(0.25, float(cell_m))
        margin_m = max(0.0, float(margin_m))
        # Court [0,18]×[0,9] plus margin ring on each side.
        span_x = 18.0 + 2.0 * margin_m
        span_y = 9.0 + 2.0 * margin_m
        nx = max(1, int(round(span_x / cell_m)))
        ny = max(1, int(round(span_y / cell_m)))
        return cls(
            cell_m=cell_m,
            margin_m=margin_m,
            nx=nx,
            ny=ny,
            mass=[0.0] * (nx * ny),
        )

    def _bin(self, x: float, y: float) -> tuple[int, int] | None:
        # Origin at (-margin, -margin).
        ix = int((x + self.margin_m) / self.cell_m)
        iy = int((y + self.margin_m) / self.cell_m)
        if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
            return None
        return ix, iy

    def add(self, x: float, y: float, dt: float) -> None:
        if dt <= 0:
            return
        cell = self._bin(x, y)
        if cell is None:
            return
        ix, iy = cell
        # Soft 3×3 blur so quantization does not kill near-cell matches.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                jx, jy = ix + dx, iy + dy
                if jx < 0 or jy < 0 or jx >= self.nx or jy >= self.ny:
                    continue
                w = 1.0 if (dx == 0 and dy == 0) else 0.35
                self.mass[jy * self.nx + jx] += dt * w

    def total(self) -> float:
        return float(sum(self.mass))

    def normalized(self) -> list[float]:
        s = self.total()
        if s <= 1e-9:
            return [0.0] * len(self.mass)
        inv = 1.0 / s
        return [m * inv for m in self.mass]

    def centroid(self) -> tuple[float, float] | None:
        s = self.total()
        if s <= 1e-9:
            return None
        cx = cy = 0.0
        for iy in range(self.ny):
            for ix in range(self.nx):
                m = self.mass[iy * self.nx + ix]
                if m <= 0:
                    continue
                # Cell center in court meters.
                x = -self.margin_m + (ix + 0.5) * self.cell_m
                y = -self.margin_m + (iy + 0.5) * self.cell_m
                cx += m * x
                cy += m * y
        return (cx / s, cy / s)


def _hist_intersection(a: list[float], b: list[float]) -> float:
    """Histogram intersection for two non-negative vectors (ideally normalized)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(sum(min(a[i], b[i]) for i in range(n)))


def _dwell_window(
    frames: list[dict[str, Any]],
    *,
    which: str,
    window_s: float,
    cell_m: float,
    margin_m: float,
) -> _DwellGrid:
    """
    Build occupancy for the head (first window_s) or tail (last window_s) of a track.
    """
    grid = _DwellGrid.empty(cell_m=cell_m, margin_m=margin_m)
    if not frames or window_s <= 0:
        return grid

    if which == "head":
        t0 = float(frames[0]["t"])
        t1 = t0 + window_s
        selected = [f for f in frames if float(f["t"]) <= t1 + 1e-9]
    else:
        t1 = float(frames[-1]["t"])
        t0 = t1 - window_s
        selected = [f for f in frames if float(f["t"]) >= t0 - 1e-9]

    if not selected:
        return grid

    for i, f in enumerate(selected):
        xy = f.get("court_xy")
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue
        if i + 1 < len(selected):
            dt = max(0.0, float(selected[i + 1]["t"]) - float(f["t"]))
        elif len(selected) >= 2:
            dt = max(0.0, float(f["t"]) - float(selected[i - 1]["t"]))
        else:
            dt = 1.0 / 8.0  # single sample — nominal SAM step
        # Cap per-sample weight so sparse tracks do not dominate.
        dt = min(dt, 0.5)
        grid.add(float(xy[0]), float(xy[1]), dt)
    return grid


@dataclass
class _Tracklet:
    track_id: int
    frames: list[dict[str, Any]] = field(default_factory=list)
    # Cached dwell views (filled by ensure_dwell).
    dwell_head: _DwellGrid | None = field(default=None, repr=False)
    dwell_tail: _DwellGrid | None = field(default=None, repr=False)
    dwell_cfg: tuple[float, float, float] | None = field(default=None, repr=False)

    @property
    def start_t(self) -> float:
        return float(self.frames[0]["t"])

    @property
    def end_t(self) -> float:
        return float(self.frames[-1]["t"])

    @property
    def first_bbox(self) -> list[float]:
        return list(self.frames[0]["bbox"])

    @property
    def last_bbox(self) -> list[float]:
        return list(self.frames[-1]["bbox"])

    def foot_at_end(self) -> tuple[float, float]:
        return _foot(self.last_bbox)

    def foot_at_start(self) -> tuple[float, float]:
        return _foot(self.first_bbox)

    def court_at_end(self) -> tuple[float, float] | None:
        xy = self.frames[-1].get("court_xy")
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            return (float(xy[0]), float(xy[1]))
        return None

    def court_at_start(self) -> tuple[float, float] | None:
        xy = self.frames[0].get("court_xy")
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            return (float(xy[0]), float(xy[1]))
        return None

    def velocity_px(self) -> tuple[float, float]:
        """Rough image velocity from the last few frames (px/s)."""
        fr = self.frames
        if len(fr) < 2:
            return (0.0, 0.0)
        n = min(5, len(fr))
        a, b = fr[-n], fr[-1]
        dt = float(b["t"]) - float(a["t"])
        if dt < 1e-3:
            return (0.0, 0.0)
        fa, fb = _foot(a["bbox"]), _foot(b["bbox"])
        return ((fb[0] - fa[0]) / dt, (fb[1] - fa[1]) / dt)

    def predicted_foot(self, t: float) -> tuple[float, float]:
        fx, fy = self.foot_at_end()
        vx, vy = self.velocity_px()
        dt = max(0.0, t - self.end_t)
        return (fx + vx * dt, fy + vy * dt)

    def ensure_dwell(
        self,
        *,
        window_s: float,
        cell_m: float,
        margin_m: float,
    ) -> None:
        cfg = (window_s, cell_m, margin_m)
        if self.dwell_cfg == cfg and self.dwell_head is not None and self.dwell_tail is not None:
            return
        self.dwell_head = _dwell_window(
            self.frames,
            which="head",
            window_s=window_s,
            cell_m=cell_m,
            margin_m=margin_m,
        )
        self.dwell_tail = _dwell_window(
            self.frames,
            which="tail",
            window_s=window_s,
            cell_m=cell_m,
            margin_m=margin_m,
        )
        self.dwell_cfg = cfg

    def dwell_sim_to(
        self,
        other: _Tracklet,
        *,
        window_s: float,
        cell_m: float,
        margin_m: float,
        min_s: float,
    ) -> float | None:
        """
        Similarity between this track's tail dwell and other's head dwell.
        None if either side lacks enough occupancy mass.
        """
        self.ensure_dwell(window_s=window_s, cell_m=cell_m, margin_m=margin_m)
        other.ensure_dwell(window_s=window_s, cell_m=cell_m, margin_m=margin_m)
        assert self.dwell_tail is not None and other.dwell_head is not None
        if self.dwell_tail.total() < min_s or other.dwell_head.total() < min_s:
            return None
        return _hist_intersection(
            self.dwell_tail.normalized(),
            other.dwell_head.normalized(),
        )


def _link_cost(
    a: _Tracklet,
    b: _Tracklet,
    *,
    max_gap_s: float,
    min_iou: float,
    max_foot_dist_px: float,
    max_court_dist_m: float,
    max_overlap_s: float = 2.0,
    dwell_window_s: float = DEFAULT_DWELL_WINDOW_S,
    dwell_cell_m: float = DEFAULT_DWELL_CELL_M,
    dwell_margin_m: float = DEFAULT_DWELL_MARGIN_M,
    dwell_min_s: float = DEFAULT_DWELL_MIN_S,
    dwell_weight: float = DEFAULT_DWELL_WEIGHT,
) -> tuple[float, bool] | None:
    """
    Lower is better. None = incompatible.
    Returns (cost, dwell_decisive) where dwell_decisive means dwell_sim pulled
    the cost below what geometry alone would have allowed / ranked.

    Uses court_xy when both ends have it; otherwise image IoU / predicted foot.
    Dwell occupancy (tail→head) rewards identities that reappear in the same
    adaptive court area after a SAM session switch.
    """
    gap = b.start_t - a.end_t
    # b must start after a starts (ordering); reject large concurrent overlaps.
    if b.start_t + 1e-3 < a.start_t:
        return None
    if gap < -max_overlap_s:
        return None
    if gap > max_gap_s:
        return None

    ca, cb = a.court_at_end(), b.court_at_start()
    iou = _bbox_iou(a.last_bbox, b.first_bbox)

    dwell_sim = a.dwell_sim_to(
        b,
        window_s=dwell_window_s,
        cell_m=dwell_cell_m,
        margin_m=dwell_margin_m,
        min_s=dwell_min_s,
    )
    use_dwell = dwell_weight > 0 and dwell_sim is not None
    dwell_ok = use_dwell and dwell_sim >= 0.15
    dwell_strong = use_dwell and dwell_sim >= 0.35

    if ca is not None and cb is not None:
        d_m = _dist(ca, cb)
        # Allow a slightly longer court hop when dwell strongly agrees.
        court_limit = max_court_dist_m * (1.5 if dwell_strong else 1.0)
        if gap < 0:
            # Overlapping chunk twins — require tight court agreement.
            if d_m > min(1.2, max_court_dist_m * 0.5) and iou < 0.25 and not dwell_strong:
                return None
            geom = d_m + 0.05
        else:
            if d_m > court_limit:
                # Hard reject far hops unless dwell is strong and still within 1.5×.
                if not (dwell_strong and d_m <= max_court_dist_m * 1.5):
                    return None
            geom = d_m + 0.15 * gap

        if use_dwell:
            cost = geom - dwell_weight * dwell_sim
            decisive = dwell_strong and d_m > max_court_dist_m * 0.6
            return (cost, decisive)
        return (geom, False)

    pred = a.predicted_foot(b.start_t)
    d_px = _dist(pred, b.foot_at_start())
    d_raw = _dist(a.foot_at_end(), b.foot_at_start())
    d_best = min(d_px, d_raw)
    foot_limit = max_foot_dist_px * (1.35 if dwell_strong else 1.0)

    if gap < 0:
        if iou < 0.25 and d_best > max_foot_dist_px * 0.6 and not dwell_ok:
            return None
        geom = (1.0 - iou) * 25.0 + d_best * 0.25
    else:
        if iou < min_iou and d_best > foot_limit:
            if not (dwell_strong and d_best <= max_foot_dist_px * 1.5):
                return None
        geom = (1.0 - iou) * 40.0 + d_best * 0.35 + gap * 8.0

    if use_dwell:
        cost = geom - dwell_weight * dwell_sim * 20.0  # px-scale costs are larger
        decisive = dwell_strong and d_best > max_foot_dist_px * 0.5
        return (cost, decisive)
    return (geom, False)


def _greedy_links(
    tracks: list[_Tracklet],
    *,
    max_gap_s: float,
    min_iou: float,
    max_foot_dist_px: float,
    max_court_dist_m: float,
    dwell_window_s: float,
    dwell_cell_m: float,
    dwell_margin_m: float,
    dwell_min_s: float,
    dwell_weight: float,
) -> tuple[list[tuple[int, int]], int]:
    """Return (pairs prev→next, count of dwell-decisive links), 1-1 greedy by cost."""
    candidates: list[tuple[float, bool, int, int]] = []
    for i, a in enumerate(tracks):
        for j, b in enumerate(tracks):
            if i == j:
                continue
            result = _link_cost(
                a,
                b,
                max_gap_s=max_gap_s,
                min_iou=min_iou,
                max_foot_dist_px=max_foot_dist_px,
                max_court_dist_m=max_court_dist_m,
                dwell_window_s=dwell_window_s,
                dwell_cell_m=dwell_cell_m,
                dwell_margin_m=dwell_margin_m,
                dwell_min_s=dwell_min_s,
                dwell_weight=dwell_weight,
            )
            if result is not None:
                cost, decisive = result
                candidates.append((cost, decisive, i, j))
    candidates.sort(key=lambda x: x[0])

    used_prev: set[int] = set()
    used_next: set[int] = set()
    links: list[tuple[int, int]] = []
    dwell_merges = 0
    for cost, decisive, i, j in candidates:
        if i in used_prev or j in used_next:
            continue
        if j in used_prev and i in used_next:
            continue
        used_prev.add(i)
        used_next.add(j)
        links.append((i, j))
        if decisive:
            dwell_merges += 1
    return links, dwell_merges


def _merge_frames(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate frames; if times overlap, keep the earlier track's samples."""
    if not a:
        return list(b)
    if not b:
        return list(a)
    end_a = float(a[-1]["t"])
    kept_b = [f for f in b if float(f["t"]) > end_a + 1e-4]
    return a + kept_b


def _on_court(xy: tuple[float, float], margin_m: float) -> bool:
    x, y = xy
    return -margin_m <= x <= 18.0 + margin_m and -margin_m <= y <= 9.0 + margin_m


def stitch_player_tracks(
    players: list[dict[str, Any]],
    *,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    min_iou: float = DEFAULT_MIN_IOU,
    max_foot_dist_px: float = DEFAULT_MAX_FOOT_DIST_PX,
    max_court_dist_m: float = DEFAULT_MAX_COURT_DIST_M,
    min_frames: int = DEFAULT_MIN_FRAMES,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    on_court_margin_m: float = DEFAULT_ON_COURT_MARGIN_M,
    min_on_court_frac: float = DEFAULT_MIN_ON_COURT_FRAC,
    max_passes: int = 8,
    image_width: int | float | None = None,
    dwell_window_s: float = DEFAULT_DWELL_WINDOW_S,
    dwell_cell_m: float = DEFAULT_DWELL_CELL_M,
    dwell_margin_m: float = DEFAULT_DWELL_MARGIN_M,
    dwell_min_s: float = DEFAULT_DWELL_MIN_S,
    dwell_weight: float = DEFAULT_DWELL_WEIGHT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Link tracklets across chunk boundaries / short occlusions.

    Returns (stitched_players, stats).
    """
    foot_px = scale_foot_dist_px(max_foot_dist_px, image_width)

    tracklets = [
        _Tracklet(track_id=int(p["track_id"]), frames=list(p.get("frames") or []))
        for p in players
        if p.get("frames")
    ]
    before = len(tracklets)
    merges = 0
    dwell_merges = 0

    for _ in range(max_passes):
        if len(tracklets) < 2:
            break
        links, dwell_n = _greedy_links(
            tracklets,
            max_gap_s=max_gap_s,
            min_iou=min_iou,
            max_foot_dist_px=foot_px,
            max_court_dist_m=max_court_dist_m,
            dwell_window_s=dwell_window_s,
            dwell_cell_m=dwell_cell_m,
            dwell_margin_m=dwell_margin_m,
            dwell_min_s=dwell_min_s,
            dwell_weight=dwell_weight,
        )
        if not links:
            break
        dwell_merges += dwell_n

        # Union-find style: follow chains prev→next, build survivor lists.
        succ = {i: j for i, j in links}
        pred = {j: i for i, j in links}
        roots = [i for i in range(len(tracklets)) if i not in pred]
        consumed: set[int] = set()
        new_tracks: list[_Tracklet] = []

        for root in roots:
            chain = [root]
            cur = root
            while cur in succ:
                cur = succ[cur]
                chain.append(cur)
            frames = list(tracklets[chain[0]].frames)
            for idx in chain[1:]:
                frames = _merge_frames(frames, tracklets[idx].frames)
                merges += 1
            new_tracks.append(
                _Tracklet(track_id=tracklets[chain[0]].track_id, frames=frames)
            )
            consumed.update(chain)

        for i, tr in enumerate(tracklets):
            if i not in consumed:
                new_tracks.append(tr)
        tracklets = new_tracks

    # Drop short / flicker tracks and mostly off-court spectators.
    kept: list[_Tracklet] = []
    dropped_short = 0
    dropped_off_court = 0
    for tr in tracklets:
        dur = tr.end_t - tr.start_t
        if len(tr.frames) < min_frames or dur < min_duration_s:
            dropped_short += 1
            continue
        court_pts = [
            (float(f["court_xy"][0]), float(f["court_xy"][1]))
            for f in tr.frames
            if isinstance(f.get("court_xy"), (list, tuple)) and len(f["court_xy"]) >= 2
        ]
        if court_pts and min_on_court_frac > 0:
            on_n = sum(1 for xy in court_pts if _on_court(xy, on_court_margin_m))
            if on_n / len(court_pts) < min_on_court_frac:
                dropped_off_court += 1
                continue
        kept.append(tr)

    # Stable renumber 1..N by first appearance.
    kept.sort(key=lambda t: (t.start_t, t.track_id))
    out: list[dict[str, Any]] = []
    for i, tr in enumerate(kept, start=1):
        out.append({"track_id": i, "frames": tr.frames})

    stats = {
        "tracks_before": before,
        "tracks_after": len(out),
        "merges": merges,
        "dwell_merges": dwell_merges,
        "dropped_short": dropped_short,
        "dropped_off_court": dropped_off_court,
        "max_gap_s": max_gap_s,
        "min_iou": min_iou,
        "max_foot_dist_px": foot_px,
        "max_foot_dist_px_base": max_foot_dist_px,
        "max_court_dist_m": max_court_dist_m,
        "min_frames": min_frames,
        "min_duration_s": min_duration_s,
        "on_court_margin_m": on_court_margin_m,
        "min_on_court_frac": min_on_court_frac,
        "dwell_window_s": dwell_window_s,
        "dwell_cell_m": dwell_cell_m,
        "dwell_min_s": dwell_min_s,
        "dwell_weight": dwell_weight,
        "image_width": image_width,
    }
    return out, stats


def stitch_players_file(
    tracks: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    players = list(tracks.get("players") or [])
    if "image_width" not in kwargs:
        iw = tracks.get("image_width")
        if iw:
            kwargs["image_width"] = iw
    stitched, stats = stitch_player_tracks(players, **kwargs)
    out = dict(tracks)
    out["players"] = stitched
    out["stitch"] = stats
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Stitch player tracklets in-place or to --out")
    p.add_argument(
        "tracks_json",
        type=Path,
        help="players.tracks.json path",
    )
    p.add_argument("--out", type=Path, default=None, help="Write here (default: overwrite)")
    p.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP_S)
    p.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    p.add_argument("--max-foot-dist", type=float, default=DEFAULT_MAX_FOOT_DIST_PX)
    p.add_argument("--max-court-dist", type=float, default=DEFAULT_MAX_COURT_DIST_M)
    p.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    p.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_S)
    p.add_argument("--dwell-window", type=float, default=DEFAULT_DWELL_WINDOW_S)
    p.add_argument("--dwell-cell", type=float, default=DEFAULT_DWELL_CELL_M)
    p.add_argument("--dwell-weight", type=float, default=DEFAULT_DWELL_WEIGHT)
    p.add_argument(
        "--backup",
        action="store_true",
        help="Write players.tracks.before_stitch.json beside the input",
    )
    args = p.parse_args()

    raw = json.loads(args.tracks_json.read_text(encoding="utf-8"))
    if args.backup:
        bak = args.tracks_json.with_name("players.tracks.before_stitch.json")
        bak.write_text(json.dumps(raw), encoding="utf-8")
        print(f"[stitch] backup -> {bak}")

    out = stitch_players_file(
        raw,
        max_gap_s=args.max_gap,
        min_iou=args.min_iou,
        max_foot_dist_px=args.max_foot_dist,
        max_court_dist_m=args.max_court_dist,
        min_frames=args.min_frames,
        min_duration_s=args.min_duration,
        dwell_window_s=args.dwell_window,
        dwell_cell_m=args.dwell_cell,
        dwell_weight=args.dwell_weight,
    )
    dest = args.out or args.tracks_json
    dest.write_text(json.dumps(out), encoding="utf-8")
    s = out["stitch"]
    print(
        f"[stitch] {s['tracks_before']} -> {s['tracks_after']} tracks "
        f"(merges={s['merges']}, dwell_merges={s.get('dwell_merges', 0)}, "
        f"dropped_short={s['dropped_short']}) -> {dest}"
    )


if __name__ == "__main__":
    main()
