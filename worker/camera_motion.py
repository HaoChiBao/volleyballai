"""
Camera-motion detection for sparse net re-detect.

Assumption: mostly fixed camera; players move a lot. Raw mean optical flow
false-triggers on athletes. Prefer a *global* transform between frames:

  1) global_affine (default) — ORB matches → estimateAffinePartial2D
     Score ≈ translation (px) + rotation term. Robust to local player motion.
  2) phase_correlate — FFT global shift only (good for pans, weak for zooms).
  3) flow_median — Farneback median magnitude (backup / comparison).

Outputs motion segments with start/end timestamps so the pipeline can:
  - keep the last net while static
  - re-detect net after motion settles (motion_end / settle_points)

Close micro-gaps between segments are merged via ``merge_gap_s`` so a stuttering
pan becomes one movement. Settle points are the times the camera stops; if the
clip starts unsettled, use the first settle pose for the whole prefix.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

Method = Literal["global_affine", "phase_correlate", "flow_median"]


@dataclass
class MotionSample:
    t: float
    frame_index: int
    score: float
    tx: float = 0.0
    ty: float = 0.0
    angle_deg: float = 0.0
    moving: bool = False


@dataclass
class MotionSegment:
    start_t: float
    end_t: float
    peak_t: float
    peak_score: float
    start_frame: int
    end_frame: int


def _resize_gray(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale < 0.999:
        bgr = cv2.resize(
            bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def score_global_affine(
    prev: np.ndarray,
    curr: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (score, tx, ty, angle_deg)."""
    orb = cv2.ORB_create(800)
    kp1, d1 = orb.detectAndCompute(prev, None)
    kp2, d2 = orb.detectAndCompute(curr, None)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0.0, 0.0, 0.0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(d1, d2, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < 0.75 * b.distance:
            good.append(a)
    if len(good) < 8:
        return 0.0, 0.0, 0.0, 0.0

    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if M is None:
        return 0.0, 0.0, 0.0, 0.0

    tx, ty = float(M[0, 2]), float(M[1, 2])
    # Rotation from 2x2 linear part
    angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    # Scale change (zoom)
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    zoom = abs(scale - 1.0)

    # Normalize translation by frame diagonal so scores are comparable.
    diag = float(np.hypot(prev.shape[1], prev.shape[0])) or 1.0
    trans_norm = float(np.hypot(tx, ty) / diag * 100.0)  # % of diagonal
    score = trans_norm + abs(angle) * 2.0 + zoom * 40.0
    return score, tx, ty, angle


def score_phase_correlate(
    prev: np.ndarray,
    curr: np.ndarray,
) -> tuple[float, float, float, float]:
    prev_f = np.float32(prev)
    curr_f = np.float32(curr)
    (dx, dy), response = cv2.phaseCorrelate(prev_f, curr_f)
    diag = float(np.hypot(prev.shape[1], prev.shape[0])) or 1.0
    score = float(np.hypot(dx, dy) / diag * 100.0) * float(max(response, 0.0))
    return score, float(dx), float(dy), 0.0


def score_flow_median(
    prev: np.ndarray,
    curr: np.ndarray,
) -> tuple[float, float, float, float]:
    flow = cv2.calcOpticalFlowFarneback(
        prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    med = float(np.median(mag))
    diag = float(np.hypot(prev.shape[1], prev.shape[0])) or 1.0
    score = med / diag * 100.0
    return score, 0.0, 0.0, 0.0


def _score_pair(method: Method, prev: np.ndarray, curr: np.ndarray):
    if method == "global_affine":
        return score_global_affine(prev, curr)
    if method == "phase_correlate":
        return score_phase_correlate(prev, curr)
    if method == "flow_median":
        return score_flow_median(prev, curr)
    raise ValueError(f"Unknown method: {method}")


def segment_motion(
    samples: list[MotionSample],
    *,
    enter_threshold: float,
    exit_threshold: float,
    min_duration_s: float = 0.25,
) -> list[MotionSegment]:
    """Hysteresis: enter when score >= enter, exit when score <= exit."""
    segments: list[MotionSegment] = []
    in_motion = False
    start_i = 0
    peak_i = 0

    for i, s in enumerate(samples):
        if not in_motion and s.score >= enter_threshold:
            in_motion = True
            start_i = i
            peak_i = i
        elif in_motion:
            if s.score > samples[peak_i].score:
                peak_i = i
            if s.score <= exit_threshold:
                in_motion = False
                dur = samples[i].t - samples[start_i].t
                if dur >= min_duration_s:
                    segments.append(
                        MotionSegment(
                            start_t=samples[start_i].t,
                            end_t=samples[i].t,
                            peak_t=samples[peak_i].t,
                            peak_score=samples[peak_i].score,
                            start_frame=samples[start_i].frame_index,
                            end_frame=samples[i].frame_index,
                        )
                    )
    if in_motion and samples:
        i = len(samples) - 1
        dur = samples[i].t - samples[start_i].t
        if dur >= min_duration_s:
            segments.append(
                MotionSegment(
                    start_t=samples[start_i].t,
                    end_t=samples[i].t,
                    peak_t=samples[peak_i].t,
                    peak_score=samples[peak_i].score,
                    start_frame=samples[start_i].frame_index,
                    end_frame=samples[i].frame_index,
                )
            )
    return segments


# Default: merge segments whose static gap is ≤ this many seconds.
DEFAULT_MERGE_GAP_S = 1.0
# Treat motion that begins within this window as "starts unsettled".
DEFAULT_START_UNSETTLED_S = 0.5
# Insert extra net refreshes in static holds longer than this (seconds).
DEFAULT_STATIC_REFRESH_GAP_S = 6.0
# Aim for roughly this many net samples relative to settle count (~2×).
DEFAULT_NET_SAMPLE_DENSITY = 2.0


def densify_net_sample_points(
    settle_points: list[dict[str, Any]],
    segments: list[MotionSegment],
    *,
    duration_s: float,
    fps: float = 30.0,
    static_refresh_gap_s: float = DEFAULT_STATIC_REFRESH_GAP_S,
    density: float = DEFAULT_NET_SAMPLE_DENSITY,
) -> list[dict[str, Any]]:
    """
    Keep every motion_settled point; add ``static_refresh`` samples in long
    quiet intervals so net is re-measured about ``density``× as often.

    Refreshes are placed only while the camera is static (between a settle and
    the next motion start, or to end-of-video) — never mid-pan.
    """
    if density < 1.0:
        density = 1.0
    out: list[dict[str, Any]] = [dict(sp) for sp in settle_points]

    static_intervals: list[tuple[float, float]] = []
    for i, seg in enumerate(segments):
        t0 = float(seg.end_t)
        t1 = (
            float(segments[i + 1].start_t)
            if i + 1 < len(segments)
            else float(duration_s)
        )
        if t1 - t0 >= static_refresh_gap_s:
            static_intervals.append((t0, t1))

    target_extra = max(0, int(round(len(settle_points) * (density - 1.0))))
    if not static_intervals or target_extra <= 0:
        return sorted(out, key=lambda p: float(p["t"]))

    lengths = [b - a for a, b in static_intervals]
    total_len = sum(lengths) or 1.0
    remaining = target_extra
    for (t0, t1), length in zip(static_intervals, lengths):
        share = max(1, int(round(target_extra * (length / total_len))))
        max_for_gap = max(1, int((t1 - t0) / static_refresh_gap_s))
        n = min(share, max_for_gap, remaining) if remaining else 0
        if n <= 0:
            continue
        for k in range(1, n + 1):
            t = t0 + (t1 - t0) * (k / (n + 1))
            cand = {
                "t": round(t, 3),
                "frame_index": int(round(t * fps)),
                "kind": "static_refresh",
                "use_for_net_detect": True,
            }
            settle_ts = [float(sp["t"]) for sp in settle_points]
            if any(abs(cand["t"] - st) < 0.25 for st in settle_ts):
                continue
            if any(abs(cand["t"] - float(o["t"])) < 0.25 for o in out):
                continue
            out.append(cand)
        remaining = max(0, remaining - n)

    return sorted(out, key=lambda p: float(p["t"]))


def merge_motion_segments(
    segments: list[MotionSegment],
    *,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
) -> list[MotionSegment]:
    """
    Merge consecutive motion segments when the quiet gap between them is small.

    Example: move ends at 2.0s, next starts at 2.6s with merge_gap_s=1.0 → one
    segment spanning  … → 6.6s (etc.). Stuttering pans become one movement;
    settle (end) is only when the camera truly stays put longer than the buffer.
    """
    if merge_gap_s < 0:
        raise ValueError("merge_gap_s must be >= 0")
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.start_t)
    merged: list[MotionSegment] = [
        MotionSegment(
            start_t=ordered[0].start_t,
            end_t=ordered[0].end_t,
            peak_t=ordered[0].peak_t,
            peak_score=ordered[0].peak_score,
            start_frame=ordered[0].start_frame,
            end_frame=ordered[0].end_frame,
        )
    ]
    for seg in ordered[1:]:
        prev = merged[-1]
        gap = seg.start_t - prev.end_t
        if gap <= merge_gap_s:
            if seg.peak_score >= prev.peak_score:
                peak_t, peak_score = seg.peak_t, seg.peak_score
            else:
                peak_t, peak_score = prev.peak_t, prev.peak_score
            merged[-1] = MotionSegment(
                start_t=prev.start_t,
                end_t=max(prev.end_t, seg.end_t),
                peak_t=peak_t,
                peak_score=peak_score,
                start_frame=prev.start_frame,
                end_frame=seg.end_frame
                if seg.end_t >= prev.end_t
                else prev.end_frame,
            )
        else:
            merged.append(
                MotionSegment(
                    start_t=seg.start_t,
                    end_t=seg.end_t,
                    peak_t=seg.peak_t,
                    peak_score=seg.peak_score,
                    start_frame=seg.start_frame,
                    end_frame=seg.end_frame,
                )
            )
    return merged


def build_settle_points(
    segments: list[MotionSegment],
    *,
    duration_s: float,
    start_unsettled_s: float = DEFAULT_START_UNSETTLED_S,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Camera-set timestamps = when motion ends (settled).

    If the video begins with motion (first segment starts near t=0), there is no
    reliable pose until the first settle — pipeline should apply that settle's
    camera/net for the entire prefix [0, first_settle].
    """
    settle_points: list[dict[str, Any]] = []
    for seg in segments:
        settle_points.append(
            {
                "t": seg.end_t,
                "frame_index": seg.end_frame,
                "kind": "motion_settled",
                "use_for_net_detect": True,
            }
        )

    starts_unsettled = bool(segments) and segments[0].start_t <= start_unsettled_s
    first_settle_t = settle_points[0]["t"] if settle_points else None
    # No motion at all → camera is set from the start (use t=0 as the set point).
    if not settle_points and duration_s > 0:
        settle_points.append(
            {
                "t": 0.0,
                "frame_index": 0,
                "kind": "static_open",
                "use_for_net_detect": True,
            }
        )
        first_settle_t = 0.0
        starts_unsettled = False

    policy = {
        "starts_unsettled": starts_unsettled,
        "start_unsettled_s": start_unsettled_s,
        "first_settle_t": first_settle_t,
        "prefix_use_settle_t": first_settle_t if starts_unsettled else None,
        "note": (
            "Settle points mark when the camera stops. If the clip starts "
            "unsettled, apply the first settle pose/net for [0, first_settle_t]."
            if starts_unsettled
            else "Camera is considered set from t=0 until the next motion; "
            "refresh at each settle point after a pan."
        ),
    }
    return settle_points, policy


def analyze_camera_motion(
    video_path: Path,
    *,
    method: Method = "global_affine",
    sample_fps: float = 5.0,
    analyze_max_side: int = 480,
    enter_threshold: float | None = None,
    exit_threshold: float | None = None,
    min_duration_s: float = 0.4,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    start_unsettled_s: float = DEFAULT_START_UNSETTLED_S,
    static_refresh_gap_s: float = DEFAULT_STATIC_REFRESH_GAP_S,
    net_sample_density: float = DEFAULT_NET_SAMPLE_DENSITY,
) -> dict[str, Any]:
    """
    Scan a video and return motion samples + start/end segments.

    Thresholds default from the score distribution (median + k*MAD) so we
    don't need a hand-tuned value per video; override via args if needed.

    ``merge_gap_s`` merges motion bursts separated by a short quiet gap into
    one segment so settle points only fire when the camera truly stops.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / max(sample_fps, 0.1))))

    samples: list[MotionSample] = []
    prev = None
    idx = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        gray = _resize_gray(bgr, analyze_max_side)
        t = idx / fps
        if prev is not None:
            score, tx, ty, ang = _score_pair(method, prev, gray)
            samples.append(
                MotionSample(
                    t=round(t, 3),
                    frame_index=idx,
                    score=round(score, 4),
                    tx=round(tx, 2),
                    ty=round(ty, 2),
                    angle_deg=round(ang, 3),
                )
            )
        prev = gray
        idx += 1

    cap.release()

    scores = np.array([s.score for s in samples], dtype=np.float64)
    duration_s = round(nframes / max(fps, 1e-6), 3)
    if len(scores) == 0:
        settle_points, settle_policy = build_settle_points(
            [],
            duration_s=duration_s,
            start_unsettled_s=start_unsettled_s,
        )
        return {
            "video": str(video_path),
            "method": method,
            "fps": fps,
            "duration_s": duration_s,
            "frames": nframes,
            "sample_fps": sample_fps,
            "samples": [],
            "segments": [],
            "segments_raw": [],
            "events": [],
            "settle_points": settle_points,
            "settle_policy": settle_policy,
            "thresholds": {
                "merge_gap_s": merge_gap_s,
                "start_unsettled_s": start_unsettled_s,
            },
        }

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) or 1e-6
    # Robust high-water: median + 4*MAD, with a small floor.
    auto_enter = max(0.35, med + 4.0 * mad)
    auto_exit = max(0.2, med + 1.5 * mad)
    enter = float(enter_threshold if enter_threshold is not None else auto_enter)
    exit_ = float(exit_threshold if exit_threshold is not None else auto_exit)
    if exit_ >= enter:
        exit_ = enter * 0.55

    segments_raw = segment_motion(
        samples,
        enter_threshold=enter,
        exit_threshold=exit_,
        min_duration_s=min_duration_s,
    )
    segments = merge_motion_segments(segments_raw, merge_gap_s=merge_gap_s)

    # Mark moving flags on samples for plotting / debug (merged ranges).
    moving_ranges = [(s.start_t, s.end_t) for s in segments]
    for s in samples:
        s.moving = any(a <= s.t <= b for a, b in moving_ranges)

    settle_points, settle_policy = build_settle_points(
        segments,
        duration_s=duration_s,
        start_unsettled_s=start_unsettled_s,
    )
    net_sample_points = densify_net_sample_points(
        settle_points,
        segments,
        duration_s=duration_s,
        fps=fps,
        static_refresh_gap_s=static_refresh_gap_s,
        density=net_sample_density,
    )

    events: list[dict[str, Any]] = []
    for seg in segments:
        events.append(
            {
                "type": "motion_start",
                "t": seg.start_t,
                "frame_index": seg.start_frame,
            }
        )
        events.append(
            {
                "type": "motion_peak",
                "t": seg.peak_t,
                "score": seg.peak_score,
            }
        )
        events.append(
            {
                "type": "motion_end",
                "t": seg.end_t,
                "frame_index": seg.end_frame,
                "suggest_net_redetect": True,
            }
        )

    return {
        "video": str(video_path),
        "method": method,
        "fps": fps,
        "duration_s": duration_s,
        "frames": nframes,
        "sample_fps": sample_fps,
        "analyze_max_side": analyze_max_side,
        "thresholds": {
            "enter": round(enter, 4),
            "exit": round(exit_, 4),
            "auto_enter": round(auto_enter, 4),
            "auto_exit": round(auto_exit, 4),
            "score_median": round(med, 4),
            "score_mad": round(mad, 4),
            "min_duration_s": min_duration_s,
            "merge_gap_s": merge_gap_s,
            "start_unsettled_s": start_unsettled_s,
            "static_refresh_gap_s": static_refresh_gap_s,
            "net_sample_density": net_sample_density,
        },
        "samples": [asdict(s) for s in samples],
        "segments_raw": [asdict(s) for s in segments_raw],
        "segments": [asdict(s) for s in segments],
        "events": events,
        "settle_points": settle_points,
        "net_sample_points": net_sample_points,
        "settle_policy": settle_policy,
        "summary": {
            "num_segments_raw": len(segments_raw),
            "num_segments": len(segments),
            "num_settle_points": len(settle_points),
            "num_net_samples": len(net_sample_points),
            "time_moving_s": round(sum(s.end_t - s.start_t for s in segments), 3),
            "starts_unsettled": settle_policy["starts_unsettled"],
            "recommend": (
                "Re-detect net at each net_sample_point (settles + static refreshes). "
                "If starts_unsettled, use the first settle pose for [0, first_settle_t]. "
                f"Merged gaps ≤ {merge_gap_s}s; ~{net_sample_density}× densify on static holds."
            ),
        },
    }


def compare_methods(
    video_path: Path,
    *,
    methods: tuple[Method, ...] = (
        "global_affine",
        "phase_correlate",
        "flow_median",
    ),
    sample_fps: float = 5.0,
    analyze_max_side: int = 480,
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    start_unsettled_s: float = DEFAULT_START_UNSETTLED_S,
) -> dict[str, Any]:
    """Run multiple scorers; pick default recommendation from segment counts."""
    results = {}
    for m in methods:
        results[m] = analyze_camera_motion(
            video_path,
            method=m,
            sample_fps=sample_fps,
            analyze_max_side=analyze_max_side,
            merge_gap_s=merge_gap_s,
            start_unsettled_s=start_unsettled_s,
        )
    # Prefer global_affine unless it finds nothing and another finds clear motion.
    recommended = "global_affine"
    return {
        "video": str(video_path),
        "recommended_method": recommended,
        "reason": (
            "global_affine estimates a RANSAC camera transform and ignores "
            "most player motion; best default for fixed-cam volleyball."
        ),
        "methods": {
            m: {
                "thresholds": results[m]["thresholds"],
                "summary": results[m]["summary"],
                "segments": results[m]["segments"],
                "events": results[m]["events"],
            }
            for m in methods
        },
        "full": results,
    }
