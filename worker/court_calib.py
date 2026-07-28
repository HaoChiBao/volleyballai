"""
Build calibration (H + camera) from Modal court.keypoints.json.

Mirrors apps/web/lib/courtFromKeypoints.ts so the worker can auto-calibrate
without calling the Next.js API.
"""

from __future__ import annotations

import math
from typing import Any


PREFERRED_NAMES = (
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
)

CORNER_NAMES = (
    "corner_top_left",
    "corner_top_right",
    "corner_bottom_right",
    "corner_bottom_left",
)


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("Singular homography system")
        M[col], M[pivot] = M[pivot], M[col]
        div = M[col][col]
        for c in range(col, n + 1):
            M[col][c] /= div
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    return [row[n] for row in M]


def compute_homography(
    image_points: list[dict[str, float]],
    court_points: list[dict[str, float]],
) -> list[float]:
    n = min(len(image_points), len(court_points))
    if n < 4:
        raise ValueError("Need at least 4 point pairs")
    A: list[list[float]] = []
    b: list[float] = []
    for i in range(n):
        x, y = image_points[i]["x"], image_points[i]["y"]
        X, Y = court_points[i]["x"], court_points[i]["y"]
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        b.append(X)
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        b.append(Y)
    # Use first 4 pairs → 8 equations for 8 DOF
    A8 = A[:8]
    b8 = b[:8]
    h = _solve_linear(A8, b8)
    return [*h, 1.0]


def invert_homography(H: list[float]) -> list[float]:
    a, b, c, d, e, f, g, h, i = H
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    F = -(a * h - b * g)
    G = b * f - c * e
    Hh = -(a * f - c * d)
    I = a * e - b * d
    det = a * A + b * B + c * C
    if abs(det) < 1e-12:
        raise ValueError("Non-invertible H")
    return [v / det for v in (A, D, G, B, E, Hh, C, F, I)]


def apply_homography(H: list[float], p: dict[str, float]) -> dict[str, float]:
    """Apply row-major 3×3 H to point {x,y}."""
    x, y = float(p["x"]), float(p["y"])
    denom = H[6] * x + H[7] * y + H[8]
    if abs(denom) < 1e-9:
        return {"x": x, "y": y}
    return {
        "x": (H[0] * x + H[1] * y + H[2]) / denom,
        "y": (H[3] * x + H[4] * y + H[5]) / denom,
    }


def project_world_to_image(
    pose: dict[str, Any],
    X: float,
    Y: float,
    Z: float,
) -> dict[str, float] | None:
    """Project court-world meters (X length, Y width, Z up) → image pixels."""
    R = pose["R"]
    t = pose["t"]
    xc = R[0] * X + R[1] * Y + R[2] * Z + t[0]
    yc = R[3] * X + R[4] * Y + R[5] * Z + t[1]
    zc = R[6] * X + R[7] * Y + R[8] * Z + t[2]
    if zc <= 1e-6:
        return None
    return {
        "x": pose["fx"] * xc / zc + pose["cx"],
        "y": pose["fy"] * yc / zc + pose["cy"],
    }


def _mat_mul3(A: list[float], B: list[float]) -> list[float]:
    out = [0.0] * 9
    for r in range(3):
        for c in range(3):
            out[r * 3 + c] = (
                A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c]
            )
    return out


def _mat_vec3(M: list[float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        M[0] * v[0] + M[1] * v[1] + M[2] * v[2],
        M[3] * v[0] + M[4] * v[1] + M[5] * v[2],
        M[6] * v[0] + M[7] * v[1] + M[8] * v[2],
    )


def _norm3(v: tuple[float, float, float]) -> float:
    return math.hypot(v[0], v[1], v[2])


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = _norm3(v) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def estimate_camera_from_H(
    H_image_to_court: list[float],
    image_width: int,
    image_height: int,
    court_length_m: float = 18.0,
    court_width_m: float = 9.0,
) -> dict[str, Any]:
    H_c2i = invert_homography(H_image_to_court)
    fx = image_width * 0.95
    fy = fx
    cx = image_width / 2
    cy = image_height / 2
    fov_y_deg = (2 * math.atan(image_height / (2 * fy)) * 180) / math.pi

    Kinv = [1 / fx, 0, -cx / fx, 0, 1 / fy, -cy / fy, 0, 0, 1]
    M = _mat_mul3(Kinv, H_c2i)
    m1 = (M[0], M[3], M[6])
    m2 = (M[1], M[4], M[7])
    m3 = (M[2], M[5], M[8])
    lam = 1 / (_norm3(m1) or 1e-9)
    r1 = _normalize3((m1[0] * lam, m1[1] * lam, m1[2] * lam))
    r2 = _normalize3((m2[0] * lam, m2[1] * lam, m2[2] * lam))
    r3 = _normalize3(_cross(r1, r2))
    r2 = _normalize3(_cross(r3, r1))
    R = [r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]]
    t = (m3[0] * lam, m3[1] * lam, m3[2] * lam)

    center = (court_length_m / 2, court_width_m / 2, 0.0)
    in_cam = _mat_vec3(R, center)
    if in_cam[2] + t[2] < 0:
        R = [-v for v in R]
        t = (-t[0], -t[1], -t[2])

    Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]
    C = _mat_vec3(Rt, (-t[0], -t[1], -t[2]))
    return {
        "position": [C[0], C[1], C[2]],
        "R": R,
        "t": [t[0], t[1], t[2]],
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image_width": image_width,
        "image_height": image_height,
        "fov_y_deg": fov_y_deg,
    }


def _visible_map(keypoints: list[dict[str, Any]], min_conf: float) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for kp in keypoints:
        if not kp.get("visible") or not kp.get("xy"):
            continue
        conf = float(kp.get("conf") or 0)
        if conf < min_conf:
            continue
        name = str(kp.get("name") or "")
        prev = out.get(name)
        if prev is None or conf > float(prev.get("conf") or 0):
            out[name] = kp
    return out


def _score_frame(frame: dict[str, Any], min_conf: float) -> float:
    m = _visible_map(frame.get("keypoints") or [], min_conf)
    score = float(frame.get("box_conf") or 0) * 2
    for name in PREFERRED_NAMES:
        kp = m.get(name)
        if kp:
            score += float(kp.get("conf") or 0) + (0.5 if name.startswith("corner_") else 0)
    if all(n in m for n in CORNER_NAMES):
        score += 3
    return score


def _pick_pairs(
    mmap: dict[str, dict],
    length_m: float,
    width_m: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[str]] | None:
    sx = length_m / 18.0
    sy = width_m / 9.0
    image: list[dict[str, float]] = []
    court: list[dict[str, float]] = []
    names: list[str] = []

    def add(name: str) -> None:
        if name in names:
            return
        kp = mmap.get(name)
        if not kp or not kp.get("xy") or not kp.get("court_m"):
            return
        xy = kp["xy"]
        cm = kp["court_m"]
        image.append({"x": float(xy[0]), "y": float(xy[1])})
        court.append({"x": float(cm["x"]) * sx, "y": float(cm["y"]) * sy})
        names.append(name)

    for name in CORNER_NAMES:
        add(name)
    if len(image) < 4:
        for name in PREFERRED_NAMES:
            if len(image) >= 4:
                break
            add(name)
    if len(image) < 4:
        return None
    return image[:4], court[:4], names[:4]


def calibration_from_keypoints(
    court_file: dict[str, Any],
    *,
    video_id: str,
    pipeline_version: str = "0.1.0",
    length_m: float = 18.0,
    width_m: float = 9.0,
    min_conf: float = 0.25,
    image_width: int | None = None,
    image_height: int | None = None,
    from_run_id: str | None = None,
) -> dict[str, Any] | None:
    frames = court_file.get("frames") or []
    if not frames:
        return None

    best = max(frames, key=lambda fr: _score_frame(fr, min_conf))
    mmap = _visible_map(best.get("keypoints") or [], min_conf)
    picked = _pick_pairs(mmap, length_m, width_m)
    if not picked:
        return None
    image_pts, court_pts, names = picked

    try:
        H = compute_homography(image_pts, court_pts)
    except ValueError:
        return None

    size = court_file.get("image_size") or {}
    w = int(image_width or size.get("width") or 1280)
    h = int(image_height or size.get("height") or 720)

    camera = None
    try:
        camera = estimate_camera_from_H(H, w, h, length_m, width_m)
    except ValueError:
        camera = None

    run = court_file.get("run") or {}
    return {
        "video_id": video_id,
        "pipeline_version": pipeline_version,
        "court": {"length_m": length_m, "width_m": width_m},
        "source": "auto_keypoints",
        "from_run_id": from_run_id or run.get("run_id"),
        "keyframes": [
            {
                "t": float(best.get("t") or 0),
                "image_points": image_pts,
                "court_points_m": court_pts,
            },
        ],
        "H": H,
        "camera": camera,
        "auto_keypoint_names": names,
    }
