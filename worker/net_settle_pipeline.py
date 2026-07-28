"""
Settle → net detect → FIVB PnP → calibration / 3D camera.

Uses existing camera_motion.json settle points on a video. Does NOT re-run
SAM / ball / YOLO. Writes:
  - net.tracks.json          (per-settle net + camera + H)
  - calibration.json         (updated from primary settle for 2D/3D overlays)
  - optional frame overlays under .data/… or video folder

Primary settle = first settle_point (also covers unsettled prefix).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from worker.court_calib import compute_homography
from worker.fivb_ratios import DEFAULT_FIVB, FivbIndoor
from worker.net_detect import detect_net_in_image
from worker.openai_court_outline import CORNER_ORDER
from worker.openai_net_to_court import derive_court_from_net, draw_net_court_overlay

PIPELINE_VERSION = "0.1.0"


def _extract_frame(video: Path, t: float, out_jpg: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    idx = max(0, int(round(t * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    if not ok:
        # Fall back to sequential seek near end.
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 1))
        ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame near t={t:.2f}s")
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_jpg), bgr)
    h, w = bgr.shape[:2]
    return w, h


def _H_from_ground(ground_lines: dict[str, Any], length_m: float, width_m: float) -> list[float]:
    boundary = ground_lines.get("boundary") or []
    if len(boundary) < 4:
        raise ValueError("Need 4 ground boundary corners for H")
    image_points = [{"x": float(p[0]), "y": float(p[1])} for p in boundary[:4]]
    court_points = [
        {"x": 0.0, "y": 0.0},
        {"x": length_m, "y": 0.0},
        {"x": length_m, "y": width_m},
        {"x": 0.0, "y": width_m},
    ]
    return compute_homography(image_points, court_points)


def load_camera_motion(video_dir: Path) -> dict[str, Any]:
    path = video_dir / "camera_motion.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run camera motion test first")
    return json.loads(path.read_text(encoding="utf-8"))


def run_net_settle_on_video(
    video_dir: Path,
    *,
    video_name: str = "work.mp4",
    max_side: int = 512,
    model: str | None = None,
    fivb: FivbIndoor = DEFAULT_FIVB,
    settle_limit: int | None = None,
    write_overlays: bool = True,
) -> dict[str, Any]:
    """
    Detect net at each settle point, solve FIVB PnP, write artifacts into video_dir.
    """
    video_dir = Path(video_dir)
    video_path = video_dir / video_name
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    motion = load_camera_motion(video_dir)
    video_id = motion.get("video_id") or video_dir.name
    settle_points = list(
        motion.get("net_sample_points")
        or motion.get("settle_points")
        or []
    )
    if not settle_points:
        # Back-compat: derive from motion_end
        settle_points = [
            {
                "t": ev["t"],
                "frame_index": ev.get("frame_index"),
                "kind": "motion_settled",
                "use_for_net_detect": True,
            }
            for ev in motion.get("events") or []
            if ev.get("type") == "motion_end"
        ]
    if settle_limit is not None:
        settle_points = settle_points[: max(0, settle_limit)]
    if not settle_points:
        raise RuntimeError("No settle / net sample points in camera_motion.json")

    settle_policy = motion.get("settle_policy") or {}
    frames_out: list[dict[str, Any]] = []
    overlay_dir = video_dir / "net_settle_overlays"
    if write_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    stills_dir = video_dir / "net_settle_frames"
    stills_dir.mkdir(parents=True, exist_ok=True)

    for i, sp in enumerate(settle_points):
        t = float(sp["t"])
        kind = sp.get("kind", "motion_settled")
        still = stills_dir / f"sample_{i:02d}_t{t:.2f}.jpg"
        print(
            f"[net-settle] {i + 1}/{len(settle_points)} t={t:.2f}s ({kind}) extract…",
            flush=True,
        )
        w, h = _extract_frame(video_path, t, still)

        print(f"[net-settle] detect net max_side={max_side}…", flush=True)
        det = detect_net_in_image(still, model=model, t=t, max_side=max_side)
        net = det["net"]

        print("[net-settle] PnP + FIVB ratios…", flush=True)
        geom = derive_court_from_net(
            net,
            image_width=w,
            image_height=h,
            length_m=fivb.length_m,
            width_m=fivb.width_m,
            net_height_m=fivb.net_height_m,
            net_depth_m=fivb.net_depth_m,
        )
        H = _H_from_ground(
            geom["ground_lines"], fivb.length_m, fivb.width_m
        )

        frame_rec: dict[str, Any] = {
            "t": t,
            "frame_index": sp.get("frame_index"),
            "trigger": "static_refresh" if kind == "static_refresh" else "settle",
            "kind": kind,
            "net": net,
            "camera": geom["camera"],
            "H": H,
            "reproj_err_px": geom.get("reproj_err_px"),
            "score": geom.get("score"),
            "mapping": geom.get("mapping"),
            "net_depth_m": (geom.get("court") or {}).get("net_depth_m"),
            "ground_lines": geom.get("ground_lines"),
            "model": det.get("model"),
            "max_side": max_side,
            "degraded": det.get("degraded"),
            "usage": det.get("usage"),
        }
        frames_out.append(frame_rec)

        if write_overlays:
            overlay_path = overlay_dir / f"sample_{i:02d}_t{t:.2f}_overlay.jpg"
            draw_net_court_overlay(
                still,
                net_labeled=net,
                geometry=geom,
                out_path=overlay_path,
            )
            print(f"[net-settle] wrote {overlay_path.name}", flush=True)

    # Primary = first settle (covers unsettled prefix per settle_policy).
    primary = frames_out[0]
    tracks = {
        "video_id": video_id,
        "pipeline_version": PIPELINE_VERSION,
        "source": "openai_net_settle",
        "model": primary.get("model"),
        "max_side": max_side,
        "fivb": fivb.ratios(),
        "settle_policy": settle_policy,
        "primary_t": primary["t"],
        "frames": frames_out,
        "summary": {
            "num_settles": len(frames_out),
            "primary_t": primary["t"],
            "primary_score": primary.get("score"),
            "primary_reproj_err_px": primary.get("reproj_err_px"),
            "starts_unsettled": settle_policy.get("starts_unsettled"),
        },
    }
    tracks_path = video_dir / "net.tracks.json"
    # Slim file for the web: drop bulky ground_lines from every frame in API? Keep them —
    # useful for overlays. They're small.
    tracks_path.write_text(json.dumps(tracks, indent=2), encoding="utf-8")
    print(f"[net-settle] wrote {tracks_path}", flush=True)

    calibration = {
        "video_id": video_id,
        "pipeline_version": PIPELINE_VERSION,
        "court": {
            "length_m": fivb.length_m,
            "width_m": fivb.width_m,
        },
        "source": "net_settle",
        "from_run_id": None,
        "keyframes": [
            {
                "t": fr["t"],
                "image_points": [
                    {
                        "x": fr["ground_lines"]["boundary"][j][0],
                        "y": fr["ground_lines"]["boundary"][j][1],
                    }
                    for j in range(4)
                ],
                "court_points_m": [
                    {"x": 0.0, "y": 0.0},
                    {"x": fivb.length_m, "y": 0.0},
                    {"x": fivb.length_m, "y": fivb.width_m},
                    {"x": 0.0, "y": fivb.width_m},
                ],
            }
            for fr in frames_out
        ],
        "H": primary["H"],
        "camera": primary["camera"],
        "segments": [],
        "net_settle": {
            "primary_t": primary["t"],
            "num_frames": len(frames_out),
            "settle_policy": settle_policy,
        },
    }
    duration = float(motion.get("duration_s") or frames_out[-1]["t"])
    # Segment i: pose from frames_out[i] applies from settle_i (or 0 for first)
    # until the next settle. Unsettled prefix uses first settle pose from t=0.
    segs: list[dict[str, Any]] = []
    for i, fr in enumerate(frames_out):
        t0 = 0.0 if i == 0 else fr["t"]
        t1 = frames_out[i + 1]["t"] if i + 1 < len(frames_out) else duration
        segs.append(
            {
                "t0": t0,
                "t1": t1,
                "keyframe_index": i,
                "settle_t": fr["t"],
            }
        )
    calibration["segments"] = segs

    cal_path = video_dir / "calibration.json"
    # Backup previous auto/manual cal once.
    backup = video_dir / "calibration.before_net_settle.json"
    if cal_path.exists() and not backup.exists():
        backup.write_text(cal_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[net-settle] backed up prior calibration → {backup.name}", flush=True)
    cal_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(f"[net-settle] wrote {cal_path}", flush=True)

    # Patch court3d.json camera if present (matched view without regenerating tracks).
    court3d_path = video_dir / "court3d.json"
    if court3d_path.exists():
        try:
            court3d = json.loads(court3d_path.read_text(encoding="utf-8"))
            court3d["camera"] = primary["camera"]
            court3d["court"] = {
                "length_m": fivb.length_m,
                "width_m": fivb.width_m,
            }
            court3d["source_camera"] = "net_settle"
            court3d_path.write_text(json.dumps(court3d), encoding="utf-8")
            print(f"[net-settle] patched camera on {court3d_path.name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[net-settle] skip court3d patch: {e}", flush=True)

    return tracks
