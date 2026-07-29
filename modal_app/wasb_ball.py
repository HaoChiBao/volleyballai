"""
WASB volleyball ball tracker (BMVC2023).

Uses pretrained HRNet weights from:
  https://github.com/nttcom/WASB-SBDT  (MIT)
  wasb_volleyball_best.pth.tar

Outputs the same frame schema as VballNet / YOLO: [{t, xy, r}, ...].
No gap-fill — raw detector+online-tracker visibility only.
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_NAME = "WASB-HRNet-volleyball"
MODEL_SOURCE = "wasb_sbdt"
DEFAULT_MODEL_PATH = Path("/models/wasb_volleyball_best.pth.tar")
DEFAULT_WASB_SRC = Path("/opt/wasb-sbdt/src")

INPUT_WIDTH = 512
INPUT_HEIGHT = 288
FRAMES_IN = 3
FRAMES_OUT = 3
SCORE_THRESHOLD = 0.5
MAX_DISP = 300.0
RADIUS_MIN = 3.0
RADIUS_MAX = 48.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Paper's "step=1" oversampling (best AP). step=3 is ~3× faster.
DEFAULT_STEP = 1

# Minimal HRNet config matching configs/model/wasb.yaml
_WASB_MODEL_CFG: dict[str, Any] = {
    "name": "hrnet",
    "frames_in": FRAMES_IN,
    "frames_out": FRAMES_OUT,
    "inp_height": INPUT_HEIGHT,
    "inp_width": INPUT_WIDTH,
    "out_height": INPUT_HEIGHT,
    "out_width": INPUT_WIDTH,
    "rgb_diff": False,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [1],
                "NUM_CHANNELS": [32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2],
                "NUM_CHANNELS": [16, 32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64],
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64, 128],
                "FUSE_METHOD": "SUM",
            },
            "DECONV": {
                "NUM_DECONVS": 0,
                "KERNEL_SIZE": [],
                "NUM_BASIC_BLOCKS": 2,
            },
            "INIT_WEIGHTS": True,
        },
    },
}


def _ensure_wasb_src(src_root: Path | None = None) -> Path:
    root = Path(src_root or os.environ.get("WASB_SRC", str(DEFAULT_WASB_SRC)))
    if not (root / "models" / "hrnet.py").exists():
        raise FileNotFoundError(
            f"WASB source not found at {root} (expected models/hrnet.py)",
        )
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def _get_dir(src_point: list[float], rot_rad: float) -> list[float]:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [
        src_point[0] * cs - src_point[1] * sn,
        src_point[0] * sn + src_point[1] * cs,
    ]


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
    center: np.ndarray,
    scale: float | np.ndarray,
    rot: float,
    output_size: list[int],
    *,
    inv: int = 0,
) -> np.ndarray:
    """Microsoft / CenterNet affine helper (same as WASB utils.image)."""
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale
    src_w = float(scale_tmp[0])
    dst_w = float(output_size[0])
    dst_h = float(output_size[1])
    rot_rad = np.pi * rot / 180.0
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + np.array(src_dir, dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def affine_transform(pt: np.ndarray, t: np.ndarray) -> np.ndarray:
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    return np.dot(t, new_pt)[:2]


def frame_affine(h: int, w: int, *, inv: int = 0) -> np.ndarray:
    c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    s = float(max(h, w))
    return get_affine_transform(c, s, 0, [INPUT_WIDTH, INPUT_HEIGHT], inv=inv)


def _decode_heatmap(
    hm: np.ndarray,
    *,
    inv_trans: np.ndarray,
    threshold: float,
    scale_r: float,
) -> list[dict[str, Any]]:
    """Connected-component decode (WASB TracknetV2Postprocessor, use_hm_weight)."""
    hm = np.asarray(hm, dtype=np.float32)
    if float(np.max(hm)) <= threshold:
        return []
    _, binary = cv2.threshold(hm, threshold, 1.0, cv2.THRESH_BINARY)
    n_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))
    dets: list[dict[str, Any]] = []
    for m in range(1, n_labels):
        ys, xs = np.where(labels == m)
        if xs.size == 0:
            continue
        ws = hm[ys, xs]
        score = float(ws.sum())
        x = float(np.sum(xs.astype(np.float32) * ws) / np.sum(ws))
        y = float(np.sum(ys.astype(np.float32) * ws) / np.sum(ws))
        xy = affine_transform(np.array([x, y], dtype=np.float32), inv_trans)
        area = float(xs.size)
        r = float(np.clip(math.sqrt(area / math.pi) * scale_r, RADIUS_MIN, RADIUS_MAX))
        dets.append({"xy": xy, "score": score, "r": r})
    return dets


class _OnlineTracker:
    """Minimal port of WASB trackers.online.OnlineTracker."""

    def __init__(self, max_disp: float = MAX_DISP) -> None:
        self._max_disp = float(max_disp)
        self._fid = 0
        self._xy: dict[int, np.ndarray] = {}
        self._visi: dict[int, bool] = {}

    def _select_not_too_far(self, dets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._fid == 0 or not self._visi.get(self._fid - 1, False):
            return dets
        prev = self._xy[self._fid - 1]
        return [
            d
            for d in dets
            if float(np.linalg.norm(d["xy"] - prev)) < self._max_disp
        ]

    def update(self, dets: list[dict[str, Any]]) -> dict[str, Any]:
        dets = self._select_not_too_far(dets)
        best_score = -np.inf
        x = y = -np.inf
        visi = False
        r = RADIUS_MIN
        for det in dets:
            score = float(det["score"])
            if score > best_score:
                best_score = score
                x, y = float(det["xy"][0]), float(det["xy"][1])
                r = float(det.get("r") or RADIUS_MIN)
                visi = True
        self._xy[self._fid] = np.array([x, y], dtype=np.float32)
        self._visi[self._fid] = visi
        self._fid += 1
        return {
            "x": x,
            "y": y,
            "visi": visi,
            "score": best_score if visi else 0.0,
            "r": r,
        }


def _load_model(model_path: Path, device: str):
    from omegaconf import OmegaConf
    from models.hrnet import HRNet  # type: ignore  # noqa: PLC0415

    import torch

    cfg = OmegaConf.create(_WASB_MODEL_CFG)
    model = HRNet(cfg)
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected WASB checkpoint format: {model_path}")
    cleaned = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state.items()
    }
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


def _preprocess_bgr(frame_bgr: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """BGR uint8 → CHW float32 ImageNet-normalized, after affine warp."""
    warped = cv2.warpAffine(
        frame_bgr,
        trans,
        (INPUT_WIDTH, INPUT_HEIGHT),
        flags=cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(rgb, (2, 0, 1))  # CHW


def track_ball_wasb(
    video_path: str | Path,
    *,
    model_path: Path | None = None,
    wasb_src: Path | None = None,
    score_threshold: float = SCORE_THRESHOLD,
    step: int = DEFAULT_STEP,
    max_disp: float = MAX_DISP,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run WASB HRNet + online tracker on a video.

    step=1: oversample overlapping 3-frame windows (paper best quality)
    step=3: non-overlapping windows (faster)
    """
    import torch

    _ensure_wasb_src(wasb_src)
    path = Path(video_path)
    weights = Path(model_path or os.environ.get("WASB_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not weights.exists():
        raise FileNotFoundError(f"WASB weights not found: {weights}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[wasb] WARNING: running on CPU (slow)", flush=True)

    step = max(1, int(step))
    model = _load_model(weights, device)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1e-3:
        fps = 30.0
    frames_bgr: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()

    n = len(frames_bgr)
    if n == 0:
        return []

    h0, w0 = frames_bgr[0].shape[:2]
    trans = frame_affine(h0, w0, inv=0)
    inv_trans = frame_affine(h0, w0, inv=1)
    scale_r = float(max(h0, w0)) / float(max(INPUT_HEIGHT, INPUT_WIDTH))

    processed = [_preprocess_bgr(f, trans) for f in frames_bgr]
    frame_dets: dict[int, list[dict[str, Any]]] = defaultdict(list)

    print(
        f"[wasb] frames={n} step={step} device={device} "
        f"size={w0}x{h0} model={weights.name}",
        flush=True,
    )

    with torch.no_grad():
        for start in range(0, max(1, n - FRAMES_IN + 1), step):
            idxs = [min(n - 1, start + k) for k in range(FRAMES_IN)]
            tensor = np.concatenate([processed[i] for i in idxs], axis=0)[
                None, ...
            ].astype(np.float32)
            inp = torch.from_numpy(tensor).to(device)
            preds = model(inp)
            hm = preds[0].sigmoid().detach().cpu().numpy()[0]  # [frames_out,H,W]
            for j in range(min(FRAMES_OUT, hm.shape[0])):
                fid = idxs[j]
                dets = _decode_heatmap(
                    hm[j],
                    inv_trans=inv_trans,
                    threshold=score_threshold,
                    scale_r=scale_r,
                )
                frame_dets[fid].extend(dets)

    tracker = _OnlineTracker(max_disp=max_disp)
    out: list[dict[str, Any]] = []
    for fid in range(n):
        result = tracker.update(frame_dets.get(fid, []))
        if not result["visi"]:
            continue
        out.append(
            {
                "t": round(fid / fps, 3),
                "xy": [round(float(result["x"]), 1), round(float(result["y"]), 1)],
                "r": round(float(result["r"]), 1),
                "score": round(float(result["score"]), 3),
            },
        )

    print(f"[wasb] detections={len(out)} / frames={n}", flush=True)
    return out
