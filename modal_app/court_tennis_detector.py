"""
TennisCourtDetector (yastrebksv) inference → normalized volleyball schema.

Architecture: TrackNet-style heatmap net, 15 channels (14 kpts + center),
input 640×360. Weights: Google Drive pretrained from upstream README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from court_normalize import (  # type: ignore
    normalize_from_tennis14,
    pack_result,
)

DEFAULT_MODEL_PATH = Path("/models/tennis_court_detector.pth")
MODEL_W = 640
MODEL_H = 360
# Upstream postprocess uses scale=2 → coords in 1280×720 space.
REF_W = 1280
REF_H = 720


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pad: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                bias=bias,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, out_channels: int = 15):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x


def _heatmap_peak(
    heatmap: np.ndarray,
    *,
    low_thresh: int = 170,
    min_radius: int = 10,
    max_radius: int = 25,
) -> tuple[float | None, float | None, float]:
    """Return (x, y in 1280×720 space, peak_conf 0..1)."""
    hm_u8 = (heatmap * 255).astype(np.uint8)
    peak = float(heatmap.max()) if heatmap.size else 0.0
    _ret, binary = cv2.threshold(hm_u8, low_thresh, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        # Fallback: argmax if peak is strong enough
        if peak < 0.35:
            return None, None, peak
        y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        return float(x * 2), float(y * 2), peak
    x_pred = float(circles[0][0][0] * 2)
    y_pred = float(circles[0][0][1] * 2)
    return x_pred, y_pred, peak


def _load_model(model_path: Path, device: str) -> BallTrackerNet:
    model = BallTrackerNet(out_channels=15)
    state = torch.load(str(model_path), map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # Some checkpoints wrap weights under a key.
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            raise
    model.to(device)
    model.eval()
    return model


def detect_tennis_court_image(
    frame_bgr: np.ndarray,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    video_id: str = "",
    pipeline_version: str = "0.1.0",
    return_overlay: bool = True,
    low_thresh: int = 170,
) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Tennis court model not found: {model_path}")

    h, w = frame_bgr.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(model_path, device)

    img = cv2.resize(frame_bgr, (MODEL_W, MODEL_H))
    inp = (img.astype(np.float32) / 255.0)
    inp_t = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(inp_t)[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

    # Scale from reference 1280×720 → original frame
    sx = w / float(REF_W)
    sy = h / float(REF_H)

    pts: list[list[float] | None] = []
    confs: list[float] = []
    # Also try a lower Hough threshold for weak heatmaps on non-tennis domains.
    for kps_num in range(14):
        x_ref, y_ref, peak = _heatmap_peak(pred[kps_num], low_thresh=low_thresh)
        if x_ref is None or y_ref is None:
            x_ref, y_ref, peak = _heatmap_peak(
                pred[kps_num],
                low_thresh=120,
                min_radius=5,
                max_radius=40,
            )
        if x_ref is None or y_ref is None:
            pts.append(None)
            confs.append(0.0)
            continue
        pts.append([x_ref * sx, y_ref * sy])
        confs.append(float(peak))

    keypoints, raw = normalize_from_tennis14(pts, confs)
    box_conf = float(np.mean([c for c in confs if c > 0]) if any(confs) else 0.0)

    return pack_result(
        model_id="tennis_court_detector",
        model_name="BallTrackerNet_heatmap15",
        model_repo="yastrebksv/TennisCourtDetector",
        video_id=video_id,
        pipeline_version=pipeline_version,
        image_size={"width": w, "height": h},
        keypoints=keypoints,
        raw_keypoints=raw,
        box_conf=box_conf,
        frame=frame_bgr,
        return_overlay=return_overlay,
        note=(
            "Trained on tennis courts; mapped to volleyball_court_v1 "
            "(corners/service≈attack/net≈sideline mid). Expect domain shift."
        ),
    )
