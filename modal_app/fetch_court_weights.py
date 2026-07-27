"""Download extra court weights into /models (runs on Modal image build / fetch fn)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MODELS = Path(os.environ.get("COURT_MODELS_DIR", "/models"))
VOLUME = Path(os.environ.get("COURT_EXTRA_DIR", "/vol/court-extra"))
TENNIS_ID = "1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"
KAGGLE_HANDLE = "pythonistasamurai/yolov8x_volleyball_analysis_models/PyTorch/default"


def fetch_tennis(dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[court-models] tennis already present ({dest.stat().st_size} bytes)")
        return
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={TENNIS_ID}"
    print("[court-models] downloading TennisCourtDetector…")
    gdown.download(url, str(dest), quiet=False)
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        raise RuntimeError(f"Tennis download failed: {dest}")
    print(f"[court-models] tennis ok ({dest.stat().st_size} bytes)")


def fetch_kaggle(dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[court-models] kaggle already present ({dest.stat().st_size} bytes)")
        return True
    import kagglehub

    user = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if user and key:
        print("[court-models] using Kaggle credentials from env")
    else:
        print("[court-models] trying unauthenticated Kaggle download (public model)…")

    print("[court-models] downloading Kaggle yolov8x court model…")
    try:
        path = Path(kagglehub.model_download(KAGGLE_HANDLE))
    except Exception as e:  # noqa: BLE001
        print(f"[court-models] Kaggle download failed: {e}")
        print(
            "[court-models] HINT: create a free Kaggle API token and pass "
            "--kaggle-username / --kaggle-key to fetch_court_models_local",
        )
        return False
    candidates = list(path.rglob("key_points_regression_model.pt"))
    if not candidates:
        candidates = [p for p in path.rglob("*.pt") if "key" in p.name.lower()]
    if not candidates:
        candidates = list(path.rglob("*.pt"))
    if not candidates:
        print(f"[court-models] No .pt files under {path}")
        return False
    src = candidates[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"[court-models] kaggle ok ({dest.stat().st_size} bytes) from {src}")
    return True


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    VOLUME.mkdir(parents=True, exist_ok=True)

    # Prefer baking tennis into image /models; also mirror onto volume.
    tennis_img = MODELS / "tennis_court_detector.pth"
    tennis_vol = VOLUME / "tennis_court_detector.pth"
    try:
        fetch_tennis(tennis_img)
        if tennis_img.exists() and (
            not tennis_vol.exists()
            or tennis_vol.stat().st_size != tennis_img.stat().st_size
        ):
            shutil.copy2(tennis_img, tennis_vol)
    except Exception as e:  # noqa: BLE001
        print(f"[court-models] tennis image path failed ({e}); trying volume")
        fetch_tennis(tennis_vol)

    kaggle_dest = VOLUME / "key_points_regression_model.pt"
    # Also try /models for image-baked copies.
    ok = fetch_kaggle(kaggle_dest)
    if ok and not (MODELS / "key_points_regression_model.pt").exists():
        try:
            shutil.copy2(kaggle_dest, MODELS / "key_points_regression_model.pt")
        except OSError:
            pass

    print("[court-models] /models:", sorted(p.name for p in MODELS.iterdir()) if MODELS.exists() else [])
    print("[court-models] /vol:", sorted(p.name for p in VOLUME.iterdir()) if VOLUME.exists() else [])
    if not tennis_img.exists() and not tennis_vol.exists():
        return 1
    if not ok:
        print("[court-models] continuing without Kaggle weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
