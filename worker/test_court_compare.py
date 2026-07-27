"""
Compare court models on Modal (volley-ref / Kaggle / TennisCourtDetector).

Usage (from repo root):
  .\\.venv\\Scripts\\python.exe -m worker.test_court_compare
  .\\.venv\\Scripts\\python.exe -m worker.test_court_compare .data/court-model-test/images
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path


def _save_compare(result: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in result.items() if k != "results"}
    models_out: dict = {}
    for mid, payload in (result.get("results") or {}).items():
        models_out[mid] = {k: v for k, v in payload.items() if k != "overlays"}
        for i, ov in enumerate(payload.get("overlays") or []):
            jpg = dest / f"{mid}_overlay_{i:02d}.jpg"
            jpg.write_bytes(base64.b64decode(ov["jpg_b64"]))
            print(f"  wrote {jpg}")
    slim["results"] = models_out
    (dest / "compare.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"  wrote {dest / 'compare.json'}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    images_dir = Path(args[0] if args else ".data/court-model-test/images")
    out_root = Path(args[1] if len(args) > 1 else ".data/court-model-compare")

    if images_dir.is_file():
        images = [images_dir]
    else:
        images = sorted(
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and p.is_file()
        )
    if not images:
        print(f"No images in {images_dir}", file=sys.stderr)
        return 1

    import modal

    app_name = __import__("os").environ.get("MODAL_APP_NAME", "volleyball-ai")
    fn = modal.Function.from_name(app_name, "compare_court_models")

    summaries = []
    for img in images:
        print(f"\n==== {img.name} ====")
        result = fn.remote(
            image_bytes=img.read_bytes(),
            video_id=img.stem,
            media_suffix=img.suffix.lower() or ".jpg",
            kaggle_confidence=0.001,
            return_overlays=True,
        )
        dest = out_root / img.stem
        _save_compare(result, dest)
        row = {"image": img.name, "summary": result.get("summary"), "errors": result.get("errors")}
        summaries.append(row)
        print("  summary:", json.dumps(result.get("summary")))
        if result.get("errors"):
            print("  errors:", json.dumps(result.get("errors")))

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
