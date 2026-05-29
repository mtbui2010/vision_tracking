"""Fine-tune YOLOv8/v11 on the MOT-derived dataset.

Wraps `ultralytics.YOLO.train` so the recipe is reproducible. Run after
`scripts/prepare_dataset.py`.

Usage:
    python scripts/train_yolo.py \
        --data datasets/MOT17_yolo/data.yaml \
        --weights yolov11n.pt \
        --epochs 30 --img 640
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--weights", default="yolov11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/tracker_lab")
    parser.add_argument("--name", default="mot17_yolov11n")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO  # type: ignore[import-not-found]

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        cos_lr=True,
    )

    if args.export_onnx:
        best = Path(args.project) / args.name / "weights" / "best.pt"
        model = YOLO(str(best))
        onnx_path = model.export(format="onnx", dynamic=True, simplify=True)
        print(f"exported ONNX -> {onnx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
