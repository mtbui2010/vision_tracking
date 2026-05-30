"""Run a fine-tuned YOLO over a MOT17 sequence's frames and emit a MOT-format
det.txt that the eval / benchmark scripts can consume.

Lets us swap the provided (FRCNN) detections out for our own and re-run the
full leaderboard — that is the "fine-tune effect" experiment.

Usage:
    python scripts/infer_detections.py \\
        --weights runs/mot17/yolov8n_10ep/weights/best.pt \\
        --frames datasets/MOT17/train/MOT17-09-FRCNN/img1 \\
        --out datasets/MOT17_yolo_dets/MOT17-09-FRCNN/det/det.txt \\
        --img 640 \\
        --score-threshold 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--class-id", type=int, default=0)
    args = parser.parse_args()

    import cv2  # type: ignore[import-not-found]
    from ultralytics import YOLO  # type: ignore[import-not-found]

    model = YOLO(args.weights)
    frame_paths = sorted(args.frames.glob("*.jpg")) or sorted(args.frames.glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"no frames under {args.frames}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, fp in enumerate(frame_paths, start=1):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        result = model(frame, imgsz=args.img, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, c in zip(xyxy, conf, cls):
            if c != args.class_id or score < args.score_threshold:
                continue
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            lines.append(f"{i},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1")
        if i % 30 == 0:
            print(f"  frame {i}/{len(frame_paths)}", flush=True)

    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out} ({len(lines)} detections over {len(frame_paths)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
