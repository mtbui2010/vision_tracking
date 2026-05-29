"""Render tracker predictions onto frames and write an MP4 demo clip.

Useful for the README and the LinkedIn / Twitter post. Reads MOT-format
detections + tracker name, plus a folder of frame images, and writes a
side-by-side or single-pane annotated video.

Usage:
    python scripts/render_demo.py \\
        --frames datasets/MOT17/train/MOT17-09-FRCNN/img1 \\
        --detections datasets/MOT17/train/MOT17-09-FRCNN/det/det.txt \\
        --tracker sort \\
        --out exports/MOT17-09_sort.mp4 \\
        --score-threshold 0.3 \\
        --fps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from services.eval_dataset import read_mot
from services.trackers.base import Detection
from services.trackers.registry import build


def _color_for(track_id: int) -> tuple[int, int, int]:
    """Deterministic distinct color per ID (HSV → BGR)."""
    import colorsys

    hue = (track_id * 0.6180339887) % 1.0  # golden ratio mod 1 → spread
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    import cv2  # type: ignore[import-not-found]

    det_seq = read_mot(args.detections)
    tracker = build(args.tracker)

    frame_paths = sorted(args.frames.glob("*.jpg")) or sorted(args.frames.glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"no frames under {args.frames}")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise SystemExit(f"cannot read first frame {frame_paths[0]}")
    h, w = first.shape[:2]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open {args.out} for writing")

    for i, frame_path in enumerate(frame_paths):
        if i >= len(det_seq.frames):
            break
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        f = det_seq.frames[i]
        scores = f.scores if f.scores is not None else np.ones(len(f.bboxes), dtype=np.float32)
        dets = [
            Detection(bbox=bbox.astype(np.float32), score=float(s))
            for bbox, s in zip(f.bboxes, scores)
            if s >= args.score_threshold
        ]
        tracks = tracker.update(dets)
        for t in tracks:
            x1, y1, x2, y2 = t.bbox.astype(int)
            color = _color_for(t.track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{t.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, f"{args.tracker} | frame {i+1}/{len(frame_paths)} | tracks: {len(tracks)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(f"wrote {args.out} ({len(frame_paths)} frames, {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
