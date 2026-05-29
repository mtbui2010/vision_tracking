"""Render side-by-side comparison of multiple trackers on the same sequence.

Reads frames + det.txt, runs each tracker independently, stitches a Nx1
or 2x2 grid of annotated views into one MP4.

Usage:
    python scripts/render_side_by_side.py \\
        --frames datasets/MOT17/train/MOT17-09-FRCNN/img1 \\
        --detections datasets/MOT17/train/MOT17-09-FRCNN/det/det.txt \\
        --trackers sort,deepsort,bytetrack,custom \\
        --out exports/MOT17-09_compare.mp4 \\
        --score-threshold 0.3 \\
        --scale 0.5 \\
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
    import colorsys

    hue = (track_id * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _draw(frame, tracks, label: str):
    import cv2  # type: ignore[import-not-found]

    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t.bbox.astype(int)
        color = _color_for(t.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"#{t.track_id}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    bar_h = 32
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(out, f"{label}  |  tracks: {len(tracks)}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _grid(panels):
    import cv2  # type: ignore[import-not-found]

    n = len(panels)
    if n == 1:
        return panels[0]
    if n == 2:
        return np.hstack(panels)
    if n == 3:
        h, w = panels[0].shape[:2]
        blank = np.zeros((h, w, 3), dtype=panels[0].dtype)
        top = np.hstack([panels[0], panels[1]])
        bot = np.hstack([panels[2], blank])
        return np.vstack([top, bot])
    # n == 4
    top = np.hstack([panels[0], panels[1]])
    bot = np.hstack([panels[2], panels[3]])
    return np.vstack([top, bot])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--trackers", required=True, help="comma-separated tracker names")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    import cv2  # type: ignore[import-not-found]

    tracker_names = [t.strip() for t in args.trackers.split(",") if t.strip()]
    trackers = [build(n) for n in tracker_names]

    det_seq = read_mot(args.detections)
    frame_paths = sorted(args.frames.glob("*.jpg")) or sorted(args.frames.glob("*.png"))
    if not frame_paths:
        raise SystemExit(f"no frames under {args.frames}")
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    if args.scale != 1.0:
        h = int(h * args.scale)
        w = int(w * args.scale)
    # Probe grid dimensions
    panels0 = [np.zeros((h, w, 3), dtype=np.uint8) for _ in trackers]
    grid0 = _grid(panels0)
    out_h, out_w = grid0.shape[:2]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"cannot open {args.out}")

    for i, frame_path in enumerate(frame_paths):
        if i >= len(det_seq.frames):
            break
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        if args.scale != 1.0:
            frame = cv2.resize(frame, (w, h))

        f = det_seq.frames[i]
        scores = f.scores if f.scores is not None else np.ones(len(f.bboxes), dtype=np.float32)
        if args.scale != 1.0:
            scaled_boxes = f.bboxes * args.scale
        else:
            scaled_boxes = f.bboxes
        dets = [
            Detection(bbox=bbox.astype(np.float32), score=float(s))
            for bbox, s in zip(scaled_boxes, scores)
            if s >= args.score_threshold
        ]

        panels = []
        for name, tr in zip(tracker_names, trackers):
            tracks = tr.update(dets, frame=frame)
            panels.append(_draw(frame, tracks, name))

        writer.write(_grid(panels))
        if i % 30 == 0:
            print(f"frame {i+1}/{len(frame_paths)}")

    writer.release()
    print(f"wrote {args.out} ({len(frame_paths)} frames @ {args.fps}fps, {out_w}x{out_h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
