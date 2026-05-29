"""Run a tracker over a MOT-format detection file and report metrics.

Usage:
    python scripts/eval.py \
        --tracker sort \
        --detections datasets/MOT17/train/MOT17-04-FRCNN/det/det.txt \
        --gt datasets/MOT17/train/MOT17-04-FRCNN/gt/gt.txt \
        --out exports/MOT17-04_sort.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from services.eval_dataset import MOTSequence, read_mot, write_mot
from services.metrics import FrameAnnotations, compute_metrics
from services.trackers.base import Detection
from services.trackers.registry import available, build


def run(
    tracker_name: str,
    detections: MOTSequence,
    score_threshold: float = 0.1,
) -> tuple[MOTSequence, float]:
    tracker = build(tracker_name)
    pred_frames: list[FrameAnnotations] = []

    start = time.perf_counter()
    for f in detections.frames:
        dets: list[Detection] = []
        scores = f.scores if f.scores is not None else np.ones(len(f.bboxes), dtype=np.float32)
        for bbox, score in zip(f.bboxes, scores):
            if score < score_threshold:
                continue
            dets.append(Detection(bbox=bbox.astype(np.float32), score=float(score)))
        tracks = tracker.update(dets)
        if tracks:
            ids = np.array([t.track_id for t in tracks], dtype=np.int64)
            boxes = np.stack([t.bbox for t in tracks])
        else:
            ids = np.zeros(0, dtype=np.int64)
            boxes = np.zeros((0, 4))
        pred_frames.append(FrameAnnotations(ids=ids, bboxes=boxes))
    elapsed = time.perf_counter() - start

    return MOTSequence(name=detections.name, frames=pred_frames), elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", required=True, choices=available())
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    args = parser.parse_args()

    det_seq = read_mot(args.detections, name=args.detections.stem)
    pred_seq, elapsed = run(args.tracker, det_seq, args.score_threshold)
    fps = sum(len(f.ids) for f in det_seq.frames) / max(elapsed, 1e-6)
    print(f"tracker={args.tracker} frames={len(pred_seq.frames)} elapsed={elapsed:.2f}s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_mot(args.out, pred_seq)
        print(f"wrote {args.out}")

    if args.gt:
        gt_seq = read_mot(args.gt, name=args.gt.stem)
        # Truncate to common length for safety.
        n = min(len(gt_seq.frames), len(pred_seq.frames))
        m = compute_metrics(gt_seq.frames[:n], pred_seq.frames[:n], iou_threshold=0.5)
        print(
            f"MOTA={m.mota:.4f} MOTP={m.motp:.4f} IDF1={m.idf1:.4f} "
            f"HOTA={m.hota:.4f} (DetA={m.deta:.4f}, AssA={m.assa:.4f})"
        )
        print(f"FP={m.fp} FN={m.fn} IDSW={m.idsw} GT={m.num_gt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
