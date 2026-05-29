"""Run every registered tracker on a curated set of hard clips, dump a JSON
leaderboard that the frontend stress-test page reads.

A "stress clip" is any sequence under stress_clips/ with the MOT directory
layout:
    stress_clips/<clip_name>/det/det.txt
    stress_clips/<clip_name>/gt/gt.txt

The output JSON is structured for direct rendering in the React table.

Usage:
    python scripts/run_stress_test.py \
        --clips stress_clips/ \
        --out frontend/public/stress_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from services.eval_dataset import MOTSequence, read_mot
from services.metrics import FrameAnnotations, compute_metrics
from services.trackers.base import Detection
from services.trackers.registry import available, build


def run_tracker_on_clip(tracker_name: str, detections: MOTSequence) -> tuple[list[FrameAnnotations], float]:
    tracker = build(tracker_name)
    pred_frames: list[FrameAnnotations] = []
    start = time.perf_counter()
    for f in detections.frames:
        dets = [
            Detection(bbox=bbox.astype(np.float32), score=1.0)
            for bbox in f.bboxes
        ]
        tracks = tracker.update(dets)
        if tracks:
            ids = np.array([t.track_id for t in tracks], dtype=np.int64)
            boxes = np.stack([t.bbox for t in tracks])
        else:
            ids = np.zeros(0, dtype=np.int64)
            boxes = np.zeros((0, 4))
        pred_frames.append(FrameAnnotations(ids=ids, bboxes=boxes))
    return pred_frames, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    clips = sorted(p for p in args.clips.iterdir() if (p / "gt" / "gt.txt").exists())
    if not clips:
        raise SystemExit(f"no MOT-format clips under {args.clips}")

    rows: list[dict] = []
    for tracker_name in available():
        agg_mota = []
        agg_idf1 = []
        agg_hota = []
        agg_idsw = 0
        agg_frames = 0
        agg_elapsed = 0.0
        for clip in clips:
            det_seq = read_mot(clip / "det" / "det.txt", name=clip.name)
            gt_seq = read_mot(clip / "gt" / "gt.txt", name=clip.name)
            preds, elapsed = run_tracker_on_clip(tracker_name, det_seq)
            n = min(len(gt_seq.frames), len(preds))
            m = compute_metrics(gt_seq.frames[:n], preds[:n], iou_threshold=0.5)
            agg_mota.append(m.mota)
            agg_idf1.append(m.idf1)
            agg_hota.append(m.hota)
            agg_idsw += m.idsw
            agg_frames += n
            agg_elapsed += elapsed
        rows.append({
            "tracker": tracker_name,
            "mota": float(np.mean(agg_mota)),
            "idf1": float(np.mean(agg_idf1)),
            "hota": float(np.mean(agg_hota)),
            "idsw": int(agg_idsw),
            "fps": agg_frames / max(agg_elapsed, 1e-6),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"clips": [c.name for c in clips], "rows": rows}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
