"""Run every registered tracker on every MOT-format sequence under --root.

Dumps:
  - pretty table to stdout
  - JSON suitable for the frontend stress-test page (`benchmark.json`)
  - markdown table for `docs/TECHNICAL_DESIGN.md` (`benchmark.md`)

Usage:
    python scripts/benchmark.py \\
        --root datasets/MOT17/train \\
        --out-json frontend/public/benchmark.json \\
        --out-md docs/_generated_benchmark.md \\
        --score-threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from services.eval_dataset import read_mot
from services.metrics import FrameAnnotations, compute_metrics
from services.trackers.base import Detection
from services.trackers.registry import available, build


@dataclass
class SeqResult:
    seq: str
    tracker: str
    mota: float
    idf1: float
    hota: float
    deta: float
    assa: float
    fp: int
    fn: int
    idsw: int
    num_gt: int
    frames: int
    elapsed: float

    @property
    def fps(self) -> float:
        return self.frames / max(self.elapsed, 1e-6)


def run_tracker_on_seq(
    tracker_name: str, seq_dir: Path, score_threshold: float
) -> SeqResult:
    det = read_mot(seq_dir / "det" / "det.txt")
    gt = read_mot(seq_dir / "gt" / "gt.txt")
    tracker = build(tracker_name)

    pred_frames: list[FrameAnnotations] = []
    start = time.perf_counter()
    for f in det.frames:
        scores = f.scores if f.scores is not None else np.ones(len(f.bboxes), dtype=np.float32)
        dets = [
            Detection(bbox=bbox.astype(np.float32), score=float(s))
            for bbox, s in zip(f.bboxes, scores)
            if s >= score_threshold
        ]
        tracks = tracker.update(dets)
        if tracks:
            ids = np.array([t.track_id for t in tracks], dtype=np.int64)
            boxes = np.stack([t.bbox for t in tracks])
        else:
            ids = np.zeros(0, dtype=np.int64)
            boxes = np.zeros((0, 4))
        pred_frames.append(FrameAnnotations(ids=ids, bboxes=boxes))
    elapsed = time.perf_counter() - start

    n = min(len(gt.frames), len(pred_frames))
    m = compute_metrics(gt.frames[:n], pred_frames[:n], iou_threshold=0.5)
    return SeqResult(
        seq=seq_dir.name,
        tracker=tracker_name,
        mota=m.mota,
        idf1=m.idf1,
        hota=m.hota,
        deta=m.deta,
        assa=m.assa,
        fp=m.fp,
        fn=m.fn,
        idsw=m.idsw,
        num_gt=m.num_gt,
        frames=n,
        elapsed=elapsed,
    )


def aggregate(results: list[SeqResult]) -> dict[str, dict]:
    by_tracker: dict[str, list[SeqResult]] = {}
    for r in results:
        by_tracker.setdefault(r.tracker, []).append(r)
    summary = {}
    for tracker, rows in by_tracker.items():
        # Sequence-length-weighted aggregation (matches MOT challenge convention)
        total_gt = sum(r.num_gt for r in rows)
        total_fp = sum(r.fp for r in rows)
        total_fn = sum(r.fn for r in rows)
        total_idsw = sum(r.idsw for r in rows)
        total_frames = sum(r.frames for r in rows)
        total_elapsed = sum(r.elapsed for r in rows)
        summary[tracker] = {
            "mota": 1.0 - (total_fp + total_fn + total_idsw) / max(total_gt, 1),
            "idf1": float(np.mean([r.idf1 for r in rows])),
            "hota": float(np.mean([r.hota for r in rows])),
            "deta": float(np.mean([r.deta for r in rows])),
            "assa": float(np.mean([r.assa for r in rows])),
            "fp": total_fp,
            "fn": total_fn,
            "idsw": total_idsw,
            "fps": total_frames / max(total_elapsed, 1e-6),
        }
    return summary


def print_table(rows: list[dict], headers: list[str]) -> None:
    widths = [
        max(len(h), max((len(str(r.get(h, ""))) for r in rows), default=0))
        for h in headers
    ]
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(h, "")).ljust(w) for h, w in zip(headers, widths)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--trackers", default=",".join(available()))
    args = parser.parse_args()

    seqs = sorted(p for p in args.root.iterdir() if (p / "det" / "det.txt").exists())
    if not seqs:
        raise SystemExit(f"no sequences under {args.root}")
    tracker_names = [t.strip() for t in args.trackers.split(",") if t.strip()]
    print(f"running {len(tracker_names)} trackers on {len(seqs)} sequences")
    print(f"score threshold = {args.score_threshold}")
    print()

    all_results: list[SeqResult] = []
    for seq in seqs:
        for t in tracker_names:
            r = run_tracker_on_seq(t, seq, args.score_threshold)
            all_results.append(r)
            print(
                f"{seq.name:18s} {t:10s} "
                f"MOTA={r.mota:+.4f} IDF1={r.idf1:.4f} HOTA={r.hota:.4f} "
                f"IDSW={r.idsw:4d} FPS={r.fps:7.0f}"
            )

    print()
    print("=== aggregate ===")
    summary = aggregate(all_results)
    rows = [{"tracker": t, **{k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}}
            for t, s in summary.items()]
    print_table(rows, ["tracker", "mota", "idf1", "hota", "deta", "assa", "fp", "fn", "idsw", "fps"])

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "sequences": [s.name for s in seqs],
                    "score_threshold": args.score_threshold,
                    "rows": rows,
                    "per_sequence": [{**r.__dict__, "fps": r.fps} for r in all_results],
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out_json}")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = ["| Tracker | MOTA | IDF1 | HOTA | DetA | AssA | IDSW | FPS |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(
                f"| {r['tracker']} | {r['mota']:.4f} | {r['idf1']:.4f} | {r['hota']:.4f} | "
                f"{r['deta']:.4f} | {r['assa']:.4f} | {r['idsw']} | {r['fps']:.0f} |"
            )
        args.out_md.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
