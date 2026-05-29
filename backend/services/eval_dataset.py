"""Read/write MOTChallenge-format annotation files.

The MOT format is one CSV row per detection/track instance:
    frame, id, x, y, w, h, conf, [class, vis]
1-indexed frames. Bbox is top-left (x, y) + (w, h).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import FrameAnnotations


@dataclass
class MOTSequence:
    name: str
    frames: list[FrameAnnotations]


def read_mot(
    path: str | Path,
    *,
    score_threshold: float | None = None,
    name: str | None = None,
) -> MOTSequence:
    rows = np.loadtxt(str(path), delimiter=",", dtype=np.float64, ndmin=2)
    if rows.size == 0:
        return MOTSequence(name=name or Path(path).stem, frames=[])

    if score_threshold is not None and rows.shape[1] >= 7:
        rows = rows[rows[:, 6] >= score_threshold]

    n_frames = int(rows[:, 0].max())
    frames: list[FrameAnnotations] = []
    rows_by_frame: dict[int, list[np.ndarray]] = {}
    for r in rows:
        f = int(r[0])
        rows_by_frame.setdefault(f, []).append(r)

    for f in range(1, n_frames + 1):
        rs = rows_by_frame.get(f, [])
        if not rs:
            frames.append(
                FrameAnnotations(
                    ids=np.zeros(0, dtype=np.int64),
                    bboxes=np.zeros((0, 4)),
                    scores=np.zeros(0, dtype=np.float32),
                )
            )
            continue
        arr = np.stack(rs)
        ids = arr[:, 1].astype(np.int64)
        x1 = arr[:, 2]
        y1 = arr[:, 3]
        x2 = x1 + arr[:, 4]
        y2 = y1 + arr[:, 5]
        bboxes = np.stack([x1, y1, x2, y2], axis=1)
        scores = arr[:, 6].astype(np.float32) if arr.shape[1] >= 7 else np.ones(len(arr), dtype=np.float32)
        frames.append(FrameAnnotations(ids=ids, bboxes=bboxes, scores=scores))

    return MOTSequence(name=name or Path(path).stem, frames=frames)


def write_mot(path: str | Path, sequence: MOTSequence) -> None:
    """Write predictions in MOT format. Score column = 1.0 (placeholder)."""
    lines = []
    for fi, frame in enumerate(sequence.frames, start=1):
        for tid, bbox in zip(frame.ids, frame.bboxes):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            lines.append(f"{fi},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    Path(path).write_text("\n".join(lines))
