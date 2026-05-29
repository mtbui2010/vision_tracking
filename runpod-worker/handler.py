"""RunPod serverless handler — single video tracking + batch MOT eval.

The handler is sync (RunPod's API expects a function). It receives a job dict
and returns a result dict. Heavy lifting happens in the same process.
"""

from __future__ import annotations

import base64
import io
import sys
import tempfile
from pathlib import Path
from typing import Any

BACKEND_PATH = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))

import cv2  # type: ignore[import-not-found]
import numpy as np

from services.detector import YOLODetector
from services.metrics import FrameAnnotations, compute_metrics
from services.eval_dataset import read_mot
from services.trackers.registry import available, build


def _decode_video(b64: str, suffix: str = ".mp4") -> Path:
    tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
    tmp.write_bytes(base64.b64decode(b64))
    return tmp


def _track_video(video_path: Path, tracker_name: str, weights: str) -> dict[str, Any]:
    detector = YOLODetector(weights=weights)
    tracker = build(tracker_name)
    cap = cv2.VideoCapture(str(video_path))
    pred_frames: list[FrameAnnotations] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame=frame)
        if tracks:
            ids = np.array([t.track_id for t in tracks], dtype=np.int64)
            boxes = np.stack([t.bbox for t in tracks])
        else:
            ids = np.zeros(0, dtype=np.int64)
            boxes = np.zeros((0, 4))
        pred_frames.append(FrameAnnotations(ids=ids, bboxes=boxes))
        frame_idx += 1
    cap.release()
    return {
        "frames": frame_idx,
        "predictions": [
            {"frame": i + 1, "ids": f.ids.tolist(), "bboxes": f.bboxes.tolist()}
            for i, f in enumerate(pred_frames)
        ],
    }


def handler(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input", {})
    action = payload.get("action", "track")

    if action == "list_trackers":
        return {"trackers": available()}

    if action == "track":
        video_path = _decode_video(payload["video_b64"], payload.get("suffix", ".mp4"))
        try:
            return _track_video(
                video_path,
                payload.get("tracker", "sort"),
                payload.get("weights", "yolov8n.pt"),
            )
        finally:
            video_path.unlink(missing_ok=True)

    if action == "batch_eval":
        det_path = Path(payload["detections_path"])
        gt_path = Path(payload["gt_path"])
        det_seq = read_mot(det_path)
        gt_seq = read_mot(gt_path)
        # caller is expected to have produced predictions already; we just compute metrics
        m = compute_metrics(gt_seq.frames, det_seq.frames)
        return {
            "mota": m.mota,
            "idf1": m.idf1,
            "hota": m.hota,
            "fp": m.fp,
            "fn": m.fn,
            "idsw": m.idsw,
        }

    return {"error": f"unknown action: {action}"}


if __name__ == "__main__":
    import runpod  # type: ignore[import-not-found]

    runpod.serverless.start({"handler": handler})
