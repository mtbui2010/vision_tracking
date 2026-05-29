"""Outbound-WebSocket GPU worker.

Connects to the backend, accepts tracking jobs, runs detector + tracker, ships
result video / metrics back.

Run with:
    python worker.py --backend ws://localhost:8000 --token dev-token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np
import websockets  # type: ignore[import-not-found]


BACKEND_PATH = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))

from services.detector import YOLODetector
from services.metrics import FrameAnnotations, compute_metrics
from services.eval_dataset import read_mot
from services.trackers.base import Detection
from services.trackers.registry import build


HEARTBEAT_EVERY = 5.0


def _gpu_info() -> dict:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return {
                "cuda": True,
                "device": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            }
    except ImportError:
        pass
    return {"cuda": False}


async def heartbeat_loop(ws: websockets.WebSocketClientProtocol, worker_id: str) -> None:
    while True:
        await ws.send(json.dumps({"type": "heartbeat", "worker_id": worker_id, "gpu_info": _gpu_info()}))
        await asyncio.sleep(HEARTBEAT_EVERY)


def run_tracking_job(
    video_path: str,
    tracker_name: str,
    weights: str,
    log: callable,
) -> dict:
    detector = YOLODetector(weights=weights)
    tracker = build(tracker_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

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
        if frame_idx % 30 == 0:
            log(f"processed {frame_idx}/{n_frames} frames")
    cap.release()
    return {
        "tracker": tracker_name,
        "frames": frame_idx,
        "fps": fps,
        "predictions": [
            {"frame": i + 1, "ids": f.ids.tolist(), "bboxes": f.bboxes.tolist()}
            for i, f in enumerate(pred_frames)
        ],
    }


async def handle_job(ws, worker_id: str, job_id: str, payload: dict) -> None:
    def log(line: str) -> None:
        asyncio.create_task(ws.send(json.dumps({"type": "log", "job_id": job_id, "line": line})))

    try:
        result = await asyncio.to_thread(
            run_tracking_job,
            payload["video"],
            payload.get("tracker", "sort"),
            payload.get("weights", "yolov8n.pt"),
            log,
        )
        await ws.send(json.dumps({"type": "done", "job_id": job_id, "result": result}))
    except Exception as exc:  # noqa: BLE001
        await ws.send(json.dumps({"type": "error", "job_id": job_id, "error": str(exc)}))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=os.getenv("BACKEND_URL", "ws://localhost:8000"))
    parser.add_argument("--token", default=os.getenv("WORKER_TOKEN", "dev-token"))
    parser.add_argument("--worker-id", default=os.getenv("WORKER_ID") or uuid.uuid4().hex)
    args = parser.parse_args()

    url = f"{args.backend.rstrip('/')}/api/worker/connect?token={args.token}&worker_id={args.worker_id}"
    print(f"connecting to {url}")

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "register", "worker_id": args.worker_id, "gpu_info": _gpu_info()}))
        hb = asyncio.create_task(heartbeat_loop(ws, args.worker_id))
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "job":
                    asyncio.create_task(handle_job(ws, args.worker_id, msg["job_id"], msg["payload"]))
        finally:
            hb.cancel()


if __name__ == "__main__":
    asyncio.run(main())
