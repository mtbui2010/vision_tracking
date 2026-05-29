"""Outbound-WebSocket endpoint for GPU workers (local PC or RunPod).

Protocol (mirrors inferix):
  worker -> backend  : {"type": "register", "worker_id": "...", "gpu_info": {...}}
                       {"type": "heartbeat", "worker_id": "..."}
                       {"type": "log", "job_id": "...", "line": "..."}
                       {"type": "done", "job_id": "...", "result": {...}}
                       {"type": "error", "job_id": "...", "error": "..."}
  backend -> worker  : {"type": "job", "job_id": "...", "action": "track", ...}
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import settings
from core.jobs import JobStatus, store
from core.worker_registry import registry


router = APIRouter(prefix="/api/worker", tags=["worker"])


@router.websocket("/connect")
async def connect(socket: WebSocket) -> None:
    token = socket.query_params.get("token")
    if token != settings.worker_token:
        await socket.close(code=1008)
        return

    await socket.accept()
    worker_id = socket.query_params.get("worker_id") or uuid.uuid4().hex
    await registry.add(worker_id, socket)
    sender_task = asyncio.create_task(_send_jobs(socket, worker_id))

    try:
        while True:
            msg = await socket.receive_json()
            kind = msg.get("type")
            if kind == "heartbeat":
                await registry.heartbeat(worker_id, msg.get("gpu_info"))
            elif kind == "log":
                job = store.get(msg["job_id"])
                if job is not None:
                    job.append_log(msg["line"])
            elif kind == "done":
                job = store.get(msg["job_id"])
                if job is not None:
                    job.status = JobStatus.DONE
                    job.result = msg.get("result")
                    job.log_event.set()
            elif kind == "error":
                job = store.get(msg["job_id"])
                if job is not None:
                    job.status = JobStatus.FAILED
                    job.error = msg.get("error", "unknown error")
                    job.log_event.set()
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        await registry.remove(worker_id)


async def _send_jobs(socket: WebSocket, worker_id: str) -> None:
    while True:
        job = await store.pop()
        job.status = JobStatus.RUNNING
        try:
            await socket.send_json({
                "type": "job",
                "job_id": job.id,
                "action": job.kind,
                "payload": job.payload,
            })
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = f"failed to dispatch: {exc}"
            return
