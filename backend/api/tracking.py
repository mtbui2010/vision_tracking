"""Tracking job submission + result retrieval."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from core.config import settings
from core.jobs import Job, JobStatus, store
from services.trackers.registry import available


router = APIRouter(prefix="/api/tracking", tags=["tracking"])


class JobOut(BaseModel):
    id: str
    kind: str
    status: JobStatus
    result: dict[str, Any] | None = None
    error: str | None = None


def _to_out(job: Job) -> JobOut:
    return JobOut(id=job.id, kind=job.kind, status=job.status, result=job.result, error=job.error)


@router.get("/trackers")
def list_trackers() -> dict[str, list[str]]:
    return {"trackers": available()}


@router.post("/jobs", response_model=JobOut)
async def submit_job(
    video: UploadFile = File(...),
    tracker: str = Form("sort"),
    compute_target: str = Form(None),
) -> JobOut:
    if tracker not in available():
        raise HTTPException(400, f"unknown tracker {tracker!r}")
    ct = compute_target or settings.default_compute_target
    if ct not in {"local", "cloud", "hybrid"}:
        raise HTTPException(400, f"invalid compute_target {ct!r}")

    settings.storage_root.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_path = settings.storage_root / f"upload_{video.filename or 'video'}{suffix}"
    video_path.write_bytes(await video.read())

    job = store.create(
        kind="track",
        payload={"video": str(video_path), "tracker": tracker, "compute_target": ct},
    )
    return _to_out(job)


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> JobOut:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _to_out(job)


@router.websocket("/jobs/{job_id}/stream")
async def stream_job(socket: WebSocket, job_id: str) -> None:
    await socket.accept()
    job = store.get(job_id)
    if job is None:
        await socket.send_json({"error": "job not found"})
        await socket.close()
        return

    sent = 0
    try:
        while True:
            while sent < len(job.logs):
                await socket.send_text(job.logs[sent])
                sent += 1
            if job.status in (JobStatus.DONE, JobStatus.FAILED):
                await socket.send_json(
                    {"status": job.status.value, "result": job.result, "error": job.error}
                )
                break
            try:
                await asyncio.wait_for(job.log_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        return
    finally:
        await socket.close()
