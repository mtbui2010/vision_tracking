"""Side-by-side comparison job — run N trackers on the same video, return metrics."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from core.config import settings
from core.jobs import JobStatus, store
from services.trackers.registry import available


router = APIRouter(prefix="/api/compare", tags=["compare"])


@router.post("/jobs")
async def submit_compare(
    video: UploadFile = File(...),
    trackers: str = Form("sort,deepsort,bytetrack"),
) -> dict[str, str]:
    names = [n.strip() for n in trackers.split(",") if n.strip()]
    unknown = [n for n in names if n not in available()]
    if unknown:
        raise HTTPException(400, f"unknown trackers: {unknown}")

    settings.storage_root.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_path = settings.storage_root / f"compare_{video.filename or 'video'}{suffix}"
    video_path.write_bytes(await video.read())

    job = store.create(
        kind="compare",
        payload={"video": str(video_path), "trackers": names},
    )
    return {"id": job.id, "status": JobStatus.QUEUED.value}
