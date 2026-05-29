"""In-memory job store and async queue.

Single-process for v1 — when we go multi-replica, swap this for Redis without
touching the API layer.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    kind: str
    payload: dict[str, Any]
    status: JobStatus = JobStatus.QUEUED
    result: dict[str, Any] | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    log_event: asyncio.Event = field(default_factory=asyncio.Event)

    def append_log(self, line: str) -> None:
        self.logs.append(line)
        self.log_event.set()
        self.log_event.clear()


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[Job] = asyncio.Queue()

    def create(self, kind: str, payload: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, payload=payload)
        self._jobs[job.id] = job
        self._queue.put_nowait(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def pop(self) -> Job:
        return await self._queue.get()

    def all(self) -> list[Job]:
        return list(self._jobs.values())


store = JobStore()
