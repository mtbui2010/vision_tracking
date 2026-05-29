"""Track connected GPU workers + their last heartbeat.

Pattern borrowed from inferix: workers connect outbound via WS, send heartbeats
every few seconds, and are marked offline if they go silent.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any


HEARTBEAT_TIMEOUT_SEC = 30.0


@dataclass
class Worker:
    id: str
    socket: Any  # WebSocket
    last_heartbeat: float
    gpu_info: dict[str, Any] | None = None


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._lock = asyncio.Lock()

    async def add(self, worker_id: str, socket: Any) -> Worker:
        async with self._lock:
            w = Worker(id=worker_id, socket=socket, last_heartbeat=time.time())
            self._workers[worker_id] = w
            return w

    async def remove(self, worker_id: str) -> None:
        async with self._lock:
            self._workers.pop(worker_id, None)

    async def heartbeat(self, worker_id: str, gpu_info: dict[str, Any] | None = None) -> None:
        async with self._lock:
            w = self._workers.get(worker_id)
            if w is not None:
                w.last_heartbeat = time.time()
                if gpu_info is not None:
                    w.gpu_info = gpu_info

    def online(self) -> list[Worker]:
        now = time.time()
        return [w for w in self._workers.values() if now - w.last_heartbeat <= HEARTBEAT_TIMEOUT_SEC]

    def any_online(self) -> Worker | None:
        for w in self.online():
            return w
        return None


registry = WorkerRegistry()
