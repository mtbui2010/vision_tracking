"""Runtime config — read once from env, used everywhere."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    storage_root: Path
    runpod_api_key: str | None
    runpod_endpoint_id: str | None
    worker_token: str
    default_compute_target: str  # "local" | "cloud" | "hybrid"


def load() -> Settings:
    return Settings(
        storage_root=Path(os.getenv("TRACKERLAB_STORAGE", "./storage")).resolve(),
        runpod_api_key=os.getenv("RUNPOD_API_KEY"),
        runpod_endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        worker_token=os.getenv("WORKER_TOKEN", "dev-token"),
        default_compute_target=os.getenv("DEFAULT_COMPUTE_TARGET", "local"),
    )


settings = load()
