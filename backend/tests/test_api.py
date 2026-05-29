"""Integration tests for the FastAPI surface using TestClient."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    from main import app

    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_list_trackers(client: TestClient) -> None:
    r = client.get("/api/tracking/trackers")
    assert r.status_code == 200
    names = r.json()["trackers"]
    assert set(names) >= {"sort", "deepsort", "bytetrack"}


def test_algorithm_associate_endpoint(client: TestClient) -> None:
    body = {
        "detections": [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 300},
            {"x1": 300, "y1": 100, "x2": 400, "y2": 300},
        ],
        "tracks": [
            {"x1": 110, "y1": 110, "x2": 210, "y2": 310},
            {"x1": 310, "y1": 110, "x2": 410, "y2": 310},
        ],
        "iou_threshold": 0.3,
    }
    r = client.post("/api/algorithm/associate", json=body)
    assert r.status_code == 200
    data = r.json()
    assert len(data["iou_matrix"]) == 2 and len(data["iou_matrix"][0]) == 2
    # Each detection should match its near-identical track
    assert sorted(tuple(m) for m in data["matches"]) == [(0, 0), (1, 1)]


def test_algorithm_kalman_endpoint(client: TestClient) -> None:
    r = client.post(
        "/api/algorithm/kalman",
        json={"bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}, "steps": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["predicted_bbox"]) == 4
    assert len(data["state"]) == 7
    assert len(data["covariance_diag"]) == 7


def test_submit_job_rejects_unknown_tracker(client: TestClient) -> None:
    fake = io.BytesIO(b"\x00\x00\x00\x18ftypmp42")
    r = client.post(
        "/api/tracking/jobs",
        data={"tracker": "nope-tracker"},
        files={"video": ("clip.mp4", fake, "video/mp4")},
    )
    assert r.status_code == 400


def test_submit_job_creates_queued_job(client: TestClient) -> None:
    fake = io.BytesIO(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
    r = client.post(
        "/api/tracking/jobs",
        data={"tracker": "sort"},
        files={"video": ("clip.mp4", fake, "video/mp4")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"queued", "running"}
    job_id = body["id"]

    r2 = client.get(f"/api/tracking/jobs/{job_id}")
    assert r2.status_code == 200 and r2.json()["id"] == job_id


def test_compare_endpoint(client: TestClient) -> None:
    fake = io.BytesIO(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
    r = client.post(
        "/api/compare/jobs",
        data={"trackers": "sort,bytetrack"},
        files={"video": ("clip.mp4", fake, "video/mp4")},
    )
    assert r.status_code == 200 and "id" in r.json()


def test_unknown_job_404(client: TestClient) -> None:
    r = client.get("/api/tracking/jobs/does-not-exist")
    assert r.status_code == 404
