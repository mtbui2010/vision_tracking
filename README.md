# Tracker Lab

> A multi-object tracking research playground — four trackers (SORT, DeepSORT, ByteTrack, and a custom appearance-gated variant) implemented **from scratch**, evaluated on MOT17 with metrics that match `py-motmetrics` to within 0.5%, and served through a FastAPI + Next.js web UI that visualizes the Kalman filter and Hungarian assignment as they run.

**Live demo:** _coming soon_
**Blog post:** [docs/BLOG_DRAFT.md](docs/BLOG_DRAFT.md)
**Tests:** 51/51 passing (`pytest tests/`)

---

## Why this project

Most tracking-demo repos wrap pretrained YOLO + a pip-installed tracker. This one does the opposite: detector is a fine-tuned off-the-shelf YOLO, but **every line of the association stack is written from scratch** — the Kalman filter, the Munkres-style Hungarian, the ReID matching cascade, the byte-association logic, and the MOTA / IDF1 / HOTA evaluation. No `filterpy`. No `scipy.optimize.linear_sum_assignment`. No `motmetrics` at runtime — only used in `tests/` as a cross-check.

The goal is to demonstrate, for CV / perception roles:

1. **Algorithmic depth.** Kalman, Hungarian, motion / appearance / IoU cost design, ReID embeddings — first principles, with derivations in `docs/notebooks/`.
2. **Evaluation rigor.** Custom MOTA / IDF1 / HOTA cross-checked against `py-motmetrics` within 0.5 % (`tests/test_metrics_motmetrics.py`).
3. **Systems engineering.** FastAPI backend + Next.js frontend + outbound-WS GPU worker + RunPod serverless — same architecture pattern used by production ML platforms.
4. **A real research finding.** ByteTrack underperforms on weak detectors because its low-confidence branch picks up false positives; an appearance gate on stage-2 fixes this and **drops ID switches by 64 %**. See the [blog draft](docs/BLOG_DRAFT.md).

## Headline numbers

**MOT17-val (7 sequences) · our YOLOv8n fine-tuned on MOT17, 30 epochs img=640:**

| Tracker | MOTA | IDF1 | HOTA | IDSW | FPS |
|---|---:|---:|---:|---:|---:|
| SORT | 0.341 | 0.434 | 0.416 | 953 | 864 |
| DeepSORT | 0.337 | 0.464 | 0.432 | 923 | 548 |
| ByteTrack | 0.290 | 0.282 | 0.286 | 574 | 717 |
| **Custom (this repo)** | 0.324 | **0.490** | **0.445** | **83** | 988 |

**DanceTrack-val (3 sequences) · same YOLOv8n 30ep:**

| Tracker | MOTA | IDF1 | HOTA | IDSW |
|---|---:|---:|---:|---:|
| SORT | 0.266 | 0.103 | 0.116 | 1266 |
| DeepSORT | 0.259 | 0.138 | 0.142 | 1197 |
| ByteTrack | 0.174 | 0.079 | 0.070 | 1053 |
| **Custom** | 0.299 | **0.198** | **0.157** | **430** |

Best IDF1 and HOTA on both benchmarks. **11× fewer ID switches than DeepSORT
on MOT17, 3× fewer on DanceTrack.** The lead grows as the detector improves
— exactly what the design predicted: an appearance gate on ByteTrack's
stage-2, plus a confidence-aware Kalman R that down-weights low-score
detections, prevents the IDSW explosion that hits IoU-only methods on dense
detector outputs.

The full per-detector breakdown (FRCNN baseline, YOLOv8n 10ep, YOLOv8n 30ep,
DanceTrack) is in [docs/TECHNICAL_DESIGN.md §4.3](docs/TECHNICAL_DESIGN.md#43-reference-numbers).
The custom tracker is implemented in [backend/services/trackers/custom.py](backend/services/trackers/custom.py); the design notes are in the [blog draft](docs/BLOG_DRAFT.md).

## What the web UI does

| Page | What it shows |
|---|---|
| `/tracker-lab` | Upload a video, pick a tracker, watch tracks render in real time. |
| `/compare` | Run 4 trackers on the same clip in parallel. Live MOTA / IDF1 / FPS. |
| `/algorithm` | One-frame stepper: Kalman state, IoU matrix, Hungarian assignment colored by accept/reject. |
| `/stress-test` | Server-rendered leaderboard from `benchmark.json` — 7 sequences × 4 trackers. |

## Architecture

```
+---------------------+        +-------------------------+
|  Next.js frontend   | <----> |   FastAPI backend       |
|  (Tracker Lab UI)   |  HTTP  |   - job orchestration   |
+---------------------+   WS   |   - eval metrics        |
                                +-----------+-------------+
                                            |
                       +--------------------+-----------------+
                       |                                      |
              +--------v---------+                  +---------v----------+
              |  Local GPU       |                  |  RunPod serverless |
              |  worker (WS-out) |                  |  worker            |
              |  - detector      |                  |  - heavy batch     |
              |  - tracker       |                  |  - batch eval      |
              +------------------+                  +--------------------+
```

The compute split (`local` / `cloud` / `hybrid`) and outbound-WS GPU worker pattern are adapted from [inferix](../inferix), an ML-serving platform I built earlier.

## Run it locally

```bash
make install           # python venv + npm install
make test              # 51 tests, ~2 s
make backend           # uvicorn at :8000
make frontend          # next dev at :3000
make worker            # local GPU worker, outbound WS
```

End-to-end eval on a real MOT17 sequence (download it first per `datasets/README.md`):

```bash
.venv/bin/python scripts/eval.py \
  --tracker custom \
  --detections datasets/MOT17/train/MOT17-09-FRCNN/det/det.txt \
  --gt datasets/MOT17/train/MOT17-09-FRCNN/gt/gt.txt \
  --score-threshold 0.3
```

Re-generate the full leaderboard table on the stress-test page:

```bash
.venv/bin/python scripts/benchmark.py \
  --root datasets/MOT17/train \
  --out-json frontend/public/benchmark.json \
  --out-md docs/_generated_benchmark.md
```

Side-by-side annotated demo MP4:

```bash
.venv/bin/python scripts/render_side_by_side.py \
  --frames datasets/MOT17/train/MOT17-09-FRCNN/img1 \
  --detections datasets/MOT17/train/MOT17-09-FRCNN/det/det.txt \
  --trackers sort,deepsort,bytetrack,custom \
  --out exports/MOT17-09_4trackers.mp4 \
  --scale 0.5
```

## Tech stack

- **Backend:** Python 3.11, FastAPI, NumPy, OpenCV, ONNX Runtime. PyTorch + Ultralytics only for the YOLO detector.
- **Frontend:** Next.js 14 (App Router), TypeScript, Canvas API for the bbox overlay, Tailwind.
- **Models:** YOLOv8n / YOLOv11 (fine-tuned on MOT17), OSNet for ReID embeddings (HashEmbedder placeholder used in tests so no PyTorch is needed for the algorithm test suite).
- **Deploy:** Docker Compose locally, Vercel for the frontend, RunPod serverless for cloud GPU.
- **Eval:** custom MOTA / IDF1 / HOTA cross-checked against `py-motmetrics` (within 0.5 %).

## Repo layout

```
backend/services/trackers/   sort / deepsort / bytetrack / custom + kalman + hungarian + iou
backend/services/metrics.py  MOTA / IDF1 / HOTA, from scratch
backend/services/reid/       OSNet wrapper + HashEmbedder placeholder
backend/api/                 FastAPI routes: tracking, compare, algorithm inspector, GPU worker WS
backend/tests/               51 tests: unit, integration, py-motmetrics cross-check
frontend/app/                Next.js pages, served at :3000 with /api/* proxied to backend
scripts/                     eval / benchmark / render / prepare_dataset / train_yolo
docs/                        TECHNICAL_DESIGN.md, BLOG_DRAFT.md, notebooks/
```

## Status

In-progress portfolio project. See [ROADMAP.md](ROADMAP.md) for the week-by-week plan and [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md) for the algorithm + evaluation design. Weeks 1–8 complete; DanceTrack eval + multi-camera ReID are stretch goals.

## About

Built by Trung Bui as a portfolio project. Open to computer vision / perception engineering roles in the US.

- Email: bmtrungvp@gmail.com
- LinkedIn: _add link_
- GitHub: _add link_
