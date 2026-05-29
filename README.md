# Tracker Lab

> A multi-object tracking research playground — four trackers (SORT, DeepSORT, ByteTrack, and a custom variant) implemented from scratch, evaluated on MOT17 / MOT20 / DanceTrack, and served through a real-time web UI that visualizes the Kalman filter and Hungarian assignment as they run.

**Live demo:** _coming soon_
**Blog post:** _coming soon_

---

## Why this project

Most "tracking demo" repositories wrap a pretrained YOLO and call `tracker.update()` on top of a third-party library. This one does the opposite: the detector is a fine-tuned off-the-shelf YOLO, but every tracker — the Kalman filter, the Hungarian assignment, the ReID matching cascade, the byte-association logic — is implemented from scratch and unit-tested.

The goal is to demonstrate:

1. **Algorithmic depth.** Kalman, Hungarian, motion / appearance / IoU cost design, and ReID embeddings are written from first principles, with notebooks deriving the math.
2. **Evaluation rigor.** MOTA, IDF1, and HOTA are computed in-repo (no `motmetrics` wrapper) and benchmarked against published numbers on MOT17-val.
3. **Systems engineering.** A FastAPI backend + Next.js frontend + GPU worker (local PC) + RunPod serverless pipeline — the same architecture pattern used to deploy real ML products.
4. **Failure-mode understanding.** A dedicated stress-test page measures ID switches under occlusion, similar appearance (DanceTrack), and small-target conditions.

## What you can do in the live demo

| Page | What it shows |
|---|---|
| `/tracker-lab` | Upload a video or stream from webcam, pick a tracker, see live tracks rendered. |
| `/compare` | Run 4 trackers on the same clip in parallel. Side-by-side overlays + MOTA / IDF1 / HOTA / FPS in real time. |
| `/algorithm` | Step through frames. Visualize Kalman predict / update ellipses, Hungarian cost matrix, association decisions. |
| `/stress-test` | Curated hard clips (heavy occlusion, similar appearance). ID-switch leaderboard per tracker. |
| `/reid-explorer` | Multi-camera person re-identification. Drop two clips from different cameras, see cross-view matches. |

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
              |  - tracker       |                  |  - ReID training   |
              +------------------+                  +--------------------+
```

The compute split (`local` / `cloud` / `hybrid`) and outbound-WebSocket GPU worker pattern are adapted from `inferix`, an ML-serving platform I built earlier.

## Tech stack

- **Backend:** Python 3.11, FastAPI, NumPy, PyTorch, ONNX Runtime, OpenCV.
- **Frontend:** Next.js 14 (App Router), TypeScript, Canvas API for overlay, Tailwind.
- **Models:** YOLOv11 (fine-tuned on MOT17 / DanceTrack), OSNet for ReID embeddings.
- **Deploy:** Docker Compose locally, Vercel for frontend, RunPod serverless for cloud GPU.
- **Eval:** custom MOTA / IDF1 / HOTA implementation; benchmarked against `py-motmetrics` for correctness.

## Status

This is an in-progress portfolio project. See [ROADMAP.md](ROADMAP.md) for the week-by-week plan and [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md) for the algorithm and evaluation design.

## About

Built by Trung Bui as a portfolio project. Open to computer vision / perception engineering roles in the US.
- Email: bmtrungvp@gmail.com
- LinkedIn: _add link_
- GitHub: _add link_
