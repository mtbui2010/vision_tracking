# vision_tracking — Tracker Lab

Portfolio project: multi-object tracking research playground built to demonstrate algorithm depth + ML systems engineering for US CV / perception roles.

See [README.md](README.md) for the recruiter pitch, [ROADMAP.md](ROADMAP.md) for the week-by-week plan, [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md) for algorithm and eval design + full benchmark numbers, [docs/BLOG_DRAFT.md](docs/BLOG_DRAFT.md) for the writeup.

## Current state (post-week-8)

- **4 trackers** (sort, deepsort, bytetrack, custom) all from scratch, all in [backend/services/trackers/](backend/services/trackers/).
- **51 / 51 tests passing** including a py-motmetrics cross-check (`tests/test_metrics_motmetrics.py` — marked `@pytest.mark.slow`, requires MOT17-09-FRCNN downloaded).
- **Two datasets benchmarked**: MOT17 (7 sequences) and DanceTrack (3 of 11 val sequences).
- **YOLOv8n fine-tuned** on MOT17 at two sizes: 10ep/img=416 and 30ep/img=640 — latest checkpoint at `runs/detect/runs/mot17/yolov8n_30ep_640/weights/best.pt` (gitignored).
- **Custom tracker is the best across the board**: see headline tables in README.

## Structure

```
vision_tracking/
├── backend/                FastAPI service (Python)
│   ├── api/                HTTP + WS handlers (tracking, compare, algorithm, worker)
│   ├── core/               jobs.py (in-memory queue), worker_registry.py, config.py
│   ├── services/
│   │   ├── trackers/       sort / deepsort / bytetrack / custom — all from scratch
│   │   │   ├── base.py     Tracker interface (Detection, Track, BaseTracker)
│   │   │   ├── kalman.py   Constant-velocity Kalman, F/H/Q/R explicit numpy
│   │   │   ├── hungarian.py Munkres O(n³), rectangular-pad via _munkres helper
│   │   │   ├── iou.py      Vectorized bbox IoU
│   │   │   ├── association.py associate_iou helper
│   │   │   └── registry.py Single source of truth for tracker construction
│   │   ├── reid/embedder.py OSNet wrapper + HashEmbedder placeholder for tests
│   │   ├── detector.py     YOLO wrapper (lazy ultralytics import — backend can run without torch)
│   │   ├── metrics.py      MOTA/IDF1/HOTA with sticky per-frame matching (matches motmetrics 0.5%)
│   │   └── eval_dataset.py MOT-format read/write with FrameAnnotations.scores
│   ├── tests/              pytest, including test_api.py (TestClient) and test_metrics_motmetrics.py (slow)
│   ├── main.py             FastAPI entrypoint with CORS + 4 routers + /health
│   └── pyproject.toml
├── frontend/               Next.js 14 App Router (TypeScript) + Tailwind
│   ├── app/                tracker-lab, compare, algorithm, stress-test pages
│   ├── components/         TrackOverlay.tsx (canvas bbox overlay)
│   ├── services/api.ts     Single typed HTTP client
│   └── public/
│       ├── benchmark.json           ← MOT17 FRCNN baseline
│       ├── benchmark_yolo.json      ← MOT17 with our YOLOv8n 30ep
│       └── benchmark_dancetrack.json← DanceTrack with same YOLO
├── gpu-worker/             Outbound-WS worker (worker.py + start.sh)
├── runpod-worker/          RunPod serverless handler.py + Dockerfile
├── docs/
│   ├── TECHNICAL_DESIGN.md          ← algorithm + eval design + ALL benchmark tables
│   ├── BLOG_DRAFT.md                ← 3 lessons + DanceTrack story
│   ├── notebooks/                   ← Kalman + metrics derivations (.md, jupytext-ready)
│   ├── _generated_benchmark.md      ← auto-written by scripts/benchmark.py (MOT17 FRCNN)
│   ├── _generated_benchmark_yolo.md ← auto-written (MOT17 30ep)
│   └── _generated_benchmark_dancetrack.md
├── scripts/
│   ├── eval.py             single tracker × single sequence
│   ├── benchmark.py        all trackers × all sequences -> JSON + MD
│   ├── prepare_dataset.py  MOT17 gt -> YOLO format (clamps OOB boxes)
│   ├── prepare_dancetrack.py Voxel51 mp4 + frames.json -> MOT layout (extracts frames via cv2)
│   ├── train_yolo.py       ultralytics wrapper
│   ├── infer_detections.py YOLO -> MOT-format det.txt
│   ├── rebenchmark_yolo.sh infer all sequences + benchmark
│   ├── render_demo.py      single MP4 with annotated boxes
│   └── render_side_by_side.py 2×2 grid of 4 trackers
├── .github/workflows/test.yml  CI: pytest fast + tsc + next build
├── datasets/                   gitignored
│   ├── MOT17/train/<seq>/  img1, det/det.txt, gt/gt.txt, seqinfo.ini (7 sequences)
│   ├── DanceTrack_voxel51/ raw downloads (mp4 + samples.json + frames.json)
│   └── DanceTrack/val/<seq>/  converted MOT layout (3 sequences)
└── Makefile                install, test, backend, frontend, eval, benchmark, demo, train, ...
```

## Architecture borrowed from inferix

This project intentionally reuses patterns from `/home/trung/trung_workdir/inferix/`:

- Outbound-WebSocket GPU worker (works behind NAT, no port forwarding).
- Compute target routing: `local` / `cloud` / `hybrid`.
- Single typed `services/api.ts` for the frontend.
- RunPod serverless handler for cloud GPU.

When extending workers, reference inferix's `gpu-worker/worker.py`, `backend/workers/job_worker.py`, and `frontend/services/api.ts`.

## Key invariants

- Bbox values are **pixel coords** `[x1, y1, x2, y2]` (unlike inferix, which uses normalized [0,1]). Reason: tracking math is more naturally expressed in pixels.
- All trackers implement `BaseTracker` (see [backend/services/trackers/base.py](backend/services/trackers/base.py)). Adding a new tracker = one new file + entry in [registry.py](backend/services/trackers/registry.py).
- Metrics must agree with `py-motmetrics` within 0.5 % on MOT17-val. Hard gate; enforced by `tests/test_metrics_motmetrics.py`.
- **No `filterpy`, no `scipy.optimize.linear_sum_assignment` in production code.** Kalman + Hungarian are hand-written. This is the demo's pitch — do not shortcut.
- `numpy<2` is pinned because `py-motmetrics 1.4` still uses `np.asfarray`. Don't upgrade numpy without removing the motmetrics dependency.
- Per-frame matching in `metrics.py` uses **sticky continuity** (prefer previously-matched gt↔pred pairs) — without it our IDF1 drifted 3 % vs motmetrics on MOT17-09. See `_sticky_match_sequence` in [backend/services/metrics.py](backend/services/metrics.py).

## Adding a tracker

1. New file `backend/services/trackers/{name}.py` subclassing `BaseTracker`.
2. Register in [backend/services/trackers/registry.py](backend/services/trackers/registry.py) `_autoload()`.
3. Write tests in `backend/tests/test_{name}.py` (mirror test_custom.py / test_bytetrack.py — 5 tests minimum).
4. Run `make benchmark` to regenerate `frontend/public/benchmark*.json`.
5. Add a row to the comparison table in [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md).
6. Reproduce paper number on MOT17-val within 2 MOTA / 2 IDF1 before claiming done.

## Common workflows

```bash
# Algorithm dev loop (no GPU needed, runs in < 3 s)
cd backend && ../.venv/bin/python -m pytest -m "not slow" -q

# Full benchmark + write frontend JSON (~10 min on CPU)
make benchmark

# Fine-tune YOLO + re-benchmark with our detector (~2 hours CPU for 30 epochs img=640)
make prepare-dataset
make train
bash scripts/rebenchmark_yolo.sh runs/detect/runs/mot17/yolov8n_30ep_640/weights/best.pt

# DanceTrack: download via Voxel51 HF mirror, convert, infer, benchmark
# (samples.json + frames.json + per-seq mp4 — see scripts/prepare_dancetrack.py)

# Side-by-side demo MP4 for the README / LinkedIn post
make demo
```

## Dataset notes

- **MOT17** mirrored via HuggingFace `Lekim89/MOT17` (val split = ByteTrack-style second half of train sequences). Per-seq folder has gt/, det/ (FRCNN detections), img1/ (~50–120 MB each), seqinfo.ini.
- **DanceTrack** mirrored via HuggingFace `Voxel51/DanceTrack` (mp4 + frames.json holds normalized bboxes + track index per frame). `scripts/prepare_dancetrack.py` extracts frames via cv2 and writes MOT-format gt.txt.
- Both datasets are gitignored; the regenerated `frontend/public/benchmark*.json` files **are** committed so the live stress-test page works without re-running the benchmark.

## Working language

The author (user) is Vietnamese. Prefer Vietnamese for conversational explanations; English for code, docstrings, README, BLOG_DRAFT, TECHNICAL_DESIGN, and any text a US recruiter will read.
