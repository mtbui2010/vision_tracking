# vision_tracking — Tracker Lab

Portfolio project: a multi-object tracking research playground built to demonstrate algorithm depth + ML systems engineering for US CV / perception roles.

See [README.md](README.md) for the recruiter pitch, [ROADMAP.md](ROADMAP.md) for the week-by-week plan, [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md) for algorithm and eval design.

## Structure

```
vision_tracking/
├── backend/              FastAPI service (Python)
│   ├── api/              HTTP + WS handlers (week 6+)
│   ├── services/
│   │   ├── trackers/     SORT, DeepSORT, ByteTrack, custom — all from scratch
│   │   │   └── base.py   Tracker interface (Detection, Track, BaseTracker)
│   │   ├── reid/         ReID embedder (OSNet)
│   │   ├── detector.py   YOLOv11 wrapper (week 5+)
│   │   └── metrics.py    MOTA / IDF1 / HOTA from scratch
│   ├── workers/          Job consumer (week 6+)
│   ├── tests/            pytest
│   ├── main.py           FastAPI entrypoint (currently /health only)
│   └── pyproject.toml
├── frontend/             Next.js App Router (TypeScript) — scaffolded week 7
├── gpu-worker/           Outbound-WS worker (user PC) — week 6
├── runpod-worker/        RunPod serverless handler — week 6
├── docs/
│   ├── TECHNICAL_DESIGN.md
│   └── notebooks/        Derivations + ablations (Kalman, metrics, ReID, ByteTrack)
├── scripts/              eval.py, prepare_dataset.py
└── datasets/             MOT17/MOT20/DanceTrack (gitignored)
```

## Architecture borrowed from inferix

This project intentionally reuses patterns from `/home/trung/trung_workdir/inferix/`:

- Outbound-WebSocket GPU worker (works behind NAT, no port forwarding).
- Compute target routing: `local` / `cloud` / `hybrid`.
- Single typed `services/api.ts` for the frontend.
- RunPod serverless handler for cloud GPU.

When implementing weeks 6–7, reference inferix's `gpu-worker/worker.py`, `backend/workers/job_worker.py`, and `frontend/services/api.ts`.

## Key invariants

- Bbox values are **pixel coords** `[x1, y1, x2, y2]` (unlike inferix, which uses normalized [0,1] for annotation). Reason: tracking math is more naturally expressed in pixels.
- All trackers implement `BaseTracker` (see [backend/services/trackers/base.py](backend/services/trackers/base.py)). Adding a new tracker = one new file + entry in a registry.
- Metrics implementations must agree with `py-motmetrics` within 0.5% on MOT17-val. This is a hard acceptance gate for week 2.
- No `filterpy`, no `scipy.optimize.linear_sum_assignment`. Kalman + Hungarian are hand-written. This is the demo's pitch — do not shortcut.

## Adding a tracker

1. New file `backend/services/trackers/{name}.py` subclassing `BaseTracker`.
2. Add to the eval registry in `scripts/eval.py`.
3. Add a row to the comparison table in [docs/TECHNICAL_DESIGN.md](docs/TECHNICAL_DESIGN.md).
4. Reproduce paper number on MOT17-val within 2 MOTA / 2 IDF1 before claiming done.

## Working language

The author (user) is Vietnamese. Prefer Vietnamese for conversational explanations; English for code, docstrings, README, and any text a US recruiter will read.
