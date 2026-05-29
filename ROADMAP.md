# Roadmap

Eight-week build plan, ~10 hours/week. Each week ships something demoable.

The order is deliberate: ship the **evaluation harness first**, so every algorithm change after that produces a number you can compare against the previous week.

## Phase 1 — Algorithm core (weeks 1–4)

### Week 1 — Kalman + Hungarian from scratch

- [ ] `services/trackers/kalman.py` — constant-velocity Kalman filter, state `[x, y, s, r, vx, vy, vs]`. No `filterpy`.
- [ ] `services/trackers/hungarian.py` — Jonker-Volgenant or simple O(n^3) Hungarian. No `scipy.optimize.linear_sum_assignment`.
- [ ] Unit tests covering: known-trajectory tracking, identity assignment on synthetic 2D points, predict / update determinism.
- [ ] `docs/notebooks/01_kalman_derivation.ipynb` — derive the constant-velocity model, explain why bbox aspect ratio is kept constant in SORT.
- **Deliverable:** all green tests + notebook walkthrough.

### Week 2 — SORT baseline + MOT17 eval pipeline

- [ ] `services/trackers/sort.py` — IoU-based association on top of Kalman + Hungarian.
- [ ] `services/metrics.py` — MOTA, IDF1 from scratch. Cross-check against `py-motmetrics` (must be ≤ 0.5% off).
- [ ] `scripts/eval.py` — run a tracker over MOT17-val, dump `tracker_name.txt` in MOT format, print metrics.
- [ ] `docs/notebooks/02_metrics.ipynb` — derive MOTA / IDF1 from confusion-matrix definitions; show why MOTA over-counts ID switches.
- **Deliverable:** SORT MOTA on MOT17-val ≥ 0.55 (paper number is ~0.59).

### Week 3 — ReID embeddings + DeepSORT

- [ ] `services/reid/embedder.py` — extract OSNet embeddings (pretrained) for each detection crop.
- [ ] `services/trackers/deepsort.py` — DeepSORT matching cascade: gated Mahalanobis distance + cosine appearance.
- [ ] Compare IDF1 vs. SORT on MOT17-val.
- [ ] `docs/notebooks/03_reid.ipynb` — visualize cosine distance distribution for same-ID vs. different-ID pairs.
- **Deliverable:** IDF1 improvement of ≥ 5 points over SORT.

### Week 4 — ByteTrack 2-stage association

- [ ] `services/trackers/bytetrack.py` — high-confidence detections matched first, then low-confidence detections matched to unmatched tracks.
- [ ] Compare across all three trackers on MOT17-val + DanceTrack-val.
- [ ] `docs/notebooks/04_bytetrack_ablation.ipynb` — score-threshold sensitivity, kept vs. discarded detections.
- **Deliverable:** ByteTrack MOTA ≥ 0.76 on MOT17-val (within 2 points of paper).

## Phase 2 — Production pipeline (weeks 5–7)

### Week 5 — Fine-tune YOLO on MOT / DanceTrack

- [ ] Re-export MOT17 / DanceTrack annotations to YOLO format (`scripts/prepare_dataset.py`).
- [ ] Fine-tune YOLOv11n / YOLOv11s on combined dataset. Export to ONNX.
- [ ] Re-run all trackers with the fine-tuned detector — verify all metrics improve.
- **Deliverable:** detector mAP@0.5 ≥ 0.85 on MOT17-val, all tracker numbers refreshed.

### Week 6 — Backend + job worker (port from inferix)

- [ ] FastAPI app with `/tracking/jobs` POST, `/jobs/{id}` GET, `/jobs/{id}/stream` WebSocket for live progress.
- [ ] Outbound-WebSocket GPU worker that runs detector + tracker on local PC.
- [ ] RunPod serverless handler for batch inference.
- [ ] Compute target routing: `local` / `cloud` / `hybrid` (same pattern as inferix).
- **Deliverable:** upload a video via API, get a tracked output video back.

### Week 7 — Frontend: tracker-lab + compare

- [ ] Next.js App Router scaffold (`create-next-app`).
- [ ] `/tracker-lab` — upload video, pick tracker, see annotated playback (Canvas overlay).
- [ ] `/compare` — 2×2 grid running 4 trackers in parallel on the same clip. Live MOTA / IDF1 / FPS counters.
- [ ] Webcam mode using `getUserMedia` + WebSocket streaming.
- **Deliverable:** demoable web app on `localhost`.

## Phase 3 — Differentiators (week 8)

### Week 8 — Algorithm visualization + stress test + blog

- [ ] `/algorithm` — interactive: pause on a frame, see Kalman predicted ellipses, Hungarian cost matrix, association decisions colored by accept / reject.
- [ ] `/stress-test` — curated DanceTrack clips with heavy occlusion; ID-switch leaderboard.
- [ ] Blog post draft: "Implementing four multi-object trackers from scratch and what I learned about identity preservation."
- [ ] Deploy: backend on a small VPS, frontend on Vercel, RunPod handler published.
- **Deliverable:** public live demo link + blog post.

## Stretch goals (weeks 9–10, optional)

- [ ] **Multi-camera ReID page** — drop two clips from different cameras, cross-view person matching. Train ReID head on Market-1501.
- [ ] **3D MOT mini-page** — KITTI subset, simple monocular depth + 3D Kalman. Specifically for AV-adjacent roles.
- [ ] **Edge deployment** — TensorRT export, benchmark FPS on Jetson Orin Nano.
- [ ] **Self-recorded multi-camera dataset** — 2–3 cameras at home or campus, annotate, evaluate cross-view ReID.

## Definition of done (recruiter-facing)

- Live demo URL on the resume that loads in < 3 seconds.
- GitHub README with embedded GIF / video.
- Blog post with at least one **non-obvious technical insight** (something a recruiter would ask about in interview).
- All numbers on the comparison page are reproducible from the eval script.
- Three written-out interview talking points:
  1. Why Hungarian for assignment, and what fails when you replace it with greedy matching.
  2. Why DeepSORT's cascade depth = max_age is a tradeoff and not "best practice."
  3. Why ByteTrack's low-score branch helps on crowded scenes but hurts on small targets.
