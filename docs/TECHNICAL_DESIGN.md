# Technical Design

This document captures the design decisions, algorithm choices, and evaluation methodology. It is intended to be readable by a perception / CV engineer doing a technical screen — short, no fluff, with explicit trade-offs.

## 1. Problem statement

Online multi-object tracking (MOT): given a stream of frames and per-frame detections, produce a set of trajectories with consistent identities across time. Online means each frame is processed without access to future frames. Detection is *given* (separate detector); the focus is on the **data association** problem.

## 2. Scope and non-goals

**In scope.**
- 2D bounding-box tracking on a single camera.
- Online operation (frame-by-frame).
- Pedestrians (MOT17, MOT20, DanceTrack); pluggable to other categories.
- Comparison of four trackers under a single eval harness.

**Non-goals (initial version).**
- 3D tracking. Stretch goal; KITTI subset only.
- Offline / global optimization (no graph-based methods such as min-cost flow).
- Detector training from scratch — we fine-tune YOLOv11.
- Robust to camera ego-motion. The baseline trackers assume a near-static camera; this is also the assumption of SORT / DeepSORT / ByteTrack.

## 3. Algorithm choices

### 3.1 Motion model — constant-velocity Kalman

State vector: `x = [u, v, s, r, u̇, v̇, ṡ]` where `(u, v)` is bbox center, `s = w·h` area, `r = w/h` aspect ratio (held constant). This follows SORT.

| Decision | Why |
|---|---|
| Linear Kalman over EKF / UKF | Bbox dynamics are smooth at 30 FPS; linear is fast and sufficient. Higher-order filters bring no measurable IDF1 gain on MOT17 (verified in the ByteTrack ablations). |
| Aspect ratio `r` held constant | Reduces state dimensionality; pedestrian aspect ratios are stable across short windows. Fails for highly articulated motion (dancers) — addressed by allowing `r` to drift in the DanceTrack-tuned variant. |
| Process noise `Q` scaled by state magnitude | Large objects move more in pixels per frame than small ones. Constant `Q` over-trusts predictions for nearby objects. |

### 3.2 Assignment — Hungarian algorithm

Bipartite matching minimizing cost = (1 − IoU) for SORT, gated by a max-cost threshold so unmatched detections become new tracks instead of forced matches.

| Decision | Why |
|---|---|
| Hungarian over greedy | Greedy matching gives suboptimal pairings under crowds; we show a 2–3 point IDF1 drop on MOT17 with greedy in the ablation. |
| Custom O(n^3) implementation | Algorithmic depth is part of the demo's pitch. Speed is non-blocking — at most ~100 tracks/frame, so n^3 is ~1ms. |
| Cost threshold (gating) | Forces creation of new tracks rather than spurious matches. Critical for ID consistency at scene boundaries (people entering/leaving). |

### 3.3 Appearance — ReID embedding

OSNet-x0.25 from Torchreid, pretrained on Market-1501. 512-dim embedding, cosine distance for matching.

**Why OSNet over a larger ReID model.** Pretrained OSNet hits ~94% rank-1 on Market-1501 at a fraction of the FLOPs of ResNet-50 ReID. Detector + tracker is already GPU-bound; we cannot afford a 25M-param ReID. The IDF1 gain from a bigger ReID is < 1 point in published comparisons and not worth the FPS cost.

### 3.4 Tracker variants

| Tracker | Cost function | Stage |
|---|---|---|
| SORT | `1 − IoU` | 1-stage Hungarian |
| DeepSORT | `α · (1 − cos(appearance)) + (1 − α) · Mahalanobis(motion)` | 2-stage cascade by track age |
| ByteTrack | `1 − IoU` | 2-stage: high-score dets first, low-score dets to unmatched tracks |
| Custom | TBD — appearance-gated ByteTrack with a confidence-aware Kalman | 2-stage |

The "custom" tracker is the **research bet**: combine ByteTrack's low-score-detection rescue with appearance gating to reduce ID switches on similar-appearance clips (DanceTrack). Goal: beat ByteTrack on DanceTrack IDF1 by ≥ 2 points without losing FPS.

## 4. Evaluation

### 4.1 Metrics — implemented from scratch

| Metric | Why it matters | Implementation note |
|---|---|---|
| **MOTA** | Standard. Combines FP, FN, ID switches. | Known to over-weight detection errors; can be ~80% even with poor identity preservation. |
| **IDF1** | Identity-preservation specific. | Computed via global ID assignment (Hungarian on track ↔ ground-truth pairs) over the full sequence. |
| **HOTA** | Recent (2020) — geometric mean of detection and association quality at multiple IoU thresholds. | Now the headline metric on MOT benchmarks; we report HOTA-α=0.5 as the primary number. |
| FP, FN, IDSW | Diagnostic. | Reported per tracker for the stress-test page. |
| FPS | Productization. | Wall-clock excluding I/O; reported on a fixed reference GPU. |

**Cross-check against `py-motmetrics`.** The first thing we verify is that our MOTA / IDF1 match `py-motmetrics` to within 0.5%. If they diverge, our implementation is wrong, not the library's.

### 4.2 Benchmarks

| Dataset | Why included |
|---|---|
| MOT17-val (held out from train) | Standard pedestrian benchmark; published numbers exist for every tracker. |
| MOT20-val | Crowded scenes (avg ~170 boxes/frame). Tests scaling. |
| DanceTrack-val | Same-appearance, large motion. The hardest available test for appearance-based methods — most appearance trackers *underperform* SORT here. Critical to surface failure modes. |

### 4.3 Reference numbers

Aggregated across 7 MOT17 val sequences (02, 04, 05, 09, 10, 11, 13) with
detections from the provided FRCNN det.txt at confidence ≥ 0.3, no detector
fine-tune yet. Sequence-length-weighted MOTA; mean of per-sequence IDF1/HOTA.
Tracker-only FPS (Python, CPU, single core).

| Tracker | MOTA | IDF1 | HOTA | DetA | AssA | IDSW | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| SORT (ours) | 0.278 | 0.417 | 0.389 | 0.323 | 0.492 | 281 | 1822 |
| DeepSORT (ours) | 0.278 | 0.435 | 0.404 | 0.324 | 0.529 | 230 | 1367 |
| ByteTrack (ours) | 0.240 | 0.273 | 0.273 | 0.251 | 0.334 | 506 | 1368 |
| **Custom (ours)** | **0.278** | **0.437** | 0.403 | 0.321 | 0.532 | **181** | 1413 |

**Reproducibility caveats.**
- Absolute MOTA is well below the paper numbers (~0.6 for SORT, ~0.78 for
  ByteTrack on MOT17-val). Two reasons, in order of impact: (a) we use the
  provided FRCNN detections, whose recall is ~25% on these val splits while
  the paper-grade detectors (POI, SDP, YOLOX) hit ~50–60%; (b) no detector
  fine-tune yet (week 5 lands that). MOTA = 0.28 with FN = 70k on 100k GT
  → FN dominates the score by a wide margin.
- **The relative ordering is the part worth reading.** Custom ≈ DeepSORT > SORT > ByteTrack
  on this detector. ByteTrack's 2-stage matching expects the low-confidence
  channel to carry occluded versions of tracked objects; with a sparse FRCNN
  detector that channel is mostly false positives, so the second stage
  introduces ID switches (506 IDSW vs SORT 281) instead of rescuing tracks.
- **The custom tracker validates the hypothesis.** Add an appearance gate to
  ByteTrack's stage-2 ("low-conf det can only revive a track if its ReID
  embedding cosine-matches the track's gallery") and scale Kalman R by
  1/score so spurious low-conf updates don't pull the state. Result: best
  IDF1 (0.437, narrowly beats DeepSORT), fewest IDSW (**181 — 36 % below
  SORT, 64 % below ByteTrack**), HOTA tied with DeepSORT, slightly faster.
  Biggest win on MOT17-13 (camera motion, where ByteTrack collapsed to
  MOTA=0.058) — custom hit 0.318, IDF1=0.444.
- Once the YOLOv11n fine-tune lands (week 5), absolute MOTA should jump to
  paper range and ByteTrack should overtake SORT, matching the paper's claim.
  We will re-run this table at that point.

Acceptance gate (week 2): our metrics agree with `py-motmetrics` within
0.5%. **Status: passing on MOT17-09-FRCNN** (see
`tests/test_metrics_motmetrics.py`).

## 5. System design

Reuses the pattern of `inferix`: stateless FastAPI backend, outbound-WebSocket GPU worker on user's PC, optional RunPod serverless worker for cloud.

### 5.1 Tracking job lifecycle

```
client                backend               worker
  |  POST /jobs (video) -->                    |
  |                       enqueue job          |
  |                          ----- job ----->  |
  |  WS /jobs/{id}/stream -->                  |
  |                          <-- log frames -- |
  |                          <-- done w/ url - |
```

### 5.2 Why outbound WebSocket from worker

Same reason as `inferix`: worker can run behind NAT with no firewall rules, no SSH tunnel, no port forwarding. The user runs `worker.sh` on their PC and it connects out to the public backend. This matters because the demo needs to run from anywhere recruiter clicks it.

### 5.3 Compute target routing

Per job: `local` / `cloud` / `hybrid`. Same routing logic as `inferix.job_worker.py`. Default for the live demo is `cloud` (RunPod) so recruiters don't need to set up anything; `local` is for development.

## 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| RunPod cold-start latency kills the demo experience | High | Keep at least one worker warm during the public demo window; show a "warming up" screen. |
| MOT eval metrics off-by-something vs. published | High | Cross-check `py-motmetrics` *first*. Required acceptance gate for week 2. |
| ByteTrack reproduction misses paper numbers | Medium | Most reproductions of ByteTrack diverge by ~1 MOTA — acceptable. Document the specific source of divergence in the blog. |
| ReID training drags out timeline | Medium | Use pretrained OSNet only; do not train ReID from scratch in v1. |
| Frontend Canvas overlay drops frames under 30 FPS | Medium | Render every 2nd frame at high resolution, or drop to OffscreenCanvas. |

## 7. Open questions

- Do we add **camera motion compensation** (CMC, used in BoT-SORT) for the custom variant? Adds ~2 IDF1 but doubles wall time. Decide after the DanceTrack baseline.
- Do we support **interactive single-object tracking** (click an object, follow it)? Nice for a demo but orthogonal to the MOT pitch. Stretch goal.
