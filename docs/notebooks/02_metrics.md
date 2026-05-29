# MOTA, IDF1, HOTA — derivations and pitfalls

> Markdown source. Render to `.ipynb` with `jupytext --to notebook`.

## 1. MOTA — Multi-Object Tracking Accuracy

Per Bernardin & Stiefelhagen (CLEAR MOT, 2008):

$$
\text{MOTA} = 1 - \frac{\sum_t (\text{FP}_t + \text{FN}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}
$$

Per frame: greedy / Hungarian assign GT to predictions by IoU above a threshold (typically 0.5).
False positives are predictions that did not match. False negatives are GTs that did not match.
ID switch: a GT that *did* match this frame got matched to a *different* prediction ID than the
last frame it was matched.

### Pitfalls we hit

- **MOTA can go negative.** When `FP + FN + IDSW > GT`. Trips up newcomers; the number is fine.
- **MOTA over-weights detection errors.** A tracker with perfect identity preservation but a
  bad detector reports a worse MOTA than a tracker that flips identities constantly but on a
  good detector. Use IDF1 / HOTA alongside.
- **IDSW depends on the matching threshold.** A loose IoU threshold (0.3) hides switches that
  show up at 0.5. Always report at multiple thresholds for fair comparison.

## 2. IDF1 — Identity F1

Per Ristani et al. (2016). Build a global GT-id × pred-id cost matrix where cost = "frames either
appears" - 2 * "frames both appear matched." Run Hungarian once over the entire sequence to assign
each GT id to one pred id.

- **IDTP** = sum of matched frames over the assignment.
- **IDFN** = total GT detections - IDTP.
- **IDFP** = total pred detections - IDTP.

$$
\text{IDF1} = \frac{2 \cdot \text{IDTP}}{2 \cdot \text{IDTP} + \text{IDFP} + \text{IDFN}}
$$

### Why this matters

IDF1 is the *only* common MOT metric that puts identity preservation on equal footing with
detection. MOTA can't distinguish "perfect IDs, missed some detections" from "every detection but
swapped IDs constantly." For a tracking demo aimed at robotics / AV recruiters, **IDF1 is the
number to lead with.**

## 3. HOTA — Higher Order Tracking Accuracy

Luiten et al. (IJCV 2020). The geometric mean of detection and association accuracy, averaged over
IoU thresholds $\alpha \in \{0.05, 0.10, \ldots, 0.95\}$:

$$
\text{HOTA} = \frac{1}{|\mathcal{A}|} \sum_{\alpha \in \mathcal{A}} \sqrt{\text{DetA}(\alpha) \cdot \text{AssA}(\alpha)}
$$

$$
\text{DetA}(\alpha) = \frac{\text{TP}(\alpha)}{\text{TP}(\alpha) + \text{FP}(\alpha) + \text{FN}(\alpha)}
$$

$$
\text{AssA}(\alpha) = \frac{1}{\text{TP}(\alpha)} \sum_{c \in \text{TP}(\alpha)} \frac{\text{TPA}(c)}{\text{TPA}(c) + \text{FPA}(c) + \text{FNA}(c)}
$$

where for each true positive $c$ (a GT-pred match), TPA counts the matches between this GT id
and this pred id; FPA / FNA the opposites.

### Why HOTA over MOTA

HOTA is now the headline MOT-benchmark number (since 2020). It decomposes cleanly:
- DetA tells you about the *detector* and matching threshold.
- AssA tells you about *identity preservation* across the sequence.
- Their geometric mean penalizes any tracker that is great at one but bad at the other.

In our comparison page, we report HOTA, DetA, AssA, and IDF1 — MOTA is shown for historical
continuity but is the *last* column.

## 4. Sanity check

```python
import numpy as np
from services.metrics import FrameAnnotations, compute_metrics

def ann(ids, boxes):
    return FrameAnnotations(ids=np.array(ids, dtype=np.int64), bboxes=np.array(boxes, dtype=float))

gt = [ann([1], [[0,0,10,10]]), ann([1], [[5,0,15,10]])]
pred_perfect = gt
pred_id_swap = [ann([1], [[0,0,10,10]]), ann([2], [[5,0,15,10]])]

m_perfect = compute_metrics(gt, pred_perfect)
m_swap = compute_metrics(gt, pred_id_swap)

print(f"perfect: MOTA={m_perfect.mota:.2f} IDF1={m_perfect.idf1:.2f} HOTA={m_perfect.hota:.2f}")
print(f"swap   : MOTA={m_swap.mota:.2f}  IDF1={m_swap.idf1:.2f}  HOTA={m_swap.hota:.2f}")
```

Expected: perfect → 1.0/1.0/~1.0, swap → ~0.5/0.5/lower.

## 5. Cross-checking against py-motmetrics

The repo's acceptance gate is that our MOTA / IDF1 agree with `py-motmetrics` to within 0.5% on
a real MOT17-val sequence. Add the cross-check in `tests/test_metrics_motmetrics.py` once you
have a sequence downloaded — keep it under `@pytest.mark.slow` so it stays out of the default run.
