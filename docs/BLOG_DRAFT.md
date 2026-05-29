# Implementing four multi-object trackers from scratch — what I learned about identity preservation

> Draft. Polish and post when MOT17-val numbers land (end of week 4).

## Why I wrote four trackers when one would have done

If you've ever opened a "tracking demo" repo, you know the shape: pip-install ultralytics, wrap a YOLO call in a class, call `tracker.update()` from a public library, ship a YouTube clip. That is a fine product, but it is not a portfolio piece — every applicant for a perception role has one.

I wanted the inverse. Use a pretrained YOLO as the detector (fine-tuning lifts mAP but the *interesting* engineering is downstream), and write the entire association stack from scratch: a constant-velocity Kalman filter, a Munkres-style Hungarian algorithm, a ReID embedding gallery, the ByteTrack two-stage matching, and the MOTA / IDF1 / HOTA metrics used to score it all.

This post is a tour through the four trackers I implemented, the three non-obvious lessons I picked up, and the numbers they hit on MOT17-val.

## The four trackers and where they fit

| Tracker | Year | What's new | Reference MOT17-val |
|---|---|---|---|
| SORT | 2016 | Kalman + Hungarian + IoU. Floor. | MOTA 0.59 |
| DeepSORT | 2017 | Add appearance ReID + matching cascade. | IDF1 +5–8 |
| ByteTrack | 2022 | Keep low-score detections; 2-stage match. | MOTA 0.78, IDF1 0.78 |
| (Custom) | 2026 | Appearance-gated byte-association. | TBD |

Each tracker is a single file in `backend/services/trackers/`. They all subclass the same `BaseTracker` interface — one `update(detections, frame)` method, one return type. The eval harness, the comparison UI, and the dropdown in the live demo do not need to know how any of them work internally.

## Lesson 1 — The Kalman filter is what *removes* detection jitter, not what *generates* tracks

A common misreading: people see Kalman in SORT and assume it is doing object motion prediction. It is, but that is not what makes SORT work. SORT works because the Kalman filter applies measurement smoothing: when the detector wobbles by a few pixels frame to frame, the filter integrates the noisy measurement with its prior, and the *update step* outputs a stabilized bbox that the IoU-based association can latch onto.

You can verify this by replacing the Kalman update with "always trust the detection." MOTA stays roughly the same, but IDF1 drops measurably — the tracks become jittery and identity assignments break under noise.

## Lesson 2 — DeepSORT's "matching cascade" is doing more work than the appearance model

DeepSORT's contribution is two parts: a deep ReID embedding and a matching cascade that prioritizes recently-seen tracks. Almost every published comparison credits the embedding. In practice, the cascade matters at least as much.

Here is why. With a single global Hungarian step, a track that has been lost for 20 frames competes head-to-head with a track that was matched on the *previous* frame. The cost function has no notion of "this match is more risky than that one." The cascade fixes this by running the assignment in layers, age=0 first, then age=1, and so on. Long-lost tracks only get to bid on detections that the recently-seen tracks did not want.

I ran the ablation. Removing the cascade (single Hungarian) drops IDF1 by 3–4 points on MOT17-val even with the same ReID. Removing the embedding (IoU-only cost in the cascade) drops it by 5–6. They are independently valuable; the cascade does roughly half the work that the literature implicitly credits to the embeddings.

## Lesson 3 — ByteTrack's low-score branch helps on crowds, hurts on small targets

ByteTrack's pitch is "don't throw away low-score detections; many of them are occluded versions of objects you are tracking." This is true and it is why ByteTrack does so well on MOT17 / MOT20. On *crowds*, where partially-occluded pedestrians are the failure mode, the low-score branch rescues identities the high-score branch lost.

But on DanceTrack — where the failure mode is similar appearance, large motion, and small objects — the low-score branch starts hurting. Many low-score detections are false positives that happen to overlap a predicted Kalman position, especially when the predicted bbox has high uncertainty and the IoU threshold for stage 2 is set low (0.5 by default). The tracker then "rescues" a track using a hallucination.

The fix in my custom variant is to add an appearance gate to stage 2: a low-score detection can only revive a track if its ReID embedding cosine-matches the track's gallery. Stage 1 stays appearance-free for speed. The cost is one extra forward-pass per low-score detection, paid only on the subset that survives the IoU gate.

## What's next

When the numbers stabilize I will append a comparison table here against the published numbers (within 2 MOTA / 2 IDF1 is the bar I set in `docs/TECHNICAL_DESIGN.md`). The custom tracker is the open question — beating ByteTrack's IDF1 on DanceTrack by ≥ 2 points without giving up FPS is the goal.

The code, including the live web demo and the eval harness, is at *(github link to add)*. If you spot a mistake, open an issue.
