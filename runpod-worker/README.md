# RunPod serverless worker

Handler for RunPod serverless. Used for cloud inference (default for the public live demo) and for heavy batch eval runs over MOT17 / MOT20.

To be implemented in **week 6** of the [ROADMAP](../ROADMAP.md), based on `inferix/runpod-worker/handler.py`.

Actions:

| Action | Input | Output |
|---|---|---|
| `track` | `{video_b64, tracker_name, detector_ckpt}` | `{mot_format_b64, metrics_json}` |
| `batch_eval` | `{dataset, tracker_name}` | `{metrics_json}` |

Cold start mitigation: preload detector + ReID into the container image (see `preload_models.py` in `inferix`).
