# GPU worker

Outbound-WebSocket worker that runs the detector + tracker on the user's local PC and ships logs / results back to the backend.

To be implemented in **week 6** of the [ROADMAP](../ROADMAP.md), reusing the pattern from `inferix/gpu-worker/`.

The worker:
- connects to `wss://backend/api/worker/connect?token=TOKEN`;
- handles `infer` and `track` job messages;
- streams `log` updates frame-by-frame;
- sends `done` with the output video URL when complete.

Liveness is heartbeat-based. The backend never dials out to the worker.
