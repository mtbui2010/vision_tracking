# Kalman filter for bbox tracking — derivation

> Source-of-truth markdown for the notebook. Run `jupytext --to notebook` to produce `.ipynb`.

## 1. The problem

We observe a bounding box every frame: $z_t = [u, v, s, r]^\top$ where $u, v$ is the center,
$s = w \cdot h$ is area, $r = w / h$ is aspect ratio. We want a smoothed estimate of the bbox
*and* its motion that can predict the next frame.

## 2. Why this state, not $[x, y, w, h]$

Aspect ratio $r$ varies very little for a pedestrian over short windows — a person seen from
multiple angles still has roughly the same $w/h$. Modeling area $s$ separately from aspect ratio
lets us put a *tighter* prior on $r$ (it should almost never change) than on $s$ (it grows /
shrinks as the person moves toward / away from the camera). With $[x_1, y_1, w, h]$ you cannot
express that asymmetry — you would need a full $4 \times 4$ covariance whose off-diagonals capture
$w$-$h$ correlation.

## 3. The constant-velocity model

State $x \in \mathbb{R}^7$:

$$
x = [u, v, s, r, \dot u, \dot v, \dot s]^\top
$$

Aspect ratio has no velocity (held constant). Transition:

$$
F = \begin{bmatrix}
1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

Measurement:

$$
H = \begin{bmatrix} I_4 & 0_{4 \times 3} \end{bmatrix}
$$

Predict:

$$
\hat x_{t|t-1} = F x_{t-1|t-1}, \quad P_{t|t-1} = F P_{t-1|t-1} F^\top + Q
$$

Update:

$$
y = z - H \hat x_{t|t-1}, \quad S = H P_{t|t-1} H^\top + R, \quad K = P_{t|t-1} H^\top S^{-1}
$$

$$
x_{t|t} = \hat x_{t|t-1} + K y, \quad P_{t|t} = (I - K H) P_{t|t-1}
$$

## 4. Noise covariances $Q$ and $R$

SORT defaults (which we adopt verbatim — the literature has not found anything that beats them):

| Quantity | Value | Why |
|---|---|---|
| $Q_{pos}$ | 1 | Position drift ~ 1 px per step prior. |
| $Q_{vel\ position}$ | 0.01 | Velocity is mostly constant — small process noise. |
| $Q_{vel\ scale}$ | $10^{-4}$ | Scale changes slowly. |
| $R_{u,v}$ | 1 | Detector center is sharp. |
| $R_{s,r}$ | 10 | Detector wobbles more on area / aspect than on center. |

## 5. Code

```python
from services.trackers.kalman import KalmanBoxTracker
import numpy as np

kf = KalmanBoxTracker(np.array([100, 100, 200, 200], dtype=float))
print("init bbox:", kf.bbox)
print("predict :", kf.predict())
kf.update(np.array([110, 100, 210, 200], dtype=float))
print("post-upd:", kf.bbox)
```

## 6. Sanity check — constant-velocity tracking

```python
import numpy as np
from services.trackers.kalman import KalmanBoxTracker
rng = np.random.default_rng(0)
bbox0 = np.array([100.0, 100.0, 150.0, 200.0])
kf = KalmanBoxTracker(bbox0)
err = []
for t in range(1, 51):
    kf.predict()
    truth = bbox0 + np.array([5 * t, 2 * t, 5 * t, 2 * t])
    noisy = truth + rng.normal(0, 1.0, 4)
    kf.update(noisy)
    err.append(np.linalg.norm(kf.bbox - truth))
print("steady-state error:", np.mean(err[-10:]))
```

You should see steady-state error well under 1 px after ~10 frames. The filter has *less* error
than the measurement noise it is fed.

## 7. Open question

Why is aspect ratio $r$ pinned constant rather than tracked with its own velocity? Because the
literature has shown that letting $r$ drift causes ID switches on rotating objects (e.g. dancers,
animals). The constant-$r$ assumption breaks for DanceTrack-style motion — this is one knob the
custom tracker can revisit (let $r$ drift but bound the rate).
