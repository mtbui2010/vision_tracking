# Datasets

This directory holds the raw MOT datasets locally. **None of it is committed** (see `.gitignore`).

## Layout

```
datasets/
├── MOT17/         from https://motchallenge.net/data/MOT17/
├── MOT20/         from https://motchallenge.net/data/MOT20/
├── DanceTrack/    from https://github.com/DanceTrack/DanceTrack
└── Market-1501/   from https://www.kaggle.com/datasets/pengcw1/market-1501
                   (used for ReID in week 3 and the multi-camera page in stretch goals)
```

## Download

Use `scripts/download_datasets.py` (to be written in week 2) which:
- pulls the official archives;
- verifies checksums;
- unpacks to the above layout.

For now, download manually from the links above and unzip into `datasets/`.

## Splits

- **MOT17**: use the first half of each training sequence for training the YOLO fine-tune, second half as `MOT17-val` for evaluation. This matches the convention used by ByteTrack.
- **DanceTrack**: official train / val / test splits.
- **Market-1501**: official split.
