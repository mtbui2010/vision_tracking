"""Convert MOT-format annotations into YOLO-format labels for detector fine-tuning.

MOT format (one row per box):
    frame, id, x, y, w, h, conf, class, vis

YOLO format (one .txt per image):
    class cx cy w h     (all normalized to [0,1])

Usage:
    python scripts/prepare_dataset.py \
        --src datasets/MOT17/train \
        --dst datasets/MOT17_yolo \
        --val-frac 0.2 \
        --class-id 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_seqinfo(seqinfo_path: Path) -> dict[str, str]:
    """Read MOTChallenge seqinfo.ini (key=value lines)."""
    info: dict[str, str] = {}
    for line in seqinfo_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def convert_sequence(
    seq_dir: Path,
    out_root: Path,
    split: str,
    class_id: int,
) -> int:
    """Convert one MOT sequence (e.g. MOT17-04-FRCNN) to YOLO labels.

    Returns the number of labeled images written.
    """
    info = parse_seqinfo(seq_dir / "seqinfo.ini")
    width = int(info["imWidth"])
    height = int(info["imHeight"])
    img_dir = seq_dir / info.get("imDir", "img1")
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        return 0

    rows = np.loadtxt(str(gt_path), delimiter=",", dtype=np.float64, ndmin=2)
    # MOT17 gt convention: column 6 (conf) is 1 for valid pedestrians, 0 to skip
    if rows.shape[1] >= 7:
        rows = rows[rows[:, 6] == 1]
    # column 7 = class (1 = pedestrian on MOT17)
    if rows.shape[1] >= 8:
        rows = rows[rows[:, 7] == 1]

    by_frame: dict[int, list[np.ndarray]] = {}
    for r in rows:
        by_frame.setdefault(int(r[0]), []).append(r)

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for frame_idx, rs in by_frame.items():
        src_img = img_dir / f"{frame_idx:06d}.jpg"
        if not src_img.exists():
            continue
        dst_img = out_img_dir / f"{seq_dir.name}_{frame_idx:06d}.jpg"
        dst_lbl = out_lbl_dir / f"{seq_dir.name}_{frame_idx:06d}.txt"
        if not dst_img.exists():
            try:
                dst_img.symlink_to(src_img.resolve())
            except FileExistsError:
                pass

        lines = []
        for r in rs:
            x, y, w, h = r[2], r[3], r[4], r[5]
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            nw = w / width
            nh = h / height
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        dst_lbl.write_text("\n".join(lines))
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sequences = sorted([p for p in args.src.iterdir() if (p / "gt").exists()])
    if not sequences:
        raise SystemExit(f"no MOT sequences under {args.src}")

    val_set = set(
        rng.choice(len(sequences), size=int(len(sequences) * args.val_frac), replace=False).tolist()
    )

    total_train = 0
    total_val = 0
    for i, seq in enumerate(sequences):
        split = "val" if i in val_set else "train"
        n = convert_sequence(seq, args.dst, split, args.class_id)
        print(f"{seq.name} -> {split} ({n} frames)")
        if split == "train":
            total_train += n
        else:
            total_val += n

    yaml = args.dst / "data.yaml"
    yaml.write_text(
        f"path: {args.dst.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: [person]\n"
    )
    summary = {"train_frames": total_train, "val_frames": total_val, "sequences": [s.name for s in sequences]}
    (args.dst / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
