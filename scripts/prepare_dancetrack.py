"""Convert Voxel51's DanceTrack export (samples.json + frames.json + mp4s) to
the MOT-format directory layout used by the rest of the eval harness.

Produces, per video:
    datasets/DanceTrack/<split>/<seq>/
        img1/000001.jpg
        gt/gt.txt        (MOT format)
        seqinfo.ini

Bbox in frames.json is normalized [x, y, w, h] in [0,1]. We denormalize using
the video frame_width/frame_height from samples.json, then write standard MOT.

Usage:
    python scripts/prepare_dancetrack.py \\
        --voxel51-dir datasets/DanceTrack_voxel51 \\
        --out-dir datasets/DanceTrack \\
        --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2  # type: ignore[import-not-found]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel51-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()

    samples_path = args.voxel51_dir / "samples.json"
    frames_path = args.voxel51_dir / "frames.json"
    samples = json.loads(samples_path.read_text())["samples"]
    print(f"loaded {len(samples)} samples from samples.json")

    by_sample_id: dict[str, dict] = {}
    for s in samples:
        if args.split not in s.get("tags", []):
            continue
        oid = s["_id"]["$oid"]
        by_sample_id[oid] = {
            "filepath": s["filepath"],
            "name": Path(s["filepath"]).stem,
            "width": s["metadata"]["frame_width"],
            "height": s["metadata"]["frame_height"],
            "frames": s["metadata"]["total_frame_count"],
        }
    print(f"{len(by_sample_id)} sequences tagged '{args.split}'")

    print(f"streaming frames.json ({frames_path.stat().st_size // 1_000_000} MB)...")
    by_seq: dict[str, dict[int, list[tuple[int, float, float, float, float]]]] = {}
    for s in by_sample_id.values():
        by_seq[s["name"]] = {}

    data = json.loads(frames_path.read_text())
    for f in data["frames"]:
        sample_id = f["_sample_id"]["$oid"]
        if sample_id not in by_sample_id:
            continue
        seq_name = by_sample_id[sample_id]["name"]
        fi = int(f["frame_number"])
        dets = f.get("gt", {}).get("detections", []) or []
        rows = []
        for d in dets:
            bx = d.get("bounding_box")
            tid = int(d.get("index", -1))
            if bx is None:
                continue
            rows.append((tid, *bx))
        by_seq[seq_name][fi] = rows

    available_mp4s = {p.stem: p for p in args.voxel51_dir.glob("*.mp4")}
    processed = 0
    for seq_name, info in by_sample_id.items():
        name = info["name"]
        mp4 = available_mp4s.get(name)
        if mp4 is None:
            print(f"skip {name} — mp4 not downloaded")
            continue
        seq_root = args.out_dir / args.split / name
        img_dir = seq_root / "img1"
        gt_dir = seq_root / "gt"
        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        w, h = info["width"], info["height"]
        # Extract frames
        cap = cv2.VideoCapture(str(mp4))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            cv2.imwrite(str(img_dir / f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cap.release()
        n_frames = idx
        print(f"{name}: extracted {n_frames} frames ({w}x{h})")

        # Write gt.txt
        lines = []
        frames_data = by_seq.get(name, {})
        for fi in sorted(frames_data.keys()):
            for tid, bx, by_, bw, bh in frames_data[fi]:
                if tid < 0:
                    continue
                x = bx * w
                y = by_ * h
                bbw = bw * w
                bbh = bh * h
                # MOT format: frame, id, x, y, w, h, conf, class, vis
                # use 1-indexed track id (MOT convention); voxel51 indexes start at 0
                lines.append(f"{fi},{tid + 1},{x:.2f},{y:.2f},{bbw:.2f},{bbh:.2f},1,1,1")
        (gt_dir / "gt.txt").write_text("\n".join(lines))

        # Write seqinfo.ini
        (seq_root / "seqinfo.ini").write_text(
            f"[Sequence]\nname={name}\nimDir=img1\nframeRate=25\n"
            f"seqLength={n_frames}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n"
        )
        processed += 1

    print(f"\nprocessed {processed} sequences -> {args.out_dir / args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
