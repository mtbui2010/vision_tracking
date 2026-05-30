#!/usr/bin/env bash
# Run a fine-tuned YOLO over every MOT17 sequence under datasets/MOT17/train/,
# emit each one's det.txt under datasets/MOT17_yolo_dets/, then re-run the
# 4-tracker benchmark on the new detections.
#
# Usage:  bash scripts/rebenchmark_yolo.sh <path/to/best.pt>
set -euo pipefail

WEIGHTS=${1:-runs/mot17/yolov8n_10ep/weights/best.pt}
SRC=datasets/MOT17/train
OUT=datasets/MOT17_yolo_dets
PY=.venv/bin/python

if [[ ! -f "$WEIGHTS" ]]; then
  echo "weights not found: $WEIGHTS" >&2
  exit 1
fi

echo "=== running YOLO over each sequence ==="
mkdir -p "$OUT"
for seq_dir in "$SRC"/*/; do
  seq=$(basename "$seq_dir")
  img1="$seq_dir/img1"
  if [[ ! -d "$img1" ]]; then
    echo "skip $seq (no img1)"; continue
  fi
  mkdir -p "$OUT/$seq/det" "$OUT/$seq/gt"
  cp -f "$seq_dir/gt/gt.txt"     "$OUT/$seq/gt/gt.txt"
  cp -f "$seq_dir/seqinfo.ini"   "$OUT/$seq/seqinfo.ini" 2>/dev/null || true
  echo "--- $seq ---"
  $PY scripts/infer_detections.py \
      --weights "$WEIGHTS" \
      --frames "$img1" \
      --out "$OUT/$seq/det/det.txt" \
      --img 640 \
      --score-threshold 0.05
done

echo
echo "=== re-benchmark on fine-tuned YOLO detections ==="
$PY scripts/benchmark.py \
    --root "$OUT" \
    --out-json frontend/public/benchmark_yolo.json \
    --out-md docs/_generated_benchmark_yolo.md \
    --score-threshold 0.3
