.PHONY: install test test-slow backend frontend worker eval benchmark demo prepare-dataset train clean help

help:
	@echo "Tracker Lab — common targets"
	@echo "  install         python venv + npm install"
	@echo "  test            run fast test suite (51 tests, ~2s)"
	@echo "  test-slow       run slow tests (motmetrics cross-check, requires MOT17-09)"
	@echo "  backend         uvicorn at :8000"
	@echo "  frontend        next dev at :3000"
	@echo "  worker          outbound-WS GPU worker (local PC)"
	@echo "  eval            run SORT on MOT17-09-FRCNN and print metrics"
	@echo "  benchmark       run all trackers on all MOT17 sequences, write JSON for stress-test page"
	@echo "  demo            render 4-tracker side-by-side MP4 on MOT17-09"
	@echo "  prepare-dataset MOT17 -> YOLO format (needs img1 frames downloaded)"
	@echo "  train           YOLOv8n fine-tune on MOT17 (CPU ok, GPU recommended)"

install:
	python3 -m venv .venv
	./.venv/bin/pip install -e backend[dev]
	./.venv/bin/pip install "numpy<2"  # py-motmetrics 1.4 still uses np.asfarray
	cd frontend && npm install --no-audit --no-fund

test:
	cd backend && ../.venv/bin/python -m pytest -v -m "not slow"

test-slow:
	cd backend && ../.venv/bin/python -m pytest -v -m slow -s

backend:
	cd backend && ../.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

worker:
	cd gpu-worker && ../.venv/bin/python worker.py --backend ws://localhost:8000 --token dev-token

eval:
	./.venv/bin/python scripts/eval.py \
		--tracker custom \
		--detections datasets/MOT17/train/MOT17-09-FRCNN/det/det.txt \
		--gt datasets/MOT17/train/MOT17-09-FRCNN/gt/gt.txt \
		--score-threshold 0.3

benchmark:
	./.venv/bin/python scripts/benchmark.py \
		--root datasets/MOT17/train \
		--out-json frontend/public/benchmark.json \
		--out-md docs/_generated_benchmark.md \
		--score-threshold 0.3

demo:
	./.venv/bin/python scripts/render_side_by_side.py \
		--frames datasets/MOT17/train/MOT17-09-FRCNN/img1 \
		--detections datasets/MOT17/train/MOT17-09-FRCNN/det/det.txt \
		--trackers sort,deepsort,bytetrack,custom \
		--out exports/MOT17-09_4trackers.mp4 \
		--scale 0.5 --max-frames 200

prepare-dataset:
	./.venv/bin/python scripts/prepare_dataset.py \
		--src datasets/MOT17/train \
		--dst datasets/MOT17_yolo \
		--val-frac 0.2

train:
	./.venv/bin/python scripts/train_yolo.py \
		--data datasets/MOT17_yolo/data.yaml \
		--weights yolov8n.pt \
		--epochs 30 --img 640 --batch 16 \
		--project runs/mot17 --name yolov8n_30ep

clean:
	rm -rf backend/.pytest_cache backend/**/__pycache__ backend/__pycache__
	rm -rf frontend/.next frontend/node_modules
	rm -rf runs/ datasets/MOT17_yolo
