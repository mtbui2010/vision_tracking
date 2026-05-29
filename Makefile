.PHONY: install test backend frontend worker eval clean

install:
	python3 -m venv .venv
	./.venv/bin/pip install -e backend[dev]
	cd frontend && npm install

test:
	cd backend && ../.venv/bin/python -m pytest -v

backend:
	cd backend && ../.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

worker:
	cd gpu-worker && python worker.py --backend ws://localhost:8000 --token dev-token

eval:
	cd backend && ../.venv/bin/python ../scripts/eval.py \
		--tracker sort \
		--detections ../datasets/MOT17/train/MOT17-04-FRCNN/det/det.txt \
		--gt ../datasets/MOT17/train/MOT17-04-FRCNN/gt/gt.txt \
		--out ../exports/MOT17-04_sort.txt

clean:
	rm -rf backend/.pytest_cache backend/__pycache__ backend/**/__pycache__
	rm -rf frontend/.next frontend/node_modules
