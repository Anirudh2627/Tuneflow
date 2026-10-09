.PHONY: install sample prepare train evaluate backend frontend seed test lint clean

## Install Python deps
install:
	pip install -r requirements.txt

## Generate the built-in sample dataset (data/sample)
sample:
	python scripts/download_data.py --source sample

## Run preprocessing -> data/processed
prepare:
	python scripts/prepare_data.py --config configs/sample.yaml

## Train all models on the sample dataset (~seconds)
train:
	python scripts/train_content.py --config configs/sample.yaml
	python scripts/train_collaborative.py --config configs/sample.yaml
	python scripts/train_session.py --config configs/sample.yaml
	python scripts/train_ranker.py --config configs/sample.yaml --ranker weighted

## Evaluate all recommenders -> reports/evaluation
evaluate:
	python scripts/evaluate.py --config configs/sample.yaml

## Seed the dev database (DATABASE_URL from env/.env)
seed:
	python scripts/seed_db.py

## Run the FastAPI backend on :8000
backend:
	uvicorn app.main:app --app-dir backend --reload --port 8000

## Run the Vite dev server on :5173
frontend:
	cd frontend && npm install && npm run dev

## Run all tests
test:
	python -m pytest -q

clean:
	rm -rf data/processed/* models/*/*/ .pytest_cache
