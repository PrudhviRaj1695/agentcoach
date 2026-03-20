.PHONY: run test lint format clean setup

setup:
	python -m venv venv
	. venv/bin/activate && pip install -e ".[dev]"
	pre-commit install

run:
	uvicorn src.api.main:app --reload --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache