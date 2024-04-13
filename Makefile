# List of targets that are not associated with files
.PHONY:	quality style install test

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .

test_cpu:
	pytest tests/ -x "cpu"

test_gpu:
	pytest tests/ -x "gpu"

install:
	pip install -e .