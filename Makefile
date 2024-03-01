# List of targets that are not associated with files
.PHONY:	quality style install test

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .

test:
	pytest tests/ -x

install:
	pip install -e .