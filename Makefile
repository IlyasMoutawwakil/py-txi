# List of targets that are not associated with files
.PHONY:	quality style install

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .

install:
	pip install -e .