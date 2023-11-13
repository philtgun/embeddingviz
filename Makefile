.PHONY: format
format:
	ruff format .

.PHONY: lint
lint:
	ruff .
	mypy .

.PHONY: initdev
initdev:
	pip install --upgrade pip
	pip install -e .[dev,test]
	pre-commit install

.PHONY: all
all: format lint
