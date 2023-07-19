.PHONY: format
format:
	black embeddingviz

.PHONY: lint
lint:
	ruff embeddingviz
	mypy embeddingviz

.PHONY: initdev
initdev:
	pip install --upgrade pip
	pip install -e .[dev,test]
	pre-commit install

.PHONY: all
all: format lint
