PWD      := $(shell pwd)
CARGO    := cargo
RUSTFMT  := rustfmt
POETRY   := poetry
PYTHON   := $(POETRY) run python
PYTEST   := $(POETRY) run pytest
PYSEN    := $(POETRY) run pysen
MODULE   := stvec

.PHONY: all
all: rust-format rust-test python-setup python-format python-lint python-test

.PHONY: build
build:
	$(POETRY) run maturin build --release

.PHONY: python-setup
python-setup:
	$(POETRY) run maturin develop --release

.PHONY: python-test
python-test:
	PYTHONPATH=$(PWD) $(PYTEST)

.PHONY: python-lint
python-lint:
	PYTHONPATH=$(PWD) $(PYSEN) run lint

.PHONY: python-format
python-format:
	PYTHONPATH=$(PWD) $(PYSEN) run format

.PHONY: rust-test
rust-test:
	$(CARGO) test --lib

.PHONY: rust-format
rust-format:
	$(RUSTFMT) $(wildcard $(PWD)/src/*.rs)

.PHONY: clean
clean: clean-pyc clean-build

.PHONY: clean-pyc
clean-pyc:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-build
clean-build:
	rm -rf target/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf pip-wheel-metadata/
	find $(MODULE) -name "*.so" -exec rm -f {} +
