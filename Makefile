PY ?= python

.PHONY: install test lint data train benchmark convergence export all figures serve clean

install:
	$(PY) -m pip install -e ".[dev,figures]"

test:
	$(PY) -m pytest
	node --test "tests/js/*.test.mjs"

lint:
	$(PY) -m ruff check src tests scripts

data:
	$(PY) -m tumorpinn data

train:
	$(PY) -m tumorpinn train

benchmark:
	$(PY) -m tumorpinn benchmark

convergence:
	$(PY) -m tumorpinn convergence

export:
	$(PY) -m tumorpinn export

all:
	$(PY) -m tumorpinn all

figures:
	$(PY) scripts/make_figures.py

serve:
	$(PY) -m http.server 8000 -d docs

clean:
	rm -rf artifacts .pytest_cache .ruff_cache build *.egg-info
