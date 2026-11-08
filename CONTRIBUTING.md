# Contributing

Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/mozhgantaghavi/tumor-growth-pinn
cd tumor-growth-pinn
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,figures]"
```

## Checks that must pass

```bash
ruff check src tests scripts          # lint
pytest                                 # Python unit tests
node --test "tests/js/*.test.mjs"      # dashboard solver tests (needs docs/data/parity.json)
```

The JavaScript solver in `docs/js/solver.js` is a port of `src/tumorpinn/solver.py`. If you change the
numerical scheme in one, change the other and regenerate the parity fixture:

```bash
python -m tumorpinn export             # rewrites docs/data/*.json, including parity.json
```

## Adding a PDE model

1. Add a `PDEModel` entry to `REGISTRY` in `src/tumorpinn/models.py` (NumPy and PyTorch reaction terms).
2. Add the same reaction term to `MODELS` in `docs/js/solver.js`.
3. Extend `tests/test_solver.py` (the Radau-reference test is parametrised over the registry).
4. Run `python -m tumorpinn all --model <name>`.

## Style

- Keep modules small and single-purpose; configuration lives in `configs/*.toml`, never in code.
- Every randomised step takes an explicit seed.
- New figures in the dashboard read their data from `docs/data/*.json`; do not hard-code results.
