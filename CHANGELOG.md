# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [1.0.0] - 2026-11-09

### Added
- Legendre-Gauss-Lobatto spectral collocation solver with no-flux boundaries (`spectral.py`, `solver.py`) for the
  Fisher-KPP and Gompertz tumour-growth models, validated against a Radau reference.
- Spatial and temporal convergence studies (spectral versus second-order finite differences; observed temporal
  order 1.9 to 2.0).
- Parameter inference on noisy space-time data: CNN ensemble, spectral least squares, PINN, and a PINN
  initialised from the CNN estimate, with a head-to-head benchmark over five noise levels.
- GPU-batched PINN growth-law fits and forecasts on three public mouse tumour datasets (`real_data.py`).
- Interactive dashboard for GitHub Pages: in-browser spectral solver, inference lab with cost landscape,
  benchmark explorer, case explorer, real-data views, numerics and training curves, light and dark themes.
- `pytest` suite (35 tests), Node tests that check the JavaScript solver against Python, CI workflow.
- Reproducible TOML configuration and a command-line pipeline (`python -m tumorpinn all`).
