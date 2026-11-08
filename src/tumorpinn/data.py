"""Synthetic dataset generation: priors, initial tumours, noise, persistence."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import get_model
from .solver import solve_batch
from .spectral import NeumannGrid


def log_uniform(rng: np.random.Generator, lo_hi, n: int) -> np.ndarray:
    lo, hi = np.log10(lo_hi[0]), np.log10(lo_hi[1])
    return 10.0 ** rng.uniform(lo, hi, n)


def sample_initial_conditions(rng: np.random.Generator, x: np.ndarray, n: int, cfg: dict) -> np.ndarray:
    """Sum of 1-2 Gaussian tumour seeds evaluated at ``x``; returns (n, len(x))."""
    ic = cfg["ic"]
    L = cfg["problem"]["length"]
    lo, hi = ic["margin"] * L, (1 - ic["margin"]) * L
    out = np.zeros((n, x.size))
    n_seeds = rng.integers(ic["n_seeds"][0], ic["n_seeds"][1] + 1, n)
    for i in range(n):
        for _ in range(n_seeds[i]):
            c = rng.uniform(lo, hi)
            a = rng.uniform(*ic["amplitude"])
            s = rng.uniform(*ic["width"])
            out[i] += a * np.exp(-0.5 * ((x - c) / s) ** 2)
    return np.clip(out, 0.0, 1.0)


def make_grid(cfg: dict, M: int | None = None) -> NeumannGrid:
    return NeumannGrid(M or cfg["grid"]["M"], cfg["problem"]["length"])


def generate(cfg: dict, n: int, seed: int) -> dict:
    """Simulate ``n`` tumours; fields are returned on the uniform observation grid."""
    rng = np.random.default_rng(seed)
    pb, gd = cfg["problem"], cfg["grid"]
    grid = make_grid(cfg)
    model = get_model(pb["model"])
    D = log_uniform(rng, pb["D_range"], n)
    rho = log_uniform(rng, pb["rho_range"], n)
    u0 = sample_initial_conditions(rng, grid.x, n, cfg)
    snaps = solve_batch(grid, u0, D, rho, model, pb["T"], gd["n_t"], gd["substeps"])
    xu, A = grid.to_uniform(gd["n_x"])
    fields = np.clip(snaps @ A.T, 0.0, 1.0)
    return {
        "fields": fields.astype(np.float32),          # (n, n_t, n_x)
        "params": np.stack([D, rho], axis=1),         # (n, 2) physical units
        "x": xu,
        "t": np.linspace(0.0, pb["T"], gd["n_t"]),
    }


def add_noise(fields: np.ndarray, sigma, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian measurement noise; ``sigma`` scalar or per-sample array."""
    sigma = np.asarray(sigma, dtype=np.float32)
    if sigma.ndim == 1:
        sigma = sigma[:, None, None]
    return (fields + sigma * rng.standard_normal(fields.shape).astype(np.float32)).astype(np.float32)


def dataset_path(cfg: dict, art: Path) -> Path:
    return art / f"dataset_{cfg['problem']['model']}.npz"


def build_splits(cfg: dict, art: Path, force: bool = False) -> dict:
    """Generate (or load cached) train / val / test splits."""
    path = dataset_path(cfg, art)
    if path.exists() and not force:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d = cfg["data"]
    splits = {}
    for i, (name, n) in enumerate([("train", d["n_train"]), ("val", d["n_val"]), ("test", d["n_test"])]):
        s = generate(cfg, n, d["seed"] + 1000 * i)
        splits[f"{name}_fields"] = s["fields"]
        splits[f"{name}_params"] = s["params"]
        splits["x"], splits["t"] = s["x"], s["t"]
    np.savez_compressed(path, **splits)
    return splits
