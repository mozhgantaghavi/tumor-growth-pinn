"""TOML configuration loading and path helpers."""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.toml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load ``configs/default.toml`` and optionally merge another file / dict."""
    with open(DEFAULT_CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, "rb") as fh:
            cfg = _merge(cfg, tomllib.load(fh))
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def artifacts_dir(cfg: dict) -> Path:
    p = ROOT / cfg["paths"]["artifacts"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def checkpoints_dir(cfg: dict) -> Path:
    p = ROOT / cfg["paths"]["checkpoints"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def docs_data_dir(cfg: dict) -> Path:
    p = ROOT / cfg["paths"]["docs_data"]
    p.mkdir(parents=True, exist_ok=True)
    return p
