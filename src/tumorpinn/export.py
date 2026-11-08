"""Serialise benchmark results into compact JSON files read by the dashboard."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import torch

from . import __version__
from .models import get_model
from .solver import solve_batch
from .spectral import NeumannGrid

# keys whose values are physical parameters: exported with more decimals than the bulky field arrays
PARAM_KEYS = {"true", "est", "pred", "alpha", "K", "log10_std"}
PARAM_DECIMALS = 8


def _round(obj, nd=4):
    if isinstance(obj, dict):
        return {k: _round(v, PARAM_DECIMALS if k in PARAM_KEYS else nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, np.ndarray):
        return _round(obj.tolist(), nd)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), nd) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def dump(path: Path, obj, nd: int = 4) -> None:
    """Write compact JSON; bulky arrays use ``nd`` decimals, physical parameters keep full precision."""
    path.write_text(json.dumps(_round(obj, nd), separators=(",", ":")), encoding="utf-8")


def _downsample_history(h: dict, keep: int = 80) -> dict:
    n = len(h["iter"])
    idx = np.unique(np.linspace(0, n - 1, min(keep, n)).astype(int))
    return {k: [v[i] for i in idx] for k, v in h.items()}


def parity_fixture(cfg: dict) -> dict:
    """Reference trajectory used to check the JavaScript solver against Python."""
    M, L, T = 40, cfg["problem"]["length"], cfg["problem"]["T"]
    g = NeumannGrid(M, L)
    seeds = [{"a": 0.7, "c": 0.3, "s": 0.08}, {"a": 0.4, "c": 0.7, "s": 0.1}]
    u0 = sum(s["a"] * np.exp(-0.5 * ((g.x - s["c"] * L) / s["s"]) ** 2) for s in seeds)
    out = {"M": M, "length": L, "T": T, "n_snap": 11, "substeps": 20, "seeds": seeds}
    xu, A = g.to_uniform(41)
    out["x"] = xu
    out["cases"] = []
    for name in ("fisher_kpp", "gompertz"):
        D, rho = 0.01, 3.0
        snaps = solve_batch(g, u0, [D], [rho], get_model(name), T, out["n_snap"], out["substeps"])[0]
        out["cases"].append({"model": name, "D": D, "rho": rho, "snapshots": snaps @ A.T})
    return out


def export_all(cfg: dict, art: Path, docs: Path, train_hist: list, bench: dict, conv: dict, timings: dict) -> None:
    levels = cfg["noise"]["levels"]
    model = get_model(cfg["problem"]["model"])
    cnn = bench["cnn"]
    slow = bench["slow"]
    meta = {
        "version": __version__,
        "model": {"name": model.name, "title": model.title, "latex": model.latex},
        "problem": cfg["problem"],
        "grid": cfg["grid"],
        "noise_levels": levels,
        "pinn_noise_levels": cfg["benchmark"]["pinn_noise_levels"],
        "counts": {"train": cfg["data"]["n_train"], "val": cfg["data"]["n_val"], "test": cfg["data"]["n_test"],
                   "nls_cases": cfg["benchmark"]["n_nls_cases"], "pinn_cases": cfg["benchmark"]["n_cases"],
                   "cnn_ensemble": cfg["cnn"]["ensemble"]},
        "pinn": {k: cfg["pinn"][k] for k in ("hidden", "layers", "n_collocation", "iters_cold", "iters_warm")},
        "timings": timings,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "torch": torch.__version__, "platform": platform.system()},
    }
    dump(docs / "meta.json", meta)

    bench_out = {"levels": levels, "methods": {}}
    for key in ("cnn",):
        bench_out["methods"][key] = {
            s: {"metrics": cnn[s]["metrics"], "ms_per_sample": cnn[s]["ms_per_sample"],
                "true": bench["test_params"], "pred": cnn[s]["pred"], "log10_std": cnn[s]["log10_std"]}
            for s in cnn
        }
    for key in ("nls", "pinn_cold", "pinn_warm"):
        bench_out["methods"][key] = {
            s: {"metrics": e[key]["metrics"], "seconds_mean": e[key]["seconds_mean"],
                "seconds_median": e[key]["seconds_median"], "field_rmse_mean": e[key]["field_rmse_mean"],
                "true": e[key]["true"], "pred": e[key]["pred"]}
            for s, e in slow["per_noise"].items() if key in e
        }
    bench_out["cnn_field_rmse"] = {s: e["cnn_field_rmse_mean"] for s, e in slow["per_noise"].items()}
    dump(docs / "benchmark.json", bench_out, 5)

    showcase = []
    for r in slow["showcase"]:
        showcase.append({
            "idx": r["idx"], "sigma": r["sigma"], "true": r["true"],
            "est": {k: r[k]["params"] for k in ("nls", "pinn_cold", "pinn_warm") if k in r}
                   | {"cnn": r["cnn_pred"] if "cnn_pred" in r else None},
            "clean": r["clean"], "obs": r["obs"],
            "recon": {k: v for k, v in r["fields"].items()},
        })
    dump(docs / "showcase.json", {"x": bench["x"], "t": bench["t"], "cases": showcase}, 3)

    dump(docs / "training.json", {
        "cnn": [{"epoch": h["epoch"], "train": h["train_loss"], "val": h["val_loss"], "seconds": h["seconds"]}
                for h in train_hist],
        "pinn": [{"idx": h["idx"], "sigma": h["sigma"],
                  "cold": _downsample_history(h["cold"]), "warm": _downsample_history(h["warm"])}
                 for h in slow["pinn_history"]],
    }, 5)
    dump(docs / "convergence.json", conv, 8)
    dump(docs / "parity.json", parity_fixture(cfg), 12)


def export_real(cfg: dict, results: dict, docs: Path) -> None:
    """Write docs/data/real.json for the "Real data" dashboard tab.

    ``results`` maps a dataset key to ``{"result": ..., "summary": ...}``.
    """
    rc = cfg["real"]
    out = {"torch": torch.__version__, "datasets": {}}
    for key, bundle in results.items():
        result, summary, ds = bundle["result"], bundle["summary"], rc["datasets"][key]
        animals = []
        for a in result["animals"]:
            fits = {}
            for mode, laws in a["fits"].items():
                fits[mode] = {law: {m: {k: v for k, v in d.items() if k != "rss"} for m, d in methods.items()}
                              for law, methods in laws.items()}
            animals.append({"id": a["id"], "group": a["group"], "t": a["t"], "v": a["v"], "n_train_holdout": a["n_train_holdout"],
                            "grid": a["grid"], "fits": fits})
        meta = {
            "key": key, "dataset": {"title": ds["title"], "short": ds["short"], "authors": ds["authors"], "doi": ds["doi"],
                                    "url": f"https://doi.org/{ds['doi']}", "license": ds["license"], "note": ds.get("note")},
            "laws": ["exponential", "logistic", "gompertz"], "holdout": rc["holdout"], "restarts": rc["restarts"],
            "iters": rc["iters"], "hidden": rc["hidden"], "layers": rc["layers"], "n_networks": result["n_networks"],
            "device": result["device"], "gpu_name": result["gpu_name"], "train_seconds": result["train_seconds"],
            "timing": result.get("timing"),
        }
        out["datasets"][key] = {"meta": meta, "summary": summary, "animals": animals}
    dump(docs / "real.json", out, 4)
