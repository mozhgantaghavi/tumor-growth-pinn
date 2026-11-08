"""Error metrics for parameter recovery."""
from __future__ import annotations

import numpy as np


def summarize(pred: np.ndarray, true: np.ndarray) -> dict:
    """Summary statistics for (n, 2) arrays of physical (D, rho)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    rel = np.abs(pred - true) / true
    dlog = np.abs(np.log10(pred) - np.log10(true))
    lt, lp = np.log10(true), np.log10(pred)
    r2 = 1.0 - ((lp - lt) ** 2).sum(0) / ((lt - lt.mean(0)) ** 2).sum(0)
    out = {"n": int(len(true))}
    for j, name in enumerate(["D", "rho"]):
        out[f"mape_{name}"] = float(100 * rel[:, j].mean())
        out[f"median_ape_{name}"] = float(100 * np.median(rel[:, j]))
        out[f"p90_ape_{name}"] = float(100 * np.percentile(rel[:, j], 90))
        out[f"mae_log10_{name}"] = float(dlog[:, j].mean())
        out[f"r2_log10_{name}"] = float(r2[j])
    return out


def field_rmse(recon: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(recon) - np.asarray(truth)) ** 2)))
