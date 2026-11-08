"""Head-to-head comparison of CNN, spectral least squares, PINN and CNN->PINN."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .cnn import predict_ensemble
from .data import add_noise
from .metrics import field_rmse, summarize
from .models import get_model
from .nls import SpectralForward, fit_nls
from .pinn import fit_pinn


def noisy_test_set(cfg: dict, splits: dict, level_index: int) -> np.ndarray:
    """Deterministic noisy copy of the test fields for one noise level."""
    sigma = cfg["noise"]["levels"][level_index]
    rng = np.random.default_rng(cfg["data"]["seed"] + 77 + level_index)
    return add_noise(splits["test_fields"], sigma, rng)


def _prior_center(cfg: dict) -> np.ndarray:
    pb = cfg["problem"]
    return (np.log10([pb["D_range"][0], pb["rho_range"][0]]) + np.log10([pb["D_range"][1], pb["rho_range"][1]])) / 2


def _bounds(cfg: dict):
    pb = cfg["problem"]
    return (np.log10([pb["D_range"][0], pb["rho_range"][0]]), np.log10([pb["D_range"][1], pb["rho_range"][1]]))


def evaluate_cnn(cfg: dict, nets, splits: dict) -> dict:
    """CNN ensemble accuracy and throughput on every noise level."""
    out = {}
    for li, sigma in enumerate(cfg["noise"]["levels"]):
        obs = noisy_test_set(cfg, splits, li)
        t0 = time.perf_counter()
        pred, unc = predict_ensemble(nets, obs, cfg)
        dt = time.perf_counter() - t0
        out[str(sigma)] = {
            "pred": pred.tolist(),
            "log10_std": unc.tolist(),
            "metrics": summarize(pred, splits["test_params"]),
            "ms_per_sample": 1000 * dt / len(obs),
        }
    return out


def _case_worker(job: dict) -> dict:
    """Run NLS (+ optionally PINN cold / warm) on one noisy observation."""
    cfg = job["cfg"]
    model = get_model(cfg["problem"]["model"])
    obs, true = job["obs"], np.asarray(job["true"])
    x, t = job["x"], job["t"]
    fwd = SpectralForward(cfg, model)
    lo, hi = _bounds(cfg)
    res = {"idx": job["idx"], "sigma": job["sigma"], "true": true.tolist(),
           "cnn_pred": np.asarray(job["cnn_pred"]).tolist()}
    r = fit_nls(obs, fwd, _prior_center(cfg), (lo, hi))
    res["nls"] = {"params": r["params"].tolist(), "seconds": r["seconds"], "nfev": r["nfev"],
                  "field_rmse": field_rmse(r["recon"], job["clean"])}
    fields = {"nls": r["recon"]}
    cnn_theta = np.log10(job["cnn_pred"])
    u0 = fwd.smooth_ic(obs[0])
    fields["cnn"] = fwd(cnn_theta, u0)
    res["cnn_field_rmse"] = field_rmse(fields["cnn"], job["clean"])
    if job["run_pinn"]:
        p = cfg["pinn"]
        rc = fit_pinn(obs, x, t, model, cfg, _prior_center(cfg), p["iters_cold"], seed=job["idx"])
        rw = fit_pinn(obs, x, t, model, cfg, cnn_theta, p["iters_warm"], seed=job["idx"], pretrain=p["pretrain_warm"],
                      lr_params=p["lr_params_warm"])
        for key, r_ in (("pinn_cold", rc), ("pinn_warm", rw)):
            res[key] = {"params": r_["params"].tolist(), "seconds": r_["seconds"],
                        "field_rmse": field_rmse(r_["recon"], job["clean"])}
            fields[key] = r_["recon"]
        res["pinn_history"] = {"cold": rc["history"], "warm": rw["history"]}
    if job["keep_fields"]:
        res["fields"] = {k: np.asarray(v, dtype=np.float32) for k, v in fields.items()}
        res["clean"] = job["clean"]
        res["obs"] = obs
    return res


def run_slow_methods(cfg: dict, splits: dict, cnn_results: dict, log=print) -> dict:
    """Run NLS on many cases and PINNs on a subset, in parallel worker processes."""
    bm = cfg["benchmark"]
    levels = cfg["noise"]["levels"]
    jobs = []
    for li, sigma in enumerate(levels):
        obs_all = noisy_test_set(cfg, splits, li)
        cnn_pred = np.array(cnn_results[str(sigma)]["pred"])
        n_nls = bm["n_nls_cases"]
        run_pinn_here = sigma in bm["pinn_noise_levels"]
        for i in range(n_nls):
            pin = run_pinn_here and i < bm["n_cases"]
            jobs.append(dict(cfg=cfg, idx=i, sigma=sigma, obs=obs_all[i], clean=splits["test_fields"][i],
                             true=splits["test_params"][i], cnn_pred=cnn_pred[i], x=splits["x"], t=splits["t"],
                             run_pinn=pin, keep_fields=pin and i < bm["showcase"]))
    # long PINN jobs first for better load balance
    jobs.sort(key=lambda j: not j["run_pinn"])
    log(f"  running {len(jobs)} jobs ({sum(j['run_pinn'] for j in jobs)} with PINN) on {bm['workers']} workers")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(bm["workers"]) as ex:
        for k, r in enumerate(ex.map(_case_worker, jobs, chunksize=1)):
            results.append(r)
            if (k + 1) % 20 == 0 or k + 1 == len(jobs):
                log(f"  {k + 1}/{len(jobs)} done ({time.time() - t0:.0f}s)")
    return aggregate(cfg, results)


def aggregate(cfg: dict, results: list[dict]) -> dict:
    out = {"per_noise": {}, "showcase": [], "pinn_history": []}
    for sigma in cfg["noise"]["levels"]:
        rs = sorted([r for r in results if r["sigma"] == sigma], key=lambda r: r["idx"])
        entry = {}
        for key in ("nls", "pinn_cold", "pinn_warm"):
            sub = [r for r in rs if key in r]
            if not sub:
                continue
            pred = np.array([r[key]["params"] for r in sub])
            tr = np.array([r["true"] for r in sub])
            entry[key] = {
                "idx": [r["idx"] for r in sub],
                "true": tr.tolist(),
                "pred": pred.tolist(),
                "metrics": summarize(pred, tr),
                "seconds_mean": float(np.mean([r[key]["seconds"] for r in sub])),
                "seconds_median": float(np.median([r[key]["seconds"] for r in sub])),
                "field_rmse_mean": float(np.mean([r[key]["field_rmse"] for r in sub])),
            }
        entry["cnn_field_rmse_mean"] = float(np.mean([r["cnn_field_rmse"] for r in rs]))
        out["per_noise"][str(sigma)] = entry
        for r in rs:
            if "fields" in r:
                out["showcase"].append(r)
            if "pinn_history" in r and len(out["pinn_history"]) < 2 * 3:
                out["pinn_history"].append({"idx": r["idx"], "sigma": sigma, **r["pinn_history"]})
    return out
