"""Static figures for the README, built from the cached benchmark artefacts.

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tumorpinn.metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "assets"
COL = {"cnn": "#2563eb", "nls": "#059669", "pinn_cold": "#d97706", "pinn_warm": "#7c3aed"}
LAB = {"cnn": "CNN", "nls": "Spectral least squares", "pinn_cold": "PINN", "pinn_warm": "CNN-init PINN"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold", "axes.titlesize": 10,
})


def fields(bench):
    cases = [c for c in bench["slow"]["showcase"] if abs(c["sigma"] - 0.05) < 1e-9] or bench["slow"]["showcase"]
    c = cases[0]
    x, t = bench["x"], bench["t"]
    panels = [("Ground truth", c["clean"]), ("Noisy observation", c["obs"]), ("Spectral least squares", c["fields"]["nls"]),
              ("CNN estimate (simulated)", c["fields"]["cnn"]), ("PINN", c["fields"]["pinn_cold"]),
              ("CNN-init PINN", c["fields"]["pinn_warm"])]
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True, sharey=True)
    for a, (name, f) in zip(ax.ravel(), panels):
        im = a.imshow(f, origin="lower", aspect="auto", extent=[x[0], x[-1], t[0], t[-1]], vmin=0, vmax=1, cmap="YlOrBr")
        a.set_title(name)
        a.grid(False)
    for a in ax[:, 0]:
        a.set_ylabel("t")
    for a in ax[1]:
        a.set_xlabel("x")
    D, rho = c["true"]
    fig.suptitle(f"Test case #{c['idx']}: D = {D:.4f}, ρ = {rho:.2f}, noise σ = {c['sigma']}", y=0.995, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="tumour density u")
    fig.savefig(OUT / "fig_fields.png", bbox_inches="tight")
    plt.close(fig)


def scatter(bench, sigma=0.05):
    key = str(sigma)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for j, name in enumerate(["D", "ρ"]):
        for m in ("cnn", "nls", "pinn_cold", "pinn_warm"):
            if m == "cnn":
                e = bench["cnn"][key]
                true, pred = np.array(bench["test_params"]), np.array(e["pred"])
            else:
                e = bench["slow"]["per_noise"].get(key, {}).get(m)
                if not e:
                    continue
                true, pred = np.array(e["true"]), np.array(e["pred"])
            ax[j].scatter(true[:, j], pred[:, j], s=10 if m == "cnn" else 18, alpha=0.5 if m == "cnn" else 0.85,
                          color=COL[m], label=LAB[m], linewidths=0)
        lo, hi = ax[j].get_xlim()
        ax[j].plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].set_xlabel(f"true {name}")
        ax[j].set_ylabel(f"estimated {name}")
    ax[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Parameter recovery at noise σ = {sigma}")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter.png", bbox_inches="tight")
    plt.close(fig)


def noise(bench):
    levels = sorted(float(k) for k in bench["cnn"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for m in ("cnn", "nls", "pinn_cold", "pinn_warm"):
        xs, ys = [], {"mape_D": [], "mape_rho": []}
        for s in levels:
            e = bench["cnn"][str(s)] if m == "cnn" else bench["slow"]["per_noise"][str(s)].get(m)
            if not e:
                continue
            xs.append(s)
            for k in ys:
                ys[k].append(e["metrics"][k])
        if xs:
            for a, k, title in zip(ax, ys, ["Diffusivity D", "Proliferation ρ"]):
                a.plot(xs, ys[k], "o-", color=COL[m], label=LAB[m], lw=2)
                a.set_title(title)
                a.set_xlabel("noise σ")
    ax[0].set_ylabel("mean absolute percentage error")
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_noise.png", bbox_inches="tight")
    plt.close(fig)


def convergence():
    cv = json.loads((ART / "convergence.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    sp = cv["spatial"]
    ax[0].loglog([r["dof"] for r in sp["spectral"]], [max(r["error"], 1e-16) for r in sp["spectral"]], "o-", color=COL["nls"],
                 lw=2, label="Legendre–Gauss–Lobatto")
    ax[0].loglog([r["dof"] for r in sp["fd2"]], [r["error"] for r in sp["fd2"]], "s-", color=COL["pinn_cold"], lw=2,
                 label="finite differences (2nd order)")
    ax[0].set_xlabel("unknowns")
    ax[0].set_ylabel("max error at t = T")
    ax[0].set_title("Spatial convergence")
    ax[0].legend(frameon=False)
    tm = cv["temporal"]["rows"]
    n = np.array([r["steps"] for r in tm])
    e = np.array([r["error"] for r in tm])
    ax[1].loglog(n, e, "o-", color=COL["cnn"], lw=2, label="CN / AB2")
    ax[1].loglog(n, e[0] * (n[0] / n) ** 2, "k--", lw=0.8, label="slope −2")
    ax[1].set_xlabel("time steps")
    ax[1].set_title("Temporal convergence")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_convergence.png", bbox_inches="tight")
    plt.close(fig)


def tables(bench):
    """Markdown result tables (pasted into the README); every method is scored on the same test cases."""
    lines = []
    true_all = np.array(bench["test_params"])
    for sigma in (0.0, 0.02, 0.05, 0.1):
        key = str(sigma)
        slow = bench["slow"]["per_noise"].get(key, {})
        n_common = min([e["metrics"]["n"] for e in slow.values() if isinstance(e, dict) and "metrics" in e] or [len(true_all)])
        cnn = bench["cnn"][key]
        rows = [("CNN (3-member ensemble)", summarize(np.array(cnn["pred"])[:n_common], true_all[:n_common]), cnn["ms_per_sample"] / 1000)]
        for m, name in (("nls", "Spectral least squares"), ("pinn_cold", "PINN"), ("pinn_warm", "CNN-init PINN")):
            e = slow.get(m)
            if e:
                pred, true = np.array(e["pred"])[:n_common], np.array(e["true"])[:n_common]
                rows.append((name, summarize(pred, true), e["seconds_mean"]))
        lines += [f"**Noise σ = {sigma}** (first {n_common} test cases, identical for all methods)", "",
                  "| Method | MAPE D | MAPE ρ | median APE D | median APE ρ | R² (log D) | R² (log ρ) | time / case |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, m, sec in rows:
            t = f"{sec * 1000:.1f} ms" if sec < 0.1 else f"{sec:.1f} s"
            lines.append(f"| {name} | {m['mape_D']:.1f}% | {m['mape_rho']:.1f}% | {m['median_ape_D']:.1f}% | "
                         f"{m['median_ape_rho']:.1f}% | {m['r2_log10_D']:.3f} | {m['r2_log10_rho']:.3f} | {t} |")
        lines.append("")
    (OUT / "results.md").write_text(chr(10).join(lines), encoding="utf-8")


def real_tables():
    """Markdown table of the real-data forecasting results (needs artifacts/real_summary.json)."""
    path = ART / "real_summary.json"
    if not path.exists():
        return
    summaries = json.loads(path.read_text())
    names = {"breast": "Breast carcinoma (vehicle controls)", "lung": "Lewis lung carcinoma", "treated": "Breast carcinoma (drug-treated)"}
    lines = ["| Dataset | mice | volumes | law | PINN (GPU) | least squares | best by BIC |", "|---|---:|---:|---|---:|---:|---:|"]
    for key, s in summaries.items():
        for i, law in enumerate(("exponential", "logistic", "gompertz")):
            first = f"| {names.get(key, key)} | {s['n_animals']} | {s['n_points']} " if i == 0 else "| | | "
            lines.append(f"{first}| {law} | {s['holdout_error']['pinn'][law]['median_pct']:.1f}% | "
                         f"{s['holdout_error']['nls'][law]['median_pct']:.1f}% | {s['winner_bic'][law]} |")
    (OUT / "results_real.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(ART / "benchmark.pkl", "rb") as fh:
        bench = pickle.load(fh)
    fields(bench)
    scatter(bench)
    noise(bench)
    convergence()
    tables(bench)
    real_tables()
    print("figures and tables written to", OUT)


if __name__ == "__main__":
    main()
