"""Growth-law inference on real mouse tumour-volume curves, with a batched GPU PINN.

Data: public Zenodo datasets of caliper tumour volumes in mice (CC BY 4.0), configured in ``configs/default.toml``
(``[real.datasets.*]``): breast carcinoma, Lewis lung carcinoma and a drug-treated breast-carcinoma cohort.

For every mouse the volume V(t) is described by an ODE d(ln V)/dt = g(ln V) with one of three growth laws
(exponential, logistic, Gompertz). Each (mouse, law, mode, restart) gets its own small network
x_theta(t) = ln V; all networks are trained *simultaneously* as one batched model (``torch.bmm``), which is
what makes the problem a natural fit for a GPU. Trainable ODE parameters (rate, carrying capacity) enter
through a physics residual at random collocation times up to the forecast target.

Two modes per mouse: "full" (all measurements) and "holdout" (last ``holdout`` measurements withheld, then
forecast). A classical analytic least-squares fit serves as baseline.
"""
from __future__ import annotations

import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

LAWS = ("exponential", "logistic", "gompertz")
T_SCALE = 40.0          # days; network input is t / T_SCALE
OUT_SCALE = 2.0         # network output is added to the log-volume mean with this scale
RES_SCALE = 10.0        # residual (1/day) is multiplied by this so it is O(1)
SIGMA_LOG = 0.15        # assumed log-normal measurement error in the data term
K_DECADES = 4.0         # carrying capacity K in [Vmax, 1e4 * Vmax]


# ----------------------------------------------------------------------------------------------- data
def _fetch(path: Path, url: str) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _read_table(path: Path, url: str) -> list[dict]:
    """Plain text table with columns ID, Time (days), Volume (mm3); tab or comma separated."""
    rows = [ln.replace(",", "	").split() for ln in _fetch(path, url).read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    by_id: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_id.setdefault(r[0], []).append((float(r[1]), float(r[2])))
    return [{"id": int(k), "group": None, "pts": by_id[k]} for k in sorted(by_id, key=int)]


def _col_index(ref: str) -> int:
    n = 0
    for ch in ref:
        n = n * 26 + ord(ch) - 64
    return n


def _read_ebos_xlsx(path: Path, url: str) -> list[dict]:
    """Zenodo 10607753 sheet: blocks headed by a 'Days Post-Implantation' row, then one row per mouse.

    Mouse label in column K, treatment group in column B, primary-tumour volumes (mm3) in columns L..AJ.
    Read with the standard library only (an .xlsx file is a zip of XML parts).
    """
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(_fetch(path, url))
    shared = ["".join(t.text or "" for t in si.iter(ns + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    animals, days = [], None
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(ns + "row"):
        cells = {}
        for c in row.findall(ns + "c"):
            v = c.find(ns + "v")
            if v is not None:
                cells[re.match(r"[A-Z]+", c.get("r")).group(0)] = shared[int(v.text)] if c.get("t") == "s" else v.text
        if int(row.get("r")) < 5:
            continue
        vol_cols = [k for k in cells if 12 <= _col_index(k) <= 36]
        if cells.get("B", "").startswith("Days Post"):
            days = {k: _num(cells[k]) for k in vol_cols if _num(cells[k]) is not None}
        elif "K" in cells and days:
            pts = [(days[k], _num(cells[k])) for k in days if k in cells and _num(cells[k]) is not None]
            animals.append({"id": cells["K"], "group": cells.get("B", ""), "pts": pts})
    return animals


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _signature(pts: list[tuple[float, float]]) -> tuple:
    return tuple(sorted((t, round(v, 3)) for t, v in pts))


def _raw_animals(ds: dict) -> list[dict]:
    reader = _read_ebos_xlsx if ds.get("format", "table") == "xlsx_ebos" else _read_table
    return reader(Path(ds["path"]), ds["url"])


def load_animals(rc: dict, key: str) -> list[dict]:
    """Load one dataset: drop duplicates of other datasets and non-positive volumes, keep long-enough curves."""
    ds = rc["datasets"][key]
    seen = {_signature(a["pts"]) for other in ds.get("exclude_duplicates_of", []) for a in _raw_animals(rc["datasets"][other])}
    animals = []
    for a in _raw_animals(ds):
        if _signature(a["pts"]) in seen:
            continue
        pts = sorted((t, v) for t, v in a["pts"] if v > 0)
        if len(pts) < rc["min_points"]:
            continue
        t = np.array([q[0] for q in pts])
        animals.append({"id": a["id"], "group": a["group"], "t": t - t[0], "t0": float(t[0]), "v": np.array([q[1] for q in pts])})
    return animals


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ------------------------------------------------------------------------------------ classical baseline
def _curve(law: str, t: np.ndarray, x0: float, a: float, lnK: float) -> np.ndarray:
    """ln V(t) for the closed-form solution of each growth law."""
    if law == "exponential":
        return x0 + a * t
    if law == "gompertz":
        return lnK + (x0 - lnK) * np.exp(-a * t)
    # logistic: V = K / (1 + (K/V0 - 1) e^{-a t})
    arg = 1.0 + (np.exp(lnK - x0) - 1.0) * np.exp(-a * t)
    return lnK - np.log(np.maximum(arg, 1e-12))


def nls_fit(t: np.ndarray, v: np.ndarray, law: str) -> dict:
    """Least-squares fit on the log scale; multi-start over the carrying capacity."""
    x = np.log(v)
    lnv_max = float(x.max())
    starts = [(x[0], 0.1, lnv_max + d) for d in (0.1, 0.5, 1.5, 3.0)] if law != "exponential" else [(x[0], 0.1, lnv_max)]
    best = None
    for x0, a, lk in starts:
        if law == "exponential":
            fun, p0, lo, hi = (lambda p: _curve(law, t, p[0], p[1], 0.0) - x), [x0, a], [x.min() - 3, 1e-4], [x.max() + 3, 5.0]
        else:
            fun = lambda p: _curve(law, t, p[0], p[1], p[2]) - x  # noqa: E731
            p0, lo, hi = [x0, a, lk], [x.min() - 3, 1e-4, lnv_max], [x.max() + 3, 5.0, lnv_max + K_DECADES * np.log(10)]
        try:
            r = least_squares(fun, np.clip(p0, lo, hi), bounds=(lo, hi), max_nfev=400)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    lnK = float(p[2]) if law != "exponential" else float("nan")
    return {"x0": float(p[0]), "alpha": float(p[1]), "lnK": lnK, "rss": float(2 * best.cost),
            "predict": lambda tt: np.exp(_curve(law, np.asarray(tt, float), p[0], p[1], lnK if law != "exponential" else 0.0))}


# ------------------------------------------------------------------------------------- batched PINN
class BatchedMLP:
    """J independent tanh MLPs evaluated with batched matrix products."""

    def __init__(self, J: int, hidden: int, layers: int, device: torch.device, gen: torch.Generator):
        dims = [1] + [hidden] * layers + [1]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            bound = float(np.sqrt(6.0 / (dims[i] + dims[i + 1])))
            self.W.append(((torch.rand(J, dims[i], dims[i + 1], generator=gen, device=device) * 2 - 1) * bound).requires_grad_())
            self.b.append(torch.zeros(J, 1, dims[i + 1], device=device, requires_grad=True))

    def parameters(self):
        return [*self.W, *self.b]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = torch.baddbmm(b, h, W)
            if i < len(self.W) - 1:
                h = torch.tanh(h)
        return h


def _pack_jobs(animals: list[dict], holdout: int, restarts: int, device: torch.device) -> dict:
    """Stack every (mode, law, animal, restart) into padded tensors."""
    meta, obs_t, obs_x, mask, tmax, vmax = [], [], [], [], [], []
    nmax = max(len(a["t"]) for a in animals)
    for mode in ("full", "holdout"):
        for li in range(len(LAWS)):
            for ai, a in enumerate(animals):
                n_tr = len(a["t"]) - (holdout if mode == "holdout" else 0)
                t = np.zeros(nmax)
                x = np.zeros(nmax)
                m = np.zeros(nmax)
                t[:n_tr], x[:n_tr], m[:n_tr] = a["t"][:n_tr], np.log(a["v"][:n_tr]), 1.0
                for r in range(restarts):
                    meta.append((mode, li, ai, r))
                    obs_t.append(t / T_SCALE)
                    obs_x.append(x)
                    mask.append(m)
                    tmax.append(a["t"][-1] / T_SCALE)
                    vmax.append(a["v"][:n_tr].max())
    f = lambda arr: torch.tensor(np.array(arr), dtype=torch.float32, device=device)  # noqa: E731
    obs_x_t, mask_t = f(obs_x)[..., None], f(mask)[..., None]
    mu = (obs_x_t * mask_t).sum(1, keepdim=True) / mask_t.sum(1, keepdim=True)
    return {
        "meta": meta, "tau": f(obs_t)[..., None], "x": obs_x_t, "mask": mask_t, "mu": mu,
        "tmax": f(tmax)[:, None, None], "lnvmax": torch.log(f(vmax))[:, None, None],
        "law": torch.tensor([m[1] for m in meta], device=device)[:, None, None],
    }


class GrowthPINN:
    """All networks, plus their trainable ODE parameters (log10 rate, carrying-capacity logit)."""

    def __init__(self, jobs: dict, hidden: int, layers: int, device: torch.device, seed: int):
        J = len(jobs["meta"])
        gen = torch.Generator(device=device).manual_seed(seed)
        self.jobs, self.device = jobs, device
        self.net = BatchedMLP(J, hidden, layers, device, gen)
        self.p_rate = (torch.rand(J, 1, 1, generator=gen, device=device) - 1.5).requires_grad_()   # log10 alpha in [-1.5, -0.5]
        self.p_k = torch.randn(J, 1, 1, generator=gen, device=device).requires_grad_()

    def x_of(self, tau: torch.Tensor) -> torch.Tensor:
        return self.jobs["mu"] + OUT_SCALE * self.net(2.0 * tau - 1.0)

    def ode_params(self):
        alpha = 10.0 ** self.p_rate
        lnK = self.jobs["lnvmax"] + K_DECADES * float(np.log(10.0)) * torch.sigmoid(self.p_k)
        return alpha, lnK

    def rate(self, x: torch.Tensor) -> torch.Tensor:
        """g(ln V) for the law of each job."""
        alpha, lnK = self.ode_params()
        law = self.jobs["law"]
        xc = torch.clamp(x, max=20.0)
        logistic = alpha * (1.0 - torch.exp(xc - lnK))
        gompertz = alpha * (lnK - xc)
        return torch.where(law == 0, alpha.expand_as(x), torch.where(law == 1, logistic, gompertz))

    def losses(self, n_col: int, grid: bool = False):
        j = self.jobs
        l_data = (j["mask"] * (self.x_of(j["tau"]) - j["x"]) ** 2).sum(1) / j["mask"].sum(1)
        if grid:
            tc = (torch.linspace(0, 1, n_col, device=self.device)[None, :, None] * j["tmax"]).expand(len(j["meta"]), -1, -1).clone()
        else:
            tc = torch.rand(len(j["meta"]), n_col, 1, device=self.device) * j["tmax"]
        tc.requires_grad_(True)
        x = self.x_of(tc)
        dx = torch.autograd.grad(x.sum(), tc, create_graph=True)[0] / T_SCALE
        res = (dx - self.rate(x)) * RES_SCALE
        return l_data.squeeze(-1) / SIGMA_LOG**2, (res**2).mean(1).squeeze(-1)

    def train(self, iters: int, pretrain: int, lr: float, n_col: int, log=None) -> float:
        opt = torch.optim.Adam([{"params": self.net.parameters(), "lr": lr}, {"params": [self.p_rate, self.p_k], "lr": 2e-2}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-5)
        sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None)
        sync()
        t0 = time.time()
        for it in range(iters):
            ld, lp = self.losses(n_col)
            loss = (ld + (lp if it >= pretrain else 0.0)).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if log and (it % 500 == 0 or it == iters - 1):
                log(f"    iter {it:5d}  mean data {ld.mean().item():.3f}  mean physics {lp.mean().item():.4f}")
        sync()
        return time.time() - t0

    @torch.no_grad()
    def predict_volume(self, t_days: torch.Tensor) -> torch.Tensor:
        """t_days: (J, N) -> V (J, N)."""
        return torch.exp(self.x_of((t_days / T_SCALE)[..., None])).squeeze(-1)


def time_per_iteration(jobs: dict, cfg_real: dict, device: torch.device, n: int = 15) -> float:
    """Milliseconds per training iteration of the full batched problem on ``device``."""
    moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in jobs.items()}
    model = GrowthPINN(moved, cfg_real["hidden"], cfg_real["layers"], device, seed=0)
    model.train(3, 0, cfg_real["lr"], cfg_real["n_collocation"])           # warm-up (kernel launch, allocator)
    return 1000.0 * model.train(n, 0, cfg_real["lr"], cfg_real["n_collocation"]) / n


# ------------------------------------------------------------------------------------------ pipeline
def run(cfg: dict, key: str, log=print, with_timing: bool = True) -> dict:
    rc = cfg["real"]
    animals = load_animals(rc, key)
    device = resolve_device(rc["device"])
    torch.manual_seed(rc["seed"])
    log(f"[real:{key}] {len(animals)} animals, {sum(len(a['t']) for a in animals)} measurements, device = {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    jobs = _pack_jobs(animals, rc["holdout"], rc["restarts"], device)
    J = len(jobs["meta"])
    log(f"[real:{key}] training {J} networks in one batch "
        f"({rc['restarts']} restarts x {len(LAWS)} laws x 2 modes x {len(animals)} animals)")
    model = GrowthPINN(jobs, rc["hidden"], rc["layers"], device, rc["seed"])
    seconds = model.train(rc["iters"], rc["pretrain"], rc["lr"], rc["n_collocation"], log=log)
    log(f"[real:{key}] PINN training finished in {seconds:.1f} s")

    # restart selection on the training objective only (no access to held-out volumes)
    ld, lp = model.losses(128, grid=True)
    score = (ld + lp).detach().cpu().numpy()
    alpha_t, lnK_t = model.ode_params()
    alpha, lnK = alpha_t.detach().cpu().numpy().ravel(), lnK_t.detach().cpu().numpy().ravel()
    best: dict[tuple, int] = {}
    for j, (mode, li, ai, _r) in enumerate(jobs["meta"]):
        key = (mode, li, ai)
        if key not in best or score[j] < score[best[key]]:
            best[key] = j

    n_grid = 60
    grids = torch.tensor(np.stack([np.linspace(0, a["t"][-1], n_grid) for a in animals]), dtype=torch.float32, device=device)
    out_animals = []
    for ai, a in enumerate(animals):
        t, v = a["t"], a["v"]
        n_tr = len(t) - rc["holdout"]
        entry = {"id": a["id"], "group": a["group"], "t": t, "v": v, "n_train_holdout": n_tr,
                 "grid": np.linspace(0, t[-1], n_grid), "fits": {}}
        for mode in ("full", "holdout"):
            entry["fits"][mode] = {}
            n_fit = len(t) if mode == "full" else n_tr
            for li, law in enumerate(LAWS):
                j = best[(mode, li, ai)]
                with torch.no_grad():
                    tt = torch.cat([grids[ai], torch.tensor(t, dtype=torch.float32, device=device)])
                    full = torch.zeros(J, tt.numel(), device=device)
                    full[j] = tt
                    vv = model.predict_volume(full)[j].cpu().numpy()
                v_curve, v_at_obs = vv[:n_grid], vv[n_grid:]
                nl = nls_fit(t[:n_fit], v[:n_fit], law)
                nl_curve = nl["predict"](entry["grid"])
                nl_obs = nl["predict"](t)
                held = slice(n_tr, None)
                entry["fits"][mode][law] = {
                    "pinn": {"curve": v_curve, "alpha": float(alpha[j]), "K": float(np.exp(lnK[j])) if li else None,
                             "rss": float(np.sum((np.log(np.maximum(v_at_obs[:n_fit], 1e-9)) - np.log(v[:n_fit])) ** 2))},
                    "nls": {"curve": nl_curve, "alpha": nl["alpha"], "K": float(np.exp(nl["lnK"])) if li else None, "rss": nl["rss"]},
                }
                if mode == "holdout":
                    entry["fits"][mode][law]["pinn"]["rel_err"] = np.abs(v_at_obs[held] - v[held]) / v[held]
                    entry["fits"][mode][law]["nls"]["rel_err"] = np.abs(nl_obs[held] - v[held]) / v[held]
        out_animals.append(entry)

    result = {"animals": out_animals, "train_seconds": seconds, "device": str(device), "n_networks": J,
              "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    if with_timing:
        result["timing"] = {"cpu_ms_per_iter": time_per_iteration(jobs, rc, torch.device("cpu"))}
        if torch.cuda.is_available():
            result["timing"]["gpu_ms_per_iter"] = time_per_iteration(jobs, rc, torch.device("cuda"))
            result["timing"]["speedup"] = result["timing"]["cpu_ms_per_iter"] / result["timing"]["gpu_ms_per_iter"]
        log(f"[real] per-iteration time: {result['timing']}")
    return result


def _bic(animal: dict, law: str) -> float:
    """Gaussian-error BIC of the classical fit on the log scale (2 parameters, or 3 with a carrying capacity)."""
    n = len(animal["t"])
    k = 2 if law == "exponential" else 3
    return n * np.log(max(animal["fits"]["full"][law]["nls"]["rss"], 1e-12) / n) + k * np.log(n)


def summarize(result: dict, holdout: int) -> dict:
    """Aggregate statistics: hold-out error per law/method, BIC winners, PINN-vs-NLS agreement."""
    animals = result["animals"]
    summary: dict = {"n_animals": len(animals), "n_points": int(sum(len(a["t"]) for a in animals)), "holdout": holdout,
                     "holdout_error": {}, "winner_holdout": {}, "winner_bic": {}, "rate_agreement": {}}
    for method in ("pinn", "nls"):
        for law in LAWS:
            errs = np.concatenate([a["fits"]["holdout"][law][method]["rel_err"] for a in animals])
            summary["holdout_error"].setdefault(method, {})[law] = {
                "median_pct": float(100 * np.median(errs)), "mean_pct": float(100 * errs.mean()),
                "p90_pct": float(100 * np.percentile(errs, 90)), "n": int(errs.size)}
        wins = {law: 0 for law in LAWS}
        for a in animals:
            wins[min(LAWS, key=lambda lw: a["fits"]["holdout"][lw][method]["rel_err"].mean())] += 1
        summary["winner_holdout"][method] = wins
    bic = {law: 0 for law in LAWS}
    for a in animals:
        bic[min(LAWS, key=lambda lw, a=a: _bic(a, lw))] += 1
    summary["winner_bic"] = bic
    for law in LAWS:
        p = np.log10([a["fits"]["full"][law]["pinn"]["alpha"] for a in animals])
        q = np.log10([a["fits"]["full"][law]["nls"]["alpha"] for a in animals])
        summary["rate_agreement"][law] = {"pearson_log10": float(np.corrcoef(p, q)[0, 1]),
                                          "median_ratio": float(np.median(10 ** (p - q)))}
    return summary
