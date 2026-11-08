"""Convergence studies: spatial (spectral vs 2nd-order finite differences) and temporal."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .models import get_model
from .solver import solve_batch, solve_reference
from .spectral import NeumannGrid


def _smooth_ic(x: np.ndarray, length: float) -> np.ndarray:
    return 0.6 * np.exp(-0.5 * ((x - 0.4 * length) / 0.1) ** 2) + 0.3 * np.exp(-0.5 * ((x - 0.7 * length) / 0.08) ** 2)


def _fd_solution(n: int, length: float, D: float, rho: float, model, T: float, xq: np.ndarray) -> np.ndarray:
    """Second-order cell-centred finite differences with mirrored (Neumann) ends."""
    h = length / n
    xc = (np.arange(n) + 0.5) * h
    Lap = (np.diag(-2.0 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1))
    Lap[0, 0] = Lap[-1, -1] = -1.0
    Lap /= h * h

    def jac(_t, u):
        eps = 1e-7
        return D * Lap + np.diag((model.reaction_np(u + eps, rho) - model.reaction_np(u, rho)) / eps)

    sol = solve_ivp(lambda _t, u: D * (Lap @ u) + model.reaction_np(u, rho), (0, T), _smooth_ic(xc, length),
                    method="Radau", rtol=1e-10, atol=1e-12, jac=jac, t_eval=[T])
    return np.interp(xq, xc, sol.y[:, -1])


def spatial_convergence(cfg: dict, D: float = 0.01, rho: float = 4.0) -> dict:
    """Max-norm error at t = T versus number of unknowns (LGL spectral and FD2)."""
    model = get_model(cfg["problem"]["model"])
    L, T = cfg["problem"]["length"], cfg["problem"]["T"]
    xq = np.linspace(0, L, 301)
    ref_grid = NeumannGrid(96, L)
    ref = solve_reference(ref_grid, _smooth_ic(ref_grid.x, L), D, rho, model, T, 2)[-1]
    ref_q = ref_grid.to_uniform(301)[1] @ ref
    spec = []
    for M in cfg["convergence"]["M_values"]:
        g = NeumannGrid(M, L)
        u = solve_reference(g, _smooth_ic(g.x, L), D, rho, model, T, 2, rtol=1e-11, atol=1e-13)[-1]
        spec.append({"dof": M + 1, "error": float(np.max(np.abs(interp_to(g, u, xq) - ref_q)))})
    fd = []
    for n in (8, 16, 32, 64, 128, 256):
        u = _fd_solution(n, L, D, rho, model, T, xq)
        fd.append({"dof": n, "error": float(np.max(np.abs(u - ref_q)))})
    return {"D": D, "rho": rho, "spectral": spec, "fd2": fd}


def interp_to(grid: NeumannGrid, u: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return grid.to_uniform(len(xq))[1] @ u


def temporal_convergence(cfg: dict, D: float = 0.01, rho: float = 4.0, M: int = 48) -> dict:
    """Error of the CN/AB2 stepper versus the number of steps (observed order ~ 2)."""
    model = get_model(cfg["problem"]["model"])
    L, T = cfg["problem"]["length"], cfg["problem"]["T"]
    g = NeumannGrid(M, L)
    u0 = _smooth_ic(g.x, L)
    ref = solve_reference(g, u0, D, rho, model, T, 2, rtol=1e-12, atol=1e-14)[-1]
    rows = []
    for n_steps in (10, 20, 40, 80, 160, 320, 640):
        u = solve_batch(g, u0, [D], [rho], model, T, 2, n_steps)[0, -1]
        rows.append({"steps": n_steps, "error": float(np.max(np.abs(u - ref)))})
    orders = [float(np.log2(rows[i]["error"] / rows[i + 1]["error"])) for i in range(len(rows) - 1)]
    return {"D": D, "rho": rho, "M": M, "rows": rows, "observed_order": orders}
