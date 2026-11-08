"""Forward solvers for u_t = D u_xx + f(u; rho) with no-flux boundaries.

``solve_batch`` is the workhorse: an LGL-collocation method of lines with a
Crank-Nicolson / Adams-Bashforth-2 (IMEX) time stepper, vectorised over a batch
of (D, rho) pairs.  ``solve_reference`` integrates the same semi-discrete system
with an implicit Radau method at tight tolerance and is used for validation.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .models import PDEModel
from .spectral import NeumannGrid


def solve_batch(
    grid: NeumannGrid,
    u0: np.ndarray,
    D: np.ndarray,
    rho: np.ndarray,
    model: PDEModel,
    T: float,
    n_snap: int,
    substeps: int,
    chunk: int = 256,
) -> np.ndarray:
    """Integrate a batch of problems; returns snapshots (B, n_snap, M+1).

    u0 : (B, M+1) or (M+1,) initial condition at the LGL nodes.
    Snapshot k is taken at t = k * T / (n_snap - 1); ``substeps`` solver steps
    separate two snapshots.
    """
    D = np.atleast_1d(np.asarray(D, dtype=float))
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    B = D.shape[0]
    u0 = np.broadcast_to(np.atleast_2d(u0), (B, grid.M + 1))
    m = grid.M - 1
    nsteps = (n_snap - 1) * substeps
    h = T / nsteps
    eye = np.eye(m)
    out = np.empty((B, n_snap, grid.M + 1))
    for s in range(0, B, chunk):
        sl = slice(s, min(B, s + chunk))
        Dc, rc = D[sl], rho[sl][:, None]
        Lh = (Dc[:, None, None] * h) * grid.L[None]
        Ainv = np.linalg.inv(eye - 0.5 * Lh)
        G = Ainv @ (eye + 0.5 * Lh)
        # project the initial data onto the Neumann-consistent subspace
        ui = u0[sl][:, 1:grid.M].copy()
        out[sl, 0] = ui @ grid.P.T
        f_prev = model.reaction_np(ui, rc)
        for n in range(nsteps):
            f_cur = model.reaction_np(ui, rc)
            fx = f_cur if n == 0 else 1.5 * f_cur - 0.5 * f_prev
            ui = (G @ ui[..., None])[..., 0] + h * (Ainv @ fx[..., None])[..., 0]
            f_prev = f_cur
            if (n + 1) % substeps == 0:
                out[sl, (n + 1) // substeps] = ui @ grid.P.T
    return out


def solve_reference(
    grid: NeumannGrid,
    u0: np.ndarray,
    D: float,
    rho: float,
    model: PDEModel,
    T: float,
    n_snap: int,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> np.ndarray:
    """High-accuracy reference snapshots (n_snap, M+1) for one parameter pair."""
    L = grid.L

    def rhs(_t, v):
        return D * (L @ v) + model.reaction_np(v, rho)

    t_eval = np.linspace(0.0, T, n_snap)
    sol = solve_ivp(rhs, (0.0, T), u0[1:grid.M], method="Radau", t_eval=t_eval,
                    rtol=rtol, atol=atol, jac=lambda _t, v: D * L + _num_jac(model, v, rho))
    return sol.y.T @ grid.P.T


def _num_jac(model: PDEModel, v: np.ndarray, rho: float) -> np.ndarray:
    eps = 1e-7
    f0 = model.reaction_np(v, rho)
    # reaction is pointwise, so the Jacobian is diagonal
    d = (model.reaction_np(v + eps, rho) - f0) / eps
    return np.diag(d)


def front_position(u: np.ndarray, x: np.ndarray, level: float = 0.5) -> np.ndarray:
    """Right-most position where u crosses ``level`` (per time row)."""
    pos = np.full(u.shape[0], np.nan)
    for k, row in enumerate(u):
        above = np.where(row >= level)[0]
        if above.size == 0:
            continue
        j = above.max()
        if j + 1 < row.size:
            x0, x1, y0, y1 = x[j], x[j + 1], row[j], row[j + 1]
            pos[k] = x0 + (level - y0) * (x1 - x0) / (y1 - y0)
        else:
            pos[k] = x[j]
    return pos
