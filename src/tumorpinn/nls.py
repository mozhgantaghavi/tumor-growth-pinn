"""Classical baseline: nonlinear least squares with the spectral solver in the loop."""
from __future__ import annotations

import time

import numpy as np
from numpy.polynomial import legendre as npleg
from scipy.optimize import least_squares

from .models import PDEModel
from .solver import solve_batch
from .spectral import NeumannGrid


class SpectralForward:
    """Forward map (log10 D, log10 rho) -> snapshots on the uniform observation grid."""

    def __init__(self, cfg: dict, model: PDEModel):
        n = cfg["nls"]
        self.cfg, self.model = cfg, model
        self.grid = NeumannGrid(n["M"], cfg["problem"]["length"])
        self.T = cfg["problem"]["T"]
        self.n_t = cfg["grid"]["n_t"]
        self.substeps = n["substeps"]
        self.xu, self.A = self.grid.to_uniform(cfg["grid"]["n_x"])
        self.deg = n["ic_degree"]
        self._V = npleg.legvander(2 * self.xu / cfg["problem"]["length"] - 1, self.deg)
        self._Vn = npleg.legvander(self.grid.xi, self.deg)
        k = np.arange(self.deg + 1)
        self._A = self._V.T @ self._V + n["ic_ridge"] * np.diag(k ** 4.0)

    def smooth_ic(self, row0: np.ndarray) -> np.ndarray:
        """Ridge-regularised Legendre fit of the (noisy) t=0 slice, sampled at the LGL nodes."""
        coef = np.linalg.solve(self._A, self._V.T @ row0)
        return np.clip(self._Vn @ coef, 0.0, 1.0)

    def __call__(self, theta: np.ndarray, u0_nodes: np.ndarray) -> np.ndarray:
        D, rho = 10.0 ** theta[0], 10.0 ** theta[1]
        snaps = solve_batch(self.grid, u0_nodes, [D], [rho], self.model, self.T, self.n_t, self.substeps)[0]
        return snaps @ self.A.T


def fit_nls(obs: np.ndarray, fwd: SpectralForward, theta0: np.ndarray, bounds: tuple) -> dict:
    """Fit (log10 D, log10 rho) to the observed image by trust-region least squares."""
    t0 = time.time()
    u0 = fwd.smooth_ic(obs[0])
    nfev = {"n": 0}

    def resid(theta):
        nfev["n"] += 1
        return (fwd(theta, u0) - obs).ravel()

    sol = least_squares(resid, theta0, bounds=bounds, x_scale=1.0, diff_step=1e-3, max_nfev=60)
    return {
        "theta": sol.x,
        "params": 10.0 ** sol.x,
        "cost": float(sol.cost),
        "nfev": nfev["n"],
        "seconds": time.time() - t0,
        "recon": fwd(sol.x, u0),
    }
