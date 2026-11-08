"""Legendre-Gauss-Lobatto (LGL) spectral collocation on an interval.

The radial/spatial direction of the Fourier-Legendre method of Taghavi &
Shahrokhi-Dehkordi (Q. Appl. Math., 2023) is reused here: functions are
represented by their values at the LGL nodes (the roots of (1 - x^2) L_M'(x))
and differentiated with the dense collocation matrix D_mk.  Homogeneous
Neumann (no-flux) conditions are imposed by eliminating the two boundary
unknowns, which leaves a square operator acting on the interior nodes only.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre as npleg


def lgl_nodes(M: int) -> np.ndarray:
    """Return the M+1 LGL nodes on [-1, 1] in ascending order."""
    if M < 2:
        raise ValueError("M must be at least 2")
    c = np.zeros(M + 1)
    c[M] = 1.0
    dc = npleg.legder(c)
    inner = np.sort(npleg.legroots(dc).real)
    # Newton polish on L_M'(x) = 0, using (1 - x^2) L'' = 2 x L' - M (M + 1) L.
    for _ in range(3):
        L = npleg.legval(inner, c)
        dL = npleg.legval(inner, dc)
        d2L = (2.0 * inner * dL - M * (M + 1) * L) / (1.0 - inner**2)
        inner = inner - dL / d2L
    return np.concatenate(([-1.0], inner, [1.0]))


def lgl_weights(M: int, x: np.ndarray | None = None) -> np.ndarray:
    """LGL quadrature weights w_m = 2 / (M (M + 1) L_M(x_m)^2)."""
    x = lgl_nodes(M) if x is None else x
    c = np.zeros(M + 1)
    c[M] = 1.0
    return 2.0 / (M * (M + 1) * npleg.legval(x, c) ** 2)


def lgl_diff_matrix(M: int, x: np.ndarray | None = None) -> np.ndarray:
    """Collocation derivative matrix D with (D v)_k = v'(x_k) on [-1, 1]."""
    x = lgl_nodes(M) if x is None else x
    c = np.zeros(M + 1)
    c[M] = 1.0
    LM = npleg.legval(x, c)
    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)
    D = (LM[:, None] / LM[None, :]) / diff
    np.fill_diagonal(D, 0.0)
    D[0, 0] = -M * (M + 1) / 4.0
    D[M, M] = M * (M + 1) / 4.0
    return D


def barycentric_weights(x: np.ndarray) -> np.ndarray:
    """Barycentric weights 1 / prod_{k != j}(x_j - x_k), computed in log space."""
    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)
    sign = np.prod(np.sign(diff), axis=1)
    logmag = -np.sum(np.log(np.abs(diff)), axis=1)
    logmag -= logmag.max()
    return sign * np.exp(logmag)


def interpolation_matrix(nodes: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Spectral (barycentric Lagrange) interpolation matrix nodes -> xq."""
    w = barycentric_weights(nodes)
    diff = xq[:, None] - nodes[None, :]
    exact = np.isclose(diff, 0.0, atol=1e-14)
    diff[exact] = 1.0
    tmp = w[None, :] / diff
    A = tmp / tmp.sum(axis=1, keepdims=True)
    rows = np.where(exact.any(axis=1))[0]
    for r in rows:
        A[r] = exact[r].astype(float)
    return A


class NeumannGrid:
    """LGL collocation grid on [0, length] with no-flux boundaries.

    Attributes
    ----------
    x        : (M+1,) physical LGL nodes.
    w        : (M+1,) quadrature weights on [0, length].
    D1, D2   : first / second derivative matrices (physical units).
    P        : (M+1, M-1) prolongation, interior values -> all nodes with
               u_x = 0 enforced at both ends.
    L        : (M-1, M-1) interior Neumann Laplacian, L = D2[int] @ P.
    """

    def __init__(self, M: int, length: float = 1.0):
        self.M = int(M)
        self.length = float(length)
        xi = lgl_nodes(self.M)
        self.xi = xi
        self.x = (xi + 1.0) * self.length / 2.0
        self.w = lgl_weights(self.M, xi) * self.length / 2.0
        self.D1 = lgl_diff_matrix(self.M, xi) * (2.0 / self.length)
        self.D2 = self.D1 @ self.D1
        b = [0, self.M]
        inner = np.arange(1, self.M)
        E = -np.linalg.solve(self.D1[np.ix_(b, b)], self.D1[np.ix_(b, inner)])
        P = np.zeros((self.M + 1, self.M - 1))
        P[0], P[self.M] = E[0], E[1]
        P[1:self.M] = np.eye(self.M - 1)
        self.P = P
        self.L = self.D2[1:self.M, :] @ P

    def to_uniform(self, n_x: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the uniform grid (n_x points) and the interpolation matrix."""
        xu = np.linspace(0.0, self.length, n_x)
        return xu, interpolation_matrix(self.x, xu)

    def integrate(self, u: np.ndarray) -> np.ndarray:
        """Spectrally accurate integral over the domain (last axis = nodes)."""
        return u @ self.w
