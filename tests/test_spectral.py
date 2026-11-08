import numpy as np
import pytest

from tumorpinn.spectral import NeumannGrid, interpolation_matrix, lgl_diff_matrix, lgl_nodes, lgl_weights


@pytest.mark.parametrize("M", [4, 8, 16, 32])
def test_nodes_symmetric_and_ascending(M):
    x = lgl_nodes(M)
    assert x[0] == -1 and x[-1] == 1
    assert np.all(np.diff(x) > 0)
    np.testing.assert_allclose(x, -x[::-1], atol=1e-13)


@pytest.mark.parametrize("M", [8, 16, 24])
def test_quadrature_exact_up_to_degree_2M_minus_1(M):
    x, w = lgl_nodes(M), lgl_weights(M)
    for k in range(0, 2 * M, 3):
        exact = 0.0 if k % 2 else 2.0 / (k + 1)
        assert abs(w @ x**k - exact) < 1e-12


def test_differentiation_is_spectrally_accurate():
    errs = []
    for M in (8, 12, 16, 20):
        x = lgl_nodes(M)
        D = lgl_diff_matrix(M, x)
        errs.append(np.max(np.abs(D @ np.exp(np.sin(x)) - np.cos(x) * np.exp(np.sin(x)))))
    assert errs[-1] < 1e-9
    assert all(a > b for a, b in zip(errs, errs[1:]))


def test_differentiation_exact_on_polynomials():
    M = 12
    x = lgl_nodes(M)
    D = lgl_diff_matrix(M, x)
    np.testing.assert_allclose(D @ x**7, 7 * x**6, atol=1e-10)
    np.testing.assert_allclose(D @ np.ones_like(x), 0.0, atol=1e-11)


def test_interpolation_reproduces_polynomials_and_nodes():
    x = lgl_nodes(16)
    xq = np.linspace(-1, 1, 41)
    A = interpolation_matrix(x, xq)
    np.testing.assert_allclose(A @ x**5, xq**5, atol=1e-12)
    np.testing.assert_allclose(interpolation_matrix(x, x), np.eye(len(x)), atol=1e-12)


def test_neumann_operator_has_no_flux_and_conserves_mass():
    g = NeumannGrid(24, 1.0)
    u_int = np.exp(-0.5 * ((g.x[1:-1] - 0.5) / 0.1) ** 2)
    u = g.P @ u_int
    assert abs((g.D1 @ u)[0]) < 1e-9 and abs((g.D1 @ u)[-1]) < 1e-9
    # d/dt of the integral of u under pure diffusion must vanish
    assert abs(g.w @ (g.D2 @ u)) < 1e-8
    # constants are in the kernel of the Neumann Laplacian
    np.testing.assert_allclose(g.L @ np.ones(g.M - 1), 0.0, atol=1e-8)
