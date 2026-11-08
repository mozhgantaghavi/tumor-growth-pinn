import numpy as np
import pytest
import torch

from tumorpinn.cnn import ParamCNN, Scaler, predict_ensemble
from tumorpinn.config import load_config
from tumorpinn.data import add_noise, generate
from tumorpinn.metrics import summarize
from tumorpinn.models import get_model
from tumorpinn.nls import SpectralForward, fit_nls
from tumorpinn.pinn import fit_pinn


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def case(cfg):
    d = generate(cfg, 2, 11)
    return d["fields"][0], d["params"][0], d


def _bounds(cfg):
    pb = cfg["problem"]
    return (np.log10([pb["D_range"][0], pb["rho_range"][0]]), np.log10([pb["D_range"][1], pb["rho_range"][1]]))


def test_nls_recovers_parameters_from_clean_data(cfg, case):
    obs, true, _ = case
    lo, hi = _bounds(cfg)
    r = fit_nls(obs, SpectralForward(cfg, get_model("fisher_kpp")), (lo + hi) / 2, (lo, hi))
    assert np.all(np.abs(np.log10(r["params"]) - np.log10(true)) < 0.06)


def test_nls_is_robust_to_moderate_noise(cfg, case):
    obs, true, _ = case
    noisy = add_noise(obs[None], 0.02, np.random.default_rng(0))[0]
    lo, hi = _bounds(cfg)
    r = fit_nls(noisy, SpectralForward(cfg, get_model("fisher_kpp")), (lo + hi) / 2, (lo, hi))
    assert np.all(np.abs(np.log10(r["params"]) - np.log10(true)) < 0.15)


def test_cnn_shapes_and_scaler_roundtrip(cfg):
    net = ParamCNN(cfg["cnn"]["width"])
    x = torch.zeros(3, 1, cfg["grid"]["n_t"], cfg["grid"]["n_x"])
    assert net(x).shape == (3, 2)
    sc = Scaler(cfg)
    p = np.array([[0.01, 3.0], [0.003, 8.0]])
    np.testing.assert_allclose(sc.decode(sc.encode(p)), p, rtol=1e-9)
    pred, std = predict_ensemble([net, ParamCNN(cfg["cnn"]["width"])], np.zeros((3, 32, 64), np.float32), cfg)
    assert pred.shape == (3, 2) and std.shape == (3, 2)


def test_pinn_reduces_data_loss_and_stays_finite(cfg, case):
    obs, true, d = case
    small = {**cfg, "pinn": {**cfg["pinn"], "n_collocation": 200, "hidden": 24, "layers": 3}}
    lo, hi = _bounds(cfg)
    r = fit_pinn(obs, d["x"], d["t"], get_model("fisher_kpp"), small, (lo + hi) / 2, iters=60, pretrain=60)
    h = r["history"]
    assert h["data"][-1] < h["data"][0]
    assert np.all(np.isfinite(r["params"])) and r["recon"].shape == obs.shape


def test_metrics_summary():
    true = np.array([[0.01, 2.0], [0.02, 4.0]])
    m = summarize(true * 1.1, true)
    assert abs(m["mape_D"] - 10.0) < 1e-9 and abs(m["mape_rho"] - 10.0) < 1e-9
    assert summarize(true, true)["mae_log10_D"] == 0.0
