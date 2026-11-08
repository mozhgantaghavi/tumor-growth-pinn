import numpy as np
import pytest
import torch

from tumorpinn import real_data as rd
from tumorpinn.config import ROOT, load_config


@pytest.fixture(scope="module")
def rc():
    cfg = load_config()
    # resolve the data paths relative to the repository, whatever the working directory is
    for ds in cfg["real"]["datasets"].values():
        ds["path"] = str(ROOT / ds["path"])
    return cfg["real"]


@pytest.mark.parametrize("key,n_animals", [("breast", 57), ("lung", 19), ("treated", 166)])
def test_datasets_load_with_expected_size(rc, key, n_animals):
    animals = rd.load_animals(rc, key)
    assert len(animals) == n_animals
    for a in animals:
        assert len(a["t"]) >= rc["min_points"]
        assert a["t"][0] == 0 and np.all(np.diff(a["t"]) > 0)
        assert np.all(a["v"] > 0)


def test_treated_excludes_animals_already_in_breast(rc):
    sig = lambda a: rd._signature(list(zip(a["t"] + a["t0"], a["v"])))  # noqa: E731
    breast = {sig(a) for a in rd.load_animals(rc, "breast")}
    assert not any(sig(a) in breast for a in rd.load_animals(rc, "treated"))


def test_treated_animals_carry_a_group_label(rc):
    groups = {a["group"] for a in rd.load_animals(rc, "treated")}
    assert any(g.startswith("Su") for g in groups) and any(g.startswith("Ax") for g in groups)


@pytest.mark.parametrize("law", rd.LAWS)
def test_classical_fit_recovers_known_growth_law(law):
    t = np.linspace(0, 20, 12)
    lnK = np.log(2000.0)
    x = rd._curve(law, t, np.log(80.0), 0.15, lnK)
    fit = rd.nls_fit(t, np.exp(x), law)
    assert fit["rss"] < 1e-8
    np.testing.assert_allclose(fit["predict"](t), np.exp(x), rtol=1e-3)
    np.testing.assert_allclose(fit["alpha"], 0.15, rtol=0.05)


def test_batched_networks_are_independent():
    gen = torch.Generator().manual_seed(0)
    net = rd.BatchedMLP(3, 8, 2, torch.device("cpu"), gen)
    x = torch.rand(3, 5, 1)
    y = net(x)
    x2 = x.clone()
    x2[1] += 1.0                                   # perturb the input of network 1 only
    y2 = net(x2)
    assert torch.allclose(y[0], y2[0]) and torch.allclose(y[2], y2[2])
    assert not torch.allclose(y[1], y2[1])


def test_growth_pinn_training_reduces_the_loss(rc):
    rc = {**rc, "restarts": 1, "hidden": 16, "layers": 2}
    animals = rd.load_animals(rc, "lung")[:3]
    jobs = rd._pack_jobs(animals, rc["holdout"], rc["restarts"], torch.device("cpu"))
    model = rd.GrowthPINN(jobs, rc["hidden"], rc["layers"], torch.device("cpu"), seed=1)
    ld0, lp0 = model.losses(32)
    model.train(150, 30, 3e-3, 32)
    ld1, lp1 = model.losses(32)
    assert (ld1 + lp1).mean() < (ld0 + lp0).mean()
    assert torch.isfinite(model.predict_volume(torch.linspace(0, 20, 5).expand(len(jobs["meta"]), -1))).all()
