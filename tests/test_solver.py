import numpy as np
import pytest

from tumorpinn.config import load_config
from tumorpinn.data import add_noise, generate, sample_initial_conditions
from tumorpinn.models import REGISTRY, get_model
from tumorpinn.solver import front_position, solve_batch, solve_reference
from tumorpinn.spectral import NeumannGrid


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _ic(x):
    return 0.6 * np.exp(-0.5 * ((x - 0.4) / 0.1) ** 2)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_matches_radau_reference(name):
    g = NeumannGrid(48, 1.0)
    model = get_model(name)
    u0 = _ic(g.x)
    ours = solve_batch(g, u0, [0.01], [3.0], model, 1.0, 6, 160)[0]
    ref = solve_reference(g, u0, 0.01, 3.0, model, 1.0, 6)
    assert np.max(np.abs(ours - ref)) < 5e-4


def test_pure_diffusion_conserves_mass():
    g = NeumannGrid(40, 1.0)
    model = get_model("fisher_kpp")
    snaps = solve_batch(g, _ic(g.x), [0.02], [0.0], model, 0.5, 5, 20)[0]
    mass = snaps @ g.w
    np.testing.assert_allclose(mass, mass[0], rtol=1e-6)


def test_temporal_order_is_two():
    g = NeumannGrid(32, 1.0)
    model = get_model("fisher_kpp")
    ref = solve_reference(g, _ic(g.x), 0.01, 3.0, model, 1.0, 2, rtol=1e-12, atol=1e-14)[-1]
    err = [np.max(np.abs(solve_batch(g, _ic(g.x), [0.01], [3.0], model, 1.0, 2, n)[0, -1] - ref)) for n in (20, 40, 80)]
    orders = [np.log2(err[i] / err[i + 1]) for i in range(2)]
    assert min(orders) > 1.7


def test_fisher_front_speed_matches_bramson_correction():
    # Smooth start (a jump would seed Gibbs ringing that the reaction amplifies) and a modest
    # rho * T so round-off in the empty region cannot grow into a spurious front.
    L, D, rho, T = 8.0, 0.05, 1.0, 12.0
    g = NeumannGrid(96, L)
    model = get_model("fisher_kpp")
    u0 = 0.999 * 0.5 * (1.0 - np.tanh((g.x - 0.4) / 0.1))
    snaps = solve_batch(g, u0, [D], [rho], model, T, 25, 100)[0]
    xu, A = g.to_uniform(400)
    pos = front_position(snaps @ A.T, xu)
    t1, t2 = 6.0, 12.0
    lam = np.sqrt(rho / D)
    expected = model.front_speed(D, rho) - 1.5 / lam * np.log(t2 / t1) / (t2 - t1)
    speed = (pos[24] - pos[12]) / (t2 - t1)
    assert abs(speed - expected) / expected < 0.05


def test_batch_equals_single(cfg):
    g = NeumannGrid(32, 1.0)
    model = get_model("fisher_kpp")
    D, rho = np.array([0.005, 0.02]), np.array([2.0, 6.0])
    u0 = np.stack([_ic(g.x), _ic(g.x) * 0.8])
    batch = solve_batch(g, u0, D, rho, model, 1.0, 8, 10)
    for i in range(2):
        single = solve_batch(g, u0[i], D[i:i + 1], rho[i:i + 1], model, 1.0, 8, 10)[0]
        np.testing.assert_allclose(batch[i], single, atol=1e-12)


def test_dataset_is_reproducible_and_bounded(cfg):
    a, b = generate(cfg, 8, 3), generate(cfg, 8, 3)
    np.testing.assert_array_equal(a["fields"], b["fields"])
    assert a["fields"].min() >= 0 and a["fields"].max() <= 1
    pb = cfg["problem"]
    assert np.all((a["params"][:, 0] >= pb["D_range"][0]) & (a["params"][:, 0] <= pb["D_range"][1]))
    assert a["fields"].shape == (8, cfg["grid"]["n_t"], cfg["grid"]["n_x"])


def test_noise_has_requested_std(cfg):
    f = np.zeros((20, 32, 64), dtype=np.float32)
    n = add_noise(f, 0.05, np.random.default_rng(0))
    assert abs(n.std() - 0.05) < 2e-3
    np.testing.assert_array_equal(add_noise(f, 0.0, np.random.default_rng(0)), f)


def test_initial_conditions_are_bounded_and_centred(cfg):
    x = np.linspace(0, 1, 101)
    ic = sample_initial_conditions(np.random.default_rng(1), x, 200, cfg)
    assert ic.min() >= 0 and ic.max() <= 1
    # seed centres keep a margin from the walls, so tumours are denser mid-domain than at the walls
    wall = 0.5 * (ic[:, 0].mean() + ic[:, -1].mean())
    assert wall < 0.5 * ic[:, 40:61].mean()
    # the tail at a wall can never exceed amplitude * exp(-margin^2 / (2 width^2)) summed over seeds
    ic_cfg = cfg["ic"]
    bound = ic_cfg["n_seeds"][1] * ic_cfg["amplitude"][1] * np.exp(-0.5 * (ic_cfg["margin"] / ic_cfg["width"][1]) ** 2)
    assert ic[:, 0].max() <= bound + 1e-9 and ic[:, -1].max() <= bound + 1e-9
