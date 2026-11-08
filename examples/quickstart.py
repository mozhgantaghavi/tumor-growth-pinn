"""Minimal tour of the Python API: simulate a tumour, add noise, recover (D, rho).

    python examples/quickstart.py
"""
import numpy as np

from tumorpinn.config import load_config
from tumorpinn.data import add_noise, generate
from tumorpinn.models import get_model
from tumorpinn.nls import SpectralForward, fit_nls

cfg = load_config()
model = get_model(cfg["problem"]["model"])

# 1. simulate one tumour with the Legendre-Gauss-Lobatto spectral solver
cohort = generate(cfg, n=1, seed=42)
truth = cohort["params"][0]
clean = cohort["fields"][0]                              # (n_t, n_x) space-time image

# 2. corrupt it with 3 % Gaussian measurement noise
observed = add_noise(clean[None], 0.03, np.random.default_rng(0))[0]

# 3. recover the parameters with the spectral least-squares baseline
pb = cfg["problem"]
lo = np.log10([pb["D_range"][0], pb["rho_range"][0]])
hi = np.log10([pb["D_range"][1], pb["rho_range"][1]])
fit = fit_nls(observed, SpectralForward(cfg, model), (lo + hi) / 2, (lo, hi))

print(f"true      D = {truth[0]:.4f}   rho = {truth[1]:.3f}")
print(f"estimated D = {fit['params'][0]:.4f}   rho = {fit['params'][1]:.3f}   ({fit['seconds']:.2f} s)")

# 4. the same call pattern works with the CNN (tumorpinn.cnn) and the PINN (tumorpinn.pinn);
#    see `python -m tumorpinn --help` for the full pipeline.
