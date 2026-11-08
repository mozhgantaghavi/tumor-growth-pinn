"""Reaction-diffusion tumor-growth models: u_t = D u_xx + f(u; rho).

Both models are dimensionless (carrying capacity 1) and share the parameter
pair (D, rho): D is the cell diffusivity, rho the proliferation rate.
New models only need an entry in ``REGISTRY`` (NumPy and PyTorch flavour).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_EPS = 1e-8


def _fisher_np(u, rho):
    return rho * u * (1.0 - u)


def _gompertz_np(u, rho):
    return -rho * u * np.log(np.clip(u, _EPS, None))


def _fisher_torch(u, rho):
    return rho * u * (1.0 - u)


def _gompertz_torch(u, rho):
    import torch

    return -rho * u * torch.log(torch.clamp(u, min=_EPS))


@dataclass(frozen=True)
class PDEModel:
    name: str
    title: str
    latex: str
    reaction_np: Callable
    reaction_torch: Callable

    def front_speed(self, D: float, rho: float) -> float:
        """Asymptotic travelling-front speed (Fisher-KPP: 2 sqrt(D rho))."""
        return 2.0 * float(np.sqrt(D * rho))


REGISTRY: dict[str, PDEModel] = {
    "fisher_kpp": PDEModel(
        "fisher_kpp",
        "Fisher-KPP (logistic growth)",
        r"u_t = D\,u_{xx} + \rho\,u\,(1-u)",
        _fisher_np,
        _fisher_torch,
    ),
    "gompertz": PDEModel(
        "gompertz",
        "Gompertz growth",
        r"u_t = D\,u_{xx} - \rho\,u\,\ln u",
        _gompertz_np,
        _gompertz_torch,
    ),
}


def get_model(name: str) -> PDEModel:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown model '{name}', choose from {sorted(REGISTRY)}") from exc
