"""Physics-informed neural network for the inverse problem.

A tanh MLP u_theta(x, t) is fitted to the noisy observations while the PDE
residual u_t - D u_xx - f(u; rho) is penalised at random collocation points.
(D, rho) are trainable scalars (kept in log10 so they stay positive).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .models import PDEModel


class PINN(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 4):
        super().__init__()
        mods, d = [], 2
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, xs, ts):
        """xs, ts are already scaled to [-1, 1]; output is a density in (0, 1)."""
        return torch.sigmoid(self.net(torch.cat([xs, ts], dim=-1)))


def fit_pinn(
    obs: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    model: PDEModel,
    cfg: dict,
    theta0: np.ndarray,
    iters: int,
    seed: int = 0,
    pretrain: int | None = None,
    record_every: int = 25,
    lr_params: float | None = None,
) -> dict:
    """Fit the PINN; ``theta0`` = initial (log10 D, log10 rho); ``lr_params`` overrides the (D, rho) learning rate."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    p = cfg["pinn"]
    L, T = x[-1] - x[0], t[-1] - t[0]
    pretrain = p["pretrain_iters"] if pretrain is None else pretrain
    Xg, Tg = np.meshgrid(x, t)                                  # (n_t, n_x)
    xo = torch.tensor(2 * Xg.ravel() / L - 1, dtype=torch.float32)[:, None]
    to = torch.tensor(2 * Tg.ravel() / T - 1, dtype=torch.float32)[:, None]
    uo = torch.tensor(obs.ravel(), dtype=torch.float32)[:, None]

    net = PINN(p["hidden"], p["layers"])
    theta = nn.Parameter(torch.tensor(theta0, dtype=torch.float32))
    opt = torch.optim.Adam([
        {"params": net.parameters(), "lr": p["lr"]},
        {"params": [theta], "lr": p["lr_params"] if lr_params is None else lr_params},
    ])
    total = pretrain + iters
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-5)
    hist = {"iter": [], "data": [], "pde": [], "log10_D": [], "log10_rho": []}
    t0 = time.time()
    for it in range(total):
        joint = it >= pretrain
        theta.requires_grad_(joint)
        u_pred = net(xo, to)
        l_data = torch.mean((u_pred - uo) ** 2)
        loss = l_data
        l_pde = torch.zeros(())
        if joint:
            xc = (torch.rand(p["n_collocation"], 1) * 2 - 1).requires_grad_(True)
            tc = (torch.rand(p["n_collocation"], 1) * 2 - 1).requires_grad_(True)
            u = net(xc, tc)
            ones = torch.ones_like(u)
            u_x, u_t = torch.autograd.grad(u, [xc, tc], ones, create_graph=True)
            u_xx = torch.autograd.grad(u_x, xc, torch.ones_like(u_x), create_graph=True)[0]
            D, rho = 10.0 ** theta[0], 10.0 ** theta[1]
            res = (2 / T) * u_t - D * (2 / L) ** 2 * u_xx - model.reaction_torch(u, rho)
            l_pde = torch.mean(res ** 2) / (1.0 + float(rho.detach()) ** 2)
            # no-flux walls
            tb = (torch.rand(256, 1) * 2 - 1)
            xb = torch.cat([-torch.ones(256, 1), torch.ones(256, 1)]).requires_grad_(True)
            ub = net(xb, torch.cat([tb, tb]))
            ux_b = torch.autograd.grad(ub, xb, torch.ones_like(ub), create_graph=True)[0]
            loss = l_data + p["w_pde"] * l_pde + p["w_bc"] * torch.mean(ux_b ** 2) * 1e-2
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % record_every == 0 or it == total - 1:
            hist["iter"].append(it)
            hist["data"].append(float(l_data))
            hist["pde"].append(float(l_pde))
            hist["log10_D"].append(float(theta[0]))
            hist["log10_rho"].append(float(theta[1]))
    with torch.no_grad():
        recon = net(xo, to).numpy().reshape(obs.shape)
    th = theta.detach().numpy().astype(float)
    return {
        "theta": th,
        "params": 10.0 ** th,
        "recon": recon,
        "history": hist,
        "seconds": time.time() - t0,
    }
