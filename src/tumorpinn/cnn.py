"""Amortised parameter inference: a CNN reads the (time x space) tumour image
and regresses (log10 D, log10 rho)."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn


class ParamCNN(nn.Module):
    def __init__(self, width: int = 32, n_out: int = 2):
        super().__init__()

        def block(cin, cout, double=True):
            layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU()]
            if double:
                layers += [nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU()]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, width, double=False), nn.MaxPool2d(2),
            block(width, 2 * width), nn.MaxPool2d(2),
            block(2 * width, 4 * width), nn.AdaptiveAvgPool2d((2, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.1), nn.Linear(4 * width * 8, 128), nn.GELU(), nn.Linear(128, n_out)
        )

    def forward(self, x):
        return self.head(self.features(x))


class Scaler:
    """Maps physical (D, rho) <-> normalised log-space targets in [-1, 1]."""

    def __init__(self, cfg: dict):
        pb = cfg["problem"]
        self.lo = np.log10([pb["D_range"][0], pb["rho_range"][0]])
        self.hi = np.log10([pb["D_range"][1], pb["rho_range"][1]])

    def encode(self, params: np.ndarray) -> np.ndarray:
        return 2.0 * (np.log10(params) - self.lo) / (self.hi - self.lo) - 1.0

    def decode(self, z: np.ndarray) -> np.ndarray:
        return 10.0 ** ((z + 1.0) / 2.0 * (self.hi - self.lo) + self.lo)


def _prep(fields: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((fields - 0.5) / 0.5).astype(np.float32))[:, None]


def train_member(args) -> tuple[dict, dict]:
    """Process-pool entry point: train one ensemble member, return (state_dict, history)."""
    cfg, splits, seed = args
    torch.set_num_threads(cfg["cnn"]["threads"])
    net, hist = train_cnn(cfg, splits, seed, log=lambda m: print(m, flush=True))
    return net.state_dict(), hist


def train_cnn(cfg: dict, splits: dict, seed: int, log=print) -> tuple[ParamCNN, dict]:
    """Train one CNN with on-the-fly noise augmentation."""
    c = cfg["cnn"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sc = Scaler(cfg)
    Xtr, Ytr = splits["train_fields"], torch.from_numpy(sc.encode(splits["train_params"]).astype(np.float32))
    Xva, Yva = splits["val_fields"], sc.encode(splits["val_params"])
    net = ParamCNN(c["width"])
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    steps = c["epochs"] * int(np.ceil(len(Xtr) / c["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps)
    sig_val = np.tile(np.array(cfg["noise"]["levels"], dtype=np.float32), len(Xva) // len(cfg["noise"]["levels"]) + 1)[: len(Xva)]
    Xva_noisy = Xva + sig_val[:, None, None] * rng.standard_normal(Xva.shape).astype(np.float32)
    hist = {"epoch": [], "train_loss": [], "val_loss": []}
    t0 = time.time()
    for ep in range(c["epochs"]):
        net.train()
        perm = rng.permutation(len(Xtr))
        run, nb = 0.0, 0
        for i in range(0, len(perm), c["batch_size"]):
            idx = perm[i:i + c["batch_size"]]
            sig = rng.uniform(0.0, cfg["noise"]["train_max"], len(idx)).astype(np.float32)
            xb = _prep(Xtr[idx] + sig[:, None, None] * rng.standard_normal(Xtr[idx].shape).astype(np.float32))
            loss = nn.functional.smooth_l1_loss(net(xb), Ytr[idx], beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run, nb = run + loss.item(), nb + 1
        net.eval()
        with torch.no_grad():
            vl = nn.functional.smooth_l1_loss(net(_prep(Xva_noisy)), torch.from_numpy(Yva.astype(np.float32)), beta=0.1).item()
        hist["epoch"].append(ep + 1)
        hist["train_loss"].append(run / nb)
        hist["val_loss"].append(vl)
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"  [cnn seed {seed}] epoch {ep + 1:3d}/{c['epochs']}  train {run / nb:.4f}  val {vl:.4f}  ({time.time() - t0:.0f}s)")
    hist["seconds"] = time.time() - t0
    return net, hist


def save_nets(nets: list[ParamCNN], path) -> None:
    torch.save([n.state_dict() for n in nets], path)


def load_nets(cfg: dict, path) -> list[ParamCNN]:
    nets = []
    for sd in torch.load(path, map_location="cpu", weights_only=True):
        net = ParamCNN(cfg["cnn"]["width"])
        net.load_state_dict(sd)
        nets.append(net.eval())
    return nets


@torch.no_grad()
def predict_ensemble(nets: list[ParamCNN], fields: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble mean parameters (physical units) and log10 std (n, 2)."""
    sc = Scaler(cfg)
    x = _prep(fields)
    zs = []
    for net in nets:
        net.eval()
        zs.append(net(x).numpy())
    zs = np.stack(zs)                      # (E, n, 2)
    logp = (zs + 1.0) / 2.0 * (sc.hi - sc.lo) + sc.lo
    return 10.0 ** logp.mean(0), logp.std(0)
