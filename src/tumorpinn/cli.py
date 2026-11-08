"""Command-line entry point: ``python -m tumorpinn <stage>``.

Stages are cached under ``artifacts/`` so any of them can be re-run alone:

    data         simulate and cache the train / val / test splits
    train        train the CNN ensemble (checkpoints/)
    benchmark    evaluate CNN, spectral least squares, PINN and CNN->PINN
    convergence  spectral / temporal convergence studies
    real         growth-law inference on real mouse tumour curves (GPU if available)
    export       write the JSON files consumed by the dashboard (docs/data/)
    all          every stage in order
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from . import real_data
from .benchmark import evaluate_cnn, run_slow_methods
from .cnn import ParamCNN, load_nets, save_nets, train_member
from .config import artifacts_dir, checkpoints_dir, docs_data_dir, load_config
from .convergence import spatial_convergence, temporal_convergence
from .data import build_splits
from .export import export_all, export_real

STAGES = ["data", "train", "benchmark", "convergence", "real", "export"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def stage_data(cfg, art):
    t0 = time.time()
    s = build_splits(cfg, art, force=True)
    _log(f"[data] train {s['train_fields'].shape}  val {s['val_fields'].shape}  test {s['test_fields'].shape}"
         f"  ({time.time() - t0:.1f}s)")
    return time.time() - t0


def stage_train(cfg, art):
    ck = checkpoints_dir(cfg)
    splits = build_splits(cfg, art)
    t0 = time.time()
    jobs = [(cfg, splits, cfg["cnn"]["seed"] + k) for k in range(cfg["cnn"]["ensemble"])]
    with ProcessPoolExecutor(len(jobs)) as ex:
        results = list(ex.map(train_member, jobs))
    nets, hists = [], []
    for sd, h in results:
        net = ParamCNN(cfg["cnn"]["width"])
        net.load_state_dict(sd)
        nets.append(net)
        hists.append(h)
    save_nets(nets, ck / f"cnn_{cfg['problem']['model']}.pt")
    (art / "train_history.json").write_text(json.dumps(hists))
    _log(f"[train] {len(nets)} CNNs, {sum(p.numel() for p in nets[0].parameters())} parameters each ({time.time() - t0:.0f}s)")
    return time.time() - t0


def stage_benchmark(cfg, art):
    splits = build_splits(cfg, art)
    nets = load_nets(cfg, checkpoints_dir(cfg) / f"cnn_{cfg['problem']['model']}.pt")
    t0 = time.time()
    cnn_res = evaluate_cnn(cfg, nets, splits)
    for s, r in cnn_res.items():
        m = r["metrics"]
        _log(f"[bench] CNN  sigma={s:>5}: MAPE D {m['mape_D']:.1f}%  rho {m['mape_rho']:.1f}%  ({r['ms_per_sample']:.2f} ms/sample)")
    slow = run_slow_methods(cfg, splits, cnn_res, log=_log)
    for s, e in slow["per_noise"].items():
        for k in ("nls", "pinn_cold", "pinn_warm"):
            if k in e:
                m = e[k]["metrics"]
                _log(f"[bench] {k:9s} sigma={s:>5}: MAPE D {m['mape_D']:.1f}%  rho {m['mape_rho']:.1f}%  "
                     f"({e[k]['seconds_mean']:.1f}s/case)")
    out = {"cnn": cnn_res, "slow": slow, "test_params": splits["test_params"].tolist(),
           "x": splits["x"], "t": splits["t"]}
    with open(art / "benchmark.pkl", "wb") as fh:
        pickle.dump(out, fh)
    return time.time() - t0


def stage_convergence(cfg, art):
    t0 = time.time()
    out = {"spatial": spatial_convergence(cfg), "temporal": temporal_convergence(cfg)}
    (art / "convergence.json").write_text(json.dumps(out))
    _log(f"[convergence] temporal observed order: {np.round(out['temporal']['observed_order'], 2).tolist()}")
    return time.time() - t0


def stage_real(cfg, art):
    t0 = time.time()
    results, summaries = {}, {}
    keys = [k for k in cfg["real"]["datasets"] if k in cfg["real"].get("only", list(cfg["real"]["datasets"]))]
    for n, key in enumerate(keys):
        timing = cfg["real"].get("timing", True) and n == 0          # device timing once, on the first dataset
        result = real_data.run(cfg, key, log=_log, with_timing=timing)
        summary = real_data.summarize(result, cfg["real"]["holdout"])
        results[key] = {"result": result, "summary": summary}
        summaries[key] = summary
        for method, laws in summary["holdout_error"].items():
            _log(f"[real:{key}] hold-out median error ({method}): "
                 + ", ".join(f"{k} {v['median_pct']:.1f}%" for k, v in laws.items()))
        _log(f"[real:{key}] BIC winners: {summary['winner_bic']}")
    export_real(cfg, results, docs_data_dir(cfg))
    (art / "real_summary.json").write_text(json.dumps(summaries, indent=1))
    return time.time() - t0


def stage_export(cfg, art, timings=None):
    docs = docs_data_dir(cfg)
    with open(art / "benchmark.pkl", "rb") as fh:
        bench = pickle.load(fh)
    hist = json.loads((art / "train_history.json").read_text())
    conv = json.loads((art / "convergence.json").read_text())
    tfile = art / "timings.json"
    timings = json.loads(tfile.read_text()) if tfile.exists() else {}
    export_all(cfg, art, docs, hist, bench, conv, timings)
    _log(f"[export] wrote {len(list(docs.glob('*.json')))} files to {docs}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tumorpinn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", default=None, help="TOML file merged over configs/default.toml")
    ap.add_argument("--model", default=None, help="override problem.model (fisher_kpp | gompertz)")
    ap.add_argument("--datasets", default=None, help="comma-separated subset of [real.datasets.*] for the real stage")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"problem": {"model": args.model}} if args.model else None)
    if args.datasets:
        cfg["real"]["only"] = args.datasets.split(",")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    art = artifacts_dir(cfg)
    tfile = art / "timings.json"
    timings = json.loads(tfile.read_text()) if tfile.exists() else {}
    fn = {"data": stage_data, "train": stage_train, "benchmark": stage_benchmark, "convergence": stage_convergence,
          "real": stage_real}
    for st in (STAGES if args.stage == "all" else [args.stage]):
        if st == "export":
            stage_export(cfg, art)
        else:
            timings[st] = round(fn[st](cfg, art), 1)
            tfile.write_text(json.dumps(timings))


if __name__ == "__main__":
    main()
