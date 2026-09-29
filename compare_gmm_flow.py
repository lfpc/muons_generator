"""GMM vs normalizing flow for the muon momentum, as a function of model size.

Both are fit in the momentum flow's space (fit_space.py) on the head of one group of the
split file and scored on the rest of it: the rate-weighted NLL of the physical (pz, pt)
density [GeV^-2], which does not depend on the fit space, plus the time to sample 1e6
muons on the GPU and on the CPU. The GMM is swept in its number of components
(--gmm_K), the flow in the width of its coupling MLPs (--flow_hidden) at --flow_layers.

Each result lands in <out_dir>/results.json as soon as it is done and finished models
are skipped on a rerun, so a sweep can be split over GPUs (--only, --device) or resumed.
Plots: outputs/figs/gmm_vs_flow_momentum_{nll,sampling_time}.png.

Run:  python compare_gmm_flow.py
      python compare_gmm_flow.py --plot_only
"""
import argparse
import fcntl
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gmm"))
import fit_space as fs
import flow_training as ft
from gen_eval import figs_dir, save_plot
from gmm_model import save_gmm
from models import NeuralSplineFlow, save_flow
from train_gmm import fit_gmm
from utils import CHUNK

GMM_COLOR, FLOW_COLOR = "#2a78d6", "#eb6834"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ft.ArgFormatter)
    p.add_argument("--data", default=ft.DEFAULT_SPLIT, help="Split muon file")
    p.add_argument("--group", default=ft.HI, help="Group to train and test on")
    p.add_argument("--train_rows", type=int, default=142_000_000, help="Training rows, from the head of --group")
    p.add_argument("--val_rows", type=int, default=2_000_000, help="Rows after those, for the flow's early stopping")
    p.add_argument("--gmm_K", type=int, nargs="+", default=[8, 32, 128, 512, 1024, 2048], help="GMM sizes")
    p.add_argument("--flow_hidden", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128], help="Flow MLP widths")
    p.add_argument("--flow_layers", type=int, default=4, help="Coupling layers of every flow")
    p.add_argument("--epochs", type=int, default=150, help="Flow epochs (passes over the training rows)")
    p.add_argument("--batch_size", type=int, default=2**19, help="Flow rows per step")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--lr_factor", type=float, default=0.5, help="LR multiplier on a plateau")
    p.add_argument("--lr_patience", type=int, default=5, help="Epochs before the LR drops")
    p.add_argument("--min_lr", type=float, default=1e-6, help="Floor of the learning rate")
    p.add_argument("--patience", type=int, default=30, help="Early-stopping patience (epochs)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="torch.compile the flow")
    p.add_argument("--time_samples", type=int, default=1_000_000, help="Samples per timing")
    p.add_argument("--cpu_threads", type=int, default=16, help="torch threads on the CPU")
    p.add_argument("--only", choices=("both", "gmm", "flow"), default="both", help="Which sweep to run")
    p.add_argument("--device", default="cuda", help="Training device")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--out_dir", default="outputs/gmm_vs_flow_momentum", help="Models and results.json")
    p.add_argument("--plot_only", action="store_true", help="Only redraw the figures")
    return p.parse_args()


def load_data(args):
    """Standardized training rows [u, v, w] (numpy), validation rows (tensor on the
    device) and test rows [u, v, log_jac, w], plus the space and its mean/std."""
    with h5py.File(args.data, "r") as f:
        g = f[args.group]
        n = g["pz"].shape[0]
        p_lo = min(np.sqrt(sum(g[c][i:i + CHUNK].astype(np.float64) ** 2 for c in ("px", "py", "pz"))).min()
                   for i in range(0, n, CHUNK))
        m = {"p_min": 0.9 * p_lo, "p_max": fs.P_MAX}

        def rows(a, b):
            out = np.empty((b - a, 4), dtype=np.float32)
            for i in range(a, b, CHUNK):
                sl, o = slice(i, min(i + CHUNK, b)), slice(i - a, min(i + CHUNK, b) - a)
                px, py, pz, w = (g[c][sl].astype(np.float64) for c in ("px", "py", "pz", "weight"))
                out[o, :2], out[o, 2] = fs.momentum_to_fit(pz, np.hypot(px, py), m)
                out[o, 3] = w
            return out

        a, b = args.train_rows, args.train_rows + args.val_rows
        train, val, test = rows(0, a), rows(a, b), rows(b, n)
    w = train[:, 3].astype(np.float64)
    mean = (w @ train[:, :2]) / w.sum()
    std = np.sqrt((w @ train[:, :2].astype(np.float64) ** 2) / w.sum() - mean ** 2)
    stdz = lambda A: np.column_stack([(A[:, :2] - mean) / std, A[:, 3]]).astype(np.float32)
    print(f"{args.group}: train {a:,} | val {b - a:,} | test {n - b:,} rows, p_min {m['p_min']:.4g} GeV")
    return stdz(train), torch.from_numpy(stdz(val)).to(args.device), test, m, mean, std


@torch.no_grad()
def test_nll(log_prob, test, mean, std, device, batch=1 << 20):
    """Rate-weighted NLL of the physical (pz, pt) density over the test rows."""
    tot = 0.0
    for i in range(0, len(test), batch):
        b = test[i:i + batch]
        x = torch.as_tensor((b[:, :2] - mean) / std, dtype=torch.float32, device=device)
        lp = log_prob(x).double().cpu().numpy() - np.log(std).sum() + b[:, 2]
        tot += float(b[:, 3].astype(np.float64) @ lp)
    return -tot / test[:, 3].astype(np.float64).sum()


def sampling_time(sample, device, n):
    """Best of 3 wall times [s] of sample(n), after a warm-up."""
    sync = torch.cuda.synchronize if str(device).startswith("cuda") else (lambda: None)
    with torch.no_grad():
        sample(10_000)
        sync()
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            sample(n)
            sync()
            best = min(best, time.perf_counter() - t0)
    return best


def run_gmm(K, data, args):
    train, _, test, m, mean, std = data
    t0 = time.time()
    gmm = fit_gmm(torch.from_numpy(train[:, :2]).to(args.device),
                  torch.from_numpy(train[:, 2]).to(args.device), K, args.device, seed=args.seed)
    train_time = time.time() - t0
    nll = test_nll(gmm.log_prob, test, mean, std, args.device)
    save_gmm(os.path.join(args.out_dir, f"gmm_K{K}.json"), gmm.weights, gmm.means_np, gmm.covs,
             mean, std, **m)
    times = {}
    for dev in (args.device, "cpu"):
        gmm.to(dev)
        mt, st = (torch.as_tensor(a, dtype=torch.float32, device=dev) for a in (mean, std))
        times[dev] = sampling_time(lambda n: gmm.sample(n, mt, st), dev, args.time_samples)
    return {"model": "gmm", "K": K, "n_params": gmm.n_params, "test_nll": nll, "train_time_s": train_time,
            "sample_time_gpu_s": times[args.device], "sample_time_cpu_s": times["cpu"]}


def run_flow(h, data, args):
    train, val, test, m, mean, std = data
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True           # training only
    flow = NeuralSplineFlow(dim=2, n_layers=args.flow_layers, hidden=(h, h), num_bins=12,
                            tail_bound=9.0).to(args.device)
    t0 = time.time()
    state, _, _ = ft.fit(flow, {"all": train}, val, args, args.device)
    train_time = time.time() - t0
    flow.load_state_dict(state)
    flow.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    nll = test_nll(flow.log_prob, test, mean, std, args.device)
    save_flow(os.path.join(args.out_dir, f"flow_h{h}_L{args.flow_layers}.pt"), flow, state, mean, std,
              columns=["logit_p", "log_tan_theta"], **m)
    times = {}
    for dev in (args.device, "cpu"):
        flow.to(dev)
        times[dev] = sampling_time(lambda n: flow.sample(n, mean=mean, std=std), dev, args.time_samples)
    return {"model": "flow", "hidden": h, "n_layers": args.flow_layers,
            "n_params": sum(p.numel() for p in flow.parameters()), "test_nll": nll,
            "train_time_s": train_time, "sample_time_gpu_s": times[args.device],
            "sample_time_cpu_s": times["cpu"]}


def load_results(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_result(path, key, res):
    """Merge one result into results.json under a lock (parallel runs share the file)."""
    with open(f"{path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        allres = {**load_results(path), key: res}
        with open(f"{path}.tmp", "w") as f:
            json.dump(allres, f, indent=1)
        os.replace(f"{path}.tmp", path)


def plot(results, args):
    rows = {k: sorted((r for r in results.values() if r["model"] == k), key=lambda r: r["n_params"])
            for k in ("gmm", "flow")}
    series = [("GMM, vary K", GMM_COLOR, "o", rows["gmm"], "K"),
              (f"Flow, {args.flow_layers} layers, vary width", FLOW_COLOR, "s", rows["flow"], "hidden")]
    os.makedirs(figs_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, color, marker, rs, knob in series:
        if rs:
            ax.plot([r["n_params"] for r in rs], [r["test_nll"] for r in rs], color=color,
                    marker=marker, lw=2, ms=7, label=label)
            for r in rs:
                ax.annotate(f"{knob[0]}={r[knob]}", (r["n_params"], r["test_nll"]), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color="#52514e")
    ax.set(xscale="log", xlabel="number of parameters", ylabel="test NLL of (pz, pt) [nats / muon]",
           title=f"Rate-weighted NLL of the momentum, {args.group}")
    ax.grid(alpha=0.3, lw=0.6)
    ax.legend(frameon=False)
    save_plot(fig, "gmm_vs_flow_momentum_nll.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    for ax, dev in zip(axes, ("gpu", "cpu")):
        for label, color, marker, rs, _ in series:
            if rs:
                ax.plot([r["n_params"] for r in rs], [r[f"sample_time_{dev}_s"] for r in rs],
                        color=color, marker=marker, lw=2, ms=7, label=label)
        ax.set(xscale="log", yscale="log", xlabel="number of parameters",
               title="GPU" if dev == "gpu" else f"CPU ({args.cpu_threads} threads)")
        ax.grid(alpha=0.3, lw=0.6, which="both")
    axes[0].set_ylabel(f"time to sample {args.time_samples:,} muons [s]")
    axes[0].legend(frameon=False)
    save_plot(fig, "gmm_vs_flow_momentum_sampling_time.png")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    res_path = os.path.join(args.out_dir, "results.json")
    if not args.plot_only:
        torch.set_num_threads(args.cpu_threads)
        key = lambda kind, v: f"gmm_{v}" if kind == "gmm" else f"flow_{v}_L{args.flow_layers}"
        todo = ([("gmm", K) for K in args.gmm_K] if args.only != "flow" else []) + \
               ([("flow", h) for h in args.flow_hidden] if args.only != "gmm" else [])
        todo = [(k, v) for k, v in todo if key(k, v) not in load_results(res_path)]
        if todo:
            data = load_data(args)
            for kind, v in todo:
                print(f"\n========== {kind} {v} ==========")
                res = (run_gmm if kind == "gmm" else run_flow)(v, data, args)
                print(json.dumps(res, indent=1))
                save_result(res_path, key(kind, v), res)
    plot(load_results(res_path), args)
