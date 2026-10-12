
"""Data loading, standardization and the training loop of the muon flow (train_flow.py;
compare_gmm_flow.py uses the loop too).

Every row carries its importance weight, and the loss is the rate-weighted NLL. The split
file (split_by_weight.py) is read as its two groups and each batch mixes them, --hi_frac
from above_W, with the weights rescaled so the loss still sees the natural mix.
"""
import argparse
import os
import time

import h5py
import numpy as np
import torch
from tqdm import tqdm

from gen_eval import figs_dir, plot_loss
from models import save_flow
from utils import CHUNK

BAR_INTERVAL = 2.0       # seconds between tqdm refreshes, so a redirected log stays short
HI, LO = "above_W", "below_W"
DEFAULT_SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "muons_data_split.h5")


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def make_parser(doc, out):
    """An argument parser with the data and optimization flags of a training script."""
    p = argparse.ArgumentParser(description=doc, formatter_class=ArgFormatter)
    p.add_argument("--data", default=DEFAULT_SPLIT,
                   help="split_by_weight.py output, or the pooled table (utils.DEFAULT_DATA)")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Cap on the rows read (None = all), taken from each group in "
                        "proportion, with the weights scaled so the rate is unchanged")
    p.add_argument("--hi_frac", type=float, default=0.5,
                   help="Split file only: fraction of each batch drawn from above_W")
    p.add_argument("--num_bins", type=int, default=12, help="Spline bins per dimension")
    p.add_argument("--tail_bound", type=float, default=6.0,
                   help="Spline box in standardized units: identity outside it, so the "
                        "flow can essentially not generate anything beyond +-tail_bound")
    p.add_argument("--epochs", type=int, default=150,
                   help="An epoch is one pass over above_W (split file) or all rows")
    p.add_argument("--batch_size", type=int, default=2**19, help="Rows per training step")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial Adam learning rate")
    p.add_argument("--lr_factor", type=float, default=0.5,
                   help="LR multiplier when the val NLL plateaus")
    p.add_argument("--lr_patience", type=int, default=5,
                   help="Epochs without improvement before the LR drops")
    p.add_argument("--min_lr", type=float, default=1e-6, help="Floor of the learning rate")
    p.add_argument("--patience", type=int, default=30, help="Early-stopping patience (epochs)")
    p.add_argument("--val_frac", type=float, default=0.05, help="Fraction of rows held out for validation")
    p.add_argument("--max_val_samples", type=int, default=3_000_000,
                   help="Cap on the validation set, so evaluating it every epoch is cheap")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the training log_prob (fuses the spline's many small "
                        "kernels)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    p.add_argument("--out", default=out, help="Output checkpoint")
    return p


def setup(args):
    """Check the common flags, seed, enable TF32, and return the device."""
    if not 0 < args.hi_frac < 1:
        raise SystemExit("--hi_frac must be strictly between 0 and 1")
    torch.manual_seed(args.seed)
    # TF32 for the coupling MLPs: several times faster than fp32 matmul, at a precision
    # that is irrelevant next to the noise of a stochastic fit
    torch.backends.cuda.matmul.allow_tf32 = True
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load(path, read, dim, num_samples=None):
    """The file as {group: (n, dim + 1) float32 array}: a single group 'all' for the
    pooled table, above_W and below_W for a split file. `read(g, sl)` returns the dim
    fit-space columns of rows `sl` of `g` (the h5py file or group).

    `num_samples` takes the head of each group in proportion to its size -- the file is
    a concatenation of equivalent jobs, so the head is a fair sub-sample -- and scales
    the weights so each group still carries its full rate."""
    with h5py.File(path, "r") as f:
        groups = {"all": f} if "pz" in f else {HI: f[HI], LO: f[LO]}
        sizes = {k: g["pz"].shape[0] for k, g in groups.items()}
        keep = min(1.0, num_samples / sum(sizes.values())) if num_samples else 1.0
        data = {}
        for k, g in groups.items():
            n = max(1, round(sizes[k] * keep))
            A = np.empty((n, dim + 1), dtype=np.float32)
            for i in tqdm(range(0, n, CHUNK), desc=f"reading {k}", unit="chunk",
                          mininterval=BAR_INTERVAL):
                sl = slice(i, min(i + CHUNK, n))
                A[sl, :dim] = read(g, sl)
                A[sl, dim] = g["weight"][sl]
            if n < sizes[k]:
                A[:, dim] *= sizes[k] / n
            data[k] = A
            print(f"  {k}: {n:,} of {sizes[k]:,} muons ({A.nbytes / 2**30:.1f} GB)")
    return data


def chunks(A):
    """Row slices (views) of A, CHUNK rows each."""
    return (A[i:i + CHUNK] for i in range(0, A.shape[0], CHUNK))


def standardize(data):
    """Standardize the fit-space columns of every group in place by their weighted
    mean/std over all groups together. Returns (mean, std, sum of weights)."""
    dim = next(iter(data.values())).shape[1] - 1
    wsum, s1, s2 = 0.0, np.zeros(dim), np.zeros(dim)
    for A in data.values():
        for c in chunks(A):
            x, w = c[:, :dim].astype(np.float64), c[:, dim].astype(np.float64)
            wsum += w.sum()
            s1 += w @ x
            s2 += w @ x ** 2
    mean = s1 / wsum
    std = np.sqrt(np.maximum(s2 / wsum - mean ** 2, 0.0)) + 1e-12
    for A in data.values():
        for c in chunks(A):
            c[:, :dim] -= mean.astype(np.float32)
            c[:, :dim] /= std.astype(np.float32)
    print(f"Fit space: mean {np.round(mean, 4)}, std {np.round(std, 4)}, "
          f"sum of weights {wsum:,.4g}")
    return mean, std, wsum


def report_coverage(data, bound, columns):
    """Weight fraction of each column outside the spline box +-bound, per group: data
    the flow will essentially never reproduce, so anything but tiny numbers here means
    the fit space (or --tail_bound) needs changing. `bound` is one value for all columns
    or one per column."""
    dim = len(columns)
    bound = np.broadcast_to(np.asarray(bound, dtype=float), (dim,))
    print("Weight outside the spline box (+-bound sigma; bound "
          + ", ".join(f"{name} {b:g}" for name, b in zip(columns, bound)) + "):")
    for k, A in data.items():
        out, wsum = np.zeros(dim), 0.0
        for c in chunks(A):
            w = c[:, dim].astype(np.float64)
            out += w @ (np.abs(c[:, :dim]) > bound)
            wsum += w.sum()
        print(f"  {k:>7}: " + ", ".join(f"{name} {f:.1e}" for name, f in zip(columns, out / wsum)))


def split_val(data, args, device):
    """Shuffle every group in place and hold out its head as validation, each group in
    proportion to its size, so the validation set keeps the natural mix and weights.
    Returns ({group: training rows}, validation rows as a tensor on `device`)."""
    rng = np.random.default_rng(args.seed)
    n_tot = sum(A.shape[0] for A in data.values())
    n_val = min(int(n_tot * args.val_frac), args.max_val_samples)
    train, val = {}, []
    for k, A in data.items():
        rng.shuffle(A, axis=0)
        m = round(n_val * A.shape[0] / n_tot)
        train[k] = A[m:]
        val.append(A[:m])
    val = torch.from_numpy(np.concatenate(val)).to(device)
    print("Train: " + " + ".join(f"{A.shape[0]:,} {k}" for k, A in train.items())
          + f" | Val: {val.shape[0]:,}")
    return train, val


def batches(n, size, rng):
    """Endless stream of `size`-row slices of range(n), contiguous, in a new random order
    and offset every pass. A group smaller than its batch share is resampled with
    replacement instead (it is then small enough for a gather to be cheap)."""
    if size >= n:
        while True:
            yield torch.from_numpy(rng.integers(0, n, size))
    while True:
        off = int(rng.integers(size))
        for s in off + size * rng.permutation((n - off) // size):
            yield slice(int(s), int(s) + size)


def compiled_log_prob(flow, enabled, batch_size, device):
    """flow.log_prob, through torch.compile when possible. The graph is built here on a
    dummy batch, so its cost is paid up front and a broken backend falls back to eager
    instead of failing mid-epoch. Only the callable is compiled, so the state_dict keeps
    its plain keys."""
    if not enabled:
        return flow.log_prob
    try:
        fn = torch.compile(flow.log_prob, dynamic=False)
        t0 = time.time()
        fn(torch.zeros(batch_size, flow.dim + flow.context_dim, device=device)).sum().backward()
        flow.zero_grad(set_to_none=True)
        print(f"torch.compile: graph built in {time.time() - t0:.0f}s")
        return fn
    except Exception as exc:
        print(f"torch.compile unavailable ({type(exc).__name__}: {exc}); running eager")
        return flow.log_prob


@torch.no_grad()
def eval_nll(flow, V, batch=1 << 20):
    """Weighted mean NLL of the rows of V ([x, w], on the device)."""
    total, wsum = 0.0, 0.0
    for i in range(0, V.shape[0], batch):
        x, w = V[i:i + batch, :-1], V[i:i + batch, -1]
        total += -(w * flow.log_prob(x)).sum().item()
        wsum += w.sum().item()
    return total / wsum


def fit(flow, train, val, args, device):
    """Weighted MLE with early stopping on the validation NLL. Each batch takes each
    group's batch share (all of it for a pooled file, hi_frac / 1 - hi_frac for a split
    one), and each group's weights are rescaled here by natural share / batch share.
    Returns (best state_dict, its val NLL, loss history)."""
    keys = list(train)
    n_tot = sum(A.shape[0] for A in train.values())
    share = {"all": 1.0} if keys == ["all"] else {HI: args.hi_frac, LO: 1 - args.hi_frac}
    sizes = {k: round(args.batch_size * share[k]) for k in keys[:-1]}
    sizes[keys[-1]] = args.batch_size - sum(sizes.values())

    rng = np.random.default_rng(args.seed + 1)
    streams, tensors = {}, {}
    for k, A in train.items():
        factor = A.shape[0] / (n_tot * share[k])
        if factor != 1.0:
            for c in chunks(A):
                c[:, -1] *= factor
        streams[k] = batches(A.shape[0], sizes[k], rng)
        tensors[k] = torch.from_numpy(A)
    steps = max(1, train[keys[0]].shape[0] // sizes[keys[0]])

    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    log_prob = compiled_log_prob(flow, args.compile, args.batch_size, device)

    history = {"train": [], "val": [], "best_epoch": 1}
    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for _ in tqdm(range(steps), desc=f"epoch {epoch}", unit="batch", leave=False,
                      mininterval=BAR_INTERVAL):
            b = torch.cat([tensors[k][next(streams[k])].to(device, non_blocking=True)
                           for k in keys])
            x, w = b[:, :-1].contiguous(), b[:, -1]
            loss = -(w * log_prob(x)).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            running += loss.item()

        val_nll = eval_nll(flow, val)
        sched.step(val_nll)
        history["train"].append(running / steps)
        history["val"].append(val_nll)
        if val_nll < best_val - 1e-4:
            best_val, bad, history["best_epoch"] = val_nll, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
        else:
            bad += 1
        print(f"epoch {epoch:3d} | train {history['train'][-1]:.4f} | val {val_nll:.4f} | "
              f"best {best_val:.4f} | lr {opt.param_groups[0]['lr']:.1e} | "
              f"patience {bad}/{args.patience} | {time.time() - t0:.0f}s")
        if bad >= args.patience:
            print("early stop")
            break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
    return best_state, best_val, history


def finish(flow, state, history, args, name, mean, std, **meta):
    """Plot the loss curve and save the checkpoint (models.save_flow)."""
    os.makedirs(figs_dir, exist_ok=True)
    plot_loss(history, f"{name}_loss.png")
    save_flow(args.out, flow, state, mean, std, **meta)
    print(f"Saved {args.out}")
