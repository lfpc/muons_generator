"""Train a SINGLE Neural Spline Flow on the *complete* muon state: (pz, pt, x, y, z).

Same recipe as train_flow.py -- all weight groups pooled, weighted maximum likelihood
with per-event weight = group_weight * N_true_group / n_loaded_group -- but the flow is
5D and learns the production position (x, y, z) from the data instead of it being faked
downstream (train_flow.py fits only (pz, pt) and the samples are placed at a fixed
x=y=0, z=-1). phi is still not modelled: px/py are recovered from pt with a uniform
phi, as before.

Only pz and pt are log-transformed with --log_transform (x, y, z are negative-valued, so
the log is undefined there); the flow carries that per-dimension mask in `log_dims`, so
sample() exponentiates just those two dimensions back to the physical scale.

Run (GPU picked automatically):  python train_flow_complete.py --log_transform
"""
import argparse
import os
import time

import numpy as np
import h5py
import torch

from utils import calc_pt
from models import NeuralSplineFlow
from train_flow import DEFAULT_DATA, WMIN, eval_nll, train_one
from train_flow_per_weight import plot_losses, figs_dir

COLS = ('pz', 'pt', 'x', 'y', 'z')
LOG_DIMS = (True, True, False, False, False)   # x, y, z are negative -> never logged
DIM = len(COLS)


def load_pooled(data_path, num_samples, log_transform, log_eps):
    """Pool (pz, pt, x, y, z) [in fit space] across all weight groups >= WMIN, plus a
    per-group rate weight = group_weight * N_true / n_loaded. Mirrors
    train_flow.load_pooled, only with the three position columns added.

    Returns X (N, 5) float32, gid (N,) int16, group_weight (G,), mean/std (1, 5) and
    total_weight (the physical rate mass, saved into the checkpoint).
    """
    print(f"Loading data from {data_path} ...")
    X_parts, weights_per_group = [], []
    with h5py.File(data_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            weight = float(group.attrs['weight'])
            if weight < WMIN:
                continue
            N_true = group['pz'].shape[0]
            sl = slice(None) if num_samples is None else slice(0, num_samples)
            n_loaded = N_true if num_samples is None else min(num_samples, N_true)
            # filled column by column into one preallocated array: no full-size
            # temporaries, which matters at ~5e8 rows x 5 columns
            Xg = np.empty((n_loaded, DIM), dtype=np.float32)
            Xg[:, 0] = group['pz'][sl]
            Xg[:, 1] = calc_pt(group['px'][sl], group['py'][sl])
            for j, col in enumerate(COLS[2:], start=2):
                Xg[:, j] = group[col][sl]
            if log_transform:
                Xg[:, 0] = np.log(Xg[:, 0])
                Xg[:, 1] = np.log(Xg[:, 1] + log_eps)
            w_event = weight * N_true / n_loaded
            print(f"  group weight={weight:g}: n_loaded={n_loaded:,} true_N={N_true:,} "
                  f"per-event weight={w_event:.6g}")
            X_parts.append(Xg)
            weights_per_group.append(w_event)

    # weighted mean/std from the still-separate per-group arrays, so a full-size
    # per-sample weight array is never materialized
    total_weight = float(sum(w * Xg.shape[0] for w, Xg in zip(weights_per_group, X_parts)))
    mean = sum(w * Xg.sum(axis=0) for w, Xg in zip(weights_per_group, X_parts)) / total_weight
    var = sum(w * ((Xg - mean) ** 2).sum(axis=0)
              for w, Xg in zip(weights_per_group, X_parts)) / total_weight
    std = np.sqrt(var) + 1e-12

    gid_parts = [np.full(Xg.shape[0], gid, dtype=np.int16) for gid, Xg in enumerate(X_parts)]
    X = np.concatenate(X_parts, axis=0)
    gid = np.concatenate(gid_parts, axis=0)
    del X_parts, gid_parts
    group_weight = np.asarray(weights_per_group, dtype=np.float64)
    print(f"Pooled {X.shape[0]:,} muons from {len(group_weight)} weight groups "
          f"({X.nbytes / 2**30:.1f} GB as float32).")
    return X, gid, group_weight, mean.reshape(1, -1), std.reshape(1, -1), total_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a single Neural Spline Flow on the complete muon state "
                    "(pz, pt, x, y, z), all weight groups pooled, via weighted MLE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Per-group loading cap (None = all; with a cap, the "
                             "per-event weight is scaled by N_true/n_loaded so the true "
                             "rate is preserved)")
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--num_bins", type=int, default=8)
    parser.add_argument("--tail_bound", type=float, default=6.0,
                        help="Spline support in standardized units; identity linear tails outside")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=2**19,
                        help="GPU memory is dominated by the coupling-MLP activations "
                             "kept for backward, which scale with this (5D transforms "
                             "more dims per layer than train_flow.py's 2D); halve it if "
                             "you still run out")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="Factor to multiply the LR by when val NLL plateaus")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="Epochs of no val improvement before reducing the LR (< --patience)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Floor for the LR schedule")
    parser.add_argument("--val_frac", type=float, default=0.02,
                        help="Fraction of the pooled data held out for validation "
                             "(capped by --max_val_samples)")
    parser.add_argument("--max_val_samples", type=int, default=3_000_000,
                        help="Upper cap on validation-set size")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument("--out", type=str, default="outputs/flow_model_complete.pt")
    parser.add_argument("--log_transform", action="store_true",
                        help="Fit pz, pt as log(pz), log(pt + log_eps); x, y, z stay raw.")
    parser.add_argument("--log_eps", type=float, default=1e-6,
                        help="Epsilon added to pt before taking the log; only used with "
                             "--log_transform.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, gid, group_weight, mean, std, total_weight = load_pooled(
        args.data, args.num_samples, args.log_transform, args.log_eps)

    X -= mean.astype(np.float32)                  # standardize in place -- no second
    X /= std.astype(np.float32)                   # full-size copy of the pooled array

    n = X.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = min(int(n * args.val_frac), args.max_val_samples)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    Xval, gidval = X[val_idx], gid[val_idx]
    Xtr, gidtr = X[train_idx], gid[train_idx]
    del X, gid, perm, val_idx, train_idx
    print(f"Train/val split: {Xtr.shape[0]:,} train, {Xval.shape[0]:,} val muons.")

    Xtr = torch.from_numpy(Xtr).to(device)
    gidtr = torch.from_numpy(gidtr).to(device)     # kept as int16 on device (compact)
    Xval = torch.from_numpy(Xval).to(device)
    gidval = torch.from_numpy(gidval).to(device)
    group_weight_t = torch.from_numpy(group_weight).float().to(device)

    log_eps = (0.0, args.log_eps) + (0.0,) * (DIM - 2)
    print(f"\nTraining {DIM}D flow on {' '.join(COLS)} over {Xtr.shape[0]:,} muons "
          f"({len(group_weight)} weight groups pooled)...")
    t0 = time.time()
    flow = NeuralSplineFlow(dim=DIM, n_layers=args.n_layers, hidden=tuple(args.hidden),
                            num_bins=args.num_bins, tail_bound=args.tail_bound,
                            log_transform=args.log_transform, log_eps=log_eps,
                            log_dims=LOG_DIMS).to(device)
    state, val_nll, history = train_one(flow, Xtr, gidtr, Xval, gidval, group_weight_t,
                                        args, device)
    print(f"  done in {time.time() - t0:.1f}s | best val_nll={val_nll:.4f}")

    os.makedirs(figs_dir, exist_ok=True)
    loss_path = figs_dir + 'flow_complete_loss.png'
    plot_losses(["combined"], {"combined": history}, loss_path)

    state = {k: v.cpu() for k, v in state.items()}    # save CPU tensors only
    cfg = dict(dim=DIM, n_layers=args.n_layers, hidden=list(args.hidden),
               num_bins=args.num_bins, tail_bound=args.tail_bound,
               log_transform=args.log_transform, log_eps=list(log_eps),
               log_dims=list(LOG_DIMS))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model_type": "nsf", "columns": list(COLS),
                "groups": {"combined": {"config": cfg, "state_dict": state,
                                        "mean": mean, "std": std, "weight": total_weight}}},
               args.out)
    print(f"\nSaved complete flow to {args.out}")
    print(f"Loss curve saved to {loss_path}")
