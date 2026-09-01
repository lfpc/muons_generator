"""Train a SINGLE Neural Spline Flow on the full dataset -- all weight groups pooled
together -- via weighted maximum likelihood, instead of one flow per weight group
(see train_flow_per_weight.py, which trains per-group flows recombined into a
rate-weighted mixture only at test time).

Each event's log-likelihood is weighted by weight_group * N_true_group / n_loaded_group
(this reduces to just the raw group `weight` attribute when the whole group is loaded,
i.e. no --num_samples cap). In expectation, minimizing the resulting weighted NLL is
equivalent MLE for exactly the same rate-weighted combined density that
train_flow_per_weight.py's mixture reproduces (see gen_eval.combined_real, which applies
the identical per-event weight for plotting/W1). The same weights are used to compute the
standardization mean/std, so a group that was deliberately oversampled to cover a rare
tail (large N, tiny weight) doesn't skew the fit space away from the physical
distribution.

The pooled dataset can be hundreds of millions of rows, so: the loader reads only
px/py/pz per group (skipping position/pdg, unused here), standardizes in place, group
ids are kept as a compact int16 array -- per-sample weights are only ever materialized
for one batch at a time, never for the full dataset -- and each epoch iterates
contiguous (already-shuffled) slices in a random order rather than building a full-size
index permutation.

Run (GPU picked automatically):  python train_flow.py
"""
import argparse
import os
import time

import numpy as np
import h5py
import torch

from utils import calc_pt
from models import NeuralSplineFlow
from train_flow_per_weight import plot_losses, figs_dir

DEFAULT_DATA = '/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample_w.h5'
WMIN = 0.0001   # skip the spurious ~zero-weight group, as in train_flow_per_weight.py


def load_pooled(data_path, num_samples, log_transform, log_eps):
    """Pool (pz, pt) [in fit space, i.e. log-transformed if requested] across all weight
    groups >= WMIN, plus a per-group rate weight = group_weight * N_true / n_loaded.

    Returns:
        X: (N, 2) float32, pooled and in fit space
        gid: (N,) int16, index into group_weight for each row
        group_weight: (G,) float64
        mean, std: (1, 2) float64, weighted by group_weight
        total_weight: sum(weight_g * N_true_g) over included groups -- the physical
            rate mass, saved into the checkpoint (same role as train_flow_per_weight's
            per-group "weight" field).
    """
    print(f"Loading data from {data_path} ...")
    X_parts, weights_per_group = [], []
    with h5py.File(data_path, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            weight = float(group.attrs['weight'])
            if weight < WMIN:
                continue
            N_true = group['px'].shape[0]
            sl = slice(None) if num_samples is None else slice(0, num_samples)
            px = group['px'][sl]
            py = group['py'][sl]
            pz = group['pz'][sl].astype(np.float32)
            n_loaded = pz.shape[0]
            pt = calc_pt(px, py).astype(np.float32)
            del px, py
            if log_transform:
                Xg = np.stack([np.log(pz), np.log(pt + log_eps)], axis=1).astype(np.float32)
            else:
                Xg = np.stack([pz, pt], axis=1)
            w_event = weight * N_true / n_loaded
            print(f"  group weight={weight:g}: n_loaded={n_loaded:,} true_N={N_true:,} "
                  f"per-event weight={w_event:.6g}")
            X_parts.append(Xg)
            weights_per_group.append(w_event)

    # weighted mean/std computed from the still-separate per-group arrays, so this
    # never materializes a full-size per-sample weight array
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


def eval_nll(flow, X, gid, group_weight, batch=65536):
    """Weighted mean NLL over X, in batches (no grad). gid/group_weight give the
    per-sample weight one batch at a time, so it's never materialized in full."""
    flow.eval()
    total, wsum = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = X[i:i + batch]
            wb = group_weight[gid[i:i + batch].long()]
            total += (-wb * flow.log_prob(xb)).sum().item()
            wsum += wb.sum().item()
    return total / wsum


def train_one(flow, Xtr, gidtr, Xval, gidval, group_weight, args, device):
    """Fit `flow` by weighted maximum likelihood, with early stopping.
    Returns (best_state_dict_on_cpu, best_val_nll, history)."""
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)

    history = {"train": [], "val": [], "best_epoch": 1}
    best_val, best_state, bad = float("inf"), None, 0
    n, bs = Xtr.shape[0], args.batch_size
    for epoch in range(1, args.epochs + 1):
        flow.train()
        # The rows were globally shuffled once when the train/val split was made, so an
        # epoch just walks contiguous slices in a random order, with a random offset each
        # epoch so the batch boundaries move. A full torch.randperm(n) would be far more
        # expensive: on CUDA it is sort-based and needs tens of GB of scratch at n~5e8
        # (and the slices here are views, so there is no per-batch gather either).
        cuts = list(range(int(torch.randint(bs, (1,)).item()), n, bs))
        spans = [(a, b) for a, b in zip([0] + cuts, cuts + [n]) if b > a]
        epoch_loss, n_batches = 0.0, 0
        for j in torch.randperm(len(spans)).tolist():
            a, b = spans[j]
            xb = Xtr[a:b]
            wb = group_weight[gidtr[a:b].long()]      # per-batch weight lookup only
            loss = -(wb * flow.log_prob(xb)).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_nll = epoch_loss / max(n_batches, 1)
        val = eval_nll(flow, Xval, gidval, group_weight)
        scheduler.step(val)                      # decay LR when val NLL plateaus
        lr_now = opt.param_groups[0]["lr"]
        history["train"].append(train_nll)
        history["val"].append(val)
        if val < best_val - 1e-4:
            best_val, bad = val, 0
            history["best_epoch"] = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % 10 == 0 or bad >= args.patience:
            print(f"  epoch {epoch:3d} | train_nll={train_nll:.4f} | val_nll={val:.4f} | "
                  f"best={best_val:.4f} | lr={lr_now:.2e} | patience {bad}/{args.patience}")
        if bad >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is None:                       # never improved (e.g. epochs=0)
        best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
    return best_state, best_val, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a single Neural Spline Flow on all weight groups pooled "
                    "together, via weighted MLE")
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
    parser.add_argument("--batch_size", type=int, default=2**20)
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
                        help="Upper cap on validation-set size, so per-epoch eval stays "
                             "cheap regardless of how large the pooled dataset is")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument("--out", type=str, default="outputs/flow_model.pt")
    parser.add_argument("--log_transform", action="store_true",
                        help="Fit the flow on log(pz), log(pt + log_eps) instead of raw (pz, pt).")
    parser.add_argument("--log_eps", type=float, default=1e-6,
                        help="Epsilon added to pt before taking the log (pt gets too close to 0 "
                             "otherwise); only used with --log_transform.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, gid, group_weight, mean, std, total_weight = load_pooled(
        args.data, args.num_samples, args.log_transform, args.log_eps)

    X -= mean.astype(np.float32)                 # standardize in place -- no second
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

    print(f"\nTraining combined flow on {Xtr.shape[0]:,} muons "
          f"({len(group_weight)} weight groups pooled)...")
    t0 = time.time()
    flow = NeuralSplineFlow(dim=2, n_layers=args.n_layers, hidden=tuple(args.hidden),
                            num_bins=args.num_bins, tail_bound=args.tail_bound,
                            log_transform=args.log_transform,
                            log_eps=(0.0, args.log_eps)).to(device)
    state, val_nll, history = train_one(flow, Xtr, gidtr, Xval, gidval, group_weight_t,
                                        args, device)
    print(f"  done in {time.time() - t0:.1f}s | best val_nll={val_nll:.4f}")

    os.makedirs(figs_dir, exist_ok=True)
    loss_path = figs_dir + 'flow_loss.png'
    plot_losses(["combined"], {"combined": history}, loss_path)
    print(f"  loss curve -> {loss_path}")

    state = {k: v.cpu() for k, v in state.items()}    # save CPU tensors only
    cfg = dict(dim=2, n_layers=args.n_layers, hidden=list(args.hidden),
              num_bins=args.num_bins, tail_bound=args.tail_bound,
              log_transform=args.log_transform, log_eps=[0.0, args.log_eps])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model_type": "nsf",
               "groups": {"combined": {"config": cfg, "state_dict": state,
                                       "mean": mean, "std": std, "weight": total_weight}}},
              args.out)
    print(f"\nSaved combined flow to {args.out}")
    print(f"Loss curve saved to {loss_path}")
