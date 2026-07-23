"""Train one Neural Spline Flow per importance-weight group.

Mirrors train_gmm.py's per-weight framework: each group is standardized, a flow is
fit by maximum likelihood, and the result is saved with its mean/std and a mixing
`weight = importance_weight * N_true` so the groups recombine into the full-rate
distribution (see test_flow.py). The spurious weight~0 group is skipped.

Run (GPU picked automatically):  python train_flow.py
"""
import argparse
import os
import time

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use('Agg')                # headless-safe: training usually runs on a server
import matplotlib.pyplot as plt

from utils import get_real_muons_per_weight, calc_pt, train_val_split
from models import NeuralSplineFlow

DEFAULT_DATA = '/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample_w.h5'
figs_dir = 'outputs/figs/'


def plot_losses(weights, histories, path):
    """Grid of train/val NLL curves, one panel per weight group. Re-drawn after
    each group finishes so the figure is available while training is still running."""
    n = len(weights)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for i, w in enumerate(weights):
        ax = axes[i // ncols][i % ncols]
        label = f"w={w:.2f}" if isinstance(w, (int, float)) else str(w)
        h = histories.get(w)
        if not h or not h["val"]:
            ax.set_title(f"{label} (pending)", fontsize=10)
            ax.axis('off')
            continue
        epochs = range(1, len(h["val"]) + 1)
        ax.plot(epochs, h["train"], label='train', color='C0')
        ax.plot(epochs, h["val"], label='val', color='C1')
        ax.axvline(h["best_epoch"], color='k', ls='--', lw=0.8, label='best')
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('epoch')
        ax.set_ylabel('NLL')
        ax.legend(fontsize=8)
    for j in range(n, nrows * ncols):        # hide unused panels
        axes[j // ncols][j % ncols].axis('off')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def eval_nll(flow, X, batch=65536):
    """Mean negative log-likelihood over X, in batches (no grad)."""
    flow.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = X[i:i + batch]
            total += (-flow.log_prob(xb).sum()).item()
            n += xb.shape[0]
    return total / n


def train_one(X, args, device):
    """Standardize X = [pz, pt], fit a flow by max-likelihood with early stopping.
    Returns (best_state_dict_on_cpu, mean(1,2), std(1,2), best_val_nll, history)
    where history = {"train": [...], "val": [...], "best_epoch": int}."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-12
    Xs = ((X - mean) / std).astype(np.float32)
    Xtr, Xval = train_val_split(Xs, val_frac=args.val_frac, seed=args.seed)
    Xtr = torch.from_numpy(Xtr).to(device)
    Xval = torch.from_numpy(Xval).to(device)

    flow = NeuralSplineFlow(dim=2, n_layers=args.n_layers, hidden=tuple(args.hidden),
                            num_bins=args.num_bins, tail_bound=args.tail_bound).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)

    history = {"train": [], "val": [], "best_epoch": 1}
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        flow.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, Xtr.shape[0], args.batch_size):
            xb = Xtr[perm[i:i + args.batch_size]]
            loss = -flow.log_prob(xb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_nll = epoch_loss / max(n_batches, 1)
        val = eval_nll(flow, Xval)
        history["train"].append(train_nll)
        history["val"].append(val)
        if val < best_val - 1e-4:
            best_val, bad = val, 0
            history["best_epoch"] = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % 10 == 0 or bad >= args.patience:
            print(f"    epoch {epoch:3d} | train_nll={train_nll:.4f} | val_nll={val:.4f} | "
                  f"best={best_val:.4f} | patience {bad}/{args.patience}")
        if bad >= args.patience:
            print(f"    early stop at epoch {epoch}")
            break

    if best_state is None:                       # never improved (e.g. epochs=0)
        best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
    return best_state, mean, std, best_val, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train per-weight Neural Spline Flows on muon data")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--num_samples", type=int, default=2_000_000,
                        help="Per-group training-sample cap (None-like large = all)")
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--num_bins", type=int, default=12)
    parser.add_argument("--tail_bound", type=float, default=14.0,
                        help="Spline support in standardized units; identity linear tails outside")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument("--out", type=str, default="outputs/flow_model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = get_real_muons_per_weight(args.data, num_samples=args.num_samples)
    # true (uncapped) per-group counts, so weight*N is the correct rate even if capped
    with h5py.File(args.data, 'r') as f:
        true_n = {float(f[g].attrs['weight']): f[g]['px'].shape[0] for g in f}

    os.makedirs(figs_dir, exist_ok=True)
    weights = [w for w in data if w >= 0.0001]    # skip the spurious ~zero-weight group
    loss_path = figs_dir + 'flow_loss.png'
    models, histories = {}, {}
    for weight in weights:
        arr = data[weight]
        n_load = arr.shape[0]
        N_true = true_n[min(true_n, key=lambda k: abs(k - weight))]
        print(f"\n========== Flow for weight {weight} (train N={n_load}, true N={N_true}) ==========")
        pt = calc_pt(arr[:, 0], arr[:, 1])
        X = np.stack([arr[:, 2], pt], axis=1).astype(np.float64)   # [pz, pt]

        t0 = time.time()
        state, mean, std, val_nll, history = train_one(X, args, device)
        print(f"  done in {time.time() - t0:.1f}s | best val_nll={val_nll:.4f}")

        histories[weight] = history
        plot_losses(weights, histories, loss_path)   # update the loss figure as we go
        print(f"  loss curves -> {loss_path}")

        models[weight] = {
            "config": dict(dim=2, n_layers=args.n_layers, hidden=list(args.hidden),
                           num_bins=args.num_bins, tail_bound=args.tail_bound),
            "state_dict": state,
            "mean": mean,
            "std": std,
            # weight*N_true: final mixing weight to rebuild the full-data distribution
            "weight": float(weight) * N_true,
        }

    # Make sure every saved tensor lives on CPU, so the checkpoint loads without
    # allocating on (or pinning to) a GPU that may be busy or absent.
    for e in models.values():
        e["state_dict"] = {k: v.cpu() for k, v in e["state_dict"].items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model_type": "nsf", "groups": models}, args.out)
    print(f"\nSaved {len(models)} per-weight flows to {args.out}")
    print(f"Loss curves saved to {loss_path}")
