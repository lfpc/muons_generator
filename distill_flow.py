"""Distill the per-weight Neural Spline Flows into a single flow.

Analogue of distill_gmm.py for flows: sample the rate-weighted mixture of the
per-weight flows (the "teacher") and fit ONE new flow (the "student") by
maximum likelihood on those samples. The result is a single network reproducing
the full combined distribution -- cheaper to sample/evaluate than the per-weight
ensemble (1 network instead of N) -- saved in a format test_flow.py loads
directly as a distilled model.

Run (GPU picked automatically):  python distill_flow.py
"""
import argparse
import os
import time

import numpy as np
import torch

from models import NeuralSplineFlow
from train_flow_per_weight import plot_losses, figs_dir, eval_nll   # reuse loss plot + val NLL
from gen_eval import (grid_points, grid_normalize, mixing_alpha, sample_mixture,
                      draw_hist2d, shared_lognorm, save_plot)
import matplotlib.pyplot as plt                  # backend already Agg (set by train_flow_per_weight import)


def load_flows(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)   # CPU-saved file
    groups = ckpt["groups"]
    for w, e in groups.items():
        flow = NeuralSplineFlow.from_config(e["config"])
        flow.load_state_dict(e["state_dict"])
        e["flow"] = flow.to(device).eval()       # move to GPU (or CPU) for sampling/density
    return groups


def flow_sample_group(entry, nw, device, chunk):
    """Sample nw physical (pz, pt) points from a group's flow, in memory-safe chunks.
    sample() destandardizes internally, and if the flow was fit with --log_transform
    it also (in eval mode) exponentiates back to the physical scale."""
    mean = np.asarray(entry["mean"], dtype=float).ravel()
    std = np.asarray(entry["std"], dtype=float).ravel()
    out = np.empty((nw, 2), dtype=np.float32)
    pos = 0
    with torch.no_grad():
        while pos < nw:
            m = min(chunk, nw - pos)
            out[pos:pos + m] = entry["flow"].sample(m, mean=mean, std=std).cpu().numpy()
            pos += m
    return out


def flow_phys_density(entry, grid, device):
    """Flow density at physical (pz, pt) grid points, including the log-transform
    change-of-variables Jacobian when the flow was fit on log values."""
    mean = np.asarray(entry["mean"], dtype=float).ravel()
    std = np.asarray(entry["std"], dtype=float).ravel()
    flow = entry["flow"]
    if flow.log_transform:
        eps = np.asarray(flow.log_eps, dtype=float).ravel()
        pz, pt = grid[:, 0], grid[:, 1]
        x = np.stack([np.log(pz), np.log(pt + eps[1])], axis=1)
        jac = 1.0 / (pz * (pt + eps[1]))
    else:
        x = grid
        jac = 1.0
    Xs = ((x - mean) / std).astype(np.float32)
    with torch.no_grad():
        lp = flow.log_prob(torch.from_numpy(Xs).to(device)).cpu().numpy()
    return np.exp(lp) / np.prod(std) * jac


def teacher_fit_sample(groups, n, device, chunk, log_transform, log_eps, seed):
    """Draw n teacher muons from the combined mixture and map them to the flow's fit
    space (log-transformed if requested). Returns an (n, 2) float64 array."""
    X = sample_mixture(groups, n,
                       lambda e, nw: flow_sample_group(e, nw, device, chunk),
                       seed=seed).astype(np.float64)
    if log_transform:
        X = np.stack([np.log(X[:, 0]), np.log(X[:, 1] + log_eps)], axis=1)
    return X


def train_student(groups, args, device):
    """Fit the student flow by streaming fresh teacher samples each epoch, so the
    total data seen (samples_per_epoch * epochs) is large but only one chunk is ever
    held in RAM. Standardization stats and the validation set are drawn once and fixed."""
    lt, eps = args.log_transform, args.log_eps

    stats = teacher_fit_sample(groups, args.stats_samples, device, args.sample_chunk, lt, eps, args.seed)
    mean = stats.mean(axis=0, keepdims=True)
    std = stats.std(axis=0, keepdims=True) + 1e-12
    del stats
    Xval = teacher_fit_sample(groups, args.val_samples, device, args.sample_chunk, lt, eps, args.seed + 1)
    Xval = torch.from_numpy(((Xval - mean) / std).astype(np.float32)).to(device)

    flow = NeuralSplineFlow(dim=2, n_layers=args.n_layers, hidden=tuple(args.hidden),
                            num_bins=args.num_bins, tail_bound=args.tail_bound,
                            log_transform=lt, log_eps=(0.0, eps)).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)

    history = {"train": [], "val": [], "best_epoch": 1}
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        # fresh chunk of teacher samples this epoch (this is what bounds RAM)
        Xc = teacher_fit_sample(groups, args.samples_per_epoch, device, args.sample_chunk,
                                lt, eps, args.seed + 100 + epoch)
        Xc = torch.from_numpy(((Xc - mean) / std).astype(np.float32)).to(device)
        flow.train()
        perm = torch.randperm(Xc.shape[0], device=device)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, Xc.shape[0], args.batch_size):
            xb = Xc[perm[i:i + args.batch_size]]
            loss = -flow.log_prob(xb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        del Xc                                       # free the chunk before drawing the next

        train_nll = epoch_loss / max(n_batches, 1)
        val = eval_nll(flow, Xval)
        scheduler.step(val)
        lr_now = opt.param_groups[0]["lr"]
        history["train"].append(train_nll)
        history["val"].append(val)
        if val < best_val - 1e-4:
            best_val, bad = val, 0
            history["best_epoch"] = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0 or bad >= args.patience:
            seen = epoch * args.samples_per_epoch / 1e6
            print(f"  epoch {epoch:3d} | train_nll={train_nll:.4f} | val_nll={val:.4f} | "
                  f"best={best_val:.4f} | lr={lr_now:.2e} | seen={seen:.0f}M | patience {bad}/{args.patience}")
        if bad >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in flow.state_dict().items()}
    return best_state, mean, std, best_val, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill the per-weight flows into a single flow")
    parser.add_argument("--models", type=str, default="outputs/flow_model.pt",
                        help="Per-weight flow file from train_flow.py")
    parser.add_argument("--out", type=str, default="outputs/flow_distilled.pt")
    parser.add_argument("--samples_per_epoch", type=int, default=5_000_000,
                        help="Fresh teacher muons drawn each epoch; total seen = "
                             "samples_per_epoch * epochs, streamed one chunk at a time (bounds RAM)")
    parser.add_argument("--stats_samples", type=int, default=2_000_000,
                        help="One-off draw used to fix the standardization mean/std")
    parser.add_argument("--val_samples", type=int, default=1_000_000,
                        help="One-off held-out draw used for validation / early stopping")
    parser.add_argument("--sample_chunk", type=int, default=1_000_000,
                        help="Teacher-sampling chunk size within a single draw (memory)")
    # student-flow hyperparameters (same knobs as train_flow.py; a bit deeper by
    # default since the student must model the full combined mixture)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--num_bins", type=int, default=12)
    parser.add_argument("--tail_bound", type=float, default=6.0,
                        help="Spline support in the fit space (small ~6 for --log_transform; "
                             "use ~14 for the raw pz/pt space)")
    parser.add_argument("--log_transform", action="store_true",
                        help="Fit the student on log(pz), log(pt + log_eps); recommended when the "
                             "per-weight flows were trained with --log_transform.")
    parser.add_argument("--log_eps", type=float, default=1e-6,
                        help="Epsilon added to pt before the log; only used with --log_transform.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if omitted)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    groups = load_flows(args.models, device)
    total_weight = float(sum(e["weight"] for e in groups.values()))
    alpha = mixing_alpha(groups)
    print(f"Loaded {len(groups)} per-weight flows. Mixing alpha: "
          f"{[round(float(a), 4) for a in alpha]}")

    # ---- fit the student by STREAMING fresh teacher samples each epoch ----
    # The full dataset is never materialized: each epoch draws its own fresh chunk,
    # so RAM is bounded to one chunk while the student sees samples_per_epoch * epochs
    # distinct muons over training.
    total_seen = args.epochs * args.samples_per_epoch / 1e6
    print(f"\nTraining the distilled (student) flow: up to {args.epochs} epochs x "
          f"{args.samples_per_epoch} fresh teacher muons = {total_seen:.0f}M samples (streamed)...")
    t0 = time.time()
    state, mean, std, val_nll, history = train_student(groups, args, device)
    print(f"  done in {time.time() - t0:.1f}s | best val_nll={val_nll:.4f}")

    os.makedirs(figs_dir, exist_ok=True)
    plot_losses(["distilled"], {"distilled": history}, figs_dir + "flow_distill_loss.png")
    print(f"  loss curve -> {figs_dir}flow_distill_loss.png")

    # ---- 3) save (single-group file; test_flow.py treats a lone group as a combined
    # flow, so no explicit "distilled" flag is needed) ----
    student_cfg = dict(dim=2, n_layers=args.n_layers, hidden=list(args.hidden),
                       num_bins=args.num_bins, tail_bound=args.tail_bound,
                       log_transform=args.log_transform, log_eps=[0.0, args.log_eps])
    state = {k: v.cpu() for k, v in state.items()}      # save CPU tensors only
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model_type": "nsf",
                "groups": {"combined": {"config": student_cfg, "state_dict": state,
                                        "mean": mean, "std": std, "weight": total_weight}}},
               args.out)
    print(f"Saved distilled flow to {args.out}")

    # ---- 4) sanity: teacher (mixture) vs student density on the shared grid ----
    grid, shape = grid_points()
    teacher = grid_normalize(sum(a * flow_phys_density(e, grid, device)
                                 for a, e in zip(alpha, groups.values()))).reshape(shape)
    student = NeuralSplineFlow.from_config(student_cfg)
    student.load_state_dict(state)
    student_entry = {"flow": student.to(device).eval(), "mean": mean, "std": std}
    student_dens = grid_normalize(flow_phys_density(student_entry, grid, device)).reshape(shape)

    norm = shared_lognorm(teacher, student_dens)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    draw_hist2d(axes[0], teacher, norm, title="teacher (per-weight mixture)")
    im = draw_hist2d(axes[1], student_dens, norm, title="student (distilled)")
    fig.colorbar(im, ax=list(axes), label="density")
    save_plot(fig, "flow_distill_check.png")
    plt.close(fig)
