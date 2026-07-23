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
from train_flow import train_one, plot_losses, figs_dir     # reuse training + loss plot (sets Agg backend)
from gen_eval import (grid_points, grid_normalize, mixing_alpha, sample_mixture,
                      draw_hist2d, shared_lognorm, save_plot)
import matplotlib.pyplot as plt                              # backend already Agg (set by train_flow import)


def load_flows(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)   # CPU-saved file
    groups = ckpt["groups"]
    for w, e in groups.items():
        flow = NeuralSplineFlow.from_config(e["config"])
        flow.load_state_dict(e["state_dict"])
        e["flow"] = flow.to(device).eval()       # move to GPU (or CPU) for sampling/density
    return groups


def flow_sample_group(entry, nw, device, chunk):
    """Sample nw physical (pz, pt) points from a group's flow, in memory-safe chunks
    (a single flow.sample over millions of points can be large in intermediate memory)."""
    mean = np.asarray(entry["mean"], dtype=float).ravel()
    std = np.asarray(entry["std"], dtype=float).ravel()
    out = np.empty((nw, 2), dtype=np.float32)
    pos = 0
    with torch.no_grad():
        while pos < nw:
            m = min(chunk, nw - pos)
            out[pos:pos + m] = entry["flow"].sample(m).cpu().numpy()
            pos += m
    return out * std + mean                          # destandardize -> physical (pz, pt)


def flow_phys_density(entry, grid, device):
    """Flow density at physical (pz, pt) grid points (change of variables)."""
    mean = np.asarray(entry["mean"], dtype=float).ravel()
    std = np.asarray(entry["std"], dtype=float).ravel()
    Xs = ((grid - mean) / std).astype(np.float32)
    with torch.no_grad():
        lp = entry["flow"].log_prob(torch.from_numpy(Xs).to(device)).cpu().numpy()
    return np.exp(lp) / np.prod(std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill the per-weight flows into a single flow")
    parser.add_argument("--models", type=str, default="outputs/flow_model.pt",
                        help="Per-weight flow file from train_flow.py")
    parser.add_argument("--out", type=str, default="outputs/flow_distilled.pt")
    parser.add_argument("--total_samples", type=int, default=5_000_000,
                        help="Samples drawn from the combined mixture to fit the student")
    parser.add_argument("--sample_chunk", type=int, default=1_000_000,
                        help="Teacher-sampling chunk size (memory)")
    # student-flow hyperparameters (same knobs as train_flow.py; a bit deeper by
    # default since the student must model the full combined mixture)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--num_bins", type=int, default=12)
    parser.add_argument("--tail_bound", type=float, default=14.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
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

    # ---- 1) sample the combined rate-weighted mixture (teacher) ----
    print(f"\nSampling {args.total_samples} points from the combined mixture...")
    t0 = time.time()
    X = sample_mixture(groups, args.total_samples,
                       lambda e, nw: flow_sample_group(e, nw, device, args.sample_chunk),
                       seed=args.seed)
    print(f"  sampled {X.shape[0]} points in {time.time() - t0:.1f}s")

    # ---- 2) fit a single student flow by max-likelihood on the mixture samples ----
    print("\nTraining the distilled (student) flow...")
    t0 = time.time()
    state, mean, std, val_nll, history = train_one(X.astype(np.float64), args, device)
    print(f"  done in {time.time() - t0:.1f}s | best val_nll={val_nll:.4f}")

    os.makedirs(figs_dir, exist_ok=True)
    plot_losses(["distilled"], {"distilled": history}, figs_dir + "flow_distill_loss.png")
    print(f"  loss curve -> {figs_dir}flow_distill_loss.png")

    # ---- 3) save (distilled format: a single-group file test_flow.py detects) ----
    student_cfg = dict(dim=2, n_layers=args.n_layers, hidden=list(args.hidden),
                       num_bins=args.num_bins, tail_bound=args.tail_bound)
    state = {k: v.cpu() for k, v in state.items()}      # save CPU tensors only
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model_type": "nsf", "distilled": True,
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
