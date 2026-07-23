"""Sample, plot and score the per-weight Neural Spline Flows from train_flow.py.

Mirrors test_gmm.py exactly (same plots + normalized-W1 metric via gen_eval); only
the density/sampling are flow-specific. Like the GMM, the flow is fit in
standardized space, and it gives an EXACT log_prob, so the 2D densities are drawn
analytically on the grid and sampling is used only for the marginals / W1 metric.
"""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from models import NeuralSplineFlow
import gen_eval as ge
from gen_eval import (grid_points, grid_normalize, mixing_alpha, sample_mixture,
                      hist2d_density, draw_hist2d, shared_lognorm, plot_compare, plot_metric,
                      save_plot, report_saved_plots, report_metrics,
                      load_real, real_group, combined_real, figs_dir)

parser = argparse.ArgumentParser(description="Sample, plot and score per-weight Neural Spline Flows")
parser.add_argument("--model", type=str, default="outputs/flow_model.pt",
                    help="Per-weight flow file from train_flow.py")
parser.add_argument("--data", type=str, default=ge.DEFAULT_DATA,
                    help="Real muon file for comparison (set to '' to skip)")
parser.add_argument("--num_samples", type=int, default=10_000_000,
                    help="Samples for the marginal/W1 comparison (and the 2D histogram unless --pdf)")
parser.add_argument("--data_samples", type=int, default=2_000_000,
                    help="Per-group cap on real muons loaded for comparison")
parser.add_argument("--pdf", action="store_true",
                    help="Draw the generated 2D histograms from the analytic flow PDF "
                         "instead of sampling and histogramming")
parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if omitted)")
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Flow-specific pieces. The flow is fit in standardized space x_s=(x-mean)/std,
# so its physical density is p_std((x-mean)/std) / (std_pz*std_pt), and physical
# samples are s*std + mean.
# --------------------------------------------------------------------------- #
def flow_phys_density(entry, grid):
    mean = np.asarray(entry["mean"], dtype=float).ravel()
    std = np.asarray(entry["std"], dtype=float).ravel()
    Xs = ((grid - mean) / std).astype(np.float32)
    with torch.no_grad():
        lp = entry["flow"].log_prob(torch.from_numpy(Xs).to(device)).cpu().numpy()
    return np.exp(lp) / np.prod(std)


def flow_density_grid(entry):
    grid, shape = grid_points()
    return grid_normalize(flow_phys_density(entry, grid)).reshape(shape)


def model_density_grid(groups):
    """Analytic 2D density of the full model: rate-weighted mixture sum_w alpha_w p_w."""
    grid, shape = grid_points()
    alpha = mixing_alpha(groups)
    dens = sum(a * flow_phys_density(e, grid) for a, e in zip(alpha, groups.values()))
    return grid_normalize(dens).reshape(shape)


def flow_sample_group(entry, nw):
    mean = np.asarray(entry["mean"], dtype=float).ravel()
    std = np.asarray(entry["std"], dtype=float).ravel()
    with torch.no_grad():
        s = entry["flow"].sample(nw).cpu().numpy()
    return s * std + mean                          # destandardize -> physical (pz, pt)


os.makedirs(figs_dir, exist_ok=True)

print(f"Loading flows from {args.model} (device={device}) ...")
ckpt = torch.load(args.model, map_location='cpu', weights_only=False)   # CPU-saved file
distilled = ckpt.get("distilled", False)     # single flow from distill_flow.py
groups = ckpt["groups"]
for w, e in groups.items():
    flow = NeuralSplineFlow.from_config(e["config"])
    flow.load_state_dict(e["state_dict"])
    e["flow"] = flow.to(device).eval()       # move to GPU (or CPU) for sampling/density
print(f"Loaded {len(groups)} flow(s){' (distilled)' if distilled else ''} on {device}.")

torch.manual_seed(0)


def sample_combined_all():
    return sample_mixture(groups, args.num_samples, flow_sample_group, seed=0)


# ---- combined generated 2D histogram ----
# --pdf: evaluate the flow PDF analytically on the grid (exact, no sampling).
# otherwise: sample the flows and histogram the samples the normal way.
if args.pdf:
    Hg = model_density_grid(groups)
    gen_all = None
else:
    gen_all = sample_combined_all()
    print(f"Generated {gen_all.shape[0]} samples for the generated histogram.")
    Hg = hist2d_density(gen_all)

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
base_title = 'Distilled NSF' if distilled else 'Combined per-weight NSF'
im = draw_hist2d(ax, Hg, shared_lognorm(Hg),
                 title=f"{base_title} ({'analytic PDF' if args.pdf else 'sampled'})")
fig.colorbar(im, ax=ax, label='density')
save_plot(fig, 'flow_generated_momentum.png')
if args.show:
    plt.show()

# ---- comparison against real data ----
if not args.data:
    print("No --data given; skipping real-vs-generated comparison and metric plot.")
    report_saved_plots()
    raise SystemExit

# Samples are needed for the 1D marginals and the W1 metric (and, without --pdf,
# already drawn above for the 2D histogram).
if gen_all is None:
    gen_all = sample_combined_all()
print(f"Using {gen_all.shape[0]} samples for the marginal/W1 comparison.")

real, real_keys, true_n = load_real(args.data, args.data_samples)
real_all, real_all_w = combined_real(real, true_n)

# combined comparison: real (weight*N) vs generated. The generated 2D panel reuses
# Hg (analytic if --pdf, else the sampled histogram of gen_all).
gen_label = 'generated (analytic PDF)' if args.pdf else f'generated (N={gen_all.shape[0]})'
w1 = plot_compare(real_all, gen_all, real_all_w,
                  'Real (weighted by weight×N) vs generated — combined (NSF)',
                  'flow_compare_combined.png', gen_density=Hg, gen_label=gen_label, show=args.show)
metrics = {'combined': w1}

# ---- per-weight breakdown (skipped for a distilled single flow) ----
if not distilled:
    for w, e in groups.items():
        if w < 0.0001:
            continue
        r = real_group(real, real_keys, w)
        g = flow_sample_group(e, r.shape[0])
        w_str = f"{w:.2f}".replace(".", "_")
        gen_density = flow_density_grid(e) if args.pdf else None
        metrics[w] = plot_compare(r, g, None, f'weight {w:.2f} (NSF)', f'flow_compare_weight_{w_str}.png',
                                  gen_density=gen_density, show=args.show)

plot_metric(metrics, fname='flow_metric_w1.png',
            title='Neural Spline Flow fit quality per weight', show=args.show)
report_metrics(metrics)
report_saved_plots()
