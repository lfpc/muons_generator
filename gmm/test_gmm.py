import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from train_gmm import sample_gmm, _log_gaussian_full
import gen_eval as ge
from gen_eval import (grid_points, grid_normalize, real_pzpt,
                      hist2d_density, draw_hist2d, shared_lognorm, plot_compare,
                      save_plot, report_saved_plots, report_metrics,
                      load_real, figs_dir)

parser = argparse.ArgumentParser(description="Sample, plot and score a trained GMM",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, default="outputs/gmm_model.pkl",
                    help="Model from train_gmm.py")
parser.add_argument("--data", type=str, default=ge.DEFAULT_DATA,
                    help="Real muon file for comparison (set to '' to skip)")
parser.add_argument("--num_samples", type=int, default=100_000_000,
                    help="Samples for the marginal/W1 comparison (and the 2D histogram unless --pdf)")
parser.add_argument("--data_samples", type=int, default=2_000_000,
                    help="Cap on real muons loaded for comparison")
parser.add_argument("--pdf", action="store_true",
                    help="Draw the generated 2D histograms from the analytic model PDF "
                         "instead of sampling and histogramming")
parser.add_argument("--show", action="store_true")
args = parser.parse_args()


# --------------------------------------------------------------------------- #
# GMM-specific pieces. The GMM is fit in standardized space x_s = (x-mean)/std,
# so its density in physical units is p_std((x-mean)/std) / (std_pz*std_pt).
# --------------------------------------------------------------------------- #
def gmm_phys_density(gmm, mean, std, grid):
    mean = np.asarray(mean, dtype=float).ravel()
    std = np.asarray(std, dtype=float).ravel()
    Xs = (grid - mean) / std
    log_prob = _log_gaussian_full(Xs, gmm.means_, gmm.covariances_) + np.log(gmm.weights_)
    return np.exp(logsumexp(log_prob, axis=1)) / np.prod(std)


def gmm_density_grid(gmm, mean, std):
    """Analytic 2D density grid of a single GMM (drop-in for a sampled histogram)."""
    grid, shape = grid_points()
    return grid_normalize(gmm_phys_density(gmm, mean, std, grid)).reshape(shape)


os.makedirs(figs_dir, exist_ok=True)

with open(args.model, 'rb') as f:
    print(f"Loading model from {args.model} ...")
    model = pickle.load(f)


def sample_all():
    pz, pt, _ = sample_gmm(model["gmm"], model["mean"], model["std"],
                           n_samples=args.num_samples, seed=0)
    return np.stack([pz, pt], axis=1)


# ---- combined generated 2D histogram ----
# --pdf: evaluate the GMM PDF analytically on the grid (exact, no sampling).
# otherwise: sample the model and histogram the samples the normal way.
if args.pdf:
    Hg = gmm_density_grid(model["gmm"], model["mean"], model["std"])
    gen_all = None
else:
    gen_all = sample_all()
    print(f"Generated {gen_all.shape[0]} samples for the generated histogram.")
    Hg = hist2d_density(gen_all)

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
im = draw_hist2d(ax, Hg, shared_lognorm(Hg),
                 title=f"GMM ({'analytic PDF' if args.pdf else 'sampled'})")
fig.colorbar(im, ax=ax, label='density')
save_plot(fig, 'generated_momentum.png')
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
    gen_all = sample_all()
print(f"Using {gen_all.shape[0]} samples for the marginal/W1 comparison.")

real, real_all_w = load_real(args.data, args.data_samples)
real_all = real_pzpt(real)

# real (rate-weighted) vs generated. The generated 2D panel reuses Hg (analytic if
# --pdf, else the sampled histogram of gen_all).
gen_label = 'generated (analytic PDF)' if args.pdf else f'generated (N={gen_all.shape[0]})'
w1 = plot_compare(real_all, gen_all, real_all_w,
                  'Real (rate-weighted) vs generated — GMM',
                  'compare_combined.png', gen_density=Hg, gen_label=gen_label, show=args.show)
metrics = {'pz, pt': w1}

report_metrics(metrics)
report_saved_plots()
