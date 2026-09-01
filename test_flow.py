"""Sample, plot and score per-weight NSFs (train_flow_per_weight.py), a distilled NSF
(distill_flow.py), or the combined-training NSF (train_flow.py) -- all share the same
single-flow-per-group checkpoint format, so this script doesn't need to know which one
produced the file. Whether a per-weight breakdown makes sense is decided purely by
counting groups: more than one means a per-weight file, otherwise it's a single
combined flow (distilled or weighted-MLE-trained, doesn't matter which).

Mirrors test_gmm.py exactly (same plots + normalized-W1 metric via gen_eval); only
the density/sampling are flow-specific. Like the GMM, the flow is fit in
standardized space, and it gives an EXACT log_prob, so the 2D densities are drawn
analytically on the grid and sampling is used only for the marginals / W1 metric.
"""
import argparse
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from models import NeuralSplineFlow
import gen_eval as ge
from gen_eval import (grid_points, grid_normalize, mixing_alpha, sample_mixture,
                      hist2d_density, hist2d_counts, draw_hist2d, shared_lognorm,
                      plot_compare, plot_metric, save_plot, report_saved_plots, report_metrics,
                      load_real, real_group, combined_real, figs_dir)

parser = argparse.ArgumentParser(description="Sample, plot and score per-weight Neural Spline Flows")
parser.add_argument("--model", type=str, default="outputs/flow_model.pt",
                    help="Per-weight flow file from train_flow.py")
parser.add_argument("--data", type=str, default=ge.DEFAULT_DATA,
                    help="Real muon file for comparison (set to '' to skip)")
parser.add_argument("--num_samples", type=int, default=None,
                    help="Samples for the marginal/W1 comparison (and the 2D histogram unless "
                         "--pdf); if omitted, matches the number of real muons loaded (--data_samples)")
parser.add_argument("--data_samples", type=int, default=None,
                    help="Per-group cap on real muons loaded for comparison (None = all)")
parser.add_argument("--pdf", action="store_true",
                    help="Draw the generated 2D histograms from the analytic flow PDF "
                         "instead of sampling and histogramming")
parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if omitted)")
parser.add_argument("--show", action="store_true")
parser.add_argument("--log_transform", action="store_true",
                    help="Force log(pz), log(pt + log_eps) semantics even if a loaded group's "
                         "checkpoint config doesn't already say so (normally auto-detected per "
                         "group from train_flow.py's --log_transform).")
parser.add_argument("--log_eps", type=float, default=1e-6,
                    help="Epsilon used with --log_transform when overriding a group's config.")
parser.add_argument("--generated_weighted_sample", action="store_true",
                    help="Add a third 2D panel to the combined comparison: generated "
                         "samples drawn at the real data's weighted count (real_w.sum(), "
                         "which reflects the true physical rate and can be tens of "
                         "billions of muons). Streamed in chunks so it never "
                         "materializes all samples at once -- still off by default "
                         "since it can be very heavy.")
parser.add_argument("--weighted_sample_chunk", type=int, default=10_000_000,
                    help="Chunk size for --generated_weighted_sample: each chunk is "
                         "sampled, histogrammed, and discarded, bounding memory "
                         "regardless of the total weighted count")
args = parser.parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_NUM_SAMPLES = 10_000_000   # fallback when --data (and thus --data_samples) is skipped


# --------------------------------------------------------------------------- #
# Flow-specific pieces. The flow is fit in standardized space x_s=(x-mean)/std,
# so its physical density is p_std((x-mean)/std) / (std_pz*std_pt), and physical
# samples are s*std + mean.
# --------------------------------------------------------------------------- #
def flow_phys_density(entry, grid):
    mean = np.asarray(entry["mean"], dtype=float).ravel()
    std = np.asarray(entry["std"], dtype=float).ravel()
    flow = entry["flow"]
    if flow.log_transform:
        eps = np.asarray(flow.log_eps, dtype=float).ravel()
        pz, pt = grid[:, 0], grid[:, 1]
        x = np.stack([np.log(pz), np.log(pt + eps[1])], axis=1)
        # change of variables: p_phys(pz,pt) = p_fit(log pz, log(pt+eps)) * |d(log)/d(pz,pt)|
        jac = 1.0 / (pz * (pt + eps[1]))
    else:
        x = grid
        jac = 1.0
    Xs = ((x - mean) / std).astype(np.float32)
    with torch.no_grad():
        lp = flow.log_prob(torch.from_numpy(Xs).to(device)).cpu().numpy()
    return np.exp(lp) / np.prod(std) * jac


def flow_density_grid(entry):
    grid, shape = grid_points()
    return grid_normalize(flow_phys_density(entry, grid)).reshape(shape)


def model_density_grid(groups):
    """Analytic 2D density of the full model: rate-weighted mixture sum_w alpha_w p_w."""
    grid, shape = grid_points()
    alpha = mixing_alpha(groups)
    dens = sum(a * flow_phys_density(e, grid) for a, e in zip(alpha, groups.values()))
    return grid_normalize(dens).reshape(shape)


SAMPLE_BATCH = 2_000_000   # cap per flow.sample() call, so huge nw doesn't OOM the GPU


def flow_sample_group(entry, nw):
    mean = np.asarray(entry["mean"], dtype=float).ravel()
    std = np.asarray(entry["std"], dtype=float).ravel()
    parts = []
    with torch.no_grad():
        for i in range(0, nw, SAMPLE_BATCH):
            n_batch = min(SAMPLE_BATCH, nw - i)
            # destandardizes internally; if the flow was fit with --log_transform, this
            # (eval-mode) also exponentiates back to the original physical (pz, pt) scale.
            parts.append(entry["flow"].sample(n_batch, mean=mean, std=std).cpu().numpy())
    return np.concatenate(parts, axis=0)


os.makedirs(figs_dir, exist_ok=True)

print(f"Loading flows from {args.model} (device={device}) ...")
ckpt = torch.load(args.model, map_location='cpu', weights_only=False)   # CPU-saved file
groups = ckpt["groups"]
multiple_models = len(groups) > 1   # per-weight file (train_flow_per_weight.py) vs a
                                     # single combined flow (train_flow.py or distill_flow.py)
for w, e in groups.items():
    flow = NeuralSplineFlow.from_config(e["config"])
    flow.load_state_dict(e["state_dict"])
    if args.log_transform:                   # force-enable if the checkpoint didn't already
        flow.log_transform = True
        flow.log_eps = (0.0, args.log_eps)
    e["flow"] = flow.to(device).eval()       # move to GPU (or CPU) for sampling/density
print(f"Loaded {len(groups)} flow(s) on {device}.")

torch.manual_seed(0)

# Load real data up front (if given) so a bare --num_samples can default to matching
# the number of real muons loaded, instead of an arbitrary fixed count.
real = real_keys = true_n = real_all = real_all_w = None
if args.data:
    real, real_keys, true_n = load_real(args.data, args.data_samples)
    real_all, real_all_w = combined_real(real, true_n)

num_samples = args.num_samples
if num_samples is None:
    num_samples = real_all.shape[0] if real_all is not None else DEFAULT_NUM_SAMPLES
    print(f"--num_samples not given; sampling {num_samples} generated muons to match the real data.")


def sample_combined_all():
    return sample_mixture(groups, num_samples, flow_sample_group, seed=0)


def sample_weighted_hist(n_target, chunk):
    """Stream n_target mixture samples in chunks, accumulating straight into a 2D
    histogram -- n_target can be tens of billions (the real data's weighted count), so
    this never materializes more than one chunk of samples at a time."""
    H = np.zeros((len(ge.PZ_BINS) - 1, len(ge.PT_BINS) - 1))
    remaining, seed = n_target, 0
    while remaining > 0:
        n_chunk = min(chunk, remaining)
        xy = sample_mixture(groups, n_chunk, flow_sample_group, seed=seed)
        H += hist2d_counts(xy)
        remaining -= n_chunk
        seed += 1
    return grid_normalize(H)


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
base_title = 'Combined per-weight NSF' if multiple_models else 'Single combined NSF'
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

# combined comparison: real (weight*N) vs generated. The generated 2D panel reuses
# Hg (analytic if --pdf, else the sampled histogram of gen_all).
gen_label = 'generated (analytic PDF)' if args.pdf else f'generated (N={gen_all.shape[0]})'

gen_weighted_density = gen_weighted_label = None
if args.generated_weighted_sample:
    n_weighted = int(round(real_all_w.sum()))
    print(f"--generated_weighted_sample: streaming {n_weighted:,} generated muons "
          f"in chunks of {args.weighted_sample_chunk:,} ...")
    t0 = time.time()
    gen_weighted_density = sample_weighted_hist(n_weighted, args.weighted_sample_chunk)
    print(f"  done in {time.time() - t0:.1f}s")
    gen_weighted_label = f'generated (weighted N={n_weighted:,})'

w1 = plot_compare(real_all, gen_all, real_all_w,
                  'Real (weighted by weight×N) vs generated — combined (NSF)',
                  'flow_compare_combined.png', gen_density=Hg, gen_label=gen_label,
                  gen_weighted_density=gen_weighted_density,
                  gen_weighted_label=gen_weighted_label, show=args.show)
metrics = {'combined': w1}

# ---- per-weight breakdown (skipped for a single combined flow) ----
if multiple_models:
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
