import argparse
import os
import pickle

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import wasserstein_distance

from train_gmm import sample_gmm
from utils import get_real_muons_per_weight, calc_pt

DEFAULT_DATA = '/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample_w.h5'
figs_dir = 'outputs/figs/'
fontsize = 14

parser = argparse.ArgumentParser(description="Sample, plot and score a trained/distilled GMM")
parser.add_argument("--model", type=str, default="outputs/gmm_model.pkl",
                    help="Model from train_gmm.py (per-weight) or distill_gmm.py (flat/distilled)")
parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                    help="Real muon file for comparison (set to '' to skip)")
parser.add_argument("--num_samples", type=int, default=100_000_000,
                    help="Samples for the combined generated distribution")
parser.add_argument("--data_samples", type=int, default=2_000_000,
                    help="Per-group cap on real muons loaded for comparison")
parser.add_argument("--show", action="store_true")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# The per-weight and distilled models describe the SAME combined distribution:
# p(x) = sum_w alpha_w p_w(x), with alpha_w proportional to the real muon rate
# (importance weight x sample count) of group w. The distilled model just
# approximates that mixture with a single, smaller GMM. So both are sampled the
# same way here; only the per-weight model can additionally be broken down by w.
# ---------------------------------------------------------------------------
def mixing_alpha(models):
    # each entry stores its final mixing weight (importance weight * N)
    rate = np.array([e["weight"] for e in models.values()], dtype=float)
    return rate / rate.sum()


def sample_combined(models, n, seed=0):
    """Draw n points from the rate-weighted mixture of the per-weight GMMs."""
    np.random.seed(seed)                       # GMM.sample() uses the global RNG
    alpha = mixing_alpha(models)
    n_per = np.random.default_rng(seed).multinomial(n, alpha)
    parts = []
    for (w, e), nw in zip(models.items(), n_per):
        if nw <= 0:
            continue
        pz, pt, _ = sample_gmm(e["gmm"], e["mean"], e["std"], n_samples=int(nw))
        parts.append(np.stack([pz, pt], axis=1))
    X = np.concatenate(parts, axis=0)
    return X[:, 0], X[:, 1]


PZ_BINS = np.linspace(0, 400, 201)
PT_BINS = np.linspace(0, 13, 201)


def hist2d_density(xy, w=None):
    """Normalized 2D pz-pt histogram on the shared grid (density integrates to 1)."""
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[PZ_BINS, PT_BINS],
                             weights=w, density=True)
    return H


def draw_hist2d(ax, H, norm, title=None):
    """Draw a precomputed density on the shared grid in log color scale."""
    im = ax.pcolormesh(PZ_BINS, PT_BINS, H.T, cmap='viridis', norm=norm)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 13)
    ax.set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
    ax.set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    return im


def shared_lognorm(*grids):
    """A single LogNorm spanning the positive range of all given density grids."""
    pos = np.concatenate([H[H > 0].ravel() for H in grids])
    return LogNorm(vmin=pos.min(), vmax=max(H.max() for H in grids))


def wstd(x, w=None):
    """Weighted standard deviation, used to normalize the W1 distance."""
    mu = np.average(x, weights=w)
    return np.sqrt(np.average((x - mu) ** 2, weights=w))


def w1_norm(real, gen, real_w=None):
    """1-D Wasserstein distance normalized by the real marginal's std (unitless)."""
    d = wasserstein_distance(real, gen, u_weights=real_w)
    return d / (wstd(real, real_w) + 1e-12)


def compare_marginals(ax_pz, ax_pt, real, gen, real_w=None, label=""):
    """Overlay real vs generated pz/pt marginals; return normalized W1 per axis."""
    for ax, i, name in ((ax_pz, 0, '$P_z$ [GeV]'), (ax_pt, 1, '$P_t$ [GeV]')):
        lo, hi = (0, 400) if i == 0 else (0, 13)
        bins = np.linspace(lo, hi, 120)
        ax.hist(real[:, i], bins=bins, weights=real_w, density=True,
                histtype='step', color='k', label='real')
        ax.hist(gen[:, i], bins=bins, density=True,
                histtype='step', color='C1', label='generated')
        ax.set_xlabel(name, fontsize=fontsize)
        ax.legend(fontsize=fontsize - 3)
        if label:
            ax.set_title(label, fontsize=fontsize - 2)
    return (w1_norm(real[:, 0], gen[:, 0], real_w),
            w1_norm(real[:, 1], gen[:, 1], real_w))


def plot_compare(real, gen, real_w, suptitle, fname):
    """2x2 real-vs-generated panel: 2D densities (shared log scale + colorbar) on
    top, pz/pt marginals below. Returns the normalized W1 per marginal."""
    Hr = hist2d_density(real, real_w)
    Hg = hist2d_density(gen)
    norm = shared_lognorm(Hr, Hg)                     # same color scale for both

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    draw_hist2d(axes[0, 0], Hr, norm, title=f'real (N={real.shape[0]})')
    im = draw_hist2d(axes[0, 1], Hg, norm, title=f'generated (N={gen.shape[0]})')
    fig.colorbar(im, ax=axes[0, :].tolist(), label='density')
    w1 = compare_marginals(axes[1, 0], axes[1, 1], real, gen, real_w)
    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize)
    fig.savefig(figs_dir + fname)
    if args.show: plt.show()
    plt.close(fig)
    return w1


os.makedirs(figs_dir, exist_ok=True)

with open(args.model, 'rb') as f:
    model = pickle.load(f)
distilled = "gmm" in model          # flat dict -> distilled; else per-weight dict

# ---- combined generated distribution (identical treatment for both models) ----
if distilled:
    pz_gen, pt_gen, _ = sample_gmm(model["gmm"], model["mean"], model["std"],
                                   n_samples=args.num_samples, seed=0)
else:
    pz_gen, pt_gen = sample_combined(model, args.num_samples, seed=0)

gen_all = np.stack([pz_gen, pt_gen], axis=1)
Hg = hist2d_density(gen_all)
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
im = draw_hist2d(ax, Hg, shared_lognorm(Hg),
                 title='Distilled GMM' if distilled else 'Combined per-weight GMM')
fig.colorbar(im, ax=ax, label='density')
fig.savefig(figs_dir + 'generated_momentum.png')
if args.show: plt.show()

# ---- comparison against real data ----
if not args.data:
    print("No --data given; skipping real-vs-generated comparison and metric plot.")
    raise SystemExit

real = get_real_muons_per_weight(args.data, num_samples=args.data_samples)
real_keys = np.array(list(real), dtype=float)


def real_pzpt(arr):
    return np.stack([arr[:, 2], calc_pt(arr[:, 0], arr[:, 1])], axis=1)


# true (uncapped) sample count per group, so the combined distribution can be
# weighted by weight*N even when each group is subsampled to --data_samples.
with h5py.File(args.data, 'r') as _f:
    true_n = {float(_f[g].attrs['weight']): _f[g]['px'].shape[0] for g in _f}


def true_count(w):
    return true_n[min(true_n, key=lambda k: abs(k - w))]


# Combined real distribution weighted by rate = weight * N. Each group is loaded
# with at most --data_samples muons, so a per-sample weight of
#   weight * N_group / n_loaded
# restores the true weight*N contribution of every group regardless of the cap.
# (skip the ~zero-weight group: it carries no muon rate)
groups = [(w, a) for w, a in real.items() if w >= 0.001]
real_all = np.concatenate([real_pzpt(a) for _, a in groups], axis=0)
real_all_w = np.concatenate(
    [np.full(a.shape[0], w * true_count(w) / a.shape[0]) for w, a in groups])

# combined comparison: real (weight*N) vs generated, 2D + marginals
w1_pz, w1_pt = plot_compare(
    real_all, gen_all, real_all_w,
    'Real (weighted by weight×N) vs generated — combined', 'compare_combined.png')

metrics = {'combined': (w1_pz, w1_pt)}

# ---- per-weight breakdown (only meaningful for the per-weight model) ----
if not distilled:
    for w, e in model.items():
        if w < 0.001:
            continue
        arr = real[real_keys[np.argmin(np.abs(real_keys - w))]]   # match real group
        r = real_pzpt(arr)
        pz, pt, _ = sample_gmm(e["gmm"], e["mean"], e["std"], n_samples=r.shape[0], seed=0)
        g = np.stack([pz, pt], axis=1)
        w_str = f"{w:.2f}".replace(".", "_")
        metrics[w] = plot_compare(r, g, None, f'weight {w:.2f}', f'compare_weight_{w_str}.png')

# ---- metric summary: normalized W1 per group ----
labels = [str(k) if k == 'combined' else f'{k:.2f}' for k in metrics]
pz_vals = [metrics[k][0] for k in metrics]
pt_vals = [metrics[k][1] for k in metrics]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 5))
ax.bar(x - 0.2, pz_vals, 0.4, label='$P_z$')
ax.bar(x + 0.2, pt_vals, 0.4, label='$P_t$')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Wasserstein-1 / std (lower is better)', fontsize=fontsize)
ax.set_title('Generative fit quality per weight', fontsize=fontsize)
ax.legend(fontsize=fontsize)
fig.tight_layout()
fig.savefig(figs_dir + 'metric_w1.png')
if args.show: plt.show()

print("\nNormalized W1 distance (lower = better):")
for k in metrics:
    lbl = k if k == 'combined' else f'weight {k:.2f}'
    print(f"  {lbl:>18}: pz={metrics[k][0]:.4f}  pt={metrics[k][1]:.4f}")
