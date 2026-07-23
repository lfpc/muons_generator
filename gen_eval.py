"""Shared evaluation/plotting helpers for the per-weight generative models.

Model-agnostic: works for any per-weight model whose entries carry a mixing
`weight` (= importance weight * N) plus a way to sample and/or evaluate density.
Imported by both test_gmm.py and test_flow.py so the two comparisons render
identically. The model-specific bits (how to sample a group, how to evaluate the
density on the grid) live in the respective test scripts.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import wasserstein_distance

from utils import get_real_muons_per_weight, calc_pt

DEFAULT_DATA = '/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample_w.h5'
figs_dir = 'outputs/figs/'
fontsize = 14

# shared 2D histogram / density grid
PZ_BINS = np.linspace(0, 400, 201)
PT_BINS = np.linspace(0, 13, 201)

saved_plots = []          # every figure written, reported by report_saved_plots()


# --------------------------------------------------------------------------- #
# mixture bookkeeping (weight = importance weight * N is stored per group)
# --------------------------------------------------------------------------- #
def mixing_alpha(models):
    """Rate-weighted mixing coefficients: each entry's stored `weight`, normalized."""
    rate = np.array([e["weight"] for e in models.values()], dtype=float)
    return rate / rate.sum()


def sample_mixture(models, n, sample_group, seed=0):
    """Draw n points from the rate-weighted mixture of per-weight models.

    sample_group(entry, n_w) must return an (n_w, 2) array of physical (pz, pt).
    """
    np.random.seed(seed)                       # some samplers use the global numpy RNG
    alpha = mixing_alpha(models)
    n_per = np.random.default_rng(seed).multinomial(n, alpha)
    parts = [sample_group(e, int(nw))
             for (w, e), nw in zip(models.items(), n_per) if nw > 0]
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- #
# 2D density grid + drawing
# --------------------------------------------------------------------------- #
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


def shared_lognorm(*grids, floor=1e-8):
    """A single LogNorm spanning the positive range of all given density grids.
    vmin is floored at vmax*floor so an analytic PDF's astronomically small tail
    values (never exactly zero) don't stretch the log scale over dozens of decades
    and wash out the structure. Sampled histograms span only a few decades, so the
    floor never clips them."""
    vmax = max(H.max() for H in grids)
    pos = np.concatenate([H[H > 0].ravel() for H in grids])
    return LogNorm(vmin=max(pos.min(), vmax * floor), vmax=vmax)


def grid_points():
    """Bin centers of the shared grid as an (Ngrid, 2) array, plus the 2D shape.
    Used by the analytic-density code in each test script."""
    pz_c = 0.5 * (PZ_BINS[:-1] + PZ_BINS[1:])
    pt_c = 0.5 * (PT_BINS[:-1] + PT_BINS[1:])
    PZ, PT = np.meshgrid(pz_c, pt_c, indexing='ij')
    return np.stack([PZ.ravel(), PT.ravel()], axis=1), PZ.shape


def grid_normalize(dens):
    """Scale so the density integrates to 1 over the grid, matching the density=True
    histograms it is compared against (which normalize over the plotted range)."""
    bin_area = (PZ_BINS[1] - PZ_BINS[0]) * (PT_BINS[1] - PT_BINS[0])
    return dens / (dens.sum() * bin_area)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# real data loading + weight*N combined weighting
# --------------------------------------------------------------------------- #
def real_pzpt(arr):
    return np.stack([arr[:, 2], calc_pt(arr[:, 0], arr[:, 1])], axis=1)


def load_real(data_path, data_samples):
    """Per-group real muons (capped at data_samples) plus the true uncapped counts."""
    real = get_real_muons_per_weight(data_path, num_samples=data_samples)
    real_keys = np.array(list(real), dtype=float)
    with h5py.File(data_path, 'r') as f:
        true_n = {float(f[g].attrs['weight']): f[g]['px'].shape[0] for g in f}
    return real, real_keys, true_n


def real_group(real, real_keys, w):
    """Real (pz, pt) array of the group whose weight is closest to w."""
    return real_pzpt(real[real_keys[np.argmin(np.abs(real_keys - w))]])


def combined_real(real, true_n, wmin=0.0001):
    """Combined real (pz, pt) and per-sample weights = weight * N_true / n_loaded,
    which restores each group's true weight*N rate despite the per-group cap.
    The ~zero-weight group (w < wmin) carries no rate and is dropped."""
    def true_count(w):
        return true_n[min(true_n, key=lambda k: abs(k - w))]
    groups = [(w, a) for w, a in real.items() if w >= wmin]
    real_all = np.concatenate([real_pzpt(a) for _, a in groups], axis=0)
    real_all_w = np.concatenate(
        [np.full(a.shape[0], w * true_count(w) / a.shape[0]) for w, a in groups])
    return real_all, real_all_w


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def save_plot(fig, fname):
    """Save a figure, track it, and report it immediately so progress is visible."""
    path = figs_dir + fname
    fig.savefig(path)
    saved_plots.append(path)
    print(f"  saved plot [{len(saved_plots)}]: {path}", flush=True)
    return path


def report_saved_plots():
    print(f"\nSaved {len(saved_plots)} plot(s):")
    for p in saved_plots:
        print(f"  {p}")


def plot_compare(real, gen, real_w, suptitle, fname, gen_density=None, gen_label=None, show=False):
    """2x2 real-vs-generated panel: 2D densities (shared log scale + colorbar) on
    top, pz/pt marginals below. The generated 2D panel shows gen_density if given
    (a precomputed grid, analytic or sampled), else a histogram of the gen samples.
    gen_label overrides the generated-panel title. The marginals and the returned
    normalized W1 always come from the gen samples."""
    Hr = hist2d_density(real, real_w)
    Hg = gen_density if gen_density is not None else hist2d_density(gen)
    norm = shared_lognorm(Hr, Hg)                     # same color scale for both

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    draw_hist2d(axes[0, 0], Hr, norm, title=f'real (N={real.shape[0]})')
    if gen_label is None:
        gen_label = 'generated (analytic PDF)' if gen_density is not None else f'generated (N={gen.shape[0]})'
    im = draw_hist2d(axes[0, 1], Hg, norm, title=gen_label)
    fig.colorbar(im, ax=axes[0, :].tolist(), label='density')
    w1 = compare_marginals(axes[1, 0], axes[1, 1], real, gen, real_w)
    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize)
    save_plot(fig, fname)
    if show:
        plt.show()
    plt.close(fig)
    return w1


def plot_metric(metrics, fname='metric_w1.png', title='Generative fit quality per weight', show=False):
    """Bar chart of the normalized W1 distance (pz, pt) per group + combined."""
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
    ax.set_title(title, fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    fig.tight_layout()
    save_plot(fig, fname)
    if show:
        plt.show()
    plt.close(fig)


def report_metrics(metrics):
    print("\nNormalized W1 distance (lower = better):")
    for k in metrics:
        lbl = k if k == 'combined' else f'weight {k:.2f}'
        print(f"  {lbl:>18}: pz={metrics[k][0]:.4f}  pt={metrics[k][1]:.4f}")
