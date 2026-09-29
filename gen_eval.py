"""Shared evaluation/plotting helpers for the generative models.

Model-agnostic: every plot takes real muons, generated muons and the real per-row
importance weights, so test_gmm.py and test_flow.py render identically. The
model-specific bits (how to sample, how to evaluate the density on the grid) live in
the respective test scripts.
"""
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import wasserstein_distance

from utils import DEFAULT_DATA, get_real_muons, calc_pt

figs_dir = 'outputs/figs/'
fontsize = 14

# shared 2D histogram / density grid
PZ_BINS = np.linspace(0, 400, 201)
PT_BINS = np.linspace(0, 13, 201)

saved_plots = []          # every figure written, reported by report_saved_plots()


# --------------------------------------------------------------------------- #
# 2D density grid + drawing
# --------------------------------------------------------------------------- #
def hist2d_density(xy, w=None):
    """Normalized 2D pz-pt histogram on the shared grid (density integrates to 1)."""
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[PZ_BINS, PT_BINS],
                             weights=w, density=True)
    return H


def hist2d_counts(xy, w=None):
    """Unnormalized 2D pz-pt histogram (raw/weighted counts) on the shared grid. Used
    to accumulate a density across many sampled chunks without ever holding all of them
    in memory at once (see test_flow.py's --generated_weighted_sample); normalize the
    accumulated total with grid_normalize() once done."""
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[PZ_BINS, PT_BINS], weights=w)
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
    floor never clips them, but pass floor=0 for the true full range."""
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
# real-side cache
#
# Everything the real data contributes to a figure is an array: the bin edges (which are
# data-derived), the rate-weighted counts, and the quantile grid the W1 metric needs.
# Producing them means reading and binning the muon file, which at 2.3e9 rows is most of
# the cost of a comparison run -- and none of it changes from one model to the next. So
# every real-side value is requested through cached(); with a store loaded from disk the
# build functions are never called and the file is never opened.
# --------------------------------------------------------------------------- #
_cache = None


def use_cache(store):
    """Install a dict as the real-side store: what is in it is reused, what is missing is
    computed and added. None (the default) computes everything, as before."""
    global _cache
    _cache = store


def cached(key, build):
    """The real-side array `build()` would produce, from the store when it holds it.

    `build` is a callable, not a value, so nothing touches the real sample on a run that
    has no real sample loaded."""
    if _cache is None:
        return build()
    if key not in _cache:
        if build is None or _cache.get('_frozen', False):
            raise KeyError(
                f"{key!r} is missing from the loaded histogram cache. It was built for a "
                "different comparison -- a (pz, pt) model where this one is complete-state, "
                "or a different --data/--data_samples. Re-run without --load_hists.")
        _cache[key] = np.asarray(build())
    return _cache[key]


def save_cache(path):
    """Write the store next to the figures, for --load_hists to pick up."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in _cache.items() if not k.startswith('_')})
    print(f"  real-side histograms cached -> {path} ({len(_cache)} arrays)")


def load_cache(path):
    with np.load(path) as f:
        store = {k: f[k] for k in f.files}
    store['_frozen'] = True          # a missing key now means the cache does not fit
    use_cache(store)
    return store


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


QUANTILE_GRID = 200_001        # points; the W1 error from the grid is then ~1e-4
QUANTILE_MAX = 20_000_000      # rows actually sorted, strided out of a larger sample


def quantiles(v, w=None, n=QUANTILE_GRID, max_sample=QUANTILE_MAX):
    """The sample's weighted quantile function on a uniform probability grid.

    W1 between two 1D samples is the mean absolute gap between their quantile functions,
    and the std follows from the grid too, so these numbers are the whole of what the
    metric needs from a sample of any size. Beyond max_sample rows the grid is estimated
    from a stride through the sample -- sorting 1e9 values to place 2e5 quantiles is
    wasted work, and the rows are in arbitrary order to begin with."""
    v = np.asarray(v, dtype=np.float64)
    if v.size > max_sample:
        step = v.size // max_sample + 1
        v, w = v[::step], (None if w is None else np.asarray(w)[::step])
    o = np.argsort(v, kind='stable')
    c = np.cumsum(np.ones(v.size) if w is None else np.asarray(w, dtype=np.float64)[o])
    return np.interp(np.linspace(0.0, 1.0, n), c / c[-1], v[o])


def w1_norm_cached(key, real_v, gen, real_w=None):
    """w1_norm with the real side reduced to a cached quantile grid. `real_v` is a
    callable, so it is not evaluated when the cache already holds the grid."""
    q = cached(key, lambda: quantiles(real_v(), real_w))
    g = quantiles(gen, None, len(q))
    return float(np.abs(q - g).mean() / (q.std() + 1e-12))


def _step(ax, h, edges, color, label):
    """A step histogram drawn from precomputed densities (what ax.hist would draw)."""
    ax.stairs(h, edges, color=color, label=label)


def compare_marginals(ax_pz, ax_pt, real, gen, real_w=None, label="", key='combined'):
    """Overlay real vs generated pz/pt marginals; return normalized W1 per axis."""
    w1 = []
    for ax, i, name in ((ax_pz, 0, '$P_z$ [GeV]'), (ax_pt, 1, '$P_t$ [GeV]')):
        lo, hi = (0, 400) if i == 0 else (0, 13)
        bins = np.linspace(lo, hi, 120)
        hr = cached(f'{key}/marginal_{i}',
                    lambda i=i, b=bins: np.histogram(real[:, i], bins=b, weights=real_w,
                                                     density=True)[0])
        _step(ax, hr, bins, 'k', 'real')
        _step(ax, np.histogram(gen[:, i], bins=bins, density=True)[0], bins, 'C1', 'generated')
        ax.set_yscale('log')
        ax.set_xlabel(name, fontsize=fontsize)
        ax.legend(fontsize=fontsize - 3)
        if label:
            ax.set_title(label, fontsize=fontsize - 2)
        w1.append(w1_norm_cached(f'{key}/w1_{i}', lambda i=i: real[:, i], gen[:, i], real_w))
    return tuple(w1)


# --------------------------------------------------------------------------- #
# real data
#
# One flat table, so there is nothing to combine: load_real returns the (N, 8) array
# and callers take the views they need -- real[:, 7] are the per-row importance
# weights every plot below wants, real_pzpt(real) the momentum pair, real[:, :7] the
# physical (px, py, pz, x, y, z, pdg) state.
# --------------------------------------------------------------------------- #
def real_pzpt(arr):
    return np.stack([arr[:, 2], calc_pt(arr[:, 0], arr[:, 1])], axis=1)


def load_real(data_path=DEFAULT_DATA, num_samples=None, group=None):
    """Real muons as (N, 8) float32 plus a view on the weight column. `group` reads a
    single group of a split file (e.g. 'above_W')."""
    real = get_real_muons(data_path, num_samples, group=group)
    return real, real[:, 7]


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


def plot_loss(history, fname='flow_loss.png'):
    """Train/val NLL against epoch, with the best epoch marked."""
    epochs = range(1, len(history["val"]) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train"], label='train', color='C0')
    ax.plot(epochs, history["val"], label='val', color='C1')
    ax.axvline(history["best_epoch"], color='k', ls='--', lw=0.8, label='best')
    ax.set_xlabel('epoch')
    ax.set_ylabel('NLL')
    ax.legend(fontsize=fontsize - 4)
    fig.tight_layout()
    save_plot(fig, fname)
    plt.close(fig)


def report_saved_plots():
    print(f"\nSaved {len(saved_plots)} plot(s):")
    for p in saved_plots:
        print(f"  {p}")


def plot_compare(real, gen, real_w, suptitle, fname, gen_density=None, gen_label=None,
                 gen_weighted_density=None, gen_weighted_label=None, show=False,
                 key='combined'):
    """2x2 (2x3 if gen_weighted_density is given) real-vs-generated panel: 2D densities
    (shared log scale + colorbar) on top, pz/pt marginals below. The generated 2D panel
    shows gen_density if given (a precomputed grid, analytic or sampled), else a
    histogram of the gen samples. gen_label overrides the generated-panel title.
    gen_weighted_density, if given, adds a third top-row panel (with gen_weighted_label
    as its title) -- typically the generated density resampled at the real data's
    weighted count, so the two are compared at matched statistics. The real panel's
    title reports real_w.sum() (the weighted/physical count) whenever real_w is given.
    The marginals and the returned normalized W1 always come from the gen samples."""
    Hr = cached(f'{key}/pz_pt', lambda: hist2d_density(real, real_w))
    Hg = gen_density if gen_density is not None else hist2d_density(gen)
    grids = [Hr, Hg] + ([gen_weighted_density] if gen_weighted_density is not None else [])
    norm = shared_lognorm(*grids)                     # same color scale for every 2D panel

    ncols = 3 if gen_weighted_density is not None else 2
    fig, axes = plt.subplots(2, ncols, figsize=(7 * ncols, 9), constrained_layout=True)

    counts = cached(f'{key}/counts',
                    lambda: np.array([real.shape[0],
                                      real_w.sum() if real_w is not None else np.nan]))
    real_title = f'real (N={int(counts[0])})'
    if np.isfinite(counts[1]):                         # real_w already weights the histogram
        real_title += f', weighted N={counts[1]:,.0f}'
    draw_hist2d(axes[0, 0], Hr, norm, title=real_title)

    if gen_label is None:
        gen_label = 'generated (analytic PDF)' if gen_density is not None else f'generated (N={gen.shape[0]})'
    im = draw_hist2d(axes[0, 1], Hg, norm, title=gen_label)

    if gen_weighted_density is not None:
        draw_hist2d(axes[0, 2], gen_weighted_density, norm, title=gen_weighted_label)
        axes[1, 2].axis('off')                         # no third marginal panel

    fig.colorbar(im, ax=axes[0, :].tolist(), label='density')
    w1 = compare_marginals(axes[1, 0], axes[1, 1], real, gen, real_w, key=key)
    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize)
    save_plot(fig, fname)
    if show:
        plt.show()
    plt.close(fig)
    return w1


# --------------------------------------------------------------------------- #
# position / charge comparison (only for a model that generates the full muon state,
# i.e. train_flow_complete.py -- train_flow_momenta.py fakes x, y, z downstream)
#
# Both `real` and `gen` here are physical (px, py, pz, x, y, z, pdg) arrays -- real[:, :7]
# and train_flow_complete.to_physical produce the same layout -- so the same derivations
# apply to each. Mirrors the plots in
# see_data.py (x-y density, z distribution) with a real-vs-generated panel each.
# --------------------------------------------------------------------------- #
Z_TARGET = -2.14          # target position, as marked in see_data.py
Z_TARGET_END = -2.14 + 1.595


def derived_position(arr):
    """(r, |dphi|, charge) from a physical array: radius, momentum azimuth relative to
    the position azimuth, and charge. These are the quantities train_flow_complete.py
    actually fits, so they are what the comparison should be made in."""
    px, py, x, y, pdg = arr[:, 0], arr[:, 1], arr[:, 3], arr[:, 4], arr[:, 6]
    dphi = np.arctan2(py, px) - np.arctan2(y, x)
    return (np.hypot(x, y),
            np.abs((dphi + np.pi) % (2 * np.pi) - np.pi),
            -np.sign(pdg))                                  # pdg = -13 * charge


def _logbins(*arrays, n=80, floor=1e-9):
    """Log-spaced bins covering the positive values of every given array."""
    pos = np.concatenate([a[a > 0] for a in arrays])
    return np.logspace(np.log10(max(pos.min(), floor)), np.log10(pos.max()), n)


def plot_position_xy(real, gen, real_w=None, fname='position_xy.png', show=False):
    """Real vs generated x-y density on a shared log color scale (see_data.py's
    position_dist, as a comparison)."""
    lim = float(cached('xy/lim', lambda: np.percentile(np.abs(real[:, 3:5]), 99.9)))
    bins = np.linspace(-lim, lim, 201)
    Hr = cached('xy/h', lambda: np.histogram2d(real[:, 3], real[:, 4], bins=[bins, bins],
                                               weights=real_w, density=True)[0])
    Hg, _, _ = np.histogram2d(gen[:, 3], gen[:, 4], bins=[bins, bins], density=True)

    norm = shared_lognorm(Hr, Hg, floor=0.0)         # full range, no decade floor
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, H, title in ((axes[0], Hr, 'real'), (axes[1], Hg, 'generated')):
        im = ax.pcolormesh(bins, bins, H.T, cmap='viridis', norm=norm)
        ax.set_xlabel('x [m]', fontsize=fontsize)
        ax.set_ylabel('y [m]', fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.set_aspect('equal')
    fig.colorbar(im, ax=axes.tolist(), label='density')
    fig.suptitle('Production position — real vs generated', fontsize=fontsize)
    save_plot(fig, fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_position_marginals(real, gen, real_w=None, fname='marginals.png', show=False):
    """1D marginals of everything the position/charge part of the flow models --
    x, y, z, r (log axis: it spans decades), |dphi| and charge -- plus the momentum
    components px, py, pz, which the position parameterization feeds into (px and py are
    pt rotated by the generated azimuth, so they are only right if r and |dphi| are).
    Returns the normalized W1 per variable (on log10 r for the radius, whose raw scale
    is dominated by its tail).

    Every bin edge comes from the real sample alone (not from real and generated
    together), so that the edges can be cached and a generated sample compared against
    them without the real data being present."""
    r_gen, dphi_gen, q_gen = derived_position(gen)

    def r_real():
        return derived_position(real)[0]

    def dphi_real():
        return derived_position(real)[1]

    def panel(ax, key, real_v, gen_v, bins, xlabel, logx=False, w1_of=None):
        """One overlaid pair; the real histogram and its quantile grid are cached."""
        hr = cached(f'marg/{key}', lambda: np.histogram(real_v(), bins=bins, weights=real_w,
                                                        density=True)[0])
        _step(ax, hr, bins, 'k', 'real')
        _step(ax, np.histogram(gen_v, bins=bins, density=True)[0], bins, 'C1', 'generated')
        ax.set_yscale('log')
        if logx:
            ax.set_xscale('log')
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.legend(fontsize=fontsize - 4)
        if w1_of is None:
            return None
        rv, gv = w1_of
        return w1_norm_cached(f'marg/w1_{key}', rv, gv, real_w)

    lim = float(cached('marg/lim', lambda: np.percentile(np.abs(real[:, 3:5]), 99.9)))
    plim = float(cached('marg/plim', lambda: np.percentile(np.abs(real[:, :2]), 99.9)))
    zb = cached('marg/zb', lambda: np.linspace(real[:, 5].min(), real[:, 5].max(), 120))
    rb = cached('marg/rb', lambda: _logbins(r_real()))
    xb = np.linspace(-lim, lim, 120)
    pb = np.linspace(-plim, plim, 120)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13.5), constrained_layout=True)
    m = {}
    m['x'] = panel(axes[0, 0], 'x', lambda: real[:, 3], gen[:, 3], xb, 'x [m]',
                   w1_of=(lambda: real[:, 3], gen[:, 3]))
    m['y'] = panel(axes[0, 1], 'y', lambda: real[:, 4], gen[:, 4], xb, 'y [m]',
                   w1_of=(lambda: real[:, 4], gen[:, 4]))
    m['z'] = panel(axes[0, 2], 'z', lambda: real[:, 5], gen[:, 5], zb, 'z [m]',
                   w1_of=(lambda: real[:, 5], gen[:, 5]))
    for zline in (Z_TARGET, Z_TARGET_END):          # target markers, as in see_data.py
        axes[0, 2].axvline(zline, color='red', linestyle='--')

    m['log10 r'] = panel(axes[1, 0], 'r', r_real, r_gen, rb, 'r = hypot(x, y) [m]',
                         logx=True,
                         w1_of=(lambda: np.log10(r_real()), np.log10(r_gen)))
    m['|dphi|'] = panel(axes[1, 1], 'dphi', dphi_real, dphi_gen, np.linspace(0, np.pi, 120),
                        r'$|\Delta\phi| = |\phi_p - \phi_r|$ [rad]',
                        w1_of=(dphi_real, dphi_gen))
    panel(axes[1, 2], 'charge', lambda: derived_position(real)[2], q_gen,
          np.array([-1.5, -0.5, 0.5, 1.5]), 'charge')

    # momentum components: px, py follow from pt and the generated azimuth, so they are
    # the check that the position parameterization reassembles correctly
    m['px'] = panel(axes[2, 0], 'px', lambda: real[:, 0], gen[:, 0], pb, '$P_x$ [GeV]',
                    w1_of=(lambda: real[:, 0], gen[:, 0]))
    m['py'] = panel(axes[2, 1], 'py', lambda: real[:, 1], gen[:, 1], pb, '$P_y$ [GeV]',
                    w1_of=(lambda: real[:, 1], gen[:, 1]))
    m['pz'] = panel(axes[2, 2], 'pz', lambda: real[:, 2], gen[:, 2],
                    np.linspace(PZ_BINS[0], PZ_BINS[-1], 120), '$P_z$ [GeV]',
                    w1_of=(lambda: real[:, 2], gen[:, 2]))

    fig.suptitle('Position, azimuth, charge and momentum — real vs generated',
                 fontsize=fontsize)
    save_plot(fig, fname)
    if show:
        plt.show()
    plt.close(fig)
    return m


def plot_radius_momentum(real, gen, real_w=None, fname='radius_Pz.png', show=False):
    """The correlations the (r, |dphi|) parameterization exists to capture: radius
    against pz, and radius against |dphi| (muons drift outward, so |dphi| concentrates
    at 0, the more so the larger the radius). Real on the left, generated on the
    right, shared color scale per row."""
    r_gen, dphi_gen, _ = derived_position(gen)
    rb = cached('rmom/rb', lambda: _logbins(derived_position(real)[0], n=100))

    rows = [('pz', lambda: real[:, 2], gen[:, 2], np.linspace(0, 400, 100), '$P_z$ [GeV]'),
            ('dphi', lambda: derived_position(real)[1], dphi_gen,
             np.linspace(0, np.pi, 100), r'$|\Delta\phi|$ [rad]')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for row, (rkey, xr, xg, xb, xlabel) in enumerate(rows):
        Hr = cached(f'rmom/{rkey}', lambda xr=xr, xb=xb: np.histogram2d(
            xr(), derived_position(real)[0], bins=[xb, rb], weights=real_w, density=True)[0])
        Hg, _, _ = np.histogram2d(xg, r_gen, bins=[xb, rb], density=True)
        norm = shared_lognorm(Hr, Hg, floor=0.0)     # full range, no decade floor
        for col, (H, title) in enumerate(((Hr, 'real'), (Hg, 'generated'))):
            ax = axes[row, col]
            im = ax.pcolormesh(xb, rb, H.T, cmap='viridis', norm=norm)
            ax.set_yscale('log')
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel('r [m]', fontsize=fontsize)
            ax.set_title(title, fontsize=fontsize)
        fig.colorbar(im, ax=axes[row, :].tolist(), label='density')
    fig.suptitle('Radius vs momentum and azimuth — real vs generated',
                 fontsize=fontsize)
    save_plot(fig, fname)
    if show:
        plt.show()
    plt.close(fig)


CORNER_LABELS = ('x [m]', 'y [m]', 'z [m]', '$P_x$ [GeV]', '$P_y$ [GeV]', '$P_z$ [GeV]')


def _corner_columns(arr):
    """(x, y, z, px, py, pz) from a physical (px, py, pz, x, y, z, pdg) array."""
    return [arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 0], arr[:, 1], arr[:, 2]]


def corner_bins(real, n=70):
    """The corner's per-variable bins, cached: they come from the real sample, and both
    the corner and plot_position_momentum bin on them."""
    return [cached(f'corner/bins_{k}',
                   lambda k=k: _corner_bins(_corner_columns(real), n)[k])
            for k in range(len(CORNER_LABELS))]


def _corner_bins(cols, n=70, tail_pct=99.9):
    """Shared bins per variable, taken from the real data. The transverse quantities are
    cut at a percentile rather than their extremes: their tails run a hundred sigma out,
    and binning to the extremes would put the whole distribution in one bin."""
    bins = []
    for k, c in enumerate(cols):
        if CORNER_LABELS[k] == '$P_z$ [GeV]':
            bins.append(np.linspace(PZ_BINS[0], PZ_BINS[-1], n))
        elif CORNER_LABELS[k] == 'z [m]':
            bins.append(np.linspace(c.min(), c.max(), n))
        else:
            lim = float(np.percentile(np.abs(c), tail_pct))
            bins.append(np.linspace(-lim, lim, n))
    return bins


def plot_corner_compare(real, gen, real_w=None, fname='corner.png', show=False):
    """Every pair of (x, y, z, px, py, pz) as a 2D histogram, real above the diagonal
    and generated below it, with the 1D marginals of both on the diagonal. Charge is
    left out: it takes two values, so its panels are stripes carrying nothing the
    charge marginal in plot_position_marginals doesn't already show.

    Cell (row i, col j) always puts variable j on x and variable i on y, so the outer
    axis labels hold for the whole row/column -- which means a mirrored pair is the
    transpose of its partner, not a copy. Each mirrored pair shares one log color scale,
    so real and generated are directly comparable; different pairs are not, and there is
    deliberately no global colorbar suggesting otherwise."""
    G = _corner_columns(gen)
    bins = corner_bins(real)
    n = len(G)
    fig, axes = plt.subplots(n, n, figsize=(2.4 * n, 2.4 * n), constrained_layout=True)

    for i in range(n):                                  # diagonal: 1D, both overlaid
        ax = axes[i, i]
        hr = cached(f'corner/diag_{i}',
                    lambda i=i: np.histogram(_corner_columns(real)[i], bins=bins[i],
                                             weights=real_w, density=True)[0])
        _step(ax, hr, bins[i], 'k', 'real')
        _step(ax, np.histogram(G[i], bins=bins[i], density=True)[0], bins[i], 'C1',
              'generated')
        ax.set_yscale('log')
        # not set_yticklabels: a log locator regenerates them at draw time. which='both'
        # because a panel spanning under a decade gets labelled minor ticks too.
        ax.tick_params(which='both', labelleft=False)
        if i == 0:
            ax.legend(fontsize=fontsize - 5)

    for a in range(n):                                  # each unordered pair once
        for b in range(a + 1, n):
            Hr = cached(f'corner/{a}_{b}', lambda a=a, b=b: np.histogram2d(
                _corner_columns(real)[b], _corner_columns(real)[a],
                bins=[bins[b], bins[a]], weights=real_w, density=True)[0])
            Hg, _, _ = np.histogram2d(G[b], G[a], bins=[bins[b], bins[a]], density=True)
            norm = shared_lognorm(Hr, Hg, floor=0.0)     # full range, no decade floor
            axes[a, b].pcolormesh(bins[b], bins[a], Hr.T, cmap='viridis', norm=norm)
            axes[b, a].pcolormesh(bins[a], bins[b], Hg, cmap='viridis', norm=norm)
            for ax, tag in ((axes[a, b], 'real'), (axes[b, a], 'gen')):
                ax.text(0.04, 0.94, tag, transform=ax.transAxes, fontsize=fontsize - 5,
                        color='w', ha='left', va='top',
                        bbox=dict(facecolor='k', alpha=0.55, edgecolor='none', pad=1.5))

    for i in range(n):                                  # labels on the outer edge only
        for j in range(n):
            ax = axes[i, j]
            # Limits set from the bins rather than left to autoscale: hist() pads its
            # data by 5% while pcolormesh() does not, so the diagonal would otherwise sit
            # on a wider x range than the 2D cells sharing its column, and the outer tick
            # labels would not apply to every cell.
            ax.set_xlim(bins[j][0], bins[j][-1])
            if i != j:                                  # diagonal y is density, not a var
                ax.set_ylim(bins[i][0], bins[i][-1])
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 6)
            if i == n - 1:
                ax.set_xlabel(CORNER_LABELS[j], fontsize=fontsize - 2)
            else:
                ax.tick_params(which='both', labelbottom=False)
            # Row 0's leftmost cell is the diagonal, whose y axis is a density, not the
            # variable -- so row 0 gets its label on cell (0, 1), the first cell in the
            # row whose y axis really is x. Without this the top row is the one row of
            # the corner with no name on it.
            if (j == 0 and i != 0) or (i == 0 and j == 1):
                ax.set_ylabel(CORNER_LABELS[i], fontsize=fontsize - 2)
            if i != j and j != 0:            # tick labels stay on column 0 only, so
                ax.tick_params(which='both', labelleft=False)   # (0, 1) is label-only

    fig.suptitle('All pairs — real (upper triangle) vs generated (lower triangle), '
                 '1D marginals on the diagonal', fontsize=fontsize)
    save_plot(fig, fname)
    if show:
        plt.show()
    plt.close(fig)


def plot_position_momentum(real, gen, real_w=None,
                           fname='position_momentum.png', show=False):
    """Transverse position against the momentum along the same axis: x vs px and y vs py,
    real beside generated. These are the pairs the (r, |dphi|) parameterization has to
    reproduce indirectly -- neither x nor px is a fitted quantity, both are reassembled
    from r, pt, |dphi| and the generated azimuth. Bins and axis ranges are the corner
    plot's, one log color scale per row so the two panels are directly comparable."""
    G = _corner_columns(gen)
    bins = corner_bins(real)
    rows = [(0, 3, 'x [m]', '$P_x$ [GeV]'), (1, 4, 'y [m]', '$P_y$ [GeV]')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for row, (ipos, imom, xlabel, ylabel) in enumerate(rows):
        Hr = cached(f'posmom/{ipos}', lambda ipos=ipos, imom=imom: np.histogram2d(
            _corner_columns(real)[ipos], _corner_columns(real)[imom],
            bins=[bins[ipos], bins[imom]], weights=real_w, density=True)[0])
        Hg, _, _ = np.histogram2d(G[ipos], G[imom], bins=[bins[ipos], bins[imom]],
                                  density=True)
        norm = shared_lognorm(Hr, Hg, floor=0.0)
        for col, (H, title) in enumerate(((Hr, 'real'), (Hg, 'generated'))):
            ax = axes[row, col]
            mesh = ax.pcolormesh(bins[ipos], bins[imom], H.T, cmap='viridis', norm=norm)
            ax.set_xlim(bins[ipos][0], bins[ipos][-1])
            ax.set_ylim(bins[imom][0], bins[imom][-1])
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.set_title(title, fontsize=fontsize)
        fig.colorbar(mesh, ax=axes[row, :].tolist(), label='density')
    fig.suptitle('Transverse position vs momentum — real vs generated', fontsize=fontsize)
    save_plot(fig, fname)
    if show:
        plt.show()
    plt.close(fig)


def report_position_metrics(metrics, cos_real, cos_gen):
    print("\nPosition / charge comparison:")
    print(f"  {'<cos dphi>':>18}: real={cos_real:+.4f}  generated={cos_gen:+.4f}")
    print("  normalized W1 (lower = better):")
    for k, v in metrics.items():
        print(f"  {k:>18}: {v:.4f}")


def report_metrics(metrics):
    print("\nNormalized W1 distance (lower = better):")
    for k, (pz, pt) in metrics.items():
        print(f"  {k:>18}: pz={pz:.4f}  pt={pt:.4f}")
