"""Plot and summarize the real muon sample: the raw columns, the quantities the
generative models actually fit (|p|, theta, |dphi|, r, charge -- see fit_space.py), and a
corner plot of each set. Every histogram is weighted by the per-row importance weight, so
what is shown is the physical (rate-weighted) distribution.

Nothing is ever loaded whole: at 2.3e9 rows the table alone is 69 GiB, and the derived
columns and the beam-spot draw pushed the old in-memory version past 190 GiB. Instead the
file is streamed twice -- pass 1 accumulates the statistics (which also fix the histogram
ranges), pass 2 accumulates the histograms themselves -- so peak memory is one chunk of a
few hundred MB, whatever the file size. Two passes over 70 GB of disk is the price; use
--num_samples while iterating on the plots.

Run:  python see_data.py                      (all rows)
      python see_data.py --num_samples 50000000
"""
import argparse
import os

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from utils import DEFAULT_DATA, CHUNK, calc_pt, calc_phi, generate_position

figs_dir = 'outputs/figs/data_file/'
fontsize = 14
BINS = 120
GEN_SAMPLES = 5_000_000        # draws used to illustrate the beam-spot generator

# name -> (axis label, binned in log10 space?). The derived entries are defined exactly
# as the generator's flows fit them (fit_space.py).
QUANTS = {
    'px': ('$P_x$ [GeV]', False), 'py': ('$P_y$ [GeV]', False), 'pz': ('$P_z$ [GeV]', False),
    'x': ('x [cm]', False), 'y': ('y [cm]', False), 'z': ('z [cm]', False),
    'pdg': ('pdg', False), 'weight': ('weight', True),
    'pt': ('$P_t$ [GeV]', False), 'phi': (r'$\phi_p$ [rad]', False),
    'p': ('$|P|$ [GeV]', False),
    'theta': (r'$\theta = \arctan(P_t / P_z)$ [rad]', True),
    'r': ('r = hypot(x, y) [cm]', True),
    'dphi': (r'$|\Delta\phi| = |\phi_p - \phi_r|$ [rad]', False),
    'charge': ('charge', False),
}
RAW = ('px', 'py', 'pz', 'x', 'y', 'z', 'pdg', 'weight')
DERIVED = ('pt', 'phi', 'p', 'theta', 'r', 'dphi', 'charge')
PLOTTED = ('px', 'py', 'pz', 'x', 'y', 'z', 'pt', 'phi', 'p', 'theta', 'r', 'dphi', 'charge',
           'weight')
CORNER_RAW = ('x', 'y', 'z', 'px', 'py', 'pz')
CORNER_GEN = ('p', 'theta', 'dphi', 'r', 'z', 'charge')  # what the generator models
NAMED_2D = (('pz', 'pt'), ('p', 'theta'), ('x', 'y'), ('r', 'pz'), ('r', 'dphi'))
FIXED = {'charge': (-1.5, 1.5, 3)}                  # charge takes two values: 3 bins, not 120
MARGINALS = (('x', 'y', 'z'), ('px', 'py', 'pz'))   # draw_marginals grid, positions over momenta
WEIGHT_BREAKS = (1e-7, 1e-2, 1.0)                   # *_per_weight.png split points
WEIGHT_LABELS = ('w < 1e-7', '1e-7 <= w < 1e-2', '1e-2 <= w < 1', 'w >= 1')
PER_WEIGHT_QUANTS = ('pz', 'pt', 'z', 'r')          # momentum_per_weight.png / position_per_weight.png


def chunks(path, num_samples=None, chunk=CHUNK):
    """Yield (quantities, weights) for one chunk of the file at a time.

    A capped read has its weights scaled by N_total/n, so the sub-sample still carries
    the full physical rate (as utils.get_real_muons does)."""
    with h5py.File(path, 'r') as f:
        n_total = f['pz'].shape[0]
        n = n_total if num_samples is None else min(num_samples, n_total)
        scale = n_total / n
        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            q = {c: f[c][sl].astype(np.float32) for c in RAW}
            q['weight'] *= scale
            q['pt'] = calc_pt(q['px'], q['py'])
            q['phi'] = calc_phi(q['px'], q['py'])
            q['p'] = np.hypot(q['pt'], q['pz'])
            q['theta'] = np.arctan2(q['pt'], q['pz'])     # the model fits log(tan theta)
            q['r'] = np.hypot(q['x'], q['y'])
            dphi = q['phi'] - np.arctan2(q['y'], q['x'])
            q['dphi'] = np.abs((dphi + np.pi) % (2 * np.pi) - np.pi)
            q['charge'] = -np.sign(q['pdg'])
            yield q, q['weight'].astype(np.float64)


class Bins:
    """Uniform bins, in log10 space for the quantities that span decades.

    Uniform means a bin index is arithmetic rather than a search, so a chunk can be
    histogrammed with np.bincount -- which matters when every one of 466 chunks feeds
    ~30 two-dimensional histograms. Against np.histogram on the same edges this is exact
    for the linear quantities; for a log-binned one a value sitting exactly on an edge
    can fall either side of it (log10(10**e) is not always e), which is one row in
    millions."""

    def __init__(self, lo, hi, nb, log=False):
        self.lo, self.hi, self.nb, self.log = float(lo), float(hi), int(nb), bool(log)
        if self.hi <= self.lo:
            self.hi = self.lo + 1.0
        self.inv = self.nb / (self.hi - self.lo)

    def index(self, v):
        # float64 for the mapping: in float32 a value sitting on a bin edge can fall on
        # either side of it, which shows up as a handful of counts moving between
        # neighbouring bins
        t = v.astype(np.float64)
        if self.log:
            np.maximum(t, 1e-300, out=t)
            np.log10(t, out=t)
        np.subtract(t, self.lo, out=t)
        np.multiply(t, self.inv, out=t)
        i = t.astype(np.int32)
        return np.clip(i, 0, self.nb - 1, out=i)

    @property
    def edges(self):
        e = np.linspace(self.lo, self.hi, self.nb + 1)
        return 10.0 ** e if self.log else e


# weight_cdf.png: a fixed domain (not derived from this file's min/max, unlike every
# other Bins here) so the axis means the same thing run to run.
CDF_WEIGHT_BINS = Bins(-19, 5, 240, log=True)


def accumulate_stats(path, num_samples):
    """N, min, max, smallest positive value, and the (weighted and unweighted) mean and
    std of every quantity, in one streaming pass. Sums are float64: at 2.3e9 rows a
    float32 accumulator loses the small terms entirely."""
    st = {k: dict(n=0, min=np.inf, max=-np.inf, posmin=np.inf,
                  s=0.0, ss=0.0, ws=0.0, wss=0.0) for k in RAW + DERIVED}
    wsum = 0.0
    wsum2 = 0.0                                      # sum(w**2), for the effective N (Kish's ESS)
    log_q = {k for k in st if QUANTS[k][1]}        # only these need a positive minimum
    for q, w in chunks(path, num_samples):
        wsum += w.sum()
        wsum2 += w @ w
        for k, a in st.items():
            v = q[k].astype(np.float64)
            v2 = v * v                                # reused by both second moments
            a['n'] += v.size
            a['min'] = min(a['min'], v.min())
            a['max'] = max(a['max'], v.max())
            if k in log_q:
                pos = v[v > 0]
                a['posmin'] = min(a['posmin'], pos.min()) if pos.size else a['posmin']
            a['s'] += v.sum()
            a['ss'] += v2.sum()
            a['ws'] += v @ w
            a['wss'] += v2 @ w
    for a in st.values():
        a['mean'] = a['s'] / a['n']
        a['std'] = np.sqrt(max(a['ss'] / a['n'] - a['mean'] ** 2, 0.0))
        a['wmean'] = a['ws'] / wsum
        a['wstd'] = np.sqrt(max(a['wss'] / wsum - a['wmean'] ** 2, 0.0))
    return st, wsum, wsum2


def report_weight_ranges(range_n, range_w, n_raw, wsum):
    """Raw-row and rate-weighted share of each WEIGHT_LABELS range, e.g. how much of the
    2.3e9 rows vs. how much of the physical rate sits below weight 1e-7 -- the same split
    momentum_per_weight.png plots pz/pt by."""
    print(f"\n{'weight range':>20} | {'raw N':>14} {'raw %':>8} | "
          f"{'sum(weight)':>14} {'weighted %':>10}")
    for lab in WEIGHT_LABELS:
        n, w = range_n[lab], range_w[lab]
        print(f"{lab:>20} | {n:>14,} {100 * n / n_raw:>7.3f}% | "
              f"{w:>14,.6g} {100 * w / wsum:>9.3f}%")


def report_stats(st, wsum, wsum2):
    """min/max/mean/std of every quantity, unweighted (as sampled) and rate-weighted
    (the physical distribution the models are fit to); and the raw, weighted, and
    effective (Kish's ESS = (sum w)^2 / sum(w^2)) sample counts."""
    n_raw = st['pz']['n']
    n_eff = wsum ** 2 / wsum2
    print(f"\n{n_raw:,} muons | sum of weights {wsum:,.6g} | "
          f"effective N {n_eff:,.6g} ({100 * n_eff / n_raw:.3g}% of raw)")
    print(f"{'quantity':>10} | {'min':>12} {'max':>12} | {'mean':>12} {'std':>12} | "
          f"{'w-mean':>12} {'w-std':>12}")
    for group, keys in (('raw', RAW), ('generated', DERIVED)):
        print(f"  -- {group} --")
        for k in keys:
            a = st[k]
            print(f"{k:>10} | {a['min']:>12.5g} {a['max']:>12.5g} | "
                  f"{a['mean']:>12.5g} {a['std']:>12.5g} | "
                  f"{a['wmean']:>12.5g} {a['wstd']:>12.5g}")


def make_bins(st):
    """One Bins per plotted quantity, from the ranges pass 1 measured."""
    b = {}
    for k in PLOTTED:
        if k in FIXED:
            b[k] = Bins(*FIXED[k])
            continue
        log = QUANTS[k][1]
        a = st[k]
        lo, hi = (np.log10(a['posmin']), np.log10(a['max'])) if log else (a['min'], a['max'])
        b[k] = Bins(lo, hi, BINS, log)
    return b


def corner_pairs():
    """Every pair the two corner plots need, deduplicated across them. Kept separate
    from NAMED_2D even where they overlap (pz-pt, x-y), so the corner store is
    self-contained and every panel of a corner comes from the same rows."""
    pairs = []
    for cols in (CORNER_RAW, CORNER_GEN):
        for i, a in enumerate(cols):
            for c in cols[i + 1:]:
                if (a, c) not in pairs and (c, a) not in pairs:
                    pairs.append((a, c))
    return pairs


def pair_list():
    """Every distinct 2D histogram the script accumulates."""
    return list(NAMED_2D) + [p for p in corner_pairs()
                             if p not in NAMED_2D and p[::-1] not in NAMED_2D]


def accumulate_hists(path, num_samples, bins, corner_samples):
    """Rate-weighted histograms, in one streaming pass.

    np.bincount on precomputed indices: each quantity's bin index is computed once per
    chunk and reused by every pair it appears in. The two corner plots need 25 further
    2D histograms between them, which is most of the work per chunk, so they are filled
    from the first `corner_samples` rows only -- a corner is about shape, and a few
    hundred million rows leave every bin with thousands of entries. The standalone 1D
    and 2D figures always see the whole stream.

    Returns (h1, h2, hc1, hc2, n_corner, h1_weight_raw, hw, range_n, range_w, cdf_raw,
    cdf_w): the full-stream histograms, the corner ones, how many rows went into the
    corners, the unweighted (per-row, one count each) histogram of the weight column --
    h1['weight'] is that same column weighted by itself, so together the two show both
    the raw and the rate-weighted proportion of each weight value -- hw, the
    rate-weighted PER_WEIGHT_QUANTS histograms split by WEIGHT_LABELS weight range (for
    momentum_per_weight.png and position_per_weight.png), range_n / range_w, the raw row
    count and summed weight in each of those same ranges (for the ratios printed
    alongside those plots), and cdf_raw / cdf_w, the raw and rate-weighted weight
    histograms over CDF_WEIGHT_BINS' fixed domain (for weight_cdf.png).
    """
    named, corner = list(NAMED_2D), corner_pairs()
    h1 = {k: np.zeros(bins[k].nb) for k in PLOTTED}
    h2 = {p: np.zeros((bins[p[0]].nb, bins[p[1]].nb)) for p in named}
    hc1 = {k: np.zeros(bins[k].nb) for k in PLOTTED}
    hc2 = {p: np.zeros((bins[p[0]].nb, bins[p[1]].nb)) for p in corner}
    h1_weight_raw = np.zeros(bins['weight'].nb)
    hw = {lab: {k: np.zeros(bins[k].nb) for k in PER_WEIGHT_QUANTS} for lab in WEIGHT_LABELS}
    range_n = {lab: 0 for lab in WEIGHT_LABELS}
    range_w = {lab: 0.0 for lab in WEIGHT_LABELS}
    cdf_raw = np.zeros(CDF_WEIGHT_BINS.nb)
    cdf_w = np.zeros(CDF_WEIGHT_BINS.nb)
    seen = 0

    def fill2d(out, pairs, idx, w):
        for (a, c) in pairs:
            nb_c = bins[c].nb
            flat = idx[a] * np.int32(nb_c) + idx[c]    # < 120*120, so int32 is plenty
            out[(a, c)] += np.bincount(flat, weights=w,
                                       minlength=bins[a].nb * nb_c).reshape(-1, nb_c)

    for q, w in chunks(path, num_samples):
        idx = {k: bins[k].index(q[k]) for k in PLOTTED}
        for k in PLOTTED:
            h1[k] += np.bincount(idx[k], weights=w, minlength=bins[k].nb)
        h1_weight_raw += np.bincount(idx['weight'], minlength=bins['weight'].nb)
        idx_cdf = CDF_WEIGHT_BINS.index(q['weight'])
        cdf_raw += np.bincount(idx_cdf, minlength=CDF_WEIGHT_BINS.nb)
        cdf_w += np.bincount(idx_cdf, weights=w, minlength=CDF_WEIGHT_BINS.nb)
        edges = (-np.inf,) + WEIGHT_BREAKS + (np.inf,)
        masks = [(w >= lo) & (w < hi) for lo, hi in zip(edges, edges[1:])]
        for lab, mask in zip(WEIGHT_LABELS, masks):
            range_n[lab] += int(mask.sum())
            range_w[lab] += w[mask].sum()
            for k in PER_WEIGHT_QUANTS:
                hw[lab][k] += np.bincount(idx[k][mask], weights=w[mask],
                                          minlength=bins[k].nb)
        fill2d(h2, named, idx, w)
        if seen < corner_samples:
            for k in PLOTTED:
                hc1[k] += np.bincount(idx[k], weights=w, minlength=bins[k].nb)
            fill2d(hc2, corner, idx, w)
            seen += w.size
    return h1, h2, hc1, hc2, seen, h1_weight_raw, hw, range_n, range_w, cdf_raw, cdf_w


def get2d(h2, a, b):
    """The (a, b) histogram, transposing the stored (b, a) one if that is how it went in."""
    return h2[(a, b)] if (a, b) in h2 else h2[(b, a)].T


def lognorm(H):
    """Log colour scale over a histogram's positive range, tolerating an empty panel."""
    pos = H[H > 0]
    return LogNorm(vmin=pos.min() if pos.size else 1e-12,
                   vmax=H.max() if H.max() > 0 else 1.0)


def density(h, edges):
    """Counts as a density, so log-spaced bins are not read as a shape."""
    total = h.sum()
    return h / (total * np.diff(edges)) if total else h


def draw_1d(h1, bins, k, fname, savefig, show, log_y=True, vlines=()):
    fig, ax = plt.subplots(figsize=(10, 5))
    e = bins[k].edges
    ax.stairs(density(h1[k], e), e, color='C0', fill=True, alpha=0.7)
    for v in vlines:
        ax.axvline(v, color='red', linestyle='--')
    ax.set_xlabel(QUANTS[k][0], fontsize=fontsize)
    ax.set_ylabel('Density', fontsize=fontsize)
    if log_y:
        ax.set_yscale('log')
    if bins[k].log:
        ax.set_xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.tight_layout()
    savefig(fname)
    if show:
        plt.show()
    plt.close(fig)


def draw_weight_dist(h_raw, h_weighted, bins, fname, savefig, show):
    """Weight histogram, side by side: the left panel is the raw row count at each weight,
    the right panel the sum of weights contributed by rows at each weight -- both plain
    sums, not densities. A density (dividing by bin width) is the wrong normalization for
    the right panel specifically: with log-spaced bins the width itself scales with the
    bin's weight value, which exactly cancels the weighting and reproduces the raw
    histogram's shape -- which is why the two panels looked identical before."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    e = bins['weight'].edges
    for ax, h, ylabel, title in (
            (axes[0], h_raw, 'Count', 'raw (per-row)'),
            (axes[1], h_weighted, 'Sum of weights', 'weighted (per-weight)')):
        ax.stairs(h, e, color='C0', fill=True, alpha=0.7)
        ax.axvline(1.0, color='red', linestyle='--')
        ax.set_xlabel(QUANTS['weight'][0], fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.tight_layout()
    savefig(fname)
    if show:
        plt.show()
    plt.close(fig)


def draw_weight_cdf(cdf_raw, cdf_w, fname, savefig, show):
    """Empirical CDF of the weight column over its fixed CDF_WEIGHT_BINS domain (1e-19
    to 1e5): the proportion of rows (raw) and of total rate (weighted) with weight < w.
    Log-log -- most of the domain above weight ~1 is flat at basically 1, so only a log
    y-axis shows how the curve actually climbs to get there, over the many decades below
    it."""
    e = CDF_WEIGHT_BINS.edges[1:]                   # cumsum up to bin i is the CDF at its right edge
    raw_cdf = np.cumsum(cdf_raw) / cdf_raw.sum()
    w_cdf = np.cumsum(cdf_w) / cdf_w.sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(e, raw_cdf, color='C0', label='raw (proportion of rows)')
    ax.plot(e, w_cdf, color='C1', label='weighted (proportion of total weight)')
    ax.axvline(1.0, color='red', linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('w', fontsize=fontsize)
    ax.set_ylabel('Proportion of data with weight < w', fontsize=fontsize)
    ax.legend(fontsize=fontsize - 2)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    savefig(fname)
    if show:
        plt.show()
    plt.close(fig)


def draw_marginals(h1, bins, fname, savefig, show):
    """Marginals of the position and momentum columns -- x, y, z on top, px, py, pz below
    -- the same layout test_flow.py's plot_position_marginals uses for its position row
    and momentum row, minus the derived/charge row and the generated overlay (there is no
    generated sample here, only the real, rate-weighted histograms already accumulated)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for i, keys in enumerate(MARGINALS):
        for j, k in enumerate(keys):
            ax = axes[i, j]
            e = bins[k].edges
            ax.stairs(density(h1[k], e), e, color='steelblue', fill=True, alpha=0.7)
            ax.set_xlabel(QUANTS[k][0], fontsize=fontsize)
            ax.set_ylabel('Density', fontsize=fontsize)
            ax.set_yscale('log')
            if bins[k].log:
                ax.set_xscale('log')
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.suptitle('Position and momentum marginals — rate-weighted', fontsize=fontsize)
    savefig(fname)
    if show:
        plt.show()
    plt.close(fig)


def _draw_per_weight(hw, bins, keys, suptitle, fname, savefig, show):
    """Marginals of `keys` split into the WEIGHT_LABELS weight ranges, each one still
    rate-weighted (np.bincount(..., weights=w) inside accumulate_hists) but
    density-normalized separately per range, so the shapes are comparable regardless of
    how much weight or how many rows fall in that range."""
    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 6), constrained_layout=True)
    colors = ('C0', 'C1', 'C2', 'C3')
    for ax, k in zip(axes, keys):
        e = bins[k].edges
        for lab, color in zip(WEIGHT_LABELS, colors):
            ax.stairs(density(hw[lab][k], e), e, color=color, label=lab)
        ax.set_xlabel(QUANTS[k][0], fontsize=fontsize)
        ax.set_ylabel('Density', fontsize=fontsize)
        ax.set_yscale('log')
        if bins[k].log:
            ax.set_xscale('log')
        ax.legend(fontsize=fontsize - 4)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.suptitle(suptitle, fontsize=fontsize)
    savefig(fname)
    if show:
        plt.show()
    plt.close(fig)


def draw_momentum_per_weight(hw, bins, fname, savefig, show):
    _draw_per_weight(hw, bins, ('pz', 'pt'),
                     'Pz / Pt marginals by weight range — rate-weighted, normalized per range',
                     fname, savefig, show)


def draw_position_per_weight(hw, bins, fname, savefig, show):
    _draw_per_weight(hw, bins, ('z', 'r'),
                     'z / r marginals by weight range — rate-weighted, normalized per range',
                     fname, savefig, show)


def draw_2d(h2, bins, a, b, fname, savefig, show, circle=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    H = get2d(h2, a, b)
    ex, ey = bins[a].edges, bins[b].edges
    im = ax.pcolormesh(ex, ey, H.T, cmap='viridis', norm=lognorm(H))
    fig.colorbar(im, ax=ax, label='Weighted counts')
    ax.set_xlabel(QUANTS[a][0], fontsize=fontsize)
    ax.set_ylabel(QUANTS[b][0], fontsize=fontsize)
    if bins[a].log:
        ax.set_xscale('log')
    if bins[b].log:
        ax.set_yscale('log')
    if circle is not None:
        ax.add_patch(plt.Circle((0, 0), circle, color='r', fill=False))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    fig.tight_layout()
    savefig(fname)
    if show:
        plt.show()
    plt.close(fig)


def draw_corner(h1, h2, bins, keys, fname, savefig, show):
    """Marginals on the diagonal, pairs below it -- the same layout as before, drawn
    from the accumulated histograms instead of from the samples."""
    n = len(keys)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), constrained_layout=True)
    images = []
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            ax = axes[i, j]
            if i == j:
                e = bins[ki].edges
                ax.stairs(density(h1[ki], e), e, color='steelblue')
                ax.set_yscale('log')
            elif i > j:
                H = get2d(h2, kj, ki)
                images.append(ax.pcolormesh(bins[kj].edges, bins[ki].edges, H.T,
                                            cmap='viridis', norm=lognorm(H)))
                if bins[ki].log:
                    ax.set_yscale('log')
            else:
                ax.axis('off')
                continue
            if bins[kj].log:
                ax.set_xscale('log')
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(QUANTS[kj][0], fontsize=fontsize - 2)
            if i != j and j == 0:
                ax.set_ylabel(QUANTS[ki][0], fontsize=fontsize - 2)
            else:
                # not set_yticklabels: a log locator regenerates them at draw time, and
                # the labels then spill over the neighbouring panel
                ax.tick_params(which='both', labelleft=False)
    if images:
        fig.colorbar(images[-1], ax=axes, fraction=0.02, pad=0.02, label='Weighted counts')
    fig.suptitle(f"{'raw columns' if keys == CORNER_RAW else 'fit-target quantities'} "
                 "— rate-weighted", fontsize=fontsize)
    savefig(fname)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default=DEFAULT_DATA, help="Muon file to plot")
    parser.add_argument('--num_samples', type=int, default=None,
                        help="Row cap (None = the whole file; it is streamed either way)")
    parser.add_argument('--corner_samples', type=int, default=200_000_000,
                        help="Rows used for the two corner plots; their 25 extra 2D "
                             "histograms are most of the per-chunk cost, and a corner "
                             "only needs enough rows to show its shape")
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    show = args.show

    os.makedirs(figs_dir, exist_ok=True)
    saved_plots = []

    def savefig(name):
        path = figs_dir + name
        plt.savefig(path)
        saved_plots.append(path)
        print(f"  saved plot [{len(saved_plots)}]: {path}", flush=True)

    print(f"Pass 1/2 (statistics) over {args.data} ...", flush=True)
    st, wsum, wsum2 = accumulate_stats(args.data, args.num_samples)
    report_stats(st, wsum, wsum2)

    print("\nPass 2/2 (histograms) ...", flush=True)
    bins = make_bins(st)
    (h1, h2, hc1, hc2, n_corner, h1_weight_raw, hw, range_n, range_w, cdf_raw,
     cdf_w) = accumulate_hists(args.data, args.num_samples, bins, args.corner_samples)
    report_weight_ranges(range_n, range_w, st['pz']['n'], wsum)

    print()
    draw_weight_dist(h1_weight_raw, h1['weight'], bins, 'weight_dist.png', savefig, show)
    draw_weight_cdf(cdf_raw, cdf_w, 'weight_cdf.png', savefig, show)
    draw_marginals(h1, bins, 'marginals.png', savefig, show)
    draw_momentum_per_weight(hw, bins, 'momentum_per_weight.png', savefig, show)
    draw_position_per_weight(hw, bins, 'position_per_weight.png', savefig, show)
    draw_2d(h2, bins, 'pz', 'pt', 'momentum.png', savefig, show)
    draw_1d(h1, bins, 'z', 'z_dist.png', savefig, show, vlines=(-2.14, -2.14 + 1.595))
    draw_2d(h2, bins, 'x', 'y', 'original_position_dist.png', savefig, show, circle=0.15)
    draw_1d(h1, bins, 'phi', 'phi_dist.png', savefig, show, log_y=False)

    # the quantities the models generate
    draw_2d(h2, bins, 'p', 'theta', 'momentum_p_theta.png', savefig, show)
    draw_1d(h1, bins, 'r', 'radius_dist.png', savefig, show)
    draw_1d(h1, bins, 'dphi', 'dphi_dist.png', savefig, show)
    draw_1d(h1, bins, 'charge', 'charge_dist.png', savefig, show, log_y=False)
    draw_2d(h2, bins, 'r', 'pz', 'radius_Pz.png', savefig, show)
    draw_2d(h2, bins, 'r', 'dphi', 'radius_dphi.png', savefig, show)

    print(f"  (corner plots from the first {n_corner:,} rows)")
    draw_corner(hc1, hc2, bins, CORNER_RAW, 'corner_plot.png', savefig, show)
    draw_corner(hc1, hc2, bins, CORNER_GEN, 'corner_plot_fit_targets.png', savefig, show)

    # the beam-spot generator, for reference: it is a generator, not data, so a few
    # million draws show its shape whatever the file size
    gen_x, gen_y, _ = generate_position(GEN_SAMPLES)
    fig, ax = plt.subplots(figsize=(10, 5))
    _, _, _, im = ax.hist2d(gen_x, gen_y, bins=200, cmap='viridis', norm=LogNorm())
    fig.colorbar(im, ax=ax, label='Counts')
    ax.set_xlabel('x [cm]', fontsize=fontsize)
    ax.set_ylabel('y [cm]', fontsize=fontsize)
    ax.add_patch(plt.Circle((0, 0), 0.15, color='r', fill=False))
    ax.set_title(f'generate_position() ({GEN_SAMPLES:,} draws)', fontsize=fontsize)
    fig.tight_layout()
    savefig('position_dist.png')
    if show:
        plt.show()
    plt.close(fig)

    print("\nSaved plots:")
    for path in saved_plots:
        print(f"  {path}")
