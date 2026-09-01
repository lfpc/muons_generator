import argparse
import os

import matplotlib
matplotlib.use('Agg')          # headless: must be set before any pyplot import (incl.
                                # the one inside see_data) or backend auto-selection can hang
import numpy as np
from matplotlib import pyplot as plt

from utils import get_real_muons_per_id, calc_pt, generate_position
from see_data import hist1d, hist2d, corner_plot

DEFAULT_DATA = 'full_sample_per_id.h5'
figs_dir = 'outputs/figs/data_per_id/'
show = False
fontsize = 14

# the 7 meaningful importance-weight values used throughout the dataset (see each
# group's attrs['weight'] in full_sample_w.h5); a per-row weight is assigned to
# whichever of these it's nearest to. The ~0 spurious weight is excluded, as elsewhere.
KNOWN_WEIGHTS = [0.0937, 1.0, 3.2616, 7.6875, 9.37, 326.16, 768.75]


def weight_composition(data, known=KNOWN_WEIGHTS):
    """counts[i, j] = rows of motherID ids[i] whose weight is nearest to known[j]."""
    ids = sorted(data)
    known = np.asarray(known)
    counts = np.zeros((len(ids), len(known)))
    for i, mid in enumerate(ids):
        w = data[mid][:, 7]
        idx = np.argmin(np.abs(w[:, None] - known[None, :]), axis=1)
        for j in range(len(known)):
            counts[i, j] = np.sum(idx == j)
    return ids, counts


def plot_z_by_weight(z, w, known=KNOWN_WEIGHTS, weighted=False, bins=100):
    """Step-style (unfilled) histogram of z, overlaid once per nearest known weight
    category, each a different color, log-scale counts. If weighted, each category's
    histogram is weighted by its own physical weight rather than raw per-event counts."""
    known = np.asarray(known)
    idx = np.argmin(np.abs(w[:, None] - known[None, :]), axis=1)
    edges = np.linspace(z.min(), z.max(), bins)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, wv in enumerate(known):
        mask = idx == j
        if not mask.any():
            continue
        wts = w[mask] if weighted else None
        ax.hist(z[mask], bins=edges, weights=wts, histtype='step', label=f'weight={wv:g}')
    ax.set_yscale('log')
    ax.set_ylabel('Counts', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    return ax


def plot_weight_composition(data, known=KNOWN_WEIGHTS):
    """Grouped bar chart: counts of each known weight category, per motherID (log y)."""
    ids, counts = weight_composition(data, known)
    x = np.arange(len(ids))
    n_w = len(known)
    width = 0.8 / n_w
    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(ids)), 6))
    for j, wv in enumerate(known):
        ax.bar(x + (j - (n_w - 1) / 2) * width, counts[:, j], width, label=f'weight={wv:g}')
    ax.set_xticks(x)
    ax.set_xticklabels([str(mid) for mid in ids])
    ax.set_xlabel('motherID', fontsize=fontsize)
    ax.set_ylabel('counts', fontsize=fontsize)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax.legend(fontsize=fontsize - 5, ncol=2)
    fig.tight_layout()


def check_origin(x, y, w, name='all', windows=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)):
    """Tell a genuine discrete atom at the origin apart from a merely narrow core.

    Prints the rows sitting at exactly 0.0 (raw and weighted), the value granularity
    that would betray a quantized coordinate, and the occupancy of windows shrinking
    by decades. Read the last line as a scaling law: for a smooth density the counts
    should fall ~10x per decade for x/y (and ~100x for r, which carries the 2D
    Jacobian); counts that plateau instead mean a real atom sitting at the origin,
    which no continuous flow can represent."""
    tot_w = w.sum()
    print(f"\n--- origin check [{name}]  N={len(x)}, N_w={tot_w:g} ---")
    for arr, lab in ((x, 'x'), (y, 'y'), (np.hypot(x, y), 'r')):
        exact = arr == 0.0
        nonzero = np.abs(arr[~exact])
        smallest = nonzero.min() if nonzero.size else float('nan')
        print(f"  {lab}: exact 0.0 -> {exact.sum()} rows ({100 * exact.mean():.4f}%), "
              f"weighted {w[exact].sum():g} ({100 * w[exact].sum() / tot_w:.4f}%)")
        print(f"     n_unique = {np.unique(arr).size}, "
              f"smallest nonzero |{lab}| = {smallest:.3e}")
        occ = ",  ".join(f"<{e:g}: {int((np.abs(arr) < e).sum())}" for e in windows)
        print(f"     |{lab}| {occ}")


def plot_sample_counts(data):
    """Grouped bar chart per motherID: raw number of samples (N) vs weighted number
    of samples (N_w, i.e. the sum of each row's weight), log y."""
    ids = sorted(data)
    N = np.array([data[mid].shape[0] for mid in ids])
    N_w = np.array([data[mid][:, 7].sum() for mid in ids])
    x = np.arange(len(ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(ids)), 6))
    ax.bar(x - width / 2, N, width, label='N (raw)')
    ax.bar(x + width / 2, N_w, width, label='N_w (weighted)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(mid) for mid in ids])
    ax.set_xlabel('motherID', fontsize=fontsize)
    ax.set_ylabel('number of samples', fontsize=fontsize)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    ax.legend(fontsize=fontsize - 5)
    fig.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot per-motherID muon distributions")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--weighted_plots", action="store_true",
                        help="Weight the histograms by the 'weight' column "
                             "(combined plots by the pooled weight, per-id plots by "
                             "that id's own weight); unweighted by default")
    args = parser.parse_args()

    os.makedirs(figs_dir, exist_ok=True)
    def savefig(name):
        plt.savefig(figs_dir + name)
        plt.close()

    data = get_real_muons_per_id(args.data)

    # concatenate all ids for full-sample plots (weight is already per-row here, unlike
    # see_data.py which had to broadcast a constant group weight across each group)
    all_arr = np.concatenate(list(data.values()), axis=0)
    all_px, all_py, all_pz = all_arr[:, 0], all_arr[:, 1], all_arr[:, 2]
    all_x, all_y, all_z = all_arr[:, 3], all_arr[:, 4], all_arr[:, 5]
    all_W = all_arr[:, 7] if args.weighted_plots else None
    all_pt = calc_pt(all_px, all_py)
    gen_x, gen_y, _ = generate_position(all_arr.shape[0])

    # is the peak at the origin an atom or just a narrow core? (always uses the real
    # weights, independently of --weighted_plots)
    check_origin(all_x, all_y, all_arr[:, 7], name='all')

    labels = ['x (cm)', 'y (cm)', 'z (cm)', '$P_x$ (GeV/c)', '$P_y$ (GeV/c)', '$P_z$ (GeV/c)']

    ######## momentum distribution (combined)
    ax = hist2d(all_pz, all_pt, W=all_W)
    ax.set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
    ax.set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 13)
    savefig('momentum.png')
    if show: plt.show()

    ##### z_dist (combined)
    target = -2.14
    ax = plot_z_by_weight(all_z, all_arr[:, 7], weighted=args.weighted_plots)
    ax.axvline(target, color='red', linestyle='--', label='Target Position')
    ax.axvline(target + 1.595, color='red', linestyle='--')
    ax.set_xlabel('z [cm]', fontsize=14)
    ax.legend(fontsize=fontsize - 4)
    savefig('z_dist.png')
    if show: plt.show()

    ###### position_dist (combined, real vs generated reference)
    hist2d(all_x, all_y, W=all_W)
    plt.xlabel('x [cm]', fontsize=14)
    plt.ylabel('y [cm]', fontsize=14)
    circ = plt.Circle((0, 0), 15, color='r', fill=False)
    plt.gca().add_patch(circ)
    savefig('original_position_dist.png')
    if show: plt.show()

    hist2d(gen_x, gen_y)
    plt.xlabel('x [cm]', fontsize=14)
    plt.ylabel('y [cm]', fontsize=14)
    circ = plt.Circle((0, 0), 0.15, color='r', fill=False)
    plt.gca().add_patch(circ)
    savefig('position_dist.png')
    if show: plt.show()

    ##### corner plot (combined)
    corner_plot([all_x, all_y, all_z, all_px, all_py, all_pz], labels, W=all_W)
    savefig('corner_plot.png')
    if show: plt.show()

    ######## prompt (motherID 1) vs not-prompt (motherID 2-30) plots (momentum, corner)
    categories = {
        'prompt': [mid for mid in data if mid == 1],
        'not_prompt': [mid for mid in data if mid != 1],
    }
    for cat, ids in categories.items():
        cat_arr = np.concatenate([data[mid] for mid in ids], axis=0)
        px, py, pz = cat_arr[:, 0], cat_arr[:, 1], cat_arr[:, 2]
        x, y, z = cat_arr[:, 3], cat_arr[:, 4], cat_arr[:, 5]
        w = cat_arr[:, 7] if args.weighted_plots else None
        pt = calc_pt(px, py)
        check_origin(x, y, cat_arr[:, 7], name=cat)

        ax = hist2d(pz, pt, W=w)
        ax.set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
        ax.set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 13)
        ax.set_title(cat, fontsize=fontsize)
        savefig(f'momentum_{cat}.png')
        if show: plt.show()

        corner_plot([x, y, z, px, py, pz], labels, W=w)
        plt.suptitle(cat, fontsize=fontsize)
        savefig(f'corner_plot_{cat}.png')
        if show: plt.show()

        hist2d(x, y, W=w)
        plt.xlabel('x [cm]', fontsize=14)
        plt.ylabel('y [cm]', fontsize=14)
        circ = plt.Circle((0, 0), 15, color='r', fill=False)
        plt.gca().add_patch(circ)
        plt.title(cat, fontsize=fontsize)
        savefig(f'position_dist_{cat}.png')
        if show: plt.show()

        hist1d(x, W=w, log=True)
        plt.xlabel('x [cm]', fontsize=14)
        plt.title(cat, fontsize=fontsize)
        savefig(f'x_dist_{cat}.png')
        if show: plt.show()

        hist1d(y, W=w, log=True)
        plt.xlabel('y [cm]', fontsize=14)
        plt.title(cat, fontsize=fontsize)
        savefig(f'y_dist_{cat}.png')
        if show: plt.show()

    ######## per-motherID plots (momentum, z, position, phi, corner)
    for mid in sorted(data):
        arr = data[mid]
        print(f"motherID: {mid}, N: {arr.shape[0]} (N_w: {arr[:, 7].sum():g})")
        px, py, pz = arr[:, 0], arr[:, 1], arr[:, 2]
        x, y, z = arr[:, 3], arr[:, 4], arr[:, 5]
        w = arr[:, 7] if args.weighted_plots else None
        pt = calc_pt(px, py)

        ax = hist2d(pz, pt, W=w)
        ax.set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
        ax.set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 13)
        plt.title(f'motherID: {mid}', fontsize=fontsize)
        savefig(f'momentum_id_{mid}.png')
        if show: plt.show()

        ax = plot_z_by_weight(z, arr[:, 7], weighted=args.weighted_plots)
        ax.axvline(target, color='red', linestyle='--', label='Target Position')
        ax.axvline(target + 1.595, color='red', linestyle='--')
        ax.set_xlabel('z [cm]', fontsize=14)
        ax.set_title(f'motherID: {mid}', fontsize=fontsize)
        ax.legend(fontsize=fontsize - 4)
        savefig(f'z_dist_id_{mid}.png')
        if show: plt.show()

        hist2d(x, y, W=w)
        plt.xlabel('x [cm]', fontsize=14)
        plt.ylabel('y [cm]', fontsize=14)
        circ = plt.Circle((0, 0), 15, color='r', fill=False)
        plt.gca().add_patch(circ)
        plt.title(f'motherID: {mid}', fontsize=fontsize)
        savefig(f'position_dist_id_{mid}.png')
        if show: plt.show()

        corner_plot([x, y, z, px, py, pz], labels, W=w)
        plt.suptitle(f'motherID: {mid}', fontsize=fontsize)
        savefig(f'corner_plot_id_{mid}.png')
        if show: plt.show()

    ##### weight composition per motherID (log-scale counts)
    plot_weight_composition(data)
    savefig('weight_composition_per_id.png')
    if show: plt.show()

    ##### raw vs weighted sample counts per motherID (log-scale counts)
    plot_sample_counts(data)
    savefig('sample_counts_per_id.png')
    if show: plt.show()
