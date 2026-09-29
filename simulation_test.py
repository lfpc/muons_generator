"""Compare the simulation (transport) of the real muon sample vs. generated muons.

The generator (generator.py) is used for everything, so what is compared are three
estimates of the same physical quantity -- the muon rate hitting the sensitive planes --
each repeated --n_runs times with a different simulation seed (mean +- std):

  real         the whole real sample, carried through the simulation with each muon's
               own importance weight; the "hits" number is the weighted sum, so it
               already is the rate.
  gen matched  N generated muons, N = number of real muons. Every generated
               muon carries weight 1, so its raw hit count is multiplied by the mean
               real weight (sum of weights / N) to become a rate.
  gen weighted generated at the full physical count, sum(weights), unweighted, hence a
               rate directly. That many muons never exist at once:
               they are generated, simulated and accumulated in batches of
               --sim_batch (the reason the un-batched version got OOM-killed).

Generated muons are checked against what the propagation accepts (pz >= 0,
|p| <= P_MAX): a muon beyond the kernel's momentum tables shows up as a CUDA "illegal
memory access". The generator's bounded momentum keeps every muon inside by
construction; any offender would be redrawn rather than clamped (sample_muons).

simulate() wraps the real (CUDA) transport.
"""
import argparse
import os
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from generator import MuonGenerator
from utils import DEFAULT_DATA, get_real_muons, generate_position
from cuda_muons_ship import run_from_params

Z_FIXED = -0.1
BEAM_RADIUS = 5.0
SIGMA = 1.6
SAMPLE_BATCH = int(5e6)        # muons generated per call
SIM_BATCH = int(5e8)           # muons built + simulated at a time
P_MAX = 400.0                  # the propagation's tables stop here: see out_of_bounds
MAX_REDRAWS = 20               # rounds of redrawing out-of-bounds muons before giving up

PHI = np.array([
    [0.0000, 115.0000, 40.0000, 40.0000, 119.0000, 119.0000, 61.5000, 61.5000, 61.5000, 61.5000, 50.0000, 50.0000, 0.0000, 0.0000, 1.9000],
    [15.0000, 151.5035, 63.1169, 63.1169, 10.0000, 10.0000, 8.1495, 9.1550, 64.1224, 63.1169, 69.4286, 69.4286, 0.0000, 0.0000, 1.9000],
    [15.0000, 224.9372, 68.0437, 68.0437, 10.0000, 10.0000, 8.4314, 8.5295, 70.6235, 70.5253, 74.8481, 74.8481, 0.0000, 0.0000, 1.9000],
    [16.1245, 225.0000, 64.8050, 64.8050, 13.9322, 13.9322, 8.9857, 9.1093, 85.8846, 85.7609, 71.2855, 71.2855, 0.0000, 0.0000, 1.9000],
    [15.0000, 223.9999, 39.0189, 39.0189, 11.0774, 11.0774, 8.3273, 8.9986, 97.5319, 96.8606, 46.2724, 46.2724, 0.0000, 0.0000, 1.9000],
    [15.0000, 120.3744, 20.0000, 20.0000, 30.0000, 30.0000, 0.0000, 0.0000, 102.7509, 102.7509, 22.0000, 22.0000, 0.0000, 0.0000, -0.0000],
    [15.0001, 168.0000, 20.0000, 30.0000, 40.0000, 40.0000, 24.9153, 9.3875, 97.1824, 102.7103, 33.0000, 33.0000, 0.0000, 0.0000, -1.2000],
    [15.0000, 193.0909, 37.9502, 67.8055, 47.7714, 47.7714, 8.0002, 8.0000, 124.7799, 94.9248, 74.5861, 74.5861, 0.0000, 0.0000, -1.9000]])
PLANES = [{'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 82},
          {'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 91}]


def simulate(muons, seed=0):
    """Propagate `muons` and return the number of hits on the sensitive planes.

    `muons` is (N, 7) [px, py, pz, x, y, z, pdg] -- the return is a plain count -- or
    (N, 8) with an importance weight appended, in which case run_from_params carries the
    weights along and the return is the weighted sum of hits, i.e. a rate."""
    output = run_from_params(
        PHI, muons,
        sensitive_plane=PLANES,
        simulate_fields=False,
        NI_from_B=True,
        use_diluted=True,
        add_cavern=True,
        return_all=False,
        histogram_dir=os.path.join(os.getenv('PROJECTS_DIR', '~/projects'),
                                   'MuonsAndMatter/cuda_muons/data'),
        seed=seed,
    )
    if 'weight' in output:
        return float(output['weight'].sum())
    return float(len(output['px']))


def smear_beam(muons, z_fixed=Z_FIXED):
    """Add the beam spot to the production position, in place (columns 3:6)."""
    if z_fixed is None:
        return muons
    x, y, z = generate_position(muons.shape[0], SmearBeamRadius=BEAM_RADIUS,
                                sigma=SIGMA, fixed_z=z_fixed)
    muons[:, 3] += x
    muons[:, 4] += y
    muons[:, 5] = z
    return muons


def load_real(path, num_samples=None, z_fixed=Z_FIXED):
    """The real sample as one (N, 8) float32 array: the physical muon state plus each
    muon's own importance weight (utils.get_real_muons rescales the weights when
    --num_samples caps the rows, so a sub-sample still carries the full rate)."""
    return smear_beam(get_real_muons(path, num_samples), z_fixed)


def out_of_bounds(muons):
    """Mask of muons the propagation cannot take: pz negative, |p| above P_MAX, or a
    non-finite component.

    Such a muon indexes the CUDA kernel's momentum-loss tables out of bounds, and the
    failure surfaces later as an "illegal memory access" with no hint of which muon
    caused it."""
    px, py, pz = muons[:, 0], muons[:, 1], muons[:, 2]
    p2 = px.astype(np.float64) ** 2 + py ** 2 + pz ** 2
    return ~np.isfinite(p2) | (pz < 0) | (p2 > P_MAX ** 2)


def check_physical(muons, label="generated"):
    """Assert that no muon is out_of_bounds, naming the offenders if any are."""
    bad = out_of_bounds(muons)
    n_bad = int(bad.sum())
    if n_bad:
        finite = np.isfinite(muons[:, :3]).all(axis=1)
        ok = bad & finite      # the non-finite rows have no range worth printing
        p = np.linalg.norm(muons[:, :3].astype(np.float64), axis=1)
        where = (f"pz in [{muons[ok, 2].min():.4g}, {muons[ok, 2].max():.4g}], "
                 f"|p| up to {p[ok].max():.4g}, " if ok.any() else "")
        raise AssertionError(
            f"{n_bad:,} of {muons.shape[0]:,} {label} muons are outside what the "
            f"simulation accepts (pz >= 0, |p| <= {P_MAX}, all finite): "
            f"{where}{int((~finite).sum()):,} non-finite")


def _draw(model, m, z_fixed):
    """m generated muons as physical (m, 7) float32, beam spot added."""
    return smear_beam(model.sample(int(m)), z_fixed)


def sample_muons(model, n, z_fixed=Z_FIXED, batch_size=SAMPLE_BATCH):
    """n in-bounds generated muons as a physical (n, 7) float32 array, batch by batch.
    Any muon outside what the simulation accepts (out_of_bounds) is redrawn: the
    density restricted to the accepted region, sampled exactly."""
    n = int(n)
    muons = np.empty((n, 7), dtype=np.float32)
    redrawn = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            m = min(batch_size, n - i)
            muons[i:i + m] = _draw(model, m, z_fixed)
        for _ in range(MAX_REDRAWS):
            idx = np.flatnonzero(out_of_bounds(muons))
            if idx.size == 0:
                break
            redrawn += idx.size
            muons[idx] = _draw(model, idx.size, z_fixed)
        else:
            check_physical(muons)       # still out of bounds after MAX_REDRAWS rounds
    if redrawn:
        print(f"  redrew {redrawn:,} of {n:,} muons ({redrawn / n:.2e}) that fell "
              f"outside pz >= 0, |p| <= {P_MAX}")
    return muons


def simulate_batched(muons_of, n_total, seed, batch=SIM_BATCH, label=""):
    """Hits over n_total muons, `batch` at a time. muons_of(start, n) supplies each
    batch, which is built, simulated and dropped before the next one exists, so the peak
    memory is a single batch regardless of n_total (~1.8e10 for the weighted count)."""
    hits, done, batch_i = 0.0, 0, 0
    while done < n_total:
        n = int(min(batch, n_total - done))
        batch_i += 1
        t0 = time()
        h = simulate(muons_of(done, n), seed=seed + batch_i)
        hits += h
        done += n
        print(f"  {label}batch {n:,} muons ({done:,}/{n_total:,}): "
              f"hits={h:.6g}  total={hits:.6g}  [{time() - t0:.0f}s]")
    return hits


def plot_summary(runs, path):
    """Bar chart of the rate estimates (mean +- std over runs)."""
    labels = list(runs)
    means = [np.mean(runs[k]) for k in labels]
    stds = [np.std(runs[k]) for k in labels]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(np.arange(len(labels)), means, 0.6, yerr=stds, capsize=4,
           color=['k', 'C1', 'C2'][:len(labels)])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('hits (rate)')
    ax.set_title('Simulated hit rate (mean +- std over runs)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare simulation hits: real weighted muons vs generated muons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--momentum_flow", default="outputs/momentum_flow.pt", help="Stage-1 checkpoint")
    parser.add_argument("--position_flow", default="outputs/position_flow.pt", help="Stage-2 checkpoint")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="HDF5 file of real muons")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Cap on real muons read (None = all)")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Simulation repetitions (different seeds) per estimate")
    parser.add_argument("--sim_batch", type=int, default=SIM_BATCH,
                        help="Muons generated + simulated at a time")
    parser.add_argument("--sample_batch", type=int, default=SAMPLE_BATCH,
                        help="Muons generated per call")
    parser.add_argument("--z_fixed", type=float, default=Z_FIXED,
                        help="z of the smeared beam spot; use --no_smear to keep the file's position")
    parser.add_argument("--no_smear", action="store_true",
                        help="Do not add the beam spot to the production position")
    parser.add_argument("--skip_weighted", action="store_true",
                        help="Skip the full weighted-count generation (the expensive estimate)")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument("--seed", type=int, default=0, help="Base simulation/sampling seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    z_fixed = None if args.no_smear else args.z_fixed

    model = MuonGenerator(args.momentum_flow, args.position_flow, device)   # fails fast
    real = load_real(args.data, args.num_samples, z_fixed)

    n_real = real.shape[0]
    sum_w = float(real[:, 7].sum(dtype=np.float64))    # ~1.8e10: accumulate in float64
    mean_w = sum_w / n_real                     # scales a weight-1 count into a rate
    n_weighted = int(round(sum_w))
    # effective sample size: how many unweighted muons the weighted sample is worth
    neff_real = sum_w ** 2 / float(np.square(real[:, 7], dtype=np.float64).sum())

    runs = {'real': [], 'gen matched': [], 'gen weighted': []}
    seed = args.seed
    for run in range(args.n_runs):
        print(f"\n########## Run {run + 1}/{args.n_runs} ##########")

        print(f"Simulating the real sample ({n_real:,} weighted muons) ...")
        rate_real = simulate_batched(lambda i, n: real[i:i + n],
                                     n_real, seed, args.sim_batch, "real ")
        runs['real'].append(rate_real)
        seed += 231

        print(f"\nSimulating {n_real:,} generated muons (matching the real count) ...")
        hits_matched = simulate_batched(
            lambda i, n: sample_muons(model, n, z_fixed, args.sample_batch),
            n_real, seed, args.sim_batch, "matched ")
        runs['gen matched'].append(hits_matched * mean_w)
        seed += 231

        if args.skip_weighted:
            continue
        print(f"\nSimulating {n_weighted:,} generated muons (the full weighted count), "
              f"in batches of {args.sim_batch:,} ...")
        hits_weighted = simulate_batched(
            lambda i, n: sample_muons(model, n, z_fixed, args.sample_batch),
            n_weighted, seed, args.sim_batch, "weighted ")
        runs['gen weighted'].append(hits_weighted)
        seed += 231

    if args.skip_weighted:
        runs.pop('gen weighted')

    print("\n==== Simulated hit rate (mean +- std over runs) ====")
    print(f"{'estimate':>14} | {'N simulated':>16} | {'N_eff':>11} | {'rate':>24}")
    n_sim = {'real': n_real, 'gen matched': n_real, 'gen weighted': n_weighted}
    n_eff = {'real': neff_real, 'gen matched': n_real, 'gen weighted': n_weighted}
    for k, v in runs.items():
        v = np.array(v)
        print(f"{k:>14} | {n_sim[k]:>16,} | {n_eff[k]:>11.4e} | "
              f"{v.mean():>12.6g} +- {v.std():<9.3g}")

    os.makedirs("outputs/figs", exist_ok=True)
    plot_summary(runs, "outputs/figs/simulation_hits.png")
    print("\nSaved plot to outputs/figs/simulation_hits.png")
