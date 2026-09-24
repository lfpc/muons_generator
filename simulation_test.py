"""Compare the simulation (transport) of the real muon sample vs. muons from the flow.

A single model is used for everything, so what is compared are three estimates of the
same physical quantity -- the muon rate hitting the sensitive planes --
each repeated --n_runs times with a different simulation seed (mean +- std):

  real         the whole real sample, carried through the simulation with each muon's
               own importance weight; the "hits" number is the weighted sum, so it
               already is the rate.
  gen matched  N muons drawn from the flow, N = number of real muons. Every generated
               muon carries weight 1, so its raw hit count is multiplied by the mean
               real weight (sum of weights / N) to become a rate.
  gen weighted the flow sampled at the full physical count, sum(weights). Unweighted, hence a rate directly. That many muons never exist at once:
               they are generated, simulated and accumulated in batches of
               --sim_batch (the reason the un-batched version got OOM-killed).

Generated muons are restricted to what the propagation accepts (pz >= 0, |p| <= P_MAX):
the flow's tails reach past the data's, and a muon beyond the
kernel's momentum tables shows up as a CUDA "illegal memory access". The offenders --
~2e-7 of them -- are redrawn rather than clamped; see sample_muons.

simulate() wraps the real (CUDA) transport.
"""
import argparse
import os
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from utils import DEFAULT_DATA, get_real_muons, generate_px_py, generate_position, generate_pdg
from models import load_flow, sample_flow
from test_momenta import momentum_model
from train_flow_complete import to_physical
from cuda_muons_ship import run_from_params

Z_FIXED = -0.1
BEAM_RADIUS = 5.0
SIGMA = 1.6
SAMPLE_BATCH = int(5e6)        # flow samples drawn per forward pass
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


def load_model(path, device):
    """The flow of a checkpoint, as a dict with the flow, the checkpoint, `physical`
    (a batch of its samples -> physical (px, py, pz, x, y, z, pdg) float32) and whether
    the flow provides the position.

    A complete-state flow (train_flow_complete.py) is undone by its own to_physical. A
    momentum-only one (train_flow_momenta.py, train_flow_ploss.py) gives (pz, pt): the
    azimuth is drawn uniformly, the charge at random, and the position is left at the
    origin for smear_beam to fill in."""
    flow, ckpt = load_flow(path, device)
    if 'r' in ckpt['columns']:
        physical, has_position = (lambda s: to_physical(s, ckpt)), True
    else:
        model = momentum_model(ckpt)
        if model is None:
            raise SystemExit(f"{path} was fit on {ckpt['columns']}; this script does not "
                             "know how to turn that back into a muon.")

        def physical(s):
            pz, pt = model.to_momenta(s, ckpt).T
            px, py = generate_px_py(pt)
            zeros = np.zeros(pz.shape[0])
            return np.column_stack([px, py, pz, zeros, zeros, zeros,
                                    generate_pdg(pz.shape[0])]).astype(np.float32)
        has_position = False
    print(f"Loaded flow on {device} (columns: {' '.join(ckpt['columns'])}; "
          f"{'position from the model' if has_position else 'position faked'}).")
    return {'flow': flow, 'ckpt': ckpt, 'physical': physical, 'has_position': has_position}


def out_of_bounds(muons):
    """Mask of muons the propagation cannot take: pz negative, |p| above P_MAX, or a
    non-finite component.

    Such a muon indexes the CUDA kernel's momentum-loss tables out of bounds, and the
    failure surfaces later as an "illegal memory access" with no hint of which muon
    caused it. The flow produces them even though the data has none (real |p| tops out
    at 386): it is a continuous density whose tails reach past the training range."""
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
    """m muons straight from the flow, as physical (m, 7) float32 with the beam spot
    already added -- no bounds enforced."""
    muons = sample_flow(model['flow'], model['ckpt'], int(m), model['physical'])
    # a flow without a position of its own gets the beam spot, --no_smear or not
    return smear_beam(muons, Z_FIXED if z_fixed is None and not model['has_position']
                      else z_fixed)


def sample_muons(model, n, z_fixed=Z_FIXED, batch_size=SAMPLE_BATCH):
    """Draw n in-bounds muons from the flow as a physical (n, 7) float32 array.

    Each batch is converted and written into the output array right away, so nothing
    beyond the result itself is held at full size (n is up to a sim batch, i.e. hundreds
    of millions). The few muons that land outside what the simulation accepts (see
    out_of_bounds) are then redrawn until they don't: that is the flow's density
    restricted to the region of interest, sampled exactly -- clamping them to |p| = P_MAX
    instead would pile a spike of muons onto the boundary, and lowering the sampling
    temperature would shrink every tail, not just the part that is out of range. Only
    ~2e-7 of the muons need it, so a couple of rounds are enough."""
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
    parser.add_argument("--model", type=str, default="outputs/flow_model_complete.pt",
                        help="Checkpoint holding the single flow")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="HDF5 file of real muons")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Cap on real muons read (None = all)")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Simulation repetitions (different seeds) per estimate")
    parser.add_argument("--sim_batch", type=int, default=SIM_BATCH,
                        help="Muons generated + simulated at a time")
    parser.add_argument("--sample_batch", type=int, default=SAMPLE_BATCH,
                        help="Flow samples per forward pass")
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

    model = load_model(args.model, device)     # before the data: fails in a second
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
