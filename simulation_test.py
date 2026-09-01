"""Compare the simulation (transport) of real muons vs. muons from the flow.

For each weight group: read the real muons, generate the same number from the
model, simulate both, and print the number of hits for each. The total is the
per-group hits times the importance weight, summed over groups.

simulate() is a placeholder for the real (containerized) simulation.
"""
import argparse
import os
import numpy as np
import torch
from time import time

from utils import (get_real_muons_per_weight, generate_px_py,
                   generate_position, generate_pdg)
from models import NeuralSplineFlow
from cuda_muons_ship import run_from_params

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATA = '/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample_w.h5'
Z_FIXED = -0.1
BEAM_RADIUS = 5.0
SIGMA = 1.6
BATCH_SIZE = int(5e6)


def simulate(muons, seed = 0):
    """DUMMY: propagate the muons and return the number of hits (a scalar).
    Replace with the real simulation. muons is an (N, 7) array with columns
    [px, py, pz, x, y, z, pdg]."""
    PROJECTS_DIR = os.getenv('PROJECTS_DIR', '~/projects')
    phi = np.array([[0.0000, 115.0000, 40.0000, 40.0000, 119.0000, 119.0000, 61.5000, 61.5000, 61.5000, 61.5000, 50.0000, 50.0000, 0.0000, 0.0000, 1.9000],
            [15.0000, 151.5035, 63.1169, 63.1169, 10.0000, 10.0000, 8.1495, 9.1550, 64.1224, 63.1169, 69.4286, 69.4286, 0.0000, 0.0000, 1.9000],
            [15.0000, 224.9372, 68.0437, 68.0437, 10.0000, 10.0000, 8.4314, 8.5295, 70.6235, 70.5253, 74.8481, 74.8481, 0.0000, 0.0000, 1.9000],
            [16.1245, 225.0000, 64.8050, 64.8050, 13.9322, 13.9322, 8.9857, 9.1093, 85.8846, 85.7609, 71.2855, 71.2855, 0.0000, 0.0000, 1.9000],
            [15.0000, 223.9999, 39.0189, 39.0189, 11.0774, 11.0774, 8.3273, 8.9986, 97.5319, 96.8606, 46.2724, 46.2724, 0.0000, 0.0000, 1.9000],
            [15.0000, 120.3744, 20.0000, 20.0000, 30.0000, 30.0000, 0.0000, 0.0000, 102.7509, 102.7509, 22.0000, 22.0000, 0.0000, 0.0000, -0.0000],
            [15.0001, 168.0000, 20.0000, 30.0000, 40.0000, 40.0000, 24.9153, 9.3875, 97.1824, 102.7103, 33.0000, 33.0000, 0.0000, 0.0000, -1.2000],
            [15.0000, 193.0909, 37.9502, 67.8055, 47.7714, 47.7714, 8.0002, 8.0000, 124.7799, 94.9248, 74.5861, 74.5861, 0.0000, 0.0000, -1.9000]])
    output = run_from_params(
                phi,
                muons,
                sensitive_plane=[{'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 82},
                                 {'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 91}],
                simulate_fields= False,
                NI_from_B = True,
                use_diluted = True,
                add_cavern = True,
                return_all=False,
                histogram_dir=os.path.join(PROJECTS_DIR, 'MuonsAndMatter/cuda_muons/data'),
                seed=seed,
            )
    n_hits = len(output['px'])
    return n_hits

def build_muons(pz, pt):
    """Assemble a full (N, 7) muon array from generated (pz, pt), filling the
    remaining coordinates with the project's generators."""
    n = len(pz)
    px, py = generate_px_py(pt)
    x, y, z = generate_position(n, SmearBeamRadius = BEAM_RADIUS, sigma = SIGMA, fixed_z=Z_FIXED)
    pdg = generate_pdg(n)
    return np.stack([px, py, pz, x, y, z, pdg], axis=1)

def real_muons_per_weight(z_fixed:float = None, num_samples:int = None):
    muons = get_real_muons_per_weight(DEFAULT_DATA, num_samples=num_samples)
    if z_fixed is not None:
        for w in muons.keys():
            x,y,z = generate_position(len(muons[w]), SmearBeamRadius = BEAM_RADIUS, sigma = SIGMA, fixed_z=z_fixed)
            muons[w][:,3] += x
            muons[w][:,4] += y
            muons[w][:,5] = z
    return muons
    


def load_flows(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    groups = ckpt['groups']
    for w, e in groups.items():
        flow = NeuralSplineFlow.from_config(e['config'])
        flow.load_state_dict(e['state_dict'])
        e['flow'] = flow.to(device).eval()
    return groups


def sample_flow(entry, n_samples:int, batch_size:int=BATCH_SIZE):
    """Sample n (pz, pt) points from a group's flow, in physical units."""
    print(f"Sampling {n_samples} points from flow (batch_size={batch_size}) ...")
    mean = np.asarray(entry['mean']).ravel()
    std = np.asarray(entry['std']).ravel()
    batch_size = int(batch_size)
    flow = entry['flow'].to(device).eval()
    with torch.no_grad():
        samples = []
        batch_size = min(batch_size, n_samples)
        for i in range(int(n_samples/batch_size)):
            samples.append(flow.sample(batch_size).cpu().numpy())
        if n_samples % batch_size > 0:
            remainder = n_samples % batch_size
            samples.append(flow.sample(remainder).cpu().numpy())
    samples = np.concatenate(samples, axis=0)*std + mean 
    samples = np.clip(samples, a_min=0, a_max=390)  # ensure pz, pt >= 0
    return samples 


from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt

def compare_1d_histograms(real_sample, gen_sample, bins=120, range_vals=None, xlabel="Value", real_w=None):
    """Overlays 1D distributions of real and generated samples."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Auto-detect range if none is provided
    if range_vals is None:
        range_vals = (min(np.min(real_sample), np.min(gen_sample)), 
                      max(np.max(real_sample), np.max(gen_sample)))
        
    b = np.linspace(range_vals[0], range_vals[1], bins)
    
    # Plot marginals
    ax.hist(real_sample, bins=b, weights=real_w, density=True, 
            histtype='step', color='k', label='Real')
    ax.hist(gen_sample, bins=b, density=True, 
            histtype='step', color='C1', label='Generated')
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    return fig, ax

def plot_summary(weights, runs_real, runs_gen, path):
    """Bar chart of per-weight hits (mean +- std over runs), real vs generated."""
    labels = [f"{w:.2f}" for w in weights]
    rm = [np.mean(runs_real[w]) for w in weights]
    rs = [np.std(runs_real[w]) for w in weights]
    gm = [np.mean(runs_gen[w]) for w in weights]
    gs = [np.std(runs_gen[w]) for w in weights]
    x = np.arange(len(weights))
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(weights)), 5))
    ax.bar(x - 0.2, rm, 0.4, yerr=rs, capsize=3, label='real')
    ax.bar(x + 0.2, gm, 0.4, yerr=gs, capsize=3, label='generated')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('weight')
    ax.set_ylabel('hits')
    ax.set_title('Simulation hits per weight (mean +- std)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare simulation hits: real vs generated muons")
    parser.add_argument("--model", type=str, default="outputs/flow_model.pt")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--num_samples", type=int, default=None, help="Muons per group (None = all)")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of runs with different seeds")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    real = real_muons_per_weight(z_fixed=-0.1, num_samples=args.num_samples)
    groups = load_flows(args.model, device)
    weights = list(real.keys())

    # A single combined flow (train_flow.py or distill_flow.py) is saved as one
    # group keyed "combined" rather than one group per weight (train_flow_per_weight.py),
    # so its key can't be compared as a float. Same flow is then used for every weight.
    multiple_models = len(groups) > 1
    combined_entry = None if multiple_models else next(iter(groups.values()))

    # accumulate hits over runs with different seeds
    runs_real = {w: [] for w in weights}
    runs_gen = {w: [] for w in weights}
    runs_flat = {w: [] for w in weights} 
    runs_flat_under = {w: [] for w in weights}    
    totals_real, totals_gen, totals_flat, totals_flat_under = [], [], [], []

    # ---- Calculate Effective Sample Size (N_eff) ----
    sum_w_orig, sum_w2_orig = 0.0, 0.0
    sum_w_flat, sum_w2_flat = 0.0, 0.0
    sum_w_fu, sum_w2_fu = 0.0, 0.0

    total_physical_events = sum(arr.shape[0] * w for w, arr in real.items())
    total_original_samples = sum(arr.shape[0] for w, arr in real.items())
    global_flat_weight = total_physical_events / total_original_samples

    # N_eff depends only on sample sizes/weights per group, not on the simulation
    # run/seed, so it must be accumulated once here rather than once per run.
    for w, arr in real.items():
        n_original = arr.shape[0]
        n_flat = max(1, int(round(n_original * w)))
        physical_fraction = (n_original * w) / total_physical_events
        n_flat_under = max(1, int(round(total_original_samples * physical_fraction)))
        w_under = global_flat_weight

        sum_w_orig += n_original * w
        sum_w2_orig += n_original * (w**2)
        sum_w_flat += n_flat
        sum_w2_flat += n_flat * (1**2)
        sum_w_fu += n_flat_under * w_under
        sum_w2_fu += n_flat_under * (w_under**2)

    batch_simulation = int(100e6)

    for run in range(args.n_runs):
        print(f"\n########## Run {run + 1}/{args.n_runs} ##########")
        total_real, total_gen, total_flat, total_flat_under = 0.0, 0.0, 0.0, 0.0

        # ---- real muons: always simulated per weight group (the h5 file, not the
        # model, is what's split by weight) ----
        for w, arr in real.items():
            print(f"\nWeight {w:.4f} | N={arr.shape[0]}")
            print(f"Simulating real muons ...")
            hits_real = simulate(arr, seed = seed)
            runs_real[w].append(hits_real)
            print(f"hits real={hits_real:.4g}")
            total_real += hits_real * w
            seed += 231

        if multiple_models:
            # ---- per-weight generation: each weight group has its own flow, so
            # "generated"/"flattened"/"flattened-under" can be compared group by group ----
            for w, arr in real.items():
                n_original = arr.shape[0]
                entry = groups[min(groups, key=lambda k: abs(float(k) - w))]

                pz, pt = sample_flow(entry, n_original, batch_size=BATCH_SIZE).T
                assert pt.min() >= 0, "Negative pt generated!"
                assert pz.min() >= 0, "Negative pz generated!"
                p = np.sqrt(pz**2 + pt**2)
                assert p.max() <= 400, "Unphysically large momentum generated! {}".format(p.max())
                print(f"\nSimulating generated muons for weight {w:.4f} ...")
                hits_gen = simulate(build_muons(pz, pt), seed = seed)

                # flattened: generate N*weight muons, then count hits with weight 1
                n_flat = max(1, int(round(n_original * w)))
                pz, pt = sample_flow(entry, n_flat, batch_size=BATCH_SIZE).T
                print(f"\nSimulating {n_flat} flattened muons ...")
                hits_flat = 0
                for i in range(0, n_flat, batch_simulation):
                    pz_batch = pz[i:i+batch_simulation]
                    pt_batch = pt[i:i+batch_simulation]
                    hits_flat_batch = simulate(build_muons(pz_batch, pt_batch), seed = seed)
                    hits_flat += hits_flat_batch

                physical_fraction = (n_original * w) / total_physical_events
                n_flat_under = max(1, int(round(total_original_samples * physical_fraction)))
                w_under = global_flat_weight
                pz, pt = sample_flow(entry, n_flat_under, batch_size=BATCH_SIZE).T
                print(f"\nSimulating {n_flat_under} flattened-under muons ...")
                hits_flat_under = 0
                for i in range(0, n_flat_under, batch_simulation):
                    pz_batch = pz[i:i+batch_simulation]
                    pt_batch = pt[i:i+batch_simulation]
                    hits_flat_under_batch = simulate(build_muons(pz_batch, pt_batch), seed = seed)
                    hits_flat_under += hits_flat_under_batch

                runs_gen[w].append(hits_gen)
                runs_flat[w].append(hits_flat)
                runs_flat_under[w].append(hits_flat_under)
                print(f"hits gen={hits_gen:.4g}  gen_flat={hits_flat:.4g}")
                total_gen += hits_gen * w
                total_flat += hits_flat            # weight = 1 (rate is baked into the count)
                total_flat_under += hits_flat_under * (w_under)  # scale to same number of muons as real/gen
                seed += 231
        else:
            # ---- single combined flow: there's no per-weight density to match, so
            # generate directly at the total physical/original scale in one shot ----
            n_flat_total = max(1, int(round(total_physical_events)))
            pz, pt = sample_flow(combined_entry, n_flat_total, batch_size=BATCH_SIZE).T
            assert pt.min() >= 0, "Negative pt generated!"
            assert pz.min() >= 0, "Negative pz generated!"
            p = np.sqrt(pz**2 + pt**2)
            assert p.max() <= 400, "Unphysically large momentum generated! {}".format(p.max())
            print(f"\nSimulating {n_flat_total} flattened muons from the combined flow ...")
            hits_flat = 0
            for i in range(0, n_flat_total, batch_simulation):
                pz_batch = pz[i:i+batch_simulation]
                pt_batch = pt[i:i+batch_simulation]
                hits_flat_batch = simulate(build_muons(pz_batch, pt_batch), seed = seed)
                hits_flat += hits_flat_batch
            total_flat = hits_flat             # weight = 1 (rate is baked into the count)
            seed += 231

            n_flat_under = total_original_samples
            w_under = global_flat_weight
            pz, pt = sample_flow(combined_entry, n_flat_under, batch_size=BATCH_SIZE).T
            print(f"\nSimulating {n_flat_under} flattened-under muons from the combined flow ...")
            hits_flat_under = 0
            for i in range(0, n_flat_under, batch_simulation):
                pz_batch = pz[i:i+batch_simulation]
                pt_batch = pt[i:i+batch_simulation]
                hits_flat_under_batch = simulate(build_muons(pz_batch, pt_batch), seed = seed)
                hits_flat_under += hits_flat_under_batch
            total_flat_under = hits_flat_under * w_under
            total_gen = total_flat_under       # same N/weighting; no per-weight "generated" exists
            seed += 231

        totals_real.append(total_real)
        totals_gen.append(total_gen)
        totals_flat.append(total_flat)
        totals_flat_under.append(total_flat_under)

    # ---- summary: mean +- std over runs ----
    # per-weight rows are raw hit counts; TOTAL is the weighted sum for real/gen
    # (hits*weight) and the plain sum for the flattened column (weight 1).
    # A combined (non per-weight) model has no per-weight generated numbers to show,
    # only overall totals -- so the per-weight table is skipped in that case.
    if multiple_models:
        print("\n==== Hits per weight (mean +- std over runs) ====")
        print(f"{'weight':>10} | {'real':>20} | {'generated':>20} | {'gen flattened':>20} | {'gen flattened-under':>20}")
        for w in weights:
            r, g, f, fu = np.array(runs_real[w]), np.array(runs_gen[w]), np.array(runs_flat[w]), np.array(runs_flat_under[w])
            print(f"{w:>10.4f} | {r.mean():>10.4g} +- {r.std():<7.3g} | "
                  f"{g.mean():>10.4g} +- {g.std():<7.3g} | {f.mean():>10.4g} +- {f.std():<7.3g} | {fu.mean():>10.4g} +- {fu.std():<7.3g}")
        print()
        tr, tg, tf, tfu = np.array(totals_real), np.array(totals_gen), np.array(totals_flat), np.array(totals_flat_under)
        print(f"{'TOTAL':>10} | {tr.mean():>10.6g} +- {tr.std():<7.3g} | "
              f"{tg.mean():>10.6g} +- {tg.std():<7.3g} | {tf.mean():>10.6g} +- {tf.std():<7.3g} | {tfu.mean():>10.6g} +- {tfu.std():<7.3g}")
    else:
        print("\n==== Hits total (mean +- std over runs) -- single combined flow, no per-weight breakdown ====")
        tr, tf, tfu = np.array(totals_real), np.array(totals_flat), np.array(totals_flat_under)
        print(f"{'':>10} | {'real':>20} | {'gen flattened':>20} | {'gen flattened-under':>20}")
        print(f"{'TOTAL':>10} | {tr.mean():>10.6g} +- {tr.std():<7.3g} | "
              f"{tf.mean():>10.6g} +- {tf.std():<7.3g} | {tfu.mean():>10.6g} +- {tfu.std():<7.3g}")

    # Calculate and print N_eff row (real-data-only stat, independent of the model)
    neff_orig = (sum_w_orig**2) / sum_w2_orig if sum_w2_orig > 0 else 0
    neff_flat = (sum_w_flat**2) / sum_w2_flat if sum_w2_flat > 0 else 0
    neff_fu = (sum_w_fu**2) / sum_w2_fu if sum_w2_fu > 0 else 0
    if multiple_models:
        print(f"{'N_eff':>10} | {neff_orig:>20.4e} | {neff_orig:>20.4e} | {neff_flat:>20.4e} | {neff_fu:>20.4e}")
    else:
        print(f"{'N_eff':>10} | {neff_orig:>20.4e} | {neff_flat:>20.4e} | {neff_fu:>20.4e}")

    os.makedirs("outputs/figs", exist_ok=True)
    if multiple_models:
        plot_summary(weights, runs_real, runs_gen, "outputs/figs/simulation_hits.png")
        print("\nSaved plot to outputs/figs/simulation_hits.png")
    else:
        print("\nSkipping per-weight plot: combined flow has no per-weight generated numbers.")
