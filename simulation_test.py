
import argparse
import json
import os
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from generator import DEFAULT_MODEL, load_generator, resolve_model
from utils import DEFAULT_DATA, get_real_muons, generate_position

Z_FIXED = -0.1
BEAM_RADIUS = 5.0
SIGMA = 1.6
CM_TO_M = 0.01                 # the file's (and the generator's) x, y, z are in cm
SAMPLE_BATCH = int(5e6)        # muons generated per call
SIM_BATCH = int(5e8)           # muons built + simulated at a time
P_MAX = 400.0                  # the propagation's tables stop here: see out_of_bounds
# settings a --resume must share with the saved runs, or they would not be one sample
RESUME_MUST_MATCH = ("model", "data", "num_samples", "smear", "z_fixed", "seed", "keep_sample")
# with --keep_sample the generated sample is rebuilt batch by batch every run, so the
# batching must not change either, or it would not be the same sample
RESUME_MUST_MATCH_KEEP = ("sim_batch", "sample_batch")
OUT_NAME = {False: "simulation_hits", True: "simulation_hits_fixed_sample"}
MAX_REDRAWS = 20               # rounds of redrawing out-of-bounds muons before giving up

PHI = np.array([[0.0, 115.0, 40.0, 40.0, 119.0, 119.0, 61.5, 61.5, 61.5, 61.5, 50.0, 50.0, 0.0, 0.0, 1.9875], 
    [15, 150, 62, 62, 11, 11, 12, 12, 62, 62.0, 69, 69, 0, 0, 1.875], 
    [15, 225, 70, 70, 12, 12, 9, 9, 70, 70.0, 77, 77, 0, 0, 1.87],
    [16, 225, 64, 64, 14, 14, 8, 8, 83, 83, 71, 71, 0, 0, 1.9], 
    [15, 225, 42, 42, 13, 13, 8, 8, 109, 109, 47, 47, 0, 0, 1.89], 
    [16, 131, 20, 20, 30, 30, 0, 0, 110.0, 110.0, 22, 22, 0, 0, 0], 
    [15, 168, 20, 30, 30, 30, 22, 12, 108.0, 108.0, 33, 33, 0, 0, -1.32], 
    [19, 200, 38, 69, 51, 51, 8, 8, 125, 94, 76, 76, 0, 0, -2.03]])
PLANES = [{'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 82},
          {'dz': 0.01, 'dx': 4, 'dy': 6, 'position': 91}]


def simulate(muons, seed=0):
    """Propagate `muons` and return (sum of w, sum of w^2) over the hits on the
    sensitive planes.

    `muons` is (N, 7) [px, py, pz, x, y, z, pdg] -- every muon has w = 1, so both sums
    are the plain count -- or (N, 8) with an importance weight appended, which
    run_from_params carries along to the hits."""
    from cuda_muons_ship import run_from_params     # here, so --plot_only runs without it
    output = run_from_params(
        PHI, muons,
        sensitive_plane=PLANES,
        field_mode='uniform',
        NI_from_B=True,
        use_diluted=True,
        add_cavern=True,
        return_all=False,
        histogram_dir=os.path.join(os.getenv('PROJECTS_DIR', '~/projects'),
                                   'MuonsAndMatter/cuda_muons/data'),
        seed=seed,
    )
    if 'weight' in output:
        w = output['weight'].double()
        return float(w.sum()), float((w * w).sum())
    n = float(len(output['px']))
    return n, n


def to_metres(muons):
    """The production point (columns 3:6) from cm to m, in place."""
    muons[:, 3:6] *= CM_TO_M
    return muons


def smear_beam(muons, z_fixed=None):
    """Add the beam spot to the production position (in m), in place (columns 3:6), and
    put every muon at z = z_fixed. z_fixed None leaves the muons as they are."""
    if z_fixed is None:
        return muons
    x, y, z = generate_position(muons.shape[0], SmearBeamRadius=BEAM_RADIUS,
                                sigma=SIGMA, fixed_z=z_fixed)
    muons[:, 3] += x
    muons[:, 4] += y
    muons[:, 5] = z
    return muons


def load_real(path, num_samples=None, z_fixed=None):
    """The real sample as one (N, 8) float32 array, positions in m: the physical muon
    state plus each muon's own importance weight (utils.get_real_muons rescales the
    weights when --num_samples caps the rows, so a sub-sample still carries the full
    rate)."""
    return smear_beam(to_metres(get_real_muons(path, num_samples)), z_fixed)


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


def _draw(model, m, z_fixed, rng):
    """m generated muons as physical (m, 7) float32, positions in m."""
    return smear_beam(to_metres(model.sample(int(m), rng=rng)), z_fixed)


def sample_muons(model, n, rng, z_fixed=None, batch_size=SAMPLE_BATCH):
    """n in-bounds generated muons as a physical (n, 7) float32 array, batch by batch.
    Any muon outside what the simulation accepts (out_of_bounds) is redrawn: the
    density restricted to the accepted region, sampled exactly."""
    n = int(n)
    muons = np.empty((n, 7), dtype=np.float32)
    redrawn = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            m = min(batch_size, n - i)
            muons[i:i + m] = _draw(model, m, z_fixed, rng)
        for _ in range(MAX_REDRAWS):
            idx = np.flatnonzero(out_of_bounds(muons))
            if idx.size == 0:
                break
            redrawn += idx.size
            muons[idx] = _draw(model, idx.size, z_fixed, rng)
        else:
            check_physical(muons)       # still out of bounds after MAX_REDRAWS rounds
    if redrawn:
        print(f"  redrew {redrawn:,} of {n:,} muons ({redrawn / n:.2e}) that fell "
              f"outside pz >= 0, |p| <= {P_MAX}")
    return muons


def simulate_batched(muons_of, n_total, seed, batch=SIM_BATCH, label=""):
    """(sum of w, sum of w^2) over the hits of n_total muons, `batch` at a time.
    muons_of(start, n) supplies each batch, which is built, simulated and dropped before
    the next one exists, so the peak memory is a single batch regardless of n_total
    (~2.7e10 for the full count)."""
    hits, hits_w2, done, batch_i = 0.0, 0.0, 0, 0
    while done < n_total:
        n = int(min(batch, n_total - done))
        batch_i += 1
        t0 = time()
        h, h2 = simulate(muons_of(done, n), seed=seed + batch_i)
        hits += h
        hits_w2 += h2
        done += n
        print(f"  {label}batch {n:,} muons ({done:,}/{n_total:,}): "
              f"hits={h:.6g}  total={hits:.6g}  [{time() - t0:.0f}s]")
    return hits, hits_w2


COLORS = {"real": "k", "gen matched": "C1", "gen N_eff": "C3", "gen full count": "C2"}
# x-axis names (the sample size is added below)
NAMES = {"real": "Real data", "gen matched": "Gen model", "gen N_eff": "Gen model",
         "gen full count": "Gen model"}


def _sci(n):
    """n as a mathtext 'a x 10^b' (3 significant digits), plain below 10^4."""
    if n < 1e4:
        return f"{n:,.0f}"
    e = int(np.floor(np.log10(n)))
    return rf"${n / 10 ** e:.3g} \times 10^{{{e}}}$"


def plot_summary(runs, n_simulated, path, keep_sample=False, model_name="Gen model"):
    """The rate over runs of each estimate, on a log scale, in three side-by-side views:
    a box plot (median solid, mean dashed, quartiles, whiskers to 1.5 IQR, outliers as
    crosses), the mean with the std over runs as error bar, and every run as a dot.
    For `real` the runs only differ by the transport's randomness (the muons are the
    same every run), so its spread understates its uncertainty: the statistical error
    is in the table."""
    labels = list(runs)
    names = {k: model_name if v == "Gen model" else v for k, v in NAMES.items()}
    rates = [np.array([r for r, _ in runs[k]]) for k in labels]
    colors = [COLORS.get(k, "C4") for k in labels]
    x = np.arange(len(labels))
    jitter = np.random.default_rng(0)            # fixed, so a redraw looks the same
    fig, ax = plt.subplots(figsize=(1.5 + 2.4 * len(labels), 5))
    if "real" in runs:                           # a reference for the other columns
        ax.axhline(np.median(rates[labels.index("real")]), color="0.75", linestyle="--",
                   linewidth=1, zorder=0, label="real data median")
    box = ax.boxplot(rates, positions=x - 0.18, widths=0.3, patch_artist=True, showmeans=True,
                     meanline=True, medianprops=dict(color="k"),
                     meanprops=dict(color="k", linestyle="--"), flierprops=dict(marker="x", ms=4))
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    for i, (r, c) in enumerate(zip(rates, colors)):
        ax.errorbar(i + 0.1, r.mean(), yerr=r.std(), fmt="o", color=c, capsize=4, ms=5,
                    label="mean, std over runs" if i == 0 else None)
        ax.plot(i + 0.3 + jitter.uniform(-0.04, 0.04, len(r)), r, ".", color=c, alpha=0.6,
                ms=5, label="single runs" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{names.get(k, k)}\n({_sci(n_simulated[k])} muons"
                        + (", weighted)" if k == "real" else ")") for k in labels])
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel("hit rate")
    ax.set_title("Simulated hit rate over runs"
                 + (", same generated sample every run" if keep_sample else "")
                 + " (box: median solid, mean dashed)", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_results(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, path)                  # never leaves a half-written file behind


def config_kind(config):
    """The kind of generator a results file was made with: 'gmm' or 'flow'."""
    return "gmm" if config.get("gmm") else "flow"      # a bool, or a spec path in old files


def report(results, plot_path):
    """Print the table and draw the plot of a results dict (save_results' layout)."""
    print("\n==== Simulated hit rate ====")
    print("stat. error: one run's sqrt(sum w^2) over the hits; N_eff hits: "
          "(sum w)^2 / sum w^2 over the hits; spread: std over runs")
    print(f"{'estimate':>14} | {'runs':>4} | {'N simulated':>16} | {'N_eff input':>11} | "
          f"{'N_eff hits':>10} | {'rate':>12} | {'stat. error':>11} | {'spread':>9}")
    n_eff = {'real': results["n_eff_input"], 'gen matched': results["n_real"]}
    runs = {}
    for k, v in results["runs"].items():
        if not v:
            continue
        rate = np.array([r["rate"] for r in v])
        var = np.array([r["var"] for r in v])
        runs[k] = list(zip(rate, var))
        n_sim = results["n_simulated"][k]
        print(f"{k:>14} | {len(v):>4} | {n_sim:>16,} | {n_eff.get(k, n_sim):>11.4e} | "
              f"{np.mean(rate ** 2 / var):>10.4g} | {rate.mean():>12.6g} | "
              f"{np.sqrt(var.mean()):>11.3g} | {rate.std():>9.3g}")
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    plot_summary(runs, results["n_simulated"], plot_path,
                 results["config"].get("keep_sample", False),
                 "GMM model" if config_kind(results["config"]) == "gmm" else "Gen model")
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare simulation hits: real weighted muons vs generated muons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=None,
                        help=f"Generator: a flow checkpoint (.pt, train_flow.py), or with --gmm a GMM muon generator spec (.json, gmm/fit_z.py). None = {DEFAULT_MODEL[False]}, or {DEFAULT_MODEL[True]} with --gmm")
    parser.add_argument("--gmm", action="store_true",
                        help="--model is a GMM muon generator: results and plot go to "
                             "gmm/outputs/ by default, named *_gmm")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="HDF5 file of real muons")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Cap on real muons read (None = all)")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Simulation repetitions (different seeds) per estimate, in this job")
    parser.add_argument("--sim_batch", type=int, default=SIM_BATCH,
                        help="Muons generated + simulated at a time")
    parser.add_argument("--sample_batch", type=int, default=SAMPLE_BATCH,
                        help="Muons generated per call")
    parser.add_argument("--smear", action="store_true",
                        help="Add another beam spot to x, y and put every muon at --z_fixed "
                             "(the file's x, y already include the beam spot)")
    parser.add_argument("--z_fixed", type=float, default=Z_FIXED,
                        help="z [m] of every muon with --smear")
    parser.add_argument("--full_count", action="store_true",
                        help="Also run the 'gen full count' estimate (~2.7e10 muons per run: slow)")
    parser.add_argument("--keep_sample", action="store_true",
                        help="Simulate the same generated sample in every run (only the "
                             "transport seed changes, as for the real sample); saved to "
                             f"{OUT_NAME[True]}.json/.png by default")
    parser.add_argument("--results", default=None,
                        help=f"Numerical results, rewritten after every run (None = "
                             f"outputs/{OUT_NAME[False]}.json, or {OUT_NAME[True]}.json "
                             "with --keep_sample)")
    parser.add_argument("--resume", action="store_true",
                        help="Add --n_runs runs to those in --results (same settings, new seeds) "
                             "instead of starting over")
    parser.add_argument("--plot", default=None,
                        help=f"Summary plot (None = outputs/figs/{OUT_NAME[False]}.png, or "
                             f"{OUT_NAME[True]}.png with --keep_sample)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Only print the table and redraw the plot from --results")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument("--seed", type=int, default=0, help="Base simulation/sampling seed")
    args = parser.parse_args()
    args.model = resolve_model(args.model, args.gmm)
    out_dir, tag = ("gmm/outputs", "_gmm") if args.gmm else ("outputs", "")
    args.results = args.results or f"{out_dir}/{OUT_NAME[args.keep_sample]}{tag}.json"
    args.plot = args.plot or f"{out_dir}/figs/{OUT_NAME[args.keep_sample]}{tag}.png"
    if args.plot_only:
        with open(args.results) as f:
            report(json.load(f), args.plot)
        raise SystemExit

    previous = None
    if args.resume:
        if not os.path.exists(args.results):
            raise SystemExit(f"--resume: no {args.results} to continue")
        with open(args.results) as f:
            previous = json.load(f)
        must = RESUME_MUST_MATCH + (RESUME_MUST_MATCH_KEEP if args.keep_sample else ())
        if "model" not in previous["config"]:          # files from before --model
            previous["config"]["model"] = (previous["config"].get("gmm")
                                           or previous["config"].get("flow", parser.get_default("model")))
        saved = {k: previous["config"].get(k, parser.get_default(k)) for k in must}  # older files
        changed = {k: (saved[k], vars(args)[k]) for k in must if saved[k] != vars(args)[k]}
        if changed:
            raise SystemExit("--resume: these settings differ from the saved runs "
                             "(saved, now): " + ", ".join(f"{k} {v}" for k, v in changed.items()))

    np.random.seed(args.seed)                  # the beam smearing of the real sample
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    z_fixed = args.z_fixed if args.smear else None

    model = load_generator(args.model, device)   # fails fast
    real = load_real(args.data, args.num_samples, z_fixed)

    n_real = real.shape[0]
    sum_w = float(real[:, 7].sum(dtype=np.float64))    # ~2.7e10: accumulate in float64
    mean_w = sum_w / n_real                     # scales a weight-1 count into a rate
    n_full = int(round(sum_w))
    # Kish effective sample size: how many weight-1 muons the whole weighted sample is
    # worth. A property of the input only; the hit rate's own error is in the table.
    neff_real = sum_w ** 2 / float(np.square(real[:, 7], dtype=np.float64).sum())
    n_neff = int(round(neff_real))
    w_neff = sum_w / n_neff                     # the weight of each 'gen N_eff' muon
    print(f"Real sample: {n_real:,} muons, sum of weights {sum_w:.4g}, "
          f"N_eff {neff_real:.4g}")

    estimates = (['real', 'gen matched', 'gen N_eff']
                 + (['gen full count'] if args.full_count else []))
    if previous is None:
        results = {"config": vars(args), "n_real": n_real, "sum_w": sum_w, "mean_w": mean_w,
                   "n_eff_input": neff_real,
                   "n_simulated": {'real': n_real, 'gen matched': n_real,
                                   'gen N_eff': n_neff, 'gen full count': n_full},
                   "n_runs_done": 0, "jobs": [],
                   "runs": {}}                  # per estimate, per run: {rate, var, seed}
    else:
        results = previous
        results.setdefault("n_runs_done", len(results["runs"]["real"]))   # older files
        results.setdefault("jobs", [])
        results["n_simulated"]["gen N_eff"] = n_neff
        print(f"Resuming {args.results}: {results['n_runs_done']} runs done")
    for k in estimates:
        results["runs"].setdefault(k, [])
    first = results["n_runs_done"]
    results["jobs"].append({"first_run": first, "n_runs": args.n_runs,
                            "full_count": args.full_count})
    runs = results["runs"]

    def simulate_generated(n, seed, label, run, estimate):
        """Hits of n generated muons. Their randomness (numpy: charge and azimuths; torch:
        the flow's latent) is seeded from [seed, run, estimate], or from [seed, estimate]
        alone with --keep_sample, which rebuilds the very same sample in every run."""
        key = [args.seed, estimate] if args.keep_sample else [args.seed, run, estimate]
        rng = np.random.default_rng(key)
        torch.manual_seed(int(np.random.SeedSequence(key).generate_state(1)[0]))
        np.random.seed(np.random.SeedSequence(key).generate_state(1))   # --smear's beam spot
        return simulate_batched(lambda i, m: sample_muons(model, m, rng, z_fixed, args.sample_batch),
                                n, seed, args.sim_batch, label)

    for run in range(first, first + args.n_runs):
        print(f"\n########## Run {run + 1} ({run - first + 1}/{args.n_runs} of this job) ##########")
        # every run has its own seeds, a function of its index only, so a resumed job
        # never repeats a run: transport seeds seed + 1000 run + 0/231/462/693 (+ batch)
        seed = args.seed + 1000 * run

        print(f"Simulating the real sample ({n_real:,} weighted muons) ...")
        rate, var = simulate_batched(lambda i, n: real[i:i + n],
                                     n_real, seed, args.sim_batch, "real ")
        runs['real'].append({"rate": rate, "var": var, "seed": seed})

        print(f"\nSimulating {n_real:,} generated muons (matching the real count) ...")
        hits, _ = simulate_generated(n_real, seed + 231, "matched ", run, 1)
        runs['gen matched'].append({"rate": hits * mean_w, "var": hits * mean_w ** 2,
                                    "hits": hits, "seed": seed + 231})

        print(f"\nSimulating {n_neff:,} generated muons (the real sample's N_eff) ...")
        hits, _ = simulate_generated(n_neff, seed + 462, "N_eff ", run, 2)
        runs['gen N_eff'].append({"rate": hits * w_neff, "var": hits * w_neff ** 2,
                                  "hits": hits, "seed": seed + 462})

        if args.full_count:
            print(f"\nSimulating {n_full:,} generated muons (the full physical count), "
                  f"in batches of {args.sim_batch:,} ...")
            rate, var = simulate_generated(n_full, seed + 693, "full count ", run, 3)
            runs['gen full count'].append({"rate": rate, "var": var, "seed": seed + 693})

        results["n_runs_done"] = run + 1
        save_results(results, args.results)          # after every run: a crash keeps them
        print(f"Saved runs 1-{run + 1} to {args.results}")

    report(results, args.plot)
