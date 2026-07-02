import os
# Cap BLAS threads BEFORE importing numpy/sklearn (see train_gmm.py).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "64")

import argparse
import pickle
from time import time

import numpy as np
from matplotlib import pyplot as plt

from utils import train_val_split
from train_gmm import fit_gmm_grid, fit_gmm_batched, _mean_std_batched, sample_gmm

figs_dir = 'outputs/figs/'


def mixing_coefficients(models):
    """Combined model = sum_w alpha_w * p_w, with alpha_w proportional to the real
    muon rate of group w, i.e. (importance weight) * (number of samples)."""
    weights = np.array(list(models.keys()), dtype=float)
    counts = [models[w].get("n") for w in models.keys()]

    if any(c is None for c in counts):
        print("WARNING: per-group sample counts ('n') not found in the model file; "
              "falling back to mixing by importance weight only. Re-run train_gmm.py "
              "to store counts for an exact combination.")
        rate = weights
    else:
        rate = weights * np.array(counts, dtype=float)

    return rate / rate.sum()


def build_distill_dataset(models, total_samples, seed=0):
    """Draw samples from the combined weighted mixture of per-group GMMs.

    Fills a single preallocated (total_samples, 2) float32 array group-by-group
    (no per-group copies or vstack), so peak memory is just the dataset itself.
    Returns the (pz, pt) samples in original (physical) units and the mixing
    coefficients. EM is order-invariant, so the samples are left grouped (unshuffled).
    """
    np.random.seed(seed)                       # GMM.sample() draws from the global RNG
    alpha = mixing_coefficients(models)

    # allocate the sample budget across groups via a multinomial draw on alpha
    rng = np.random.default_rng(seed)
    n_per_group = rng.multinomial(total_samples, alpha)

    X = np.empty((total_samples, 2), dtype=np.float32)
    pos = 0
    for (weight, entry), n_w in zip(models.items(), n_per_group):
        if n_w <= 0:
            continue
        pz, pt, _ = sample_gmm(entry["gmm"], entry["mean"], entry["std"], n_samples=int(n_w))
        X[pos:pos + n_w, 0] = pz
        X[pos:pos + n_w, 1] = pt
        pos += n_w
        print(f"  weight {weight}: alpha={alpha[list(models).index(weight)]:.4g}, drew {n_w} samples")

    return X, alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill the per-weight GMMs into a single GMM")
    parser.add_argument("--models", type=str, default="outputs/gmm_model.pkl", help="Per-weight GMM file from train_gmm.py")
    parser.add_argument("--out", type=str, default="outputs/gmm_distilled.pkl", help="Output path for the distilled GMM")
    parser.add_argument("--total_samples", type=int, default=20_000_000, help="Samples drawn from the combined mixture for distillation")
    parser.add_argument("--components", type=int, nargs="+", default=[2, 4, 6, 8, 12, 16, 24, 32], help="Candidate component counts for the distilled GMM")
    parser.add_argument("--select", choices=["bic", "val_ll"], default="bic", help="Model-selection criterion (BIC favors fewer components)")
    parser.add_argument("--batch_size", type=int, default=5_000_000, help="Chunk size for streaming EM")
    parser.add_argument("--select_samples", type=int, default=2_000_000, help="Subsample size used to pick the best K")
    parser.add_argument("--max_iter", type=int, default=100, help="Max EM iterations for the full-data fit")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance on avg log-likelihood")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction for K selection")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.models, "rb") as f:
        models = pickle.load(f)

    n_src_components = sum(e["gmm"].n_components for e in models.values())
    print(f"Loaded {len(models)} per-weight GMMs, {n_src_components} components total.")

    # ---- 1) sample the combined weighted mixture ----
    print(f"\nSampling {args.total_samples} points from the combined mixture...")
    X, alpha = build_distill_dataset(models, args.total_samples, seed=args.seed)

    # standardization stats over all generated data (single streaming pass)
    mean, std = _mean_std_batched(X, args.batch_size)

    # ---- 2) choose K on a subsample using the in-memory sklearn grid ----
    n = X.shape[0]
    m = min(n, args.select_samples)
    if n > m:
        idx = np.sort(np.random.default_rng(args.seed).choice(n, size=m, replace=False))
        Xsub = np.asarray(X[idx], dtype=np.float64)
    else:
        Xsub = np.asarray(X, dtype=np.float64)
    Xsub = (Xsub - mean) / std
    X_train, X_val = train_val_split(Xsub, val_frac=args.val_frac, seed=args.seed)

    print(f"\nSelecting K on {m} samples...")
    best, results = fit_gmm_grid(X_train, X_val, n_components_list=args.components, random_state=args.seed)
    chosen = min(results, key=lambda r: r["bic"]) if args.select == "bic" else best
    K = chosen["K"]
    print(f"  Selected K={K} (val_ll={chosen['val_ll']:.6f}, BIC={chosen['bic']:.1f}) "
          f"-- reducing from {n_src_components} to {K} components.")

    # ---- 3) final fit on ALL generated samples via streaming EM ----
    print(f"\nFitting distilled GMM on all {n} samples (batched)...")
    t0 = time()
    init = (chosen["model"].weights_, chosen["model"].means_, chosen["model"].covariances_)
    gmm, train_ll = fit_gmm_batched(
            X, mean, std, K,
            batch_size=args.batch_size, init=init,
            max_iter=args.max_iter, tol=args.tol, seed=args.seed,
        )
    print(f"Full-data fit complete in {time() - t0:.1f}s | avg_ll={train_ll:.6f}")

    # ---- 4) save (flat format, directly usable by test_gmm.py) ----
    with open(args.out, "wb") as f:
        pickle.dump({"gmm": gmm, "mean": mean, "std": std}, f)
    print(f"Saved distilled GMM to {args.out}")

    # ---- 5) quick sanity plot of the distilled samples ----
    pz_gen, pt_gen, _ = sample_gmm(gmm, mean, std, n_samples=200_000, seed=args.seed)
    from see_data import hist2d
    fontsize = 14
    ax = hist2d(pz_gen, pt_gen)
    ax.set_xlabel('$P_z$ [GeV]', fontsize=fontsize)
    ax.set_ylabel('$P_t$ [GeV]', fontsize=fontsize)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 13)
    plt.title(f'Distilled GMM (K={chosen["K"]})', fontsize=fontsize)
    plt.savefig(figs_dir + 'distilled_momentum.png')
