import os
# Cap BLAS threads BEFORE importing numpy/sklearn: OpenBLAS was built for a
# max of 128 threads and segfaults if more cores are available.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "64")

import numpy as np
from time import time
import torch
from utils import *
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from scipy.linalg import solve_triangular

figs_dir = 'outputs/figs/'
show = False


def fit_gmm_grid(X_train_s, X_val_s, n_components_list, random_state=0):
    results = []

    for K in n_components_list:
        print(f"Fitting GMM with K={K} components...")
        t0 = time()
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            reg_covar=1e-5,
            max_iter=200,
            n_init=3,
            init_params="kmeans",
            random_state=random_state,
            verbose=0,
        )
        gmm.fit(X_train_s)
        print(f"Fitting complete. Time taken: {time() - t0:.3f} seconds")
        train_ll = gmm.score(X_train_s)   # average log-likelihood
        val_ll = gmm.score(X_val_s)       # average log-likelihood
        bic = gmm.bic(X_val_s)
        aic = gmm.aic(X_val_s)

        results.append({
            "K": K,
            "model": gmm,
            "train_ll": train_ll,
            "val_ll": val_ll,
            "bic": bic,
            "aic": aic,
        })

        print(
            f"K={K:2d} | train_ll={train_ll:.6f} | "
            f"val_ll={val_ll:.6f} | BIC={bic:.1f} | AIC={aic:.1f}"
        )

    def pick_best_simple(results, delta=0.01):
        best_val_ll = max(d["val_ll"] for d in results)
        candidates = [d for d in results if d["val_ll"] >= best_val_ll - delta]
        return min(candidates, key=lambda d: d["K"])
    best = max(results, key=lambda d: d["val_ll"])
    return best, results


# ----------------------------------------------------------------------------
# Memory-efficient (streaming) GMM training.
#
# EM for a GMM only needs, per component k, the summed sufficient statistics
#   N_k   = sum_n r_nk
#   Sx_k  = sum_n r_nk x_n
#   Sxx_k = sum_n r_nk x_n x_n^T
# The E-step is independent per sample, so we can stream the data in chunks,
# accumulate these sums, and do the M-step from them. The result is identical
# to a full-batch fit, but peak memory is O(batch_size * K) instead of O(N * K).
# ----------------------------------------------------------------------------

def _iter_batches(X, mean, std, batch_size):
    """Yield standardized float64 chunks of X without copying the whole array."""
    for start in range(0, X.shape[0], batch_size):
        Xb = np.asarray(X[start:start + batch_size], dtype=np.float64)
        yield (Xb - mean) / std


def _mean_std_batched(X, batch_size):
    """Population mean/std over all of X in a single streaming pass."""
    d = X.shape[1]
    s = np.zeros(d)
    ss = np.zeros(d)
    n = 0
    for start in range(0, X.shape[0], batch_size):
        Xb = np.asarray(X[start:start + batch_size], dtype=np.float64)
        s += Xb.sum(axis=0)
        ss += (Xb * Xb).sum(axis=0)
        n += Xb.shape[0]
    mean = s / n
    var = np.maximum(ss / n - mean ** 2, 0.0)
    std = np.sqrt(var) + 1e-12
    return mean.reshape(1, -1), std.reshape(1, -1)


def _log_gaussian_full(X, means, covs):
    """Per-component log N(x | mean_k, cov_k) for full covariances. (n, K)."""
    n, d = X.shape
    K = means.shape[0]
    out = np.empty((n, K))
    half_log_2pi = 0.5 * d * np.log(2.0 * np.pi)
    for k in range(K):
        L = np.linalg.cholesky(covs[k])
        diff = (X - means[k]).T                         # (d, n)
        sol = solve_triangular(L, diff, lower=True)     # (d, n)
        maha = np.einsum('dn,dn->n', sol, sol)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        out[:, k] = -(half_log_2pi + 0.5 * log_det + 0.5 * maha)
    return out


def _precisions_cholesky_full(covs):
    """Cholesky of the precisions, matching sklearn's GaussianMixture attribute."""
    K, d, _ = covs.shape
    prec_chol = np.empty_like(covs)
    for k in range(K):
        L = np.linalg.cholesky(covs[k])
        prec_chol[k] = solve_triangular(L, np.eye(d), lower=True).T
    return prec_chol


def _pack_gmm(weights, means, covs, reg_covar, n_iter, ll):
    """Wrap fitted params into a sklearn GaussianMixture so .sample()/.score() work."""
    K, d = means.shape
    gmm = GaussianMixture(n_components=K, covariance_type="full", reg_covar=reg_covar)
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covs
    gmm.precisions_cholesky_ = _precisions_cholesky_full(covs)
    gmm.converged_ = True
    gmm.n_iter_ = n_iter
    gmm.lower_bound_ = ll
    return gmm


def _init_from_subsample(X, mean, std, K, reg_covar, seed, max_subsample=1_000_000):
    """Initialize means/covs/weights by fitting sklearn on a standardized subsample."""
    n = X.shape[0]
    if n > max_subsample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_subsample, replace=False))
        Xs = np.asarray(X[idx], dtype=np.float64)
    else:
        Xs = np.asarray(X, dtype=np.float64)
    Xs = (Xs - mean) / std
    g = GaussianMixture(n_components=K, covariance_type="full", reg_covar=reg_covar,
                        max_iter=100, n_init=1, init_params="kmeans", random_state=seed)
    g.fit(Xs)
    return g.weights_.copy(), g.means_.copy(), g.covariances_.copy()


def fit_gmm_batched(X, mean, std, n_components, batch_size=5_000_000, init=None,
                    max_iter=100, tol=1e-4, reg_covar=1e-6, seed=0, verbose=True):
    """Full-data GMM fit via streaming EM. X is standardized per-batch with mean/std.

    Returns a sklearn GaussianMixture (params live in standardized space) and the
    final average log-likelihood.
    """
    d = X.shape[1]
    K = n_components
    if init is None:
        weights, means, covs = _init_from_subsample(X, mean, std, K, reg_covar, seed)
    else:
        weights, means, covs = (np.array(init[0], dtype=np.float64),
                                np.array(init[1], dtype=np.float64),
                                np.array(init[2], dtype=np.float64))

    lower_bound = -np.inf
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        Nk = np.zeros(K)
        Sx = np.zeros((K, d))
        Sxx = np.zeros((K, d, d))
        ll_sum = 0.0
        total_n = 0
        log_w = np.log(weights)

        # ---- E-step + sufficient-statistic accumulation, streamed over chunks ----
        for Xb in _iter_batches(X, mean, std, batch_size):
            log_prob = _log_gaussian_full(Xb, means, covs) + log_w
            log_norm = logsumexp(log_prob, axis=1)
            ll_sum += float(log_norm.sum())
            total_n += Xb.shape[0]
            resp = np.exp(log_prob - log_norm[:, None])         # (b, K)
            Nk += resp.sum(axis=0)
            for k in range(K):
                rk = resp[:, k]
                Sx[k] += rk @ Xb
                Sxx[k] += (Xb * rk[:, None]).T @ Xb

        # ---- M-step from accumulated statistics ----
        Nk_safe = np.maximum(Nk, 1e-12)
        weights = Nk / total_n
        means = Sx / Nk_safe[:, None]
        for k in range(K):
            covs[k] = Sxx[k] / Nk_safe[k] - np.outer(means[k], means[k])
            covs[k].flat[::d + 1] += reg_covar              # add reg to diagonal

        avg_ll = ll_sum / total_n
        change = avg_ll - lower_bound
        if verbose:
            print(f"    [batched EM] iter {it:3d} | avg_ll={avg_ll:.6f} | delta={change:.2e}")
        lower_bound = avg_ll
        if abs(change) < tol:
            if verbose:
                print(f"    [batched EM] converged at iter {it}")
            break

    return _pack_gmm(weights, means, covs, reg_covar, n_iter, lower_bound), lower_bound


def sample_gmm(gmm, mean, std, n_samples, seed=0):
    Xgen_s, comp = gmm.sample(n_samples)
    Xgen = Xgen_s * std + mean

    pz_gen = Xgen[:, 0]
    pt_gen = Xgen[:, 1]
    #pt_gen = np.expm1(Xgen[:, 1])
    #pt_gen = np.clip(pt_gen, 0.0, None)

    return pz_gen, pt_gen, comp

if __name__ == "__main__":
    import argparse
    import pickle
    parser = argparse.ArgumentParser(description="Train GMM on muon data")
    parser.add_argument("--num_samples", type=int, default=None, help="Max samples per weight group to load (None = all)")
    parser.add_argument("--components", type=int, nargs="+", default=[2,4,8,16,24], help="List of number of GMM components to try")
    parser.add_argument("--batch_size", type=int, default=5_000_000, help="Chunk size for streaming EM")
    parser.add_argument("--select_samples", type=int, default=2_000_000, help="Subsample size used to pick the best K")
    parser.add_argument("--max_iter", type=int, default=100, help="Max EM iterations for the full-data fit")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance on avg log-likelihood")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction for K selection")
    args = parser.parse_args()

    data = get_real_muons_per_weight('/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample_w.h5', num_samples=args.num_samples)

    models = {}
    for weight, arr in data.items():
        n = arr.shape[0]
        print(f"\n========== Training GMM for weight {weight} (N={n}) ==========")
        pt = calc_pt(arr[:, 0], arr[:, 1])
        X = np.stack([arr[:, 2], pt], axis=1)               # [pz, pt]

        # standardization stats over all data (single streaming pass)
        mean, std = _mean_std_batched(X, args.batch_size)

        # ---- 1) choose K on a subsample using the in-memory sklearn grid ----
        m = min(n, args.select_samples)
        if n > m:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(n, size=m, replace=False))
            Xsub = np.asarray(X[idx], dtype=np.float64)
        else:
            Xsub = np.asarray(X, dtype=np.float64)
        Xsub = (Xsub - mean) / std
        X_train, X_val = train_val_split(Xsub, val_frac=args.val_frac, seed=0)
        print(f"  Selecting K on {m} samples...")
        best, _ = fit_gmm_grid(X_train, X_val, n_components_list=args.components, random_state=0)
        K = best["K"]
        print(f"  Selected K={K} (val_ll={best['val_ll']:.6f}, BIC={best['bic']:.1f})")

        # ---- 2) final fit on ALL data via streaming EM, warm-started from the K pick ----
        t0 = time()
        init = (best["model"].weights_, best["model"].means_, best["model"].covariances_)
        gmm, train_ll = fit_gmm_batched(
                X, mean, std, K,
                batch_size=args.batch_size, init=init,
                max_iter=args.max_iter, tol=args.tol, seed=0,
            )
        print(f"  Full-data fit complete in {time() - t0:.1f}s | avg_ll={train_ll:.6f}")

        models[weight] = {
            "gmm": gmm,
            "mean": mean,
            "std": std,
            "n": n,             # sample count, needed to weight groups when combining
        }

    # save one model per weight group
    with open("outputs/gmm_model.pkl", "wb") as f:
        pickle.dump(models, f)

