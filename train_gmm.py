import numpy as np
from time import time
import torch
from utils import *
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

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
    parser = argparse.ArgumentParser(description="Train GMM on muon data")
    parser.add_argument("--num_samples", type=int, default=500_000_000, help="Number of samples to use for training")
    parser.add_argument("--components", type=int, nargs="+", default=[2,4,8,16,24], help="List of number of GMM components to try")
    args = parser.parse_args()

    px,py,pz,_x,_y,z,pdg,weight = get_real_muons('/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample.h5', num_samples = args.num_samples).T
    pt = calc_pt(px, py)
    data = np.stack([pz, pt], axis=1)
    X_train, X_val = train_val_split(data, val_frac=0.2, seed=0)
    X_train, X_val, mean, std = standardize_train_val(X_train, X_val)

    t0 = time()
    print("Fitting GMM...")
    best, results = fit_gmm_grid(
            X_train,
            X_val,
            n_components_list=args.components,
            random_state=0,
        )
    print("Fitting complete. Time taken: {:.3f} seconds".format(time() - t0))

    gmm = best["model"]
    print("\nBest model")
    print(f"K={best['K']}, val_ll={best['val_ll']:.6f}, BIC={best['bic']:.1f}")

    # Validation metric
    val_ll = gmm.score(X_val)
    print(f"\nFinal validation average log-likelihood: {val_ll:.6f}")

    # Generate samples
    pz_gen, pt_gen, comp = sample_gmm(gmm, mean, std, n_samples=100_000, seed=0)


    import pickle

    # save
    obj = {
        "gmm": gmm,
        "mean": mean,
        "std": std,
    }
    with open("outputs/gmm_model.pkl", "wb") as f:
        pickle.dump(obj, f)

