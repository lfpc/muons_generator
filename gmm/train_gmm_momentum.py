"""Fit a GMM to the muon momentum (pz, pt) on the entire dataset, in the bounded momentum
space of train_flow_ploss.py:

    u = logit((|p| - p_min) / (p_max - p_min)),   v = log(tan theta) = log(pt / pz),

standardized by their rate-weighted mean/std, so every sample has p_min < |p| < p_max and
pz, pt > 0 by construction. compare_gmm_flow.py --space momentum found this space and a
GMM of K ~ 32-1024 components to match or beat the flows on (pz, pt).

All rows of the file are used, with their own rate weights: for the pooled table that is
both weight groups of split_by_weight.py (the old train_flow_split.py pooled them too, its
hi_frac mixing being only a variance reduction for SGD; weighted EM over every row is the
exact maximum likelihood fit). EM runs on the GPU (compare_gmm_flow.fit_gmm), streaming
chunks from host memory, since 2.3e9 rows do not fit on the card.

The model is written in gmm_model.py's JSON format; load it with gmm_model.GMM.load.
Test it with  python test_gmm.py --model <out> --data ../muons_data.h5 --group ''

Run:  python train_gmm_momentum.py --K 32
"""
import argparse
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from compare_gmm_flow import fit_gmm, TorchGMM
from gmm_model import LOGIT_COLUMNS, logit_space, save_gmm
from utils import CHUNK, DEFAULT_DATA


def read_momentum(path, group, num_samples, p_max):
    """[u, v, log_jac, weight] float32 for the rows of `path` (or of one group of it),
    plus p_min: two streamed passes, the first for the smallest |p|."""
    with h5py.File(path, "r") as f:
        g = f[group] if group else f
        n = g["pz"].shape[0] if not num_samples else min(num_samples, g["pz"].shape[0])
        p_lo = np.inf
        for i in tqdm(range(0, n, CHUNK), desc="p_min", unit="chunk", mininterval=2.0):
            sl = slice(i, min(i + CHUNK, n))
            px, py, pz = (g[c][sl].astype(np.float64) for c in ("px", "py", "pz"))
            if not (pz.min() > 0 and np.hypot(px, py).min() > 0):
                raise SystemExit("pz and pt must be > 0 for the logit momentum space")
            p_lo = min(p_lo, np.sqrt(px ** 2 + py ** 2 + pz ** 2).min())
        p_min = 0.9 * p_lo                       # as train_flow_ploss.set_p_min
        T = np.empty((n, 4), dtype=np.float32)
        for i in tqdm(range(0, n, CHUNK), desc="reading", unit="chunk", mininterval=2.0):
            sl = slice(i, min(i + CHUNK, n))
            px, py, pz, w = (g[c][sl].astype(np.float64) for c in ("px", "py", "pz", "weight"))
            uv, log_jac = logit_space(pz, np.hypot(px, py), p_min, p_max)
            T[sl, :2], T[sl, 2], T[sl, 3] = uv, log_jac, w
    return T, p_min


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Muon file: the pooled table (all rows), or a split file with --group")
    parser.add_argument("--group", default=None, help="Group of --data to read (None = the file root)")
    parser.add_argument("--num_samples", type=int, default=None, help="Row cap (None = all)")
    parser.add_argument("--K", type=int, default=32, help="Number of GMM components")
    parser.add_argument("--p_max", type=float, default=400.0, help="Upper bound on |p| [GeV]")
    parser.add_argument("--reg_covar", type=float, default=1e-5,
                        help="Added to every covariance diagonal (standardized units)")
    parser.add_argument("--em_max_iter", type=int, default=500)
    parser.add_argument("--em_tol", type=float, default=1e-5,
                        help="EM stops when the weighted mean log-likelihood gains less")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/gmm_momentum_K32.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    t0 = time.time()
    T, p_min = read_momentum(args.data, args.group, args.num_samples, args.p_max)
    w = T[:, 3].astype(np.float64)
    wsum = w.sum()
    mean = (w @ T[:, :2].astype(np.float64)) / wsum
    std = np.sqrt(np.maximum((w @ T[:, :2].astype(np.float64) ** 2) / wsum - mean ** 2, 0.0)) + 1e-12
    del w
    print(f"Loaded {T.shape[0]:,} muons in {time.time() - t0:.0f}s: p_min {p_min:.4g} GeV, "
          f"sum of weights {wsum:.4g}, fit-space mean {np.round(mean, 4)}, std {np.round(std, 4)}")

    # standardized (u, v) and the weights, as CPU tensors streamed to the GPU by fit_gmm
    X = torch.from_numpy(((T[:, :2] - mean.astype(np.float32)) / std.astype(np.float32)))
    W = torch.from_numpy(np.ascontiguousarray(T[:, 3]))
    t0 = time.time()
    gmm = fit_gmm(X, W, args.K, args, args.device)
    print(f"EM done in {(time.time() - t0) / 60:.1f} min")

    # rate-weighted NLL of the physical (pz, pt) density over the same rows [GeV^-2]
    tot, const, step = 0.0, -float(np.log(std).sum()), 1 << 22
    for i in tqdm(range(0, X.shape[0], step), desc="NLL", unit="batch", mininterval=2.0):
        lp = gmm.log_prob(X[i:i + step].to(args.device)).double().cpu().numpy()
        tot += float(T[i:i + step, 3].astype(np.float64) @ (lp + const + T[i:i + step, 2]))
    print(f"Rate-weighted NLL of (pz, pt) over the training rows: {-tot / wsum:.5f} nats/muon")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_gmm(args.out, gmm.weights, gmm.means_np, gmm.covs, mean, std,
             columns=LOGIT_COLUMNS, p_min=p_min, p_max=args.p_max)
    print(f"Saved {args.out}")
