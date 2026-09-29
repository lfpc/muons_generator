"""Fit the momentum GMM (gmm_model.py) by weighted EM on the GPU, on every row of the data.

The fit space is the momentum flow's (u, v) (fit_space.py), standardized by its
rate-weighted mean/std. EM uses every row with its own rate weight -- the exact weighted
maximum likelihood fit -- starting from k-means on a weight-resampled subsample. The
rows stay in host memory and are streamed to the GPU chunk by chunk (2.3e9 rows do not
fit on the card).

Run: python train_gmm.py --K 32
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fit_space as fs
from gmm_model import save_gmm
from utils import CHUNK, DEFAULT_DATA


class TorchGMM:
    """Full-covariance mixture on a torch device: log_prob and sampling in batches."""
    ELEMENTS = 1 << 27            # cap on rows * K * d per chunk (0.5 GB in float32)

    def __init__(self, weights, means, covs, device="cpu"):
        self.weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
        self.means_np = np.asarray(means, dtype=np.float64)
        self.covs = np.asarray(covs, dtype=np.float64)
        self.K, self.d = self.means_np.shape
        self.to(device)

    def to(self, device):
        eye, jitter = np.eye(self.d), 0.0
        while True:                       # a nearly singular covariance gets a tiny jitter
            try:
                L = np.linalg.cholesky(self.covs + jitter * eye)
                break
            except np.linalg.LinAlgError:
                jitter = max(1e-8, 10 * jitter)
        self.covs = self.covs + jitter * eye
        Linv = np.linalg.inv(L)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.device = device
        self.w, self.means, self.L = t(self.weights), t(self.means_np), t(L)
        # L_k^-1 (x - mu_k) for every k at once: x @ W - b
        self.W = t(np.concatenate(list(np.transpose(Linv, (0, 2, 1))), axis=1))    # (d, K d)
        self.b = t(np.einsum("ked,kd->ke", Linv, self.means_np).ravel())
        log_det = 2.0 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
        self.log_norm = t(np.log(self.weights) - 0.5 * (self.d * np.log(2 * np.pi) + log_det))
        return self

    @property
    def n_params(self):
        return self.K - 1 + self.K * self.d + self.K * self.d * (self.d + 1) // 2

    def chunks(self, n):
        step = max(1024, self.ELEMENTS // (self.K * self.d))
        return range(0, n, step), step

    def component_log_prob(self, x):
        """log pi_k N(x | mu_k, cov_k), (n, K)."""
        sol = (x @ self.W - self.b).view(x.shape[0], self.K, self.d)
        return self.log_norm - 0.5 * sol.pow(2).sum(-1)

    @torch.no_grad()
    def log_prob(self, x):
        starts, step = self.chunks(x.shape[0])
        return torch.cat([torch.logsumexp(self.component_log_prob(x[i:i + step]), 1) for i in starts])

    @torch.no_grad()
    def sample(self, n, mean, std):
        comp = torch.distributions.Categorical(probs=self.w).sample((n,))
        eps = torch.randn(n, self.d, device=self.device)
        return (self.means[comp] + torch.einsum("nij,nj->ni", self.L[comp], eps)) * std + mean


def kmeans_init(X, K, reg, iters=25, seed=0):
    """k-means++ then Lloyd on an (unweighted, already weight-resampled) subsample;
    returns (weights, means, covariances) of the clusters."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    n, d = X.shape
    C = X[torch.randint(n, (1,), generator=g, device=X.device)]
    d2 = (X - C[0]).pow(2).sum(1)
    for _ in range(1, K):
        idx = torch.multinomial(d2 / d2.sum(), 1, generator=g)
        C = torch.cat([C, X[idx]])
        d2 = torch.minimum(d2, (X - X[idx]).pow(2).sum(1))
    for _ in range(iters):
        lab = torch.cat([torch.cdist(X[i:i + (1 << 16)], C).argmin(1) for i in range(0, n, 1 << 16)])
        cnt = torch.bincount(lab, minlength=K).double()
        S = torch.zeros(K, d, dtype=torch.float64, device=X.device).index_add_(0, lab, X.double())
        C[cnt > 0] = (S[cnt > 0] / cnt[cnt > 0, None]).float()
    Xd, cnt = X.double(), torch.bincount(lab, minlength=K).double().clamp_min(1.0)
    mu = torch.zeros(K, d, dtype=torch.float64, device=X.device).index_add_(0, lab, Xd) / cnt[:, None]
    diff = Xd - mu[lab]
    cov = torch.zeros(K, d * d, dtype=torch.float64, device=X.device).index_add_(
        0, lab, (diff[:, :, None] * diff[:, None, :]).flatten(1)).view(K, d, d) / cnt[:, None, None]
    cov += reg * torch.eye(d, dtype=torch.float64, device=X.device)
    cov[torch.bincount(lab, minlength=K) < 2 * d] = torch.cov(Xd.T) / K ** (2.0 / d)
    return (cnt / cnt.sum()).cpu().numpy(), mu.cpu().numpy(), cov.cpu().numpy()


def fit_gmm(X, w, K, device, reg_covar=1e-5, max_iter=500, tol=1e-5, seed=0):
    """Weighted EM over all rows of X (standardized float32 tensor, on `device` or on the
    CPU, streamed chunk by chunk). Sums in float64, TF32 off."""
    torch.backends.cuda.matmul.allow_tf32 = False
    wn = w.cpu().numpy().astype(np.float64)
    idx = np.random.default_rng(seed).choice(len(wn), size=min(1_000_000, len(wn)), p=wn / wn.sum())
    del wn
    gmm = TorchGMM(*kmeans_init(X[torch.from_numpy(idx).to(X.device)].to(device), K, reg_covar,
                                seed=seed), device)
    d, wsum, prev = X.shape[1], w.sum(dtype=torch.float64).item(), -np.inf
    starts, step = gmm.chunks(X.shape[0])
    bar = tqdm(range(max_iter), desc=f"EM K={K}", mininterval=2.0)
    for it in bar:
        Nk = torch.zeros(K, dtype=torch.float64, device=device)
        Sx = torch.zeros(K, d, dtype=torch.float64, device=device)
        Sxx = torch.zeros(K, d * d, dtype=torch.float64, device=device)
        ll = 0.0
        for i in starts:
            xb, wb = X[i:i + step].to(device), w[i:i + step].to(device)
            lp = gmm.component_log_prob(xb)
            lse = torch.logsumexp(lp, 1)
            ll += (wb.double() * lse.double()).sum().item()
            resp = torch.exp(lp - lse[:, None]) * wb[:, None]
            Nk += resp.sum(0).double()
            Sx += (resp.T @ xb).double()
            Sxx += (resp.T @ (xb[:, :, None] * xb[:, None, :]).flatten(1)).double()
        ll /= wsum
        Nk, Sx, Sxx = (a.cpu().numpy() for a in (Nk, Sx, Sxx))
        alive = Nk > 1e-12 * wsum                  # an emptied component keeps its last state
        mu = np.where(alive[:, None], Sx / np.maximum(Nk, 1e-300)[:, None], gmm.means_np)
        cov = Sxx.reshape(K, d, d) / np.maximum(Nk, 1e-300)[:, None, None] - mu[:, :, None] * mu[:, None, :]
        cov = np.where(alive[:, None, None], 0.5 * (cov + cov.transpose(0, 2, 1)) + reg_covar * np.eye(d), gmm.covs)
        gmm = TorchGMM(np.where(alive, Nk, 0.0) + 1e-300, mu, cov, device)
        bar.set_postfix(ll=f"{ll:.6f}", refresh=False)
        if ll - prev < tol:
            break
        prev = ll
    bar.close()
    print(f"  EM K={K}: {it + 1} iterations, weighted mean log-likelihood {ll:.5f} (fit space)")
    return gmm


def read_momentum(path, group, num_samples, p_max):
    """[u, v, log_jac, weight] float32 of the rows of `path` (or one group of it), and
    p_min = 0.9 x the smallest |p|: two streamed passes."""
    with h5py.File(path, "r") as f:
        g = f[group] if group else f
        n = min(num_samples or g["pz"].shape[0], g["pz"].shape[0])
        p_lo = np.inf
        for i in tqdm(range(0, n, CHUNK), desc="p_min", mininterval=2.0):
            sl = slice(i, min(i + CHUNK, n))
            p_lo = min(p_lo, np.sqrt(sum(g[c][sl].astype(np.float64) ** 2 for c in ("px", "py", "pz"))).min())
        m = {"p_min": 0.9 * p_lo, "p_max": p_max}
        T = np.empty((n, 4), dtype=np.float32)
        for i in tqdm(range(0, n, CHUNK), desc="reading", mininterval=2.0):
            sl = slice(i, min(i + CHUNK, n))
            px, py, pz, w = (g[c][sl].astype(np.float64) for c in ("px", "py", "pz", "weight"))
            T[sl, :2], T[sl, 2] = fs.momentum_to_fit(pz, np.hypot(px, py), m)
            T[sl, 3] = w
    return T, m


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    parser.add_argument("--data", default=DEFAULT_DATA, help="Muon file (the pooled table: every row)")
    parser.add_argument("--group", default=None, help="Group of a split --data to read (None = the file root)")
    parser.add_argument("--num_samples", type=int, default=None, help="Row cap (None = all)")
    parser.add_argument("--K", type=int, default=32, help="Number of components")
    parser.add_argument("--p_max", type=float, default=fs.P_MAX, help="Upper bound on |p| [GeV]")
    parser.add_argument("--reg_covar", type=float, default=1e-5, help="Added to every covariance diagonal")
    parser.add_argument("--max_iter", type=int, default=500, help="EM iteration cap")
    parser.add_argument("--tol", type=float, default=1e-5, help="Stop when the mean log-likelihood gains less")
    parser.add_argument("--device", default="cuda", help="EM device")
    parser.add_argument("--out", default="outputs/gmm_model.json", help="Output JSON")
    args = parser.parse_args()

    t0 = time.time()
    T, m = read_momentum(args.data, args.group, args.num_samples, args.p_max)
    w = T[:, 3].astype(np.float64)
    mean = (w @ T[:, :2].astype(np.float64)) / w.sum()
    std = np.sqrt((w @ T[:, :2].astype(np.float64) ** 2) / w.sum() - mean ** 2)
    print(f"Loaded {len(T):,} muons in {time.time() - t0:.0f}s, p_min {m['p_min']:.4g} GeV")

    X = torch.from_numpy((T[:, :2] - mean.astype(np.float32)) / std.astype(np.float32))
    gmm = fit_gmm(X, torch.from_numpy(np.ascontiguousarray(T[:, 3])), args.K, args.device,
                  args.reg_covar, args.max_iter, args.tol)

    tot = 0.0                                  # rate-weighted NLL of (pz, pt) [GeV^-2]
    for i in range(0, len(X), 1 << 22):
        lp = gmm.log_prob(X[i:i + (1 << 22)].to(args.device)).double().cpu().numpy()
        tot += float(w[i:i + (1 << 22)] @ (lp - np.log(std).sum() + T[i:i + (1 << 22), 2]))
    print(f"Rate-weighted NLL of (pz, pt) over the training rows: {-tot / w.sum():.5f} nats/muon")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_gmm(args.out, gmm.weights, gmm.means_np, gmm.covs, mean, std, **m)
    print(f"Saved {args.out}")
