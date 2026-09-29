"""GMM vs Neural Spline Flow on the complete muon state, as a function of model size.

Both models are fit in train_flow_complete.py's fit space (bounded momentum u, v, |dphi|,
asinh r, asinh z, dequantized charge) on the head of one group of the split file, but
they are *scored* on the physical data (px, py, pz, x, y, z, charge) of the rest of that
group: the rate-weighted NLL of the physical density [GeV^-3 m^-3] times the charge
probability, which does not depend on the fit space. It follows the generative model
exactly as to_physical() samples it:

  - the global azimuth phi_r is uniform and dphi = phi_p - phi_r has a random sign
    (the model only sees |dphi|); x, y and px, py enter through r dr dphi_r, pt dpt dphi_p;
  - to_physical folds any |dphi| outside [0, pi] back in, so the density at |dphi| = d
    sums the model at d, -d and 2 pi - d;
  - the charge is the sign of the dequantized column, so P(charge | rest) integrates the
    model over that half-line. It is estimated by stratified importance sampling over the
    dequantization interval (--is_draws). That misses whatever the model puts beyond it
    (|charge column| > 2, which still yields a valid sign): ~0.02% for a flow, but a few
    % for a GMM, so the GMM is scored with the exact integral (conditional Gaussian CDF)
    and the IS number is kept alongside as a check on the estimator.

The flow is swept in the width of its coupling MLPs (--flow_hidden) and in the number of
coupling layers (--flow_layers, every combination of the two is run), the GMM in its
number of components (--gmm_K). Every model's result lands in <out_dir>/results.json as
soon as it is done and finished models are skipped on a rerun, so the sweep can be split
over GPUs (--only, --device) or resumed. Plots: outputs/figs/<out_dir name>_*.png.

--space momentum runs the same comparison on the momentum alone: both models are fit in
train_flow_ploss.py's (u, v) = (logit of the bounded |p|, log tan theta) and scored on the
rate-weighted NLL of the physical (pz, pt) density [GeV^-2], through the same exact
Jacobian; there is no charge or dphi to handle there.

Run:  python compare_gmm_flow.py
      python compare_gmm_flow.py --space momentum
      python compare_gmm_flow.py --only flow --device cuda:1     (in parallel)
      python compare_gmm_flow.py --plot_only
"""
import argparse
import fcntl
import json
import math
import os
import time

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import flow_training as ft
import train_flow_ploss as momentum
from gen_eval import figs_dir, save_plot
from models import NeuralSplineFlow, save_flow
from utils import CHUNK

SPACES = {"complete": ("logit_p", "log_tan_theta", "dphi", "r", "z", "charge"),
          "momentum": momentum.COLUMNS}
COLS = SPACES["complete"]            # set from --space in __main__
DIM = len(COLS)
PHYS_LABEL = {"complete": "(x, y, z, px, py, pz, charge)", "momentum": "(pz, pt)"}
TWO_PI = 2.0 * math.pi
GMM_COLOR, FLOW_COLOR = "#2a78d6", "#eb6834"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=ft.ArgFormatter)
    p.add_argument("--space", choices=tuple(SPACES), default="complete",
                   help="complete: the whole muon state; momentum: (pz, pt) only")
    p.add_argument("--data", default=ft.DEFAULT_SPLIT, help="Split muon file")
    p.add_argument("--group", default=ft.HI, help="Group to train and test on")
    p.add_argument("--train_rows", type=int, default=20_000_000,
                   help="Training rows, from the head of --group")
    p.add_argument("--val_rows", type=int, default=2_000_000,
                   help="Rows after the training ones, for the flow's early stopping")
    p.add_argument("--test_rows", type=int, default=None,
                   help="Test rows after those (None = the whole rest of --group)")
    p.add_argument("--gmm_K", type=int, nargs="+", default=[8, 32, 128, 512, 2048, 4096])
    p.add_argument("--flow_hidden", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256],
                   help="Widths of the coupling MLPs")
    p.add_argument("--flow_depth", type=int, default=2, help="Hidden layers per coupling MLP")
    p.add_argument("--flow_layers", type=int, nargs="+", default=[12],
                   help="Coupling layers; the flows run are every --flow_hidden x --flow_layers")
    p.add_argument("--num_bins", type=int, default=12)
    p.add_argument("--tail_bound", type=float, default=6.0)
    p.add_argument("--epochs", type=int, default=150,
                   help="Flow epochs, each one pass over the training rows")
    p.add_argument("--batch_size", type=int, default=2**19)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=30, help="Flow early-stopping patience")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the flow for training and for the test NLL")
    p.add_argument("--reg_covar", type=float, default=1e-5,
                   help="Added to every GMM covariance diagonal (standardized units)")
    p.add_argument("--em_max_iter", type=int, default=500)
    p.add_argument("--em_tol", type=float, default=1e-5,
                   help="EM stops when the weighted mean log-likelihood gains less")
    p.add_argument("--is_draws", type=int, default=8,
                   help="Importance draws over the charge dequantization interval")
    p.add_argument("--eval_batch", type=int, default=1 << 20)
    p.add_argument("--time_samples", type=int, default=1_000_000,
                   help="Samples per sampling-time measurement")
    p.add_argument("--time_repeats", type=int, default=3, help="Best of this many")
    p.add_argument("--cpu_threads", type=int, default=16, help="torch threads on the CPU")
    p.add_argument("--p_max", type=float, default=400.0, help="Upper bound on |p| [GeV]")
    p.add_argument("--r_scale", type=float, default=2.0, help="asinh crossover of r [m]")
    p.add_argument("--z_scale", type=float, default=20.0, help="asinh crossover of z [m]")
    p.add_argument("--only", choices=("both", "gmm", "flow"), default="both")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None,
                   help="Models and results.json (default outputs/gmm_vs_flow, "
                        "outputs/gmm_vs_flow_momentum for --space momentum)")
    p.add_argument("--plot_only", action="store_true")
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = "outputs/gmm_vs_flow" + ("_momentum" if args.space == "momentum" else "")
    return args


# --------------------------------------------------------------------------- #
# data: physical columns -> fit space, with the log-Jacobian back to physical
# --------------------------------------------------------------------------- #
PHYS = ("px", "py", "pz", "x", "y", "z", "pdg", "weight")


def read_rows(g, start, stop, fn, width, desc):
    """fn(dict of physical columns) for rows [start, stop) of g, chunk by chunk, into
    one (stop - start, width) float32 array."""
    out = np.empty((stop - start, width), dtype=np.float32)
    for i in tqdm(range(start, stop, CHUNK), desc=desc, unit="chunk", mininterval=ft.BAR_INTERVAL):
        sl = slice(i, min(i + CHUNK, stop))
        out[i - start:sl.stop - start] = fn({c: g[c][sl].astype(np.float64) for c in PHYS})
    return out


def space_params(g, args):
    """The fit space's constants, from the whole group so no test muon falls outside
    them: p_min just below the smallest |p| (as set_p_min), r_max the largest radius."""
    p_lo, r_hi, r_lo, n = np.inf, 0.0, np.inf, g["pz"].shape[0]
    full = DIM > 2
    for i in tqdm(range(0, n, CHUNK), desc="bounds", unit="chunk", mininterval=ft.BAR_INTERVAL):
        sl = slice(i, min(i + CHUNK, n))
        px, py, pz = (g[c][sl].astype(np.float64) for c in ("px", "py", "pz"))
        momentum.read_momentum(pz, np.hypot(px, py), args.p_max)      # checks the bounds
        p_lo = min(p_lo, np.sqrt(px ** 2 + py ** 2 + pz ** 2).min())
        if full:
            r = np.hypot(g["x"][sl].astype(np.float64), g["y"][sl].astype(np.float64))
            r_hi, r_lo = max(r_hi, r.max()), min(r_lo, r.min())
    if not full:
        return {"columns": list(COLS), "p_min": 0.9 * p_lo, "p_max": args.p_max}
    if r_lo <= 0:
        raise SystemExit("a muon at r = 0: its physical density (1/r) is undefined")
    return {"columns": list(COLS), "p_min": 0.9 * p_lo, "p_max": args.p_max,
            "r_scale": args.r_scale, "r_max": float(r_hi),
            "z0": float(np.median(g["z"][:CHUNK])), "z_scale": args.z_scale}


def fit_columns(c, m):
    """(u, v, |dphi|, asinh r, asinh z) [unstandardized], q = -sign(pdg), and the
    log-density conversion log p_phys = log p_fit + logJ (p_fit over those five columns
    with the charge handled apart; phi_r uniform and the sign of dphi a coin flip)."""
    px, py, pz, x, y, z = (c[k] for k in ("px", "py", "pz", "x", "y", "z"))
    pt, r = np.hypot(px, py), np.hypot(x, y)
    dphi = np.arctan2(py, px) - np.arctan2(y, x)
    s = np.column_stack([momentum.fit_space(pz, pt, m),
                         np.abs((dphi + np.pi) % TWO_PI - np.pi),
                         np.arcsinh(r / m["r_scale"]),
                         np.arcsinh((z - m["z0"]) / m["z_scale"])])
    logJ = (momentum.log_jacobian(pz, pt, m)
            - np.log(pt) - np.log(r)                               # (px,py), (x,y) polar
            - 0.5 * np.log(r ** 2 + m["r_scale"] ** 2)
            - 0.5 * np.log((z - m["z0"]) ** 2 + m["z_scale"] ** 2)
            - np.log(TWO_PI) - np.log(2.0))                        # phi_r, sign of dphi
    return s, -np.sign(c["pdg"]), logJ


def momentum_columns(c, m):
    """(u, v) and log p(pz, pt) = log p(u, v) + logJ."""
    pz, pt = c["pz"], np.hypot(c["px"], c["py"])
    return momentum.fit_space(pz, pt, m), momentum.log_jacobian(pz, pt, m)


def train_reader(m, rng):
    """Training / validation rows: [5 columns, dequantized charge, weight]
    ([u, v, weight] in the momentum space)."""
    def fn(c):
        if DIM == 2:
            return np.column_stack([momentum_columns(c, m)[0], c["weight"]])
        s, q, _ = fit_columns(c, m)
        return np.column_stack([s, q + rng.uniform(-1.0, 1.0, q.shape[0]), c["weight"]])
    return fn


def test_reader(m):
    """Test rows: [5 unstandardized columns, q, logJ, weight] ([u, v, logJ, weight] in
    the momentum space)."""
    def fn(c):
        if DIM == 2:
            s, logJ = momentum_columns(c, m)
            return np.column_stack([s, logJ, c["weight"]])
        s, q, logJ = fit_columns(c, m)
        return np.column_stack([s, q, logJ, c["weight"]])
    return fn


# --------------------------------------------------------------------------- #
# GMM in torch: weighted EM on the GPU, log_prob, sampling
# --------------------------------------------------------------------------- #
class TorchGMM:
    ELEMENTS = 1 << 27            # cap on rows * K * d per chunk (0.5 GB in float32)

    def __init__(self, weights, means, covs, device="cpu"):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means_np = np.asarray(means, dtype=np.float64)
        self.covs = np.asarray(covs, dtype=np.float64)
        self.K, self.d = self.means_np.shape
        self.to(device)

    def to(self, device):
        eye, jitter = np.eye(self.d), 0.0
        while True:
            try:
                L = np.linalg.cholesky(self.covs + jitter * eye)
                break
            except np.linalg.LinAlgError:
                jitter = max(1e-8, 10 * jitter)
        self.covs = self.covs + jitter * eye
        Linv = np.linalg.inv(L)
        log_det = 2.0 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.device = device
        self.w = t(self.weights / self.weights.sum())
        self.means, self.L, self.Linv = t(self.means_np), t(L), t(Linv)
        # L^-1 (x - mu_k) for every k at once: x @ W - b, W = [L_k^-T]_k
        self.W = t(np.concatenate(list(np.transpose(Linv, (0, 2, 1))), axis=1))   # (d, K d)
        self.b = t(np.einsum("ked,kd->ke", Linv, self.means_np).ravel())
        self.log_norm = t(np.log(self.weights / self.weights.sum())
                          - 0.5 * (self.d * np.log(2 * np.pi) + log_det))
        # the first five dims alone, and the charge (last dim) conditional on them
        self.log_norm5 = t(np.log(self.weights / self.weights.sum())
                           - 0.5 * ((self.d - 1) * np.log(2 * np.pi) + log_det)
                           + np.log(L[:, -1, -1]))
        self.cond_row, self.cond_sd = t(L[:, -1, :-1]), t(L[:, -1, -1])
        return self

    @property
    def n_params(self):
        return self.K - 1 + self.K * self.d + self.K * self.d * (self.d + 1) // 2

    def _chunks(self, n):
        step = max(1024, self.ELEMENTS // (self.K * self.d))
        return range(0, n, step), step

    def component_log_prob(self, x):
        """log pi_k N(x | mu_k, cov_k), (n, K), x one chunk."""
        sol = (x @ self.W - self.b).view(x.shape[0], self.K, self.d)
        return self.log_norm - 0.5 * sol.pow(2).sum(-1)

    @torch.no_grad()
    def log_prob(self, x):
        starts, step = self._chunks(x.shape[0])
        return torch.cat([torch.logsumexp(self.component_log_prob(x[i:i + step]), 1)
                          for i in starts])

    @torch.no_grad()
    def log_prob_charge(self, x5, q, thr):
        """log of the model integrated over the charge half-line of sign q (cut at thr,
        standardized), at the first five standardized dims x5: exact, through the
        charge's conditional Gaussian in every component."""
        starts, step = self._chunks(x5.shape[0])
        out = []
        x5 = torch.cat([x5, x5.new_zeros(x5.shape[0], 1)], 1)    # L^-1 is lower triangular:
        for i in starts:                                        # the first 5 rows ignore dim 6
            xb, qb = x5[i:i + step], q[i:i + step, None]
            sol = (xb @ self.W - self.b).view(xb.shape[0], self.K, self.d)[..., :-1]
            cond_mean = self.means[:, -1] + (sol * self.cond_row).sum(-1)
            lp = (self.log_norm5 - 0.5 * sol.pow(2).sum(-1)
                  + torch.special.log_ndtr(qb * (cond_mean - thr) / self.cond_sd))
            out.append(torch.logsumexp(lp, 1))
        return torch.cat(out)

    @torch.no_grad()
    def sample(self, n, mean, std):
        comp = torch.distributions.Categorical(probs=self.w).sample((n,))
        eps = torch.randn(n, self.d, device=self.device)
        x = self.means[comp] + torch.einsum("nij,nj->ni", self.L[comp], eps)
        return x * std + mean


def _kmeans_init(X, K, reg, iters=25, seed=0):
    """k-means++ then Lloyd on an (unweighted, already weight-resampled) subsample."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    n = X.shape[0]
    C = X[torch.randint(n, (1,), generator=g, device=X.device)]
    d2 = (X - C[0]).pow(2).sum(1)
    for _ in range(1, K):
        idx = torch.multinomial(d2 / d2.sum(), 1, generator=g)
        C = torch.cat([C, X[idx]])
        d2 = torch.minimum(d2, (X - X[idx]).pow(2).sum(1))
    for _ in range(iters):
        lab = torch.cat([torch.cdist(X[i:i + (1 << 16)], C).argmin(1) for i in range(0, n, 1 << 16)])
        cnt = torch.bincount(lab, minlength=K).double()
        S = torch.zeros(K, X.shape[1], dtype=torch.float64, device=X.device).index_add_(0, lab, X.double())
        keep = cnt > 0
        C[keep] = (S[keep] / cnt[keep, None]).float()
    Xd = X.double()
    cnt = torch.bincount(lab, minlength=K).double().clamp_min(1.0)
    mu = torch.zeros(K, X.shape[1], dtype=torch.float64, device=X.device).index_add_(0, lab, Xd) / cnt[:, None]
    diff = Xd - mu[lab]
    cov = torch.zeros(K, X.shape[1] ** 2, dtype=torch.float64, device=X.device).index_add_(
        0, lab, (diff[:, :, None] * diff[:, None, :]).flatten(1)).view(K, X.shape[1], X.shape[1])
    cov = cov / cnt[:, None, None] + reg * torch.eye(X.shape[1], dtype=torch.float64, device=X.device)
    small = torch.bincount(lab, minlength=K) < 2 * X.shape[1]
    cov[small] = torch.cov(Xd.T) / K ** (2.0 / X.shape[1])
    return (cnt / cnt.sum()).cpu().numpy(), mu.cpu().numpy(), cov.cpu().numpy()


def fit_gmm(X, w, K, args, device):
    """Weighted EM over all of X (standardized float32 tensor, on `device` or on the CPU
    for data too big for the GPU, then streamed to it chunk by chunk), started from
    k-means on a weight-resampled subsample. Sums in float64, TF32 off."""
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(args.seed)
    wn = w.cpu().numpy().astype(np.float64)
    idx = rng.choice(len(wn), size=min(1_000_000, len(wn)), p=wn / wn.sum())
    del wn
    sub = X[torch.from_numpy(idx).to(X.device)].to(device)
    weights, means, covs = _kmeans_init(sub, K, args.reg_covar, seed=args.seed)
    gmm = TorchGMM(weights, means, covs, device)
    d, eye = X.shape[1], np.eye(X.shape[1])
    wsum, prev = w.sum(dtype=torch.float64).item(), -np.inf
    starts, step = gmm._chunks(X.shape[0])
    bar = tqdm(range(args.em_max_iter), desc=f"EM K={K}", mininterval=ft.BAR_INTERVAL)
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
        alive = (Nk > 1e-12 * wsum).cpu().numpy()
        Nk_, Sx_, Sxx_ = (a.cpu().numpy() for a in (Nk, Sx, Sxx))
        mu = np.where(alive[:, None], Sx_ / np.maximum(Nk_, 1e-300)[:, None], gmm.means_np)
        cov = Sxx_.reshape(K, d, d) / np.maximum(Nk_, 1e-300)[:, None, None] - mu[:, :, None] * mu[:, None, :]
        cov = np.where(alive[:, None, None], 0.5 * (cov + cov.transpose(0, 2, 1)) + args.reg_covar * eye, gmm.covs)
        wt = np.where(alive, Nk_, 0.0) + 1e-300
        gmm = TorchGMM(wt / wt.sum(), mu, cov, device)
        bar.set_postfix(ll=f"{ll:.6f}", refresh=False)
        if ll - prev < args.em_tol:
            break
        prev = ll
    bar.close()
    print(f"  EM K={K}: {it + 1} iterations, train log-likelihood {ll:.5f} (fit space)")
    return gmm


# --------------------------------------------------------------------------- #
# scoring on the physical data, and sampling time
# --------------------------------------------------------------------------- #
@torch.no_grad()
def physical_nll(log_prob, T, mean, std, args, device, gmm_exact=None):
    """Rate-weighted mean -log p(px, py, pz, x, y, z, charge) over the test rows T
    ([5 unstandardized columns, q, logJ, weight], CPU). log_prob maps standardized
    (n, 6) rows to log densities. With gmm_exact (a TorchGMM) also returns the NLL with
    the charge integrated exactly."""
    torch.backends.cuda.matmul.allow_tf32 = False
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    if DIM == 2:                                     # (pz, pt): nothing to integrate
        tot = wsum = 0.0
        const = -float(np.log(std).sum())
        for i in tqdm(range(0, T.shape[0], args.eval_batch), desc="test NLL", unit="batch",
                      mininterval=ft.BAR_INTERVAL):
            b = torch.from_numpy(T[i:i + args.eval_batch]).to(device)
            w = b[:, 3].double()
            ll = log_prob((b[:, :2] - mean_t) / std_t).double() + const + b[:, 2].double()
            tot += (w * ll).sum().item()
            wsum += w.sum().item()
        return -tot / wsum, None
    thr = float(-mean[5] / std[5])                   # charge = 0 in standardized units
    M = args.is_draws
    const = math.log(2.0 / M) - float(np.log(std).sum())      # IS width 2, 1/std Jacobian
    const_exact = -float(np.log(std[:5]).sum())
    g = torch.Generator(device=device).manual_seed(args.seed + 7)
    tot = tot_exact = wsum = 0.0
    for i in tqdm(range(0, T.shape[0], args.eval_batch), desc="test NLL", unit="batch",
                  mininterval=ft.BAR_INTERVAL):
        b = torch.from_numpy(T[i:i + args.eval_batch]).to(device)
        s5, q, logJ, w = b[:, :5], b[:, 5], b[:, 6], b[:, 7].double()
        terms, terms_exact = [], []
        for pre in (s5[:, 2], -s5[:, 2], TWO_PI - s5[:, 2]):          # |dphi| fold
            x5 = s5.clone()
            x5[:, 2] = pre
            x5 = (x5 - mean_t[:5]) / std_t[:5]
            if gmm_exact is not None:
                terms_exact.append(gmm_exact.log_prob_charge(x5, q, thr))
            for j in range(M):                                          # stratified u in (0, 2)
                u = 2.0 * (j + torch.rand(q.shape[0], device=device, generator=g)) / M
                c = ((q * u - mean_t[5]) / std_t[5])[:, None]
                terms.append(log_prob(torch.cat([x5, c], 1)))
        ll = torch.logsumexp(torch.stack(terms), 0).double() + const + logJ.double()
        tot += (w * ll).sum().item()
        wsum += w.sum().item()
        if gmm_exact is not None:
            ll = torch.logsumexp(torch.stack(terms_exact), 0).double() + const_exact + logJ.double()
            tot_exact += (w * ll).sum().item()
    return -tot / wsum, (-tot_exact / wsum if gmm_exact is not None else None)


def sampling_time(sample, device, args):
    """Best-of wall time [s] of sample(n) for n = --time_samples, after a warm-up."""
    sync = torch.cuda.synchronize if str(device).startswith("cuda") else (lambda: None)
    with torch.no_grad():
        sample(min(args.time_samples, 10_000))
        sync()
        best = np.inf
        for _ in range(args.time_repeats):
            t0 = time.perf_counter()
            sample(args.time_samples)
            sync()
            best = min(best, time.perf_counter() - t0)
    return best


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_result(path, key, res):
    """Merge one model's result into results.json, under a lock, so parallel runs on
    different GPUs don't drop each other's entries."""
    with open(f"{path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        allres = load_results(path)
        allres[key] = res
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(allres, f, indent=1)
        os.replace(tmp, path)


def fixed_batch(fn, n):
    """fn with every call padded to n rows, so a torch.compile(dynamic=False) graph
    built for n is reused on the last, shorter batch too."""
    def f(x):
        m = x.shape[0]
        return fn(x) if m == n else fn(torch.cat([x, x.new_zeros(n - m, x.shape[1])]))[:m]
    return f


def prepare_data(args, device):
    with h5py.File(args.data, "r") as f:
        g = f[args.group]
        n = g["pz"].shape[0]
        a, b = args.train_rows, args.train_rows + args.val_rows
        c = n if args.test_rows is None else min(n, b + args.test_rows)
        if c <= b:
            raise SystemExit(f"{args.group} has {n:,} rows: nothing left to test on")
        m = space_params(g, args)
        rng = np.random.default_rng(args.seed)
        train = read_rows(g, 0, a, train_reader(m, rng), DIM + 1, "train rows")
        val = read_rows(g, a, b, train_reader(m, rng), DIM + 1, "val rows")
        test = read_rows(g, b, c, test_reader(m), 8 if DIM > 2 else 4, "test rows")
    mean, std, wsum = ft.standardize({"all": train})
    val[:, :DIM] = (val[:, :DIM] - mean.astype(np.float32)) / std.astype(np.float32)
    print(f"{args.group}: train {a:,} | val {b - a:,} | test {c - b:,} rows "
          f"(test sum of weights {test[:, -1].sum(dtype=np.float64):,.4g})")
    return m, train, torch.from_numpy(val).to(device), test, mean, std, wsum


def run_gmm(K, data, args, device):
    m, train, _, test, mean, std, _ = data
    X = torch.from_numpy(train[:, :DIM]).to(device)
    w = torch.from_numpy(train[:, DIM]).to(device)
    t0 = time.time()
    gmm = fit_gmm(X, w, K, args, device)
    train_time = time.time() - t0
    del X, w
    nll, nll_exact = physical_nll(gmm.log_prob, test, mean, std, args, device,
                                  gmm_exact=gmm if DIM > 2 else None)
    print(f"  GMM K={K}: test NLL {nll:.5f}"
          + (f" (charge exact: {nll_exact:.5f})" if nll_exact is not None else ""))
    with open(os.path.join(args.out_dir, f"gmm_K{K}.json"), "w") as f:
        json.dump({"weights": gmm.weights.tolist(), "means": gmm.means_np.tolist(),
                   "covariances": gmm.covs.tolist(), "mean": mean.tolist(), "std": std.tolist(),
                   **m}, f)
    times = {}
    for dev in (device, "cpu"):
        gmm.to(dev)
        mt = torch.as_tensor(mean, dtype=torch.float32, device=dev)
        st = torch.as_tensor(std, dtype=torch.float32, device=dev)
        times["gpu" if dev != "cpu" else "cpu"] = sampling_time(lambda n: gmm.sample(n, mt, st), dev, args)
    res = {"model": "gmm", "K": K, "n_params": gmm.n_params, "test_nll": nll,
           "train_time_s": train_time,
           "sample_time_gpu_s": times["gpu"], "sample_time_cpu_s": times["cpu"]}
    if nll_exact is not None:
        res["test_nll_charge_exact"] = nll_exact
    return res


def run_flow(cfg, data, args, device):
    h, L = cfg
    m, train, val, test, mean, std, wsum = data
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True           # training only, as ft.setup
    flow = NeuralSplineFlow(dim=DIM, n_layers=L, hidden=(h,) * args.flow_depth,
                            num_bins=args.num_bins, tail_bound=args.tail_bound).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"  flow hidden={h} layers={L}: {n_params:,} parameters")
    t0 = time.time()
    state, val_nll, _ = ft.fit(flow, {"all": train}, val, args, device)
    train_time = time.time() - t0
    flow.load_state_dict(state)
    flow.eval()
    log_prob = flow.log_prob
    if args.compile:                   # 3-8x faster on the ~1e9 evaluations of the test set
        log_prob = fixed_batch(torch.compile(flow.log_prob, dynamic=False), args.eval_batch)
    nll, _ = physical_nll(log_prob, test, mean, std, args, device)
    print(f"  flow hidden={h} layers={L}: test NLL {nll:.5f}")
    save_flow(os.path.join(args.out_dir, f"flow_h{h}_L{L}.pt"), flow, state, mean, std,
              weight=wsum, **m)
    times = {}
    for dev in (device, "cpu"):
        flow.to(dev)
        times["gpu" if dev != "cpu" else "cpu"] = sampling_time(
            lambda n: flow.sample(n, mean=mean, std=std), dev, args)
    return {"model": "flow", "hidden": h, "n_layers": L, "n_params": n_params, "test_nll": nll,
            "val_nll_fit_space": val_nll, "train_time_s": train_time,
            "sample_time_gpu_s": times["gpu"], "sample_time_cpu_s": times["cpu"]}


def result_key(kind, v):
    return f"gmm_{v}" if kind == "gmm" else f"flow_{v[0]}_L{v[1]}"


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
# categorical slots in fixed order: GMM, then one per flow sweep
SWEEP_COLORS = ("#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
SWEEP_MARKERS = ("s", "^", "D", "v", "P", "X")


def plot_series(results):
    """[(label, color, marker, rows sorted by n_params, point label fn)]: the GMM K sweep,
    the flow width sweep at the base depth (the most common number of coupling layers),
    and one depth sweep per width that has runs at other depths (its base-depth run
    included, as the sweep's end point)."""
    gmm = sorted((r for r in results.values() if r["model"] == "gmm"), key=lambda r: r["n_params"])
    flows = [r for r in results.values() if r["model"] == "flow"]
    for r in flows:
        r.setdefault("n_layers", 12)          # results from before the depth sweep
    out = [("GMM, vary K", GMM_COLOR, "o", gmm, lambda r: f"K={r['K']}")]
    depths = [r["n_layers"] for r in flows]
    base_L = max(set(depths), key=depths.count) if flows else 12
    base = sorted((r for r in flows if r["n_layers"] == base_L), key=lambda r: r["n_params"])
    sweeps = [(f"Flow, {base_L} layers, vary width", base, lambda r: f"h={r['hidden']}")]
    for h in sorted({r["hidden"] for r in flows if r["n_layers"] != base_L}):
        rows = sorted((r for r in flows if r["hidden"] == h), key=lambda r: r["n_params"])
        sweeps.append((f"Flow, width {h}, vary layers", rows,
                       lambda r: f"L={r['n_layers']}" if r["n_layers"] != base_L else None))
    for i, (label, rows, fn) in enumerate(sweeps):
        out.append((label, SWEEP_COLORS[i], SWEEP_MARKERS[i], rows, fn))
    return [s for s in out if s[3]]


def plot(results, args):
    """Figures named after --out_dir, so separate sweeps keep separate plots."""
    os.makedirs(figs_dir, exist_ok=True)
    fig_name = os.path.basename(os.path.normpath(args.out_dir))
    series = plot_series(results)

    # left: everything; right: zoom on the flows (and the GMMs that reach their level),
    # where the curves bunch up and the point labels need the room
    fig, (ax, zx) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    flow_y = [r["test_nll"] for _, _, _, rows, _ in series[1:] for r in rows]
    for i, (label, color, marker, rows, point_label) in enumerate(series):
        x = [r["n_params"] for r in rows]
        y = [r.get("test_nll_charge_exact", r["test_nll"]) for r in rows]
        for a in (ax, zx):
            a.plot(x, y, color=color, marker=marker, lw=2, ms=7, label=label)
        for r, yy in zip(rows, y):
            if not point_label(r):
                continue
            if i == 0:
                ax.annotate(point_label(r), (r["n_params"], yy), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color="#52514e")
            dy = (10, -14, 10, -14, 10, -14)[i % 6] if i else 8   # alternate sides per series
            zx.annotate(point_label(r), (r["n_params"], yy), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8, color="#52514e")
    for a in (ax, zx):
        a.set_xscale("log")
        a.set_xlabel("number of parameters")
        a.grid(alpha=0.3, lw=0.6)
    if flow_y:
        hi = max(flow_y)                 # and any GMM that gets below the flows' worst
        gmm_y = [r.get("test_nll_charge_exact", r["test_nll"]) for r in series[0][3]]
        lo = min(flow_y + [y for y in gmm_y if y <= hi])
        pad = 0.35 * (hi - lo) + 1e-4
        zx.set_ylim(lo - pad, hi + 2 * pad)
        zx.set_title("zoom on the flows' range")
    ax.set_ylabel("test NLL, physical space [nats / muon]")
    ax.set_title(f"Rate-weighted NLL of {PHYS_LABEL[args.space]}, {args.group}")
    ax.legend(frameon=False)
    save_plot(fig, f"{fig_name}_nll.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    for ax, dev in zip(axes, ("gpu", "cpu")):
        for label, color, marker, rows, _ in series:
            ax.plot([r["n_params"] for r in rows], [r[f"sample_time_{dev}_s"] for r in rows],
                    color=color, marker=marker, lw=2, ms=7, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of parameters")
        ax.set_title(dev.upper() if dev == "gpu" else f"CPU ({args.cpu_threads} threads)")
        ax.grid(alpha=0.3, lw=0.6, which="both")
    axes[0].set_ylabel(f"time to sample {args.time_samples:,} muons [s]")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Sampling time (fit space, best of repeats)")
    save_plot(fig, f"{fig_name}_sampling_time.png")


if __name__ == "__main__":
    args = parse_args()
    COLS = SPACES[args.space]
    DIM = len(COLS)
    os.makedirs(args.out_dir, exist_ok=True)
    res_path = os.path.join(args.out_dir, "results.json")
    if not args.plot_only:
        torch.set_num_threads(args.cpu_threads)
        todo = ([("gmm", K) for K in args.gmm_K] if args.only in ("both", "gmm") else []) + \
               ([("flow", (h, L)) for h in args.flow_hidden for L in args.flow_layers]
                if args.only in ("both", "flow") else [])
        done = load_results(res_path)
        todo = [(k, v) for k, v in todo if result_key(k, v) not in done]
        print(f"To run: {', '.join(f'{k} {v}' for k, v in todo) or 'nothing'}")
        if todo:
            data = prepare_data(args, args.device)
            for kind, v in todo:
                print(f"\n========== {kind} {v} ==========")
                res = (run_gmm if kind == "gmm" else run_flow)(v, data, args, args.device)
                res["n_test"] = int(data[3].shape[0])
                print(json.dumps(res, indent=1))
                save_result(res_path, result_key(kind, v), res)
    plot(load_results(res_path), args)
