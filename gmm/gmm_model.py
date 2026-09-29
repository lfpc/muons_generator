"""Portable GMM: save/load as JSON, sample and evaluate with numpy (or torch) only.

The model lives in a standardized fit space x_s = (f(pz, pt) - mean) / std, where f is
  - (log pz, log pt) (train_gmm.py; files without a "columns" entry), or
  - (u, v) = (logit((|p| - p_min) / (p_max - p_min)), log(pt / pz)), the bounded momentum
    space of train_flow_ploss.py (columns ["logit_p", "log_tan_theta"], with p_min and
    p_max stored alongside; train_gmm_momentum.py), where every sample has
    p_min < |p| < p_max and pz, pt > 0 by construction.
The file holds just what defines it: mixture weights, means, covariances and the
standardization mean/std (plus metadata such as the physical rate "weight" and the
training "group"). Everything else (Cholesky factors, precisions) is derived on load.
"""
import json

import numpy as np

_ARRAYS = ("weights", "means", "covariances", "mean", "std")
LOGIT_COLUMNS = ["logit_p", "log_tan_theta"]


def logit_space(pz, pt, p_min, p_max):
    """(pz, pt) -> (u, v), and log |d(u, v) / d(pz, pt)|."""
    p = np.hypot(pz, pt)
    uv = np.column_stack([np.log(p - p_min) - np.log(p_max - p), np.log(pt / pz)])
    log_jac = (np.log(p_max - p_min) - np.log(p - p_min) - np.log(p_max - p)
               + np.log(p) - np.log(pz) - np.log(pt))
    return uv, log_jac


def logit_inverse(uv, p_min, p_max):
    """(u, v) -> (pz, pt)."""
    p = p_min + (p_max - p_min) / (1.0 + np.exp(-uv[:, 0]))
    pz = p / np.sqrt(1.0 + np.exp(2.0 * uv[:, 1]))
    return pz, pz * np.exp(uv[:, 1])


def save_gmm(path, weights, means, covariances, mean, std, **meta):
    """Write the GMM to `path` as JSON. Floats round-trip exactly (repr)."""
    arrays = dict(weights=weights, means=means, covariances=covariances, mean=mean, std=std)
    out = {k: np.asarray(v, dtype=np.float64).tolist() for k, v in arrays.items()}
    out.update(meta)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


class GMM:
    """Full-covariance Gaussian mixture in a standardized (pz, pt) fit space."""

    def __init__(self, weights, means, covariances, mean, std, **meta):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        self.means = np.asarray(means, dtype=np.float64)                  # (K, d)
        self.covariances = np.asarray(covariances, dtype=np.float64)      # (K, d, d)
        self.mean = np.asarray(mean, dtype=np.float64).ravel()            # (d,)
        self.std = np.asarray(std, dtype=np.float64).ravel()              # (d,)
        self.meta = meta
        self.logit = meta.get("columns") == LOGIT_COLUMNS
        self.chol = np.linalg.cholesky(self.covariances)                  # (K, d, d), cov = L L^T
        self.chol_inv = np.linalg.inv(self.chol)                          # L^-1
        d = self.means.shape[1]
        log_det = 2.0 * np.log(np.diagonal(self.chol, axis1=1, axis2=2)).sum(axis=1)
        self._log_norm = np.log(self.weights) - 0.5 * (d * np.log(2.0 * np.pi) + log_det)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    @property
    def n_components(self):
        return self.weights.shape[0]

    # ---- standardized space ---- #
    def component_log_prob(self, Xs):
        """log(pi_k N(x_s | mu_k, cov_k)) per component, (n, K)."""
        Xs = np.asarray(Xs, dtype=np.float64)
        out = np.empty((Xs.shape[0], self.n_components))
        for k in range(self.n_components):
            sol = (Xs - self.means[k]) @ self.chol_inv[k].T               # L^-1 (x - mu)
            out[:, k] = self._log_norm[k] - 0.5 * np.einsum('nd,nd->n', sol, sol)
        return out

    def log_prob_std(self, Xs):
        """Mixture log-density in standardized space, (n,)."""
        lp = self.component_log_prob(Xs)
        m = lp.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(lp - m).sum(axis=1, keepdims=True))).ravel()

    def sample_std(self, n_samples, seed=0, shuffle=True, temperature=1.0):
        """(Xs, comp): n_samples points in standardized space and their components.
        Components are drawn as multinomial counts and filled block by block;
        `shuffle` interleaves them (otherwise the output is sorted by component).
        `temperature` scales every component's spread (covariance x T^2) around its
        mean, keeping the mixture weights: T > 1 broadens the distribution, T < 1
        sharpens it."""
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n_samples, self.weights)
        d = self.means.shape[1]
        Xs = np.empty((n_samples, d))
        comp = np.repeat(np.arange(self.n_components), counts)
        start = 0
        for k, c in enumerate(counts):
            z = rng.standard_normal((c, d))
            Xs[start:start + c] = self.means[k] + temperature * (z @ self.chol[k].T)
            start += c
        if shuffle:
            perm = rng.permutation(n_samples)
            Xs, comp = Xs[perm], comp[perm]
        return Xs, comp

    # ---- physical (pz, pt) space ---- #
    def sample(self, n_samples, seed=0, shuffle=True, temperature=1.0):
        """(pz, pt, comp) in physical units (temperature: see sample_std)."""
        Xs, comp = self.sample_std(n_samples, seed=seed, shuffle=shuffle, temperature=temperature)
        X = Xs * self.std + self.mean
        if self.logit:
            pz, pt = logit_inverse(X, self.meta["p_min"], self.meta["p_max"])
            return pz, pt, comp
        X = np.exp(X)
        return X[:, 0], X[:, 1], comp

    def pdf(self, x):
        """Density in physical (pz, pt) units at x of shape (n, 2): the standardized
        density times the Jacobian of the fit space (1 / (prod(std) * pz * pt) for the
        log space, |d(u, v) / d(pz, pt)| / prod(std) for the logit one)."""
        x = np.asarray(x, dtype=np.float64)
        if self.logit:
            uv, log_jac = logit_space(x[:, 0], x[:, 1], self.meta["p_min"], self.meta["p_max"])
            Xs = (uv - self.mean) / self.std
            return np.exp(self.log_prob_std(Xs) + log_jac) / np.prod(self.std)
        Xs = (np.log(x) - self.mean) / self.std
        return np.exp(self.log_prob_std(Xs)) / (np.prod(self.std) * np.prod(x, axis=1))

    # ---- torch ---- #
    def torch_dist(self, device=None, dtype=None):
        """Equivalent torch.distributions mixture in standardized space (sample /
        log_prob on GPU); map samples back with x_s * std + mean, then exp (log space)
        or logit_inverse (logit space)."""
        import torch
        from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
        dtype = dtype or torch.float64
        t = lambda a: torch.as_tensor(a, device=device, dtype=dtype)
        return MixtureSameFamily(Categorical(probs=t(self.weights)),
                                 MultivariateNormal(t(self.means), scale_tril=t(self.chol)))
