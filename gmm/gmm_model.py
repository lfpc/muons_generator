"""A Gaussian mixture for the muon momentum, as a numpy-only baseline to the momentum
flow: load from JSON, sample and evaluate without scikit-learn or torch.

It lives in the momentum flow's space (fit_space.py): (u, v), standardized by the stored
mean/std, so every sample has p_min < |p| < p_max and pz, pt > 0. The JSON holds the
mixture (weights, means, covariances), mean, std and p_min, p_max; train_gmm.py writes it.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fit_space import fit_to_momentum, momentum_to_fit


def save_gmm(path, weights, means, covariances, mean, std, p_min, p_max):
    arrays = dict(weights=weights, means=means, covariances=covariances, mean=mean, std=std)
    out = {k: np.asarray(v, dtype=np.float64).tolist() for k, v in arrays.items()}
    with open(path, "w") as f:
        json.dump({**out, "p_min": float(p_min), "p_max": float(p_max)}, f, indent=1)


class GMM:
    def __init__(self, weights, means, covariances, mean, std, p_min, p_max, **_):
        self.weights = np.asarray(weights) / np.sum(weights)
        self.means = np.asarray(means, dtype=np.float64)                  # (K, 2)
        self.covariances = np.asarray(covariances, dtype=np.float64)      # (K, 2, 2)
        self.mean, self.std = np.ravel(mean), np.ravel(std)
        self.bounds = {"p_min": p_min, "p_max": p_max}
        self.chol = np.linalg.cholesky(self.covariances)                  # cov = L L^T
        self.chol_inv = np.linalg.inv(self.chol)
        log_det = 2.0 * np.log(np.diagonal(self.chol, axis1=1, axis2=2)).sum(axis=1)
        self._log_norm = np.log(self.weights) - 0.5 * (2 * np.log(2.0 * np.pi) + log_det)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    @property
    def n_components(self):
        return len(self.weights)

    def log_prob_std(self, xs):
        """Mixture log density at standardized points xs, (n,)."""
        lp = np.stack([self._log_norm[k] - 0.5 * np.sum(((xs - self.means[k]) @ self.chol_inv[k].T) ** 2, 1)
                       for k in range(self.n_components)], axis=1)
        return np.logaddexp.reduce(lp, axis=1)

    def sample(self, n, seed=0, temperature=1.0):
        """(pz, pt). `temperature` scales every component's spread around its mean
        (covariance x T^2), keeping the mixture weights. `seed` is an int or a
        np.random.Generator, which is then drawn from directly."""
        rng = np.random.default_rng(seed)
        comp = rng.choice(self.n_components, size=n, p=self.weights)
        z = rng.standard_normal((n, 2))
        xs = self.means[comp] + temperature * np.einsum("nij,nj->ni", self.chol[comp], z)
        return fit_to_momentum(xs * self.std + self.mean, self.bounds)

    def log_pdf(self, pz, pt):
        """Log density in (pz, pt) [GeV^-2]."""
        uv, log_jac = momentum_to_fit(pz, pt, self.bounds)
        return self.log_prob_std((uv - self.mean) / self.std) + log_jac - np.log(self.std).sum()

    def pdf(self, pzpt):
        """Density in (pz, pt) [GeV^-2]."""
        return np.exp(self.log_pdf(pzpt[:, 0], pzpt[:, 1]))
