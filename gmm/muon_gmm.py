"""The GMM muon generator: complete muons from simple independent pieces, as in the
MuonShieldChallenge project (MuonsGenerator.py), but in this repo's units and frame:

    momentum (pz, pt)  the momentum GMM (gmm_model.py, train_gmm.py)
    azimuth phi_p      uniform in [0, 2 pi): px = pt cos phi_p, py = pt sin phi_p
    x, y               the beam spot: a ring of radius `radius` at a uniform azimuth of its
                       own, smeared by a Gaussian of width `sigma` in x and in y
    z                  a Beta distribution on [loc, loc + scale] (fit_z.py)
    charge q           Bernoulli, P(mu+) = mu_plus_fraction (fit_z.py: the data's
                       rate-weighted fraction; MuonShieldChallenge uses 0.5)

All pieces are independent, so the angles are symmetric by construction. Output and
log_prob follow generator.MuonGenerator, so test_generator.py and simulation_test.py
take either: (n, 7) float32 [px, py, pz, x, y, z, pdg], momenta in GeV and positions in
cm in the data's frame (MuonShieldChallenge converts to m and shifts z by -2.14 m).

A spec JSON (fit_z.py writes it) names the parts; paths are relative to the spec:

    {"gmm": "gmm_model.json", "z": "z_beta_fit.json", "mu_plus_fraction": 0.529,
     "radius": 5.0, "sigma": 1.6}

    gen = GMMMuonGenerator.load("gmm/outputs/muons_gmm.json")
    muons = gen.sample(1_000_000)
    log_p = gen.log_prob(muons)       # GeV^-3 cm^-3 times P(charge), like the flow's
"""
import json
import os
import sys

import numpy as np
from scipy import special, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from gmm_model import GMM                     # noqa: E402

TWO_PI = 2.0 * np.pi


class GMMMuonGenerator:
    def __init__(self, gmm, z_beta, mu_plus_fraction, radius=5.0, sigma=1.6):
        self.gmm = gmm
        self.z = stats.beta(z_beta["a"], z_beta["b"], loc=z_beta["loc"], scale=z_beta["scale"])
        self.z_beta = z_beta
        self.mu_plus = float(mu_plus_fraction)
        self.radius, self.sigma = float(radius), float(sigma)

    @classmethod
    def load(cls, spec_path):
        with open(spec_path) as f:
            spec = json.load(f)
        here = os.path.dirname(os.path.abspath(spec_path))
        with open(os.path.join(here, spec["z"])) as f:
            z_beta = json.load(f)
        return cls(GMM.load(os.path.join(here, spec["gmm"])), z_beta, spec["mu_plus_fraction"],
                   spec.get("radius", 5.0), spec.get("sigma", 1.6))

    def __repr__(self):
        b = self.z_beta
        return (f"GMMMuonGenerator(K={self.gmm.n_components}, z ~ Beta({b['a']:.4g}, {b['b']:.4g}) "
                f"on [{b['loc']:.4g}, {b['loc'] + b['scale']:.4g}] cm, ring {self.radius:g} cm "
                f"+ {self.sigma:g} cm, P(mu+) {self.mu_plus:.4f})")

    # ---- sampling ---- #
    def sample(self, n, momentum_temperature=1.0, rng=None, batch=1 << 22):
        """(n, 7) float32 [px, py, pz, x, y, z, pdg]. `momentum_temperature` is the GMM's:
        it widens every component around its centre (not a latent, as in the flow)."""
        rng = np.random.default_rng() if rng is None else rng
        out = np.empty((n, 7), dtype=np.float32)
        for i in range(0, n, batch):
            k = min(batch, n - i)
            pz, pt = self.gmm.sample(k, seed=rng, temperature=momentum_temperature)
            phi_p, phi_r = rng.uniform(0.0, TWO_PI, (2, k))
            q = np.where(rng.random(k) < self.mu_plus, 1.0, -1.0)
            out[i:i + k] = np.column_stack([
                pt * np.cos(phi_p), pt * np.sin(phi_p), pz,
                self.radius * np.cos(phi_r) + self.sigma * rng.standard_normal(k),
                self.radius * np.sin(phi_r) + self.sigma * rng.standard_normal(k),
                self.z.rvs(k, random_state=rng), -13.0 * q])
        return out

    # ---- density ---- #
    def log_prob(self, muons, batch=1 << 22):
        """log p(px, py, pz, x, y, z) + log P(charge), float64 [GeV^-3 cm^-3]."""
        out = np.empty(len(muons))
        s2 = self.sigma ** 2
        for i in range(0, len(muons), batch):
            px, py, pz, x, y, z, pdg = (muons[i:i + batch, j].astype(np.float64) for j in range(7))
            pt, r = np.hypot(px, py), np.hypot(x, y)
            # (pz, pt, phi_p) -> (px, py, pz): a factor 1 / (2 pi pt)
            lp_mom = self.gmm.log_pdf(pz, pt) - np.log(TWO_PI * pt)
            # ring (x) Gaussian, averaged over the ring's azimuth:
            # p(x, y) = exp(-(r^2 + R^2) / 2 s^2) I0(r R / s^2) / (2 pi s^2); i0e for stability
            lp_xy = (-(r - self.radius) ** 2 / (2 * s2) + np.log(special.i0e(r * self.radius / s2))
                     - np.log(TWO_PI * s2))
            lp_q = np.where(pdg < 0, np.log(self.mu_plus), np.log(1.0 - self.mu_plus))
            out[i:i + batch] = lp_mom + lp_xy + self.z.logpdf(z) + lp_q
        return out
