"""The two-stage muon generator.

    stage 1  momentum (pz, pt)   momentum flow (train_momentum_flow.py), optionally at a
                                 latent temperature -- that flow models nothing else,
                                 so the temperature acts on the momentum alone
             charge q            Bernoulli, P(mu+) = the data's rate-weighted fraction
             azimuth phi_p       uniform in [0, 2 pi): px = pt cos phi_p, py = pt sin phi_p
    stage 2  position (x, y, z)  position flow (train_position_flow.py), given (pz, pt, q),
                                 placed at the azimuth phi_p - dphi

    gen = MuonGenerator(device="cuda")
    muons = gen.sample(1_000_000)          # (n, 7) float32: px, py, pz, x, y, z, pdg
    log_p = gen.log_prob(muons)            # exact log density of physical muons

log_prob is the density in (px, py, pz, x, y, z) [GeV^-3 m^-3] times the probability
of the charge, i.e. what the rate-weighted test NLL of the generator is made of.
"""
import numpy as np
import torch

import fit_space as fs
from models import load_flow


class MuonGenerator:
    def __init__(self, momentum_flow="outputs/momentum_flow.pt",
                 position_flow="outputs/position_flow.pt", device="cpu"):
        self.momentum, self.m = load_flow(momentum_flow, device)
        self.position, self.p = load_flow(position_flow, device)
        for k in ("p_min", "p_max"):               # stage 2 is only defined inside these
            if not np.isclose(self.m[k], self.p[k], rtol=1e-6):
                raise ValueError(f"{k}: momentum flow {self.m[k]:g}, position flow {self.p[k]:g} GeV")
        self.device = device
        self.mu_plus = self.p["mu_plus_fraction"]

    # ---- sampling ---- #
    @torch.no_grad()
    def sample(self, n, momentum_temperature=1.0, rng=None, batch=1 << 21):
        rng = np.random.default_rng() if rng is None else rng
        out = np.empty((n, 7), dtype=np.float32)
        for i in range(0, n, batch):
            k = min(batch, n - i)
            pz, pt = self.sample_momentum(k, momentum_temperature)
            q = np.where(rng.random(k) < self.mu_plus, 1.0, -1.0)
            out[i:i + k] = self.sample_position(pz, pt, q, rng.uniform(0.0, fs.TWO_PI, k), rng)
        return out

    @torch.no_grad()
    def sample_momentum(self, n, temperature=1.0):
        """(pz, pt), float64."""
        uv = self.momentum.sample(n, temperature=temperature, mean=self.m["mean"], std=self.m["std"])
        return fs.fit_to_momentum(uv.double().cpu().numpy(), self.m)

    @torch.no_grad()
    def sample_position(self, pz, pt, q, phi_p, rng):
        """Complete muons (px, py, pz, x, y, z, pdg) for a given stage 1."""
        mean, std = self.p["mean"][:3], self.p["std"][:3]
        abc = self.position.sample(0, mean=mean, std=std, context=self._context(pz, pt, q))
        x, y, z = fs.fit_to_position(abc.double().cpu().numpy(), phi_p, self.p, rng)
        return np.column_stack([pt * np.cos(phi_p), pt * np.sin(phi_p), pz,
                                x, y, z, -13.0 * q]).astype(np.float32)

    def _context(self, pz, pt, q):
        # float rounding can put a sampled |p| a hair outside (p_min, p_max)
        eps = 1e-9 * (self.p["p_max"] - self.p["p_min"])
        p = np.clip(np.hypot(pz, pt), self.p["p_min"] + eps, self.p["p_max"] - eps)
        ctx = np.column_stack([fs.logit_p(p, self.p), np.log(pt / pz), q])
        ctx = (ctx - self.p["mean"][3:]) / self.p["std"][3:]
        return torch.as_tensor(ctx, dtype=torch.float32, device=self.device)

    # ---- density ---- #
    @torch.no_grad()
    def log_prob(self, muons, batch=1 << 20):
        """log p(px, py, pz, x, y, z) + log P(charge) of physical muons, float64."""
        out = np.empty(len(muons))
        for i in range(0, len(muons), batch):
            px, py, pz, x, y, z, pdg = (muons[i:i + batch, j].astype(np.float64) for j in range(7))
            pt, q = np.hypot(px, py), -np.sign(pdg)
            uv, jac_mom = fs.momentum_to_fit(pz, pt, self.m)
            lp_mom = self._log_prob(self.momentum, uv, self.m) + jac_mom
            lp_q = np.where(q > 0, np.log(self.mu_plus), np.log(1.0 - self.mu_plus))
            abc, jac_pos = fs.position_to_fit(px, py, x, y, z, self.p)
            ctx = self._context(pz, pt, q)
            # the sampler folds |dphi| into [0, pi] and reflects r >= 0: sum over preimages
            lp_pos = np.logaddexp.reduce([
                self._log_prob(self.position, np.column_stack([a, b, abc[:, 2]]), self.p, ctx)
                for a in (abc[:, 0], -abc[:, 0], fs.TWO_PI - abc[:, 0])
                for b in (abc[:, 1], -abc[:, 1])]) + jac_pos
            out[i:i + batch] = lp_mom + lp_q + lp_pos - np.log(fs.TWO_PI) - np.log(pt)  # phi_p
        return out

    def _log_prob(self, flow, s, ck, context=None):
        """log density of fit-space points s (unstandardized)."""
        d = s.shape[1]
        xs = torch.as_tensor((s - ck["mean"][:d]) / ck["std"][:d], dtype=torch.float32,
                             device=self.device)
        return flow.log_prob(xs, context).double().cpu().numpy() - np.log(ck["std"][:d]).sum()
