"""The two-stage muon generator, on one flow (models.MuonFlow, train_flow.py):

    stage 1  momentum (pz, pt)   the flow's momentum part, optionally at a latent
                                 temperature -- that part models nothing else, so the
                                 temperature acts on the momentum alone
             charge q            Bernoulli, P(mu+) = the data's rate-weighted fraction
             azimuth phi_p       uniform in [0, 2 pi): px = pt cos phi_p, py = pt sin phi_p
    stage 2  position (x, y, z)  the flow's position part, given (pz, pt, q), placed at
                                 the azimuth phi_p - dphi

    gen = MuonGenerator(device="cuda")
    muons = gen.sample(1_000_000)          # (n, 7) float32: px, py, pz, x, y, z, pdg
    log_p = gen.log_prob(muons)            # exact log density of physical muons

log_prob is the density in (px, py, pz, x, y, z) [GeV^-3 cm^-3] times the probability
of the charge, i.e. what the rate-weighted test NLL of the generator is made of.
"""
import json
import os
import sys

import numpy as np
import torch

import fit_space as fs
from models import load_flow


class MuonGenerator:
    def __init__(self, flow="outputs/muon_flow.pt", device="cpu"):
        self.flow, self.m = load_flow(flow, device)
        self.mean, self.std = self.m["mean"], self.m["std"]
        self.device = device
        self.mu_plus = self.m["mu_plus_fraction"]

    # ---- sampling ---- #
    @torch.no_grad()
    def sample(self, n, momentum_temperature=1.0, rng=None, batch=1 << 21):
        rng = np.random.default_rng() if rng is None else rng
        out = np.empty((n, 7), dtype=np.float32)
        for i in range(0, n, batch):
            k = min(batch, n - i)
            q = np.where(rng.random(k) < self.mu_plus, 1.0, -1.0)
            qs = torch.as_tensor((q - self.mean[5]) / self.std[5], dtype=torch.float32,
                                 device=self.device)
            s = self.flow.sample(qs, momentum_temperature).double().cpu().numpy()
            s = s * self.std + self.mean
            pz, pt = fs.fit_to_momentum(s[:, 0:2], self.m)
            phi_p = rng.uniform(0.0, fs.TWO_PI, k)
            x, y, z = fs.fit_to_position(s[:, 2:5], phi_p, self.m, rng)
            out[i:i + k] = np.column_stack([pt * np.cos(phi_p), pt * np.sin(phi_p), pz,
                                            x, y, z, -13.0 * q])
        return out

    # ---- density ---- #
    @torch.no_grad()
    def log_prob(self, muons, batch=1 << 20):
        """log p(px, py, pz, x, y, z) + log P(charge) of physical muons, float64."""
        out = np.empty(len(muons))
        for i in range(0, len(muons), batch):
            px, py, pz, x, y, z, pdg = (muons[i:i + batch, j].astype(np.float64) for j in range(7))
            pt, q = np.hypot(px, py), -np.sign(pdg)
            uv, jac_mom = fs.momentum_to_fit(pz, pt, self.m)
            abc, jac_pos = fs.position_to_fit(px, py, x, y, z, self.m)
            xs = self._standardize(np.column_stack([uv, abc, q]))
            lp_mom = self.flow.log_prob_momentum(xs).double().cpu().numpy()
            lp_q = np.where(q > 0, np.log(self.mu_plus), np.log(1.0 - self.mu_plus))
            # the sampler folds |dphi| into [0, pi] and reflects r >= 0: sum over preimages
            lp_pos = []
            for a in (abc[:, 0], -abc[:, 0], fs.TWO_PI - abc[:, 0]):
                for b in (abc[:, 1], -abc[:, 1]):
                    xs = self._standardize(np.column_stack([uv, a, b, abc[:, 2], q]))
                    lp_pos.append(self.flow.log_prob_position(xs).double().cpu().numpy())
            lp_pos = np.logaddexp.reduce(lp_pos)
            out[i:i + batch] = (lp_mom + jac_mom + lp_q + lp_pos + jac_pos
                                - np.log(self.std[:5]).sum()     # the standardization
                                - np.log(fs.TWO_PI) - np.log(pt))  # phi_p
        return out

    def _standardize(self, s):
        return torch.as_tensor((s - self.mean) / self.std, dtype=torch.float32, device=self.device)


def model_kind(path):
    """'flow' for a flow checkpoint (.pt, train_flow.py), 'gmm' for a GMM muon generator
    spec (.json, gmm/fit_z.py)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        return "flow"
    if ext == ".json":
        with open(path) as f:
            spec = json.load(f)
        if not {"gmm", "z", "mu_plus_fraction"} <= set(spec):
            raise ValueError(f"{path} is not a GMM muon generator spec (gmm/fit_z.py writes "
                             "gmm/outputs/muons_gmm.json); a momentum-only GMM "
                             "(gmm_model.json) cannot generate complete muons")
        return "gmm"
    raise ValueError(f"{path}: unknown model file (a flow .pt or a GMM spec .json)")


def load_generator(path="outputs/muon_flow.pt", device="cpu"):
    """The generator stored in `path`, whichever kind it is (model_kind): the flow
    (MuonGenerator) or the GMM muon generator (gmm/muon_gmm.py). Both give
    sample(n, momentum_temperature, rng), log_prob(muons) and mu_plus."""
    if model_kind(path) == "flow":
        return MuonGenerator(path, device)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gmm"))
    from muon_gmm import GMMMuonGenerator
    return GMMMuonGenerator.load(path)


DEFAULT_MODEL = {False: "outputs/muon_flow.pt", True: "gmm/outputs/muons_gmm.json"}


def resolve_model(model, gmm):
    """The model path of a script's --model / --gmm flags: --model, or the default of the
    kind --gmm selects. The file must be of that kind, so outputs are never tagged wrong."""
    model = model or DEFAULT_MODEL[gmm]
    kind = model_kind(model)
    if (kind == "gmm") != gmm:
        raise SystemExit(f"--model {model} is a {kind} model: "
                         + ("drop --gmm" if gmm else "add --gmm to use a GMM spec"))
    return model
