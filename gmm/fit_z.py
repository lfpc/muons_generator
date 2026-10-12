"""Fit the non-momentum parts of the GMM muon generator (muon_gmm.py) and write its spec:

  z       a Beta distribution on [z_min, z_max]: with t = (z - z_min) / (z_max - z_min),
          Beta(a, b) is fit to t by weighted maximum likelihood, i.e. maximizing
          sum_i w_i log Beta(t_i; a, b). The limits are the data's range widened by --pad
          of it per side, so no muon sits on an endpoint.
  charge  the rate-weighted mu+ fraction, P(mu+) of the generator's Bernoulli.

x, y are the fixed beam spot (--radius, --sigma), as in MuonShieldChallenge.

The file is streamed in chunks (2.3e9 rows do not fit in memory): one pass for the range
of z and the charge fraction, one for the fit's sufficient statistics (the weighted means
of log t and log(1 - t)) and the histogram of the figure. Positions are in cm.

Writes --out (the Beta fit), --spec (the generator's spec, pointing at --gmm and --out)
and gmm/outputs/figs/z_beta_fit.png (the weighted real z against the fit).

Run:  python gmm/fit_z.py
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize, special, stats
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gen_eval as ge                        # noqa: E402
from flow_training import ArgFormatter       # noqa: E402
from utils import CHUNK, DEFAULT_DATA        # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
p.add_argument("--data", default=DEFAULT_DATA, help="Muon file (the pooled table)")
p.add_argument("--num_samples", type=int, default=0, help="Rows read from the head of the file (0 = all)")
p.add_argument("--pad", type=float, default=1e-3, help="Limits widened by this fraction of the range")
p.add_argument("--radius", type=float, default=5.0, help="Beam-spot ring radius [cm]")
p.add_argument("--sigma", type=float, default=1.6, help="Beam-spot Gaussian smear [cm]")
p.add_argument("--gmm", default=os.path.join(HERE, "outputs/gmm_model.json"),
               help="Momentum GMM the spec points at (train_gmm.py)")
p.add_argument("--out", default=os.path.join(HERE, "outputs/z_beta_fit.json"), help="Beta fit of z")
p.add_argument("--spec", default=os.path.join(HERE, "outputs/muons_gmm.json"),
               help="Spec of the GMM muon generator (muon_gmm.py)")
p.add_argument("--bins", type=int, default=400, help="Bins of the figure")
args = p.parse_args()


def chunks(f, n, columns):
    for i in tqdm(range(0, n, 4 * CHUNK), unit="chunk", mininterval=2.0):
        sl = slice(i, min(i + 4 * CHUNK, n))
        yield [f[c][sl].astype(np.float64) for c in columns]


with h5py.File(args.data, "r") as f:
    n = f["z"].shape[0] if not args.num_samples else min(args.num_samples, f["z"].shape[0])
    print(f"{n:,} muons from {args.data}")

    # pass 1: the range of z and the charge fraction
    z_min, z_max, w_tot, w_plus = np.inf, -np.inf, 0.0, 0.0
    for z, pdg, w in chunks(f, n, ("z", "pdg", "weight")):
        z_min, z_max = min(z_min, z.min()), max(z_max, z.max())
        w_tot += w.sum()
        w_plus += w[pdg < 0].sum()                # pdg -13 is mu+
    pad = args.pad * (z_max - z_min)
    lo, hi = z_min - pad, z_max + pad
    mu_plus = w_plus / w_tot

    # pass 2: the weighted means of log t, log(1 - t), and the histogram
    edges = np.linspace(lo, hi, args.bins + 1)
    hist, s_log, s_log1m = np.zeros(args.bins), 0.0, 0.0
    for z, w in chunks(f, n, ("z", "weight")):
        t = (z - lo) / (hi - lo)
        s_log += w @ np.log(t)
        s_log1m += w @ np.log1p(-t)
        hist += np.histogram(z, edges, weights=w)[0]
mlog, mlog1m = s_log / w_tot, s_log1m / w_tot

nll = lambda x: special.betaln(*np.exp(x)) - (np.exp(x[0]) - 1) * mlog - (np.exp(x[1]) - 1) * mlog1m
a, b = np.exp(optimize.minimize(nll, np.zeros(2), method="Nelder-Mead",
                                options={"xatol": 1e-8, "fatol": 1e-12}).x)
fit = stats.beta(a, b, loc=lo, scale=hi - lo)
print(f"z: Beta(a={a:.6g}, b={b:.6g}) on [{lo:.6g}, {hi:.6g}] cm; P(mu+) = {mu_plus:.6f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"a": a, "b": b, "loc": lo, "scale": hi - lo}, f, indent=2)
spec_dir = os.path.dirname(os.path.abspath(args.spec))
with open(args.spec, "w") as f:
    json.dump({"description": "GMM muon generator (gmm/muon_gmm.py); positions in cm",
               "gmm": os.path.relpath(os.path.abspath(args.gmm), spec_dir),
               "z": os.path.relpath(os.path.abspath(args.out), spec_dir),
               "mu_plus_fraction": mu_plus, "radius": args.radius, "sigma": args.sigma}, f, indent=2)
print(f"Saved {args.out} and {args.spec}")

ge.figs_dir = os.path.join(HERE, "outputs/figs/")
os.makedirs(ge.figs_dir, exist_ok=True)
density = hist / (hist.sum() * np.diff(edges))
centres = 0.5 * (edges[1:] + edges[:-1])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, scale in zip(axes, ("linear", "log")):
    ax.stairs(density, edges, color="#52514e", lw=1.2, label="real (weighted)")
    ax.plot(centres, fit.pdf(centres), color="#eb6834", lw=1.5, label="Beta fit")
    ax.set(xlabel="z [cm]", ylabel="density [1/cm]", yscale=scale, xlim=(lo, hi))
axes[0].legend(frameon=False)
fig.suptitle(f"z: real vs Beta(a={a:.4g}, b={b:.4g}) on [{lo:.4g}, {hi:.4g}] cm")
fig.tight_layout()
ge.save_plot(fig, "z_beta_fit.png")
