"""Fit a weighted Beta distribution to the muon z, scaled onto [z_min, z_max].

With t = (z - z_min) / (z_max - z_min), the Beta(a, b) is fit to t by weighted maximum
likelihood, i.e. maximizing sum_i w_i log Beta(t_i; a, b). The limits are the data's range
widened by --pad of it per side, so no muon sits on an endpoint. Saves the fit as JSON and
plots the (weighted) real z against z sampled from the fit.

Run:  python fit_z.py
"""
import argparse
import json

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize, special, stats

from flow_training import ArgFormatter
from gen_eval import save_plot
from utils import DEFAULT_DATA, get_real_muons

p = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
p.add_argument("--data", default=DEFAULT_DATA, help="Muon file")
p.add_argument("--num_samples", type=int, default=0, help="Rows read from the head of the file (0 = all)")
p.add_argument("--n_gen", type=int, default=10_000_000, help="z values sampled from the fit")
p.add_argument("--pad", type=float, default=1e-3, help="Limits widened by this fraction of the range")
p.add_argument("--out", default="outputs/z_beta_fit.json")
args = p.parse_args()

z, w = get_real_muons(args.data, args.num_samples, columns=("z", "weight")).astype(np.float64).T
pad = args.pad * (z.max() - z.min())
lo, hi = z.min() - pad, z.max() + pad
t = (z - lo) / (hi - lo)

# weighted MLE: the log-likelihood only needs the weighted means of log t and log(1 - t)
mlog, mlog1m = np.average(np.log(t), weights=w), np.average(np.log1p(-t), weights=w)
nll = lambda x: special.betaln(*np.exp(x)) - (np.exp(x[0]) - 1) * mlog - (np.exp(x[1]) - 1) * mlog1m
a, b = np.exp(optimize.minimize(nll, np.zeros(2), method="Nelder-Mead",
                                options={"xatol": 1e-8, "fatol": 1e-12}).x)
fit = stats.beta(a, b, loc=lo, scale=hi - lo)
print(f"Beta(a={a:.6g}, b={b:.6g}) on [{lo:.6g}, {hi:.6g}] m")

with open(args.out, "w") as f:
    json.dump({"a": a, "b": b, "loc": lo, "scale": hi - lo}, f, indent=2)
print(f"Saved {args.out}")

gen = fit.rvs(args.n_gen, random_state=0)
bins = np.linspace(lo, hi, 400)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, scale in zip(axes, ("linear", "log")):
    ax.hist(z, bins, weights=w, density=True, histtype="step", color="#52514e", lw=1.2, label="real (weighted)")
    ax.hist(gen, bins, density=True, histtype="step", color="#eb6834", lw=1.5, label="generated (Beta fit)")
    ax.set(xlabel="z [m]", ylabel="density [1/m]", yscale=scale, xlim=(lo, hi))
axes[0].legend(frameon=False)
fig.suptitle(f"z: real vs Beta(a={a:.4g}, b={b:.4g}) on [{lo:.4g}, {hi:.4g}] m")
fig.tight_layout()
save_plot(fig, "z_beta_fit.png")
