"""Compare the two-stage generator with the real muons.

Figures (outputs/figs/): the (pz, pt) density and its marginals (combined.png), the
production point in x-y (position_xy.png), 1D marginals of every variable
(marginals.png), radius against pz and |dphi| (radius_Pz.png), transverse position
against momentum (position_momentum.png) and all pairs (corner.png). Metrics: the
normalized W1 distance of each marginal and the rate-weighted NLL of real muons under
the generator (generator.log_prob).

The real side of every figure is cached (--hists) and reused with --load_hists, so the
2.3e9-row file is read once.

Run: python test_generator.py              (first run: reads the data, writes the cache)
     python test_generator.py --load_hists
"""
import argparse
import os

import numpy as np
import torch

import gen_eval as ge
from generator import MuonGenerator
from utils import get_real_muons


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument("--momentum_flow", default="outputs/momentum_flow.pt", help="Stage-1 checkpoint")
parser.add_argument("--position_flow", default="outputs/position_flow.pt", help="Stage-2 checkpoint")
parser.add_argument("--momentum_temperature", type=float, default=1.0,
                    help="Latent temperature of the momentum flow")
parser.add_argument("--data", default=ge.DEFAULT_DATA, help="Real muon file")
parser.add_argument("--data_samples", type=int, default=None, help="Cap on the real muons read (None = all)")
parser.add_argument("--num_samples", type=int, default=None,
                    help="Generated muons (None = as many as real ones)")
parser.add_argument("--nll_rows", type=int, default=5_000_000,
                    help="Real muons (from the head of --data) the NLL is computed on")
parser.add_argument("--hists", default="outputs/real_hists.npz", help="Cache of the real side of every figure")
parser.add_argument("--load_hists", action="store_true", help="Reuse --hists instead of reading --data")
parser.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
parser.add_argument("--show", action="store_true", help="Show the figures as well")
args = parser.parse_args()
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(ge.figs_dir, exist_ok=True)

gen_model = MuonGenerator(args.momentum_flow, args.position_flow, device)
print(f"Generator: {args.momentum_flow} (temperature {args.momentum_temperature:g}) + "
      f"{args.position_flow}, mu+ fraction {gen_model.mu_plus:.4f}, on {device}")

# ---- real side: the data, or the cache of everything the figures need from it ----
if args.load_hists:
    ge.load_cache(args.hists)
    real = real_w = None
    n_real = int(ge.cached("combined/counts", None)[0])
    print(f"Reusing the real histograms in {args.hists} ({n_real:,} muons)")
else:
    ge.use_cache({})                      # filled as the figures are drawn, saved at the end
    real, real_w = ge.load_real(args.data, args.data_samples)
    n_real = real.shape[0]
    ge.cached("combined/counts", lambda: np.array([n_real, real_w.sum()]))

# ---- generated side ----
torch.manual_seed(0)
gen = gen_model.sample(args.num_samples or n_real, args.momentum_temperature,
                       rng=np.random.default_rng(0))
print(f"Generated {gen.shape[0]:,} muons")

# ---- figures and metrics ----
real_phys = None if real is None else real[:, :7]
w1_pzpt = ge.plot_compare(None if real is None else ge.real_pzpt(real), ge.real_pzpt(gen), real_w,
                          "Real (rate-weighted) vs generated", "combined.png",
                          gen_label=f"generated (N={gen.shape[0]:,})", show=args.show)
ge.plot_position_xy(real_phys, gen, real_w, show=args.show)
w1 = ge.plot_position_marginals(real_phys, gen, real_w, show=args.show)
ge.plot_radius_momentum(real_phys, gen, real_w, show=args.show)
ge.plot_position_momentum(real_phys, gen, real_w, show=args.show)
ge.plot_corner_compare(real_phys, gen, real_w, show=args.show)
cos_real = float(ge.cached("marg/cos_dphi", lambda: np.average(
    np.cos(ge.derived_position(real_phys)[1]), weights=real_w)))
ge.report_position_metrics(w1, cos_real, float(np.cos(ge.derived_position(gen)[1]).mean()))
ge.report_metrics({"pz, pt": w1_pzpt})

nll_data = get_real_muons(args.data, args.nll_rows)
w = nll_data[:, 7].astype(np.float64)
nll = -(w @ gen_model.log_prob(nll_data[:, :7])) / w.sum()
print(f"\nRate-weighted NLL of {len(nll_data):,} real muons: {nll:.5f} nats/muon "
      "(density in GeV^-3 m^-3 times P(charge))")

if not args.load_hists:
    ge.save_cache(args.hists)
ge.report_saved_plots()
