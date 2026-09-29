"""Sample, plot and score a complete-state flow (train_flow_complete.py) against the real
muons: first the (pz, pt) comparison of test_momenta.py, then production position,
momentum azimuth and charge (gen_eval's plot_position_* panels).

A conditional flow (train_flow_conditional.py) is sampled in its two stages: momentum from
the momentum flow --momentum_model (at latent temperature --momentum_temperature), charge
and momentum azimuth, then the position from the conditional flow given them.

The real side of every figure is cached (--hists) and reused with --load_hists, so
another model is compared without re-reading the 2.3e9-row file.

Run: python test_flow.py --load_hists             (the two-stage pair, the default)
     python test_flow.py --model <joint complete-state flow>.pt --load_hists
"""
import argparse
import os

import numpy as np
import torch

import gen_eval as ge
from models import load_flow, sample_flow
from test_momenta import ArgFormatter, add_comparison_args, load_reference, compare_momenta
from train_flow_complete import to_physical
import train_flow_conditional as conditional

parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument("--model", default="outputs/flow_model_conditional.pt")
parser.add_argument("--momentum_model", default="outputs/flow_model_ploss.pt",
                    help="Conditional --model only: the momentum flow (train_flow_ploss.py) "
                         "of its first stage")
parser.add_argument("--momentum_temperature", type=float, default=1.0,
                    help="Conditional --model only: latent temperature of the momentum flow")
add_comparison_args(parser, "outputs/real_hists.npz")
args = parser.parse_args()
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(ge.figs_dir, exist_ok=True)

flow, ckpt = load_flow(args.model, device)
if "r" not in ckpt["columns"]:
    raise SystemExit(f"{args.model} is not a complete-state flow (columns {ckpt['columns']}); "
                     "a momentum-only one goes to test_momenta.py")
print(f"Loaded {args.model}: fit space {ckpt['columns']}, on {device}")
momentum_flow = None
if "context_columns" in ckpt:
    momentum_flow, momentum_ckpt = load_flow(args.momentum_model, device)
    conditional.check_pair(momentum_ckpt, ckpt)
    print(f"  conditional on {ckpt['context_columns']}: momentum from {args.momentum_model} "
          f"(temperature {args.momentum_temperature:g}), mu+ fraction {ckpt['mu_plus_fraction']:.4f}")

real, real_w, n_real, n_weighted = load_reference(args)
torch.manual_seed(0)
# seeded: to_physical draws the global azimuth and the dphi sign, so without it two runs
# of the same model would give different x, y and the figures would never compare
rng = np.random.default_rng(0)


def sample_muons(n):
    if momentum_flow is not None:
        return conditional.generate(momentum_flow, momentum_ckpt, flow, ckpt, n,
                                    momentum_temperature=args.momentum_temperature, rng=rng)
    return sample_flow(flow, ckpt, n, lambda s: to_physical(s, ckpt, rng))


gen = sample_muons(args.num_samples or n_real)
w1 = compare_momenta(ge.real_pzpt(gen), lambda n: ge.real_pzpt(sample_muons(n)), args,
                     real, real_w, n_weighted)

# real[:, :7] is the physical state in the layout to_physical also produces
real_phys = real[:, :7] if real is not None else None
print(f"Comparing positions: {n_real:,} real vs {gen.shape[0]:,} generated.")
ge.plot_position_xy(real_phys, gen, real_w, show=args.show)
pos_metrics = ge.plot_position_marginals(real_phys, gen, real_w, show=args.show)
ge.plot_radius_momentum(real_phys, gen, real_w, show=args.show)
ge.plot_position_momentum(real_phys, gen, real_w, show=args.show)
ge.plot_corner_compare(real_phys, gen, real_w, show=args.show)
cos_real = float(ge.cached("marg/cos_dphi", lambda: np.average(
    np.cos(ge.derived_position(real_phys)[1]), weights=real_w)))
ge.report_position_metrics(pos_metrics, cos_real, float(np.cos(ge.derived_position(gen)[1]).mean()))

ge.report_metrics({"pz, pt": w1})
if not args.load_hists:
    ge.save_cache(args.hists)
ge.report_saved_plots()
