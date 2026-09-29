"""Compare the momentum GMM (gmm_model.py) with the real muons: the (pz, pt) density and
its marginals (combined_gmm.png), their normalized W1 and the rate-weighted NLL.

The real side comes from the same cache as test_generator.py (--hists, --load_hists).

Run: python test_gmm.py --load_hists
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gen_eval as ge
from gmm_model import GMM
from utils import get_real_muons

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument("--model", default="outputs/gmm_model.json", help="GMM from train_gmm.py")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--data", default=ge.DEFAULT_DATA, help="Real muon file")
parser.add_argument("--data_samples", type=int, default=None, help="Cap on the real muons read (None = all)")
parser.add_argument("--num_samples", type=int, default=100_000_000, help="Generated muons")
parser.add_argument("--nll_rows", type=int, default=5_000_000, help="Real muons the NLL is computed on")
parser.add_argument("--hists", default=os.path.join(ROOT, "outputs/real_hists.npz"),
                    help="Cache of the real side of the figures")
parser.add_argument("--load_hists", action="store_true", help="Reuse --hists instead of reading --data")
parser.add_argument("--show", action="store_true", help="Show the figures as well")
args = parser.parse_args()
ge.figs_dir = os.path.join(ROOT, "outputs/figs/")
os.makedirs(ge.figs_dir, exist_ok=True)

gmm = GMM.load(args.model)
print(f"Loaded {args.model}: K={gmm.n_components}")
if args.load_hists:
    ge.load_cache(args.hists)
    real = real_w = None
else:
    ge.use_cache({})
    real, real_w = ge.load_real(args.data, args.data_samples)

pz, pt = gmm.sample(args.num_samples, temperature=args.temperature)
w1 = ge.plot_compare(None if real is None else ge.real_pzpt(real), np.column_stack([pz, pt]), real_w,
                     f"Real (rate-weighted) vs GMM (K={gmm.n_components})", "combined_gmm.png",
                     gen_label=f"GMM (N={args.num_samples:,})", show=args.show)
ge.report_metrics({"pz, pt": w1})

data = get_real_muons(args.data, args.nll_rows, columns=("px", "py", "pz", "weight")).astype(np.float64)
w = data[:, 3].astype(np.float64)
nll = -(w @ np.log(gmm.pdf(np.column_stack([data[:, 2], np.hypot(data[:, 0], data[:, 1])])))) / w.sum()
print(f"Rate-weighted NLL of (pz, pt) over {len(data):,} real muons: {nll:.5f} nats/muon")
if not args.load_hists:
    ge.save_cache(args.hists)
ge.report_saved_plots()
