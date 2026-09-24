"""Sample, plot and score a momentum-only flow -- train_flow_ploss.py's or
train_flow_momenta.py's, told apart by the checkpoint's `columns` -- against the real
muons: the rate-weighted (pz, pt) density, its marginals and their normalized W1
(gen_eval.plot_compare), the same comparison test_flow.py starts with for a
complete-state model.

Everything the real data contributes to the figures is cached (--hists) and reused with
--load_hists, so another model is compared without re-reading the 2.3e9-row file. A
cache written by test_flow.py works here too: it holds the same momentum entries, plus
the position ones.

Run: python test_momenta.py --model outputs/flow_model_ploss.pt
     python test_momenta.py --model outputs/flow_model_ploss.pt --hists outputs/real_hists.npz --load_hists
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import gen_eval as ge
import train_flow_momenta
import train_flow_ploss
from models import load_flow, sample_flow

MOMENTUM_MODELS = {m.COLUMNS: m for m in (train_flow_momenta, train_flow_ploss)}


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def momentum_model(ckpt):
    """The training module whose fit space the checkpoint uses; None if it is not a
    momentum-only flow."""
    return MOMENTUM_MODELS.get(tuple(ckpt["columns"]))


def add_comparison_args(parser, hists):
    parser.add_argument("--data", default=ge.DEFAULT_DATA, help="Real muon file")
    parser.add_argument("--data_samples", type=int, default=None,
                        help="Cap on the real muons read (None = all)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Generated muons (None = as many as real ones, which for the "
                             "whole file is billions)")
    parser.add_argument("--hists", default=hists,
                        help="Cache of the real side of every figure")
    parser.add_argument("--load_hists", action="store_true",
                        help="Reuse --hists instead of reading --data")
    parser.add_argument("--generated_weighted_sample", action="store_true",
                        help="Add a 2D panel of generated muons at the real data's weighted "
                             "count (tens of billions, streamed in chunks)")
    parser.add_argument("--weighted_sample_chunk", type=int, default=10_000_000)
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument("--show", action="store_true")


def load_reference(args):
    """The real side: (real (N, 8) array, its weights, N, weighted N). With --load_hists
    the file is not read (the arrays are None): the cache holds everything the figures
    need."""
    if args.load_hists:
        ge.load_cache(args.hists)
        n_real, n_weighted = ge.cached("combined/counts", None)
        print(f"Reusing the real histograms in {args.hists}: {int(n_real):,} muons, "
              f"weighted {n_weighted:,.0f}")
        return None, None, int(n_real), float(n_weighted)
    ge.use_cache({})                     # filled as the figures are drawn, saved at the end
    real, w = ge.load_real(args.data, args.data_samples)
    ge.cached("combined/counts", lambda: np.array([real.shape[0], w.sum()]))
    return real, w, real.shape[0], float(w.sum(dtype=np.float64))


def compare_momenta(gen, sample_pzpt, args, real, real_w, n_weighted, gen_density=None):
    """The (pz, pt) figures: the generated density alone (generated_momentum.png) and
    real vs generated with marginals (combined.png). `gen` is the generated (pz, pt)
    sample, `sample_pzpt(n)` draws more (for --generated_weighted_sample), `gen_density`
    replaces the histogram of `gen` in the 2D panels. Returns the W1 per axis."""
    p = np.hypot(gen[:, 0], gen[:, 1])
    print(f"Generated {gen.shape[0]:,}: pz [{gen[:, 0].min():.4g}, {gen[:, 0].max():.4g}] | "
          f"pt [{gen[:, 1].min():.3g}, {gen[:, 1].max():.4g}] | |p| [{p.min():.4g}, {p.max():.4g}]")
    Hg = gen_density if gen_density is not None else ge.hist2d_density(gen)
    label = ("generated (analytic PDF)" if gen_density is not None
             else f"generated (N={gen.shape[0]:,})")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ge.draw_hist2d(ax, Hg, ge.shared_lognorm(Hg), title=label)
    fig.colorbar(im, ax=ax, label="density")
    ge.save_plot(fig, "generated_momentum.png")
    if args.show:
        plt.show()
    plt.close(fig)

    weighted = weighted_label = None
    if args.generated_weighted_sample:
        n_left = int(round(n_weighted))
        print(f"Streaming {n_left:,} generated muons in chunks of {args.weighted_sample_chunk:,} ...")
        t0 = time.time()
        H = np.zeros((len(ge.PZ_BINS) - 1, len(ge.PT_BINS) - 1))
        weighted_label = f"generated (weighted N={n_left:,})"
        while n_left > 0:
            n = min(args.weighted_sample_chunk, n_left)
            H += ge.hist2d_counts(sample_pzpt(n))
            n_left -= n
        weighted = ge.grid_normalize(H)
        print(f"  done in {time.time() - t0:.0f}s")

    return ge.plot_compare(None if real is None else ge.real_pzpt(real), gen, real_w,
                           "Real (rate-weighted) vs generated — NSF", "combined.png",
                           gen_density=Hg, gen_label=label, gen_weighted_density=weighted,
                           gen_weighted_label=weighted_label, show=args.show)


def density_grid(flow, ckpt, model, device):
    """The flow's exact (pz, pt) density on gen_eval's grid: the fit-space density at the
    mapped grid points times the Jacobian of the map, and zero where the map is undefined
    (outside train_flow_ploss's p_min < |p| < p_max)."""
    grid, shape = ge.grid_points()
    pz, pt = grid[:, 0], grid[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = model.fit_space(pz, pt, ckpt)
        log_jac = model.log_jacobian(pz, pt, ckpt)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(log_jac)
    xs = ((x[ok] - ckpt["mean"]) / ckpt["std"]).astype(np.float32)
    with torch.no_grad():
        log_p = flow.log_prob(torch.from_numpy(xs).to(device)).cpu().numpy()
    dens = np.zeros(grid.shape[0])
    dens[ok] = np.exp(log_p - np.log(ckpt["std"]).sum() + log_jac[ok])
    return ge.grid_normalize(dens).reshape(shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
    parser.add_argument("--model", default="outputs/flow_model_ploss.pt")
    parser.add_argument("--pdf", action="store_true",
                        help="Draw the generated 2D density from the flow's exact PDF "
                             "instead of a histogram of the samples")
    add_comparison_args(parser, "outputs/real_hists_momenta.npz")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ge.figs_dir, exist_ok=True)

    flow, ckpt = load_flow(args.model, device)
    model = momentum_model(ckpt)
    if model is None:
        raise SystemExit(f"{args.model} is not a momentum-only flow (columns "
                         f"{ckpt['columns']}); a complete-state one goes to test_flow.py")
    print(f"Loaded {args.model}: {model.__name__} fit space {ckpt['columns']}, on {device}")

    real, real_w, n_real, n_weighted = load_reference(args)
    torch.manual_seed(0)

    def sample_pzpt(n):
        return sample_flow(flow, ckpt, n, lambda s: model.to_momenta(s, ckpt))

    gen = sample_pzpt(args.num_samples or n_real)
    w1 = compare_momenta(gen, sample_pzpt, args, real, real_w, n_weighted,
                         gen_density=density_grid(flow, ckpt, model, device) if args.pdf else None)
    ge.report_metrics({"pz, pt": w1})
    if not args.load_hists:
        ge.save_cache(args.hists)
    ge.report_saved_plots()
