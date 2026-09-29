"""Stage 2 of the muon generator: a flow for the production point given the momentum
and the charge, p(a, b, c | u, v, q) in fit_space.py's variables:

    a = |dphi| (position azimuth relative to the momentum's), b = asinh r, c = asinh z,
    conditioned on u, v (the momentum, as in train_momentum_flow.py) and q = +-1.

Stage 1 (generator.py) draws what this flow is conditioned on: the momentum from the
momentum flow, the charge from a Bernoulli with the data's rate-weighted mu+ fraction
(stored here) and the momentum azimuth phi_p uniformly. The beam is azimuthally
symmetric, so only the position's azimuth relative to phi_p, dphi, is physical; its
sign is independent of the charge in the data (P(dphi > 0) = 0.501 for mu+ and mu-)
and is drawn at random.

The momentum bounds (p_min, p_max) are taken from --momentum_flow, so the two flows
always agree on them. Depth matters here, width hardly: 6 coupling layers of width 16
(7.8k parameters) reach the best test NLL of the depth x width sweep we ran.

Run: python train_position_flow.py
"""
import h5py
import numpy as np
import torch

import fit_space as fs
import flow_training as ft
from models import NeuralSplineFlow
from utils import CHUNK

TARGET = ("dphi", "asinh_r", "asinh_z")
CONTEXT = ("logit_p", "log_tan_theta", "charge")


def reader(m):
    """read(g, sl) for flow_training.load: [a, b, c, u, v, q] of rows sl."""
    def read(g, sl):
        px, py, pz, x, y, z, pdg = (g[k][sl].astype(np.float64)
                                    for k in ("px", "py", "pz", "x", "y", "z", "pdg"))
        uv, _ = fs.momentum_to_fit(pz, np.hypot(px, py), m)
        abc, _ = fs.position_to_fit(px, py, x, y, z, m)
        return np.column_stack([abc, uv, -np.sign(pdg)])
    return read


def mu_plus_fraction(data):
    """Rate-weighted fraction of mu+ (q = +1) over all groups: stage 1's Bernoulli."""
    num = den = 0.0
    for A in data.values():
        for c in ft.chunks(A):
            w = c[:, -1].astype(np.float64)
            num += w @ (c[:, 5] > 0)
            den += w.sum()
    return num / den


if __name__ == "__main__":
    parser = ft.make_parser(__doc__, "outputs/position_flow.pt")
    parser.add_argument("--momentum_flow", default="outputs/momentum_flow.pt",
                        help="Stage-1 checkpoint: this flow takes its p_min, p_max")
    parser.add_argument("--r_scale", type=float, default=fs.R_SCALE, help="asinh crossover of r [m]")
    parser.add_argument("--z_scale", type=float, default=fs.Z_SCALE, help="asinh crossover of z [m]")
    parser.set_defaults(n_layers=6, hidden=[16, 16], tail_bound=9.0)
    args = parser.parse_args()
    device = ft.setup(args)

    mck = torch.load(args.momentum_flow, map_location="cpu", weights_only=False)
    with h5py.File(args.data, "r") as f:           # z0 only has to sit inside the bulk of z
        z = f["z"] if "z" in f else f[ft.LO]["z"]
        z0 = float(np.median(z[:CHUNK]))
    m = {"p_min": mck["p_min"], "p_max": mck["p_max"],
         "r_scale": args.r_scale, "z0": z0, "z_scale": args.z_scale}

    data = ft.load(args.data, reader(m), len(TARGET) + len(CONTEXT), args.num_samples)
    if not all(np.isfinite(A[:, 3]).all() for A in data.values()):
        raise SystemExit(f"a muon has |p| outside the momentum flow's ({m['p_min']:g}, {m['p_max']:g}) GeV")
    m["r_max"] = float(m["r_scale"] * np.sinh(max(A[:, 1].max() for A in data.values())))
    m["mu_plus_fraction"] = mu_plus_fraction(data)
    print(f"z0 {z0:.4g} m, largest radius {m['r_max']:.4g} m, mu+ fraction {m['mu_plus_fraction']:.4f}")

    mean, std, wsum = ft.standardize(data)          # target and context columns alike
    ft.report_coverage(data, args.tail_bound, TARGET + CONTEXT)
    train, val = ft.split_val(data, args, device)
    del data
    flow = NeuralSplineFlow(dim=len(TARGET), context_dim=len(CONTEXT), n_layers=args.n_layers,
                            hidden=tuple(args.hidden), num_bins=args.num_bins,
                            tail_bound=args.tail_bound).to(device)
    state, val_nll, history = ft.fit(flow, train, val, args, device)
    print(f"best val NLL {val_nll:.4f}")
    ft.finish(flow, state, history, args, "position_flow", mean, std, columns=list(TARGET),
              context_columns=list(CONTEXT), weight=wsum, **m)
