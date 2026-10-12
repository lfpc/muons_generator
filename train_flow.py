import h5py
import numpy as np
import torch

import fit_space as fs
import flow_training as ft
from models import MuonFlow
from utils import CHUNK

COLUMNS = ("logit_p", "log_tan_theta", "dphi", "asinh_r", "asinh_z", "charge")


def reader(m):
    """read(g, sl) for flow_training.load: [|p|, v, a, b, c, q] of rows sl; |p| becomes
    u once p_min is known."""
    def read(g, sl):
        px, py, pz, x, y, z, pdg = (g[k][sl].astype(np.float64)
                                    for k in ("px", "py", "pz", "x", "y", "z", "pdg"))
        pt = np.hypot(px, py)
        if not (pz.min() > 0 and pt.min() > 0):
            raise SystemExit("the data has a muon with pz <= 0 or pt = 0")
        abc, _ = fs.position_to_fit(px, py, x, y, z, m)
        return np.column_stack([np.hypot(pz, pt), np.log(pt / pz), abc, -np.sign(pdg)])
    return read


def mu_plus_fraction(data):
    """Rate-weighted fraction of mu+ (q = +1) over all groups: the generator's Bernoulli."""
    num = den = 0.0
    for A in data.values():
        for c in ft.chunks(A):
            w = c[:, -1].astype(np.float64)
            num += w @ (c[:, 5] > 0)
            den += w.sum()
    return num / den


@torch.no_grad()
def stage_nlls(flow, V, batch=1 << 20):
    """Weighted mean NLL of each stage on the validation rows V ([x, w])."""
    tot, wsum = np.zeros(2), 0.0
    for i in range(0, V.shape[0], batch):
        x, w = V[i:i + batch, :-1], V[i:i + batch, -1]
        tot -= [(w * flow.log_prob_momentum(x)).sum().item(),
                (w * flow.log_prob_position(x)).sum().item()]
        wsum += w.sum().item()
    return tot / wsum


if __name__ == "__main__":
    parser = ft.make_parser(__doc__, "outputs/muon_flow.pt")
    parser.add_argument("--p_max", type=float, default=fs.P_MAX, help="Upper bound on |p| [GeV]")
    parser.add_argument("--p_min", type=float, default=None,
                        help="Lower bound on |p| [GeV]; None = 0.9 x the data's smallest")
    parser.add_argument("--r_scale", type=float, default=fs.R_SCALE, help="asinh crossover of r [cm]")
    parser.add_argument("--z_scale", type=float, default=fs.Z_SCALE, help="asinh crossover of z [cm]")
    parser.add_argument("--mom_layers", type=int, default=4, help="Coupling layers of the momentum part")
    parser.add_argument("--mom_hidden", type=int, nargs="+", default=[16, 16],
                        help="Widths of each coupling MLP of the momentum part")
    parser.add_argument("--pos_layers", type=int, default=6, help="Coupling layers of the position part")
    parser.add_argument("--pos_hidden", type=int, nargs="+", default=[16, 16],
                        help="Widths of each coupling MLP of the position part")
    parser.add_argument("--mom_tail_bound", type=float, default=14.0,
                        help="Spline box of the momentum part, in standardized units: the "
                             "|p| tail must die out inside it (at 9 the box ends at 370 GeV "
                             "and the flow's tail collapses from ~330 GeV)")
    parser.add_argument("--mom_num_bins", type=int, default=24,
                        help="Spline bins of the momentum part (more bins for its wider box)")
    parser.set_defaults(tail_bound=14.0)
    for flag in ("--tail_bound", "--num_bins"):        # the shared flags: the position part's
        parser._option_string_actions[flag].help += " (position part; the momentum part has --mom_*)"
    args = parser.parse_args()
    device = ft.setup(args)

    with h5py.File(args.data, "r") as f:           # z0 only has to sit inside the bulk of z
        z = f["z"] if "z" in f else f[ft.LO]["z"]
        z0 = float(np.median(z[:CHUNK]))
    m = {"p_max": args.p_max, "r_scale": args.r_scale, "z0": z0, "z_scale": args.z_scale}

    data = ft.load(args.data, reader(m), len(COLUMNS), args.num_samples)
    p_lo = min(A[:, 0].min() for A in data.values())
    p_hi = max(A[:, 0].max() for A in data.values())
    m["p_min"] = 0.9 * p_lo if args.p_min is None else args.p_min
    if not m["p_min"] < p_lo <= p_hi < m["p_max"]:
        raise SystemExit(f"data |p| in [{p_lo:.4g}, {p_hi:.4g}] GeV is not inside "
                         f"(p_min, p_max) = ({m['p_min']:.4g}, {m['p_max']:.4g})")
    for A in data.values():
        for c in ft.chunks(A):
            c[:, 0] = fs.logit_p(c[:, 0].astype(np.float64), m)
    m["r_max"] = float(m["r_scale"] * np.sinh(max(A[:, 3].max() for A in data.values())))
    m["mu_plus_fraction"] = mu_plus_fraction(data)
    print(f"|p| in [{p_lo:.4g}, {p_hi:.4g}] GeV, p_min {m['p_min']:.4g} GeV, z0 {z0:.4g} cm, "
          f"largest radius {m['r_max']:.4g} cm, mu+ fraction {m['mu_plus_fraction']:.4f}")

    mean, std, wsum = ft.standardize(data)
    bounds = [args.mom_tail_bound] * 2 + [args.tail_bound] * (len(COLUMNS) - 2)
    ft.report_coverage(data, bounds, COLUMNS)
    u_edge = mean[0] + np.array([-1, 1]) * args.mom_tail_bound * std[0]
    p_edge = m["p_min"] + (m["p_max"] - m["p_min"]) / (1 + np.exp(-u_edge))
    print(f"Momentum spline box: |p| in [{p_edge[0]:.4g}, {p_edge[1]:.4g}] GeV")
    train, val = ft.split_val(data, args, device)
    del data
    flow = MuonFlow(momentum=dict(n_layers=args.mom_layers, hidden=args.mom_hidden,
                                  num_bins=args.mom_num_bins, tail_bound=args.mom_tail_bound),
                    position=dict(n_layers=args.pos_layers, hidden=args.pos_hidden,
                                  num_bins=args.num_bins, tail_bound=args.tail_bound)).to(device)
    print(f"Parameters: momentum {sum(p.numel() for p in flow.momentum.parameters()):,}, "
          f"position {sum(p.numel() for p in flow.position.parameters()):,}")
    state, val_nll, history = ft.fit(flow, train, val, args, device)
    flow.load_state_dict(state)
    nll_mom, nll_pos = stage_nlls(flow, val)
    print(f"best val NLL {val_nll:.4f} = momentum {nll_mom:.4f} + position {nll_pos:.4f}")
    ft.finish(flow, state, history, args, "muon_flow", mean, std,
              columns=list(COLUMNS), weight=wsum, **m)
