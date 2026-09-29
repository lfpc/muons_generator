"""Stage 1 of the muon generator: a flow for the momentum, p(u, v), with

    u = logit((|p| - p_min) / (p_max - p_min)),   v = log(pt / pz)

(fit_space.py), so every generated muon has p_min < |p| < p_max and pz, pt > 0.
p_max is the 400 GeV beam; p_min defaults to 0.9 x the smallest |p| in the data.
A small flow is enough: 4 coupling layers of width 16 (3.7k parameters) match a
12 x 128 one (258k) -- see compare_gmm_flow.py.

Run: python train_momentum_flow.py
"""
import numpy as np

import fit_space as fs
import flow_training as ft

COLUMNS = ("logit_p", "log_tan_theta")


def read(g, sl):
    """[|p|, v] of rows sl; |p| becomes u once p_min is known."""
    pz = g["pz"][sl].astype(np.float64)
    pt = np.hypot(g["px"][sl], g["py"][sl]).astype(np.float64)
    if not (pz.min() > 0 and pt.min() > 0):
        raise SystemExit("the data has a muon with pz <= 0 or pt = 0")
    return np.column_stack([np.hypot(pz, pt), np.log(pt / pz)])


if __name__ == "__main__":
    parser = ft.make_parser(__doc__, "outputs/momentum_flow.pt")
    parser.add_argument("--p_max", type=float, default=fs.P_MAX, help="Upper bound on |p| [GeV]")
    parser.add_argument("--p_min", type=float, default=None,
                        help="Lower bound on |p| [GeV]; None = 0.9 x the data's smallest")
    parser.set_defaults(n_layers=4, hidden=[16, 16], tail_bound=9.0)
    args = parser.parse_args()
    device = ft.setup(args)

    data = ft.load(args.data, read, len(COLUMNS), args.num_samples)
    p_lo = min(A[:, 0].min() for A in data.values())
    p_hi = max(A[:, 0].max() for A in data.values())
    m = {"p_min": 0.9 * p_lo if args.p_min is None else args.p_min, "p_max": args.p_max}
    if not m["p_min"] < p_lo <= p_hi < m["p_max"]:
        raise SystemExit(f"data |p| in [{p_lo:.4g}, {p_hi:.4g}] GeV is not inside "
                         f"(p_min, p_max) = ({m['p_min']:.4g}, {m['p_max']:.4g})")
    print(f"|p| in [{p_lo:.4g}, {p_hi:.4g}] GeV, p_min {m['p_min']:.4g} GeV")
    for A in data.values():
        for c in ft.chunks(A):
            c[:, 0] = fs.logit_p(c[:, 0].astype(np.float64), m)

    mean, std, wsum = ft.standardize(data)
    ft.report_coverage(data, args.tail_bound, COLUMNS)
    train, val = ft.split_val(data, args, device)
    del data
    flow = ft.make_flow(len(COLUMNS), args, device)
    state, val_nll, history = ft.fit(flow, train, val, args, device)
    print(f"best val NLL {val_nll:.4f}")
    ft.finish(flow, state, history, args, "momentum_flow", mean, std,
              columns=list(COLUMNS), weight=wsum, **m)
