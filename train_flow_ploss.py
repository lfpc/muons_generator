"""Train a 2D Neural Spline Flow on the muon momentum, in coordinates that make the
physical bounds part of the model instead of something to clip or reject afterwards.

Every muon comes from a 400 GeV proton, so |p| < p_max, and the data has a threshold at
a few GeV (by default p_min is set just below the smallest |p| in the data). The flow is
fit on

    u = logit((|p| - p_min) / (p_max - p_min))    how much momentum was lost from the beam
    v = log(tan theta) = log(pt / pz)             the polar angle, in the log domain

and a sample maps back through |p| = p_min + (p_max - p_min) * sigmoid(u), tan theta = e^v,
so every generated muon has p_min < |p| < p_max and pz, pt > 0 by construction.

Why a logit and not just log(p_max - |p|): most muons have |p| of a few tens of GeV,
where log(p_max - |p|) is nearly flat (std 0.06, the 373 GeV tail 45 sigma out), so the
bulk of the data gets crushed into a sliver of the fit space. The logit is
~ -log(p_max - |p|) near the beam momentum, which is where the bound is needed, and
~ log(|p| - p_min) in the bulk, which also turns the steep onset just above threshold
into a gentle slope the spline can follow. log(tan theta) is ~ log theta for the small
angles here and runs to +inf at theta = pi/2, so pz can never come out negative.

train_flow_complete.py uses the same two columns for its momentum. Test this model with
test_momenta.py. Data, batching and the fit itself: flow_training.py.

Run: python train_flow_ploss.py --hi_frac 0.5
"""
import numpy as np

import flow_training as ft

COLUMNS = ("logit_p", "log_tan_theta")


def logit_p(p, m):
    """|p| -> u; `m` holds p_min, p_max (a checkpoint does)."""
    return np.log(p - m["p_min"]) - np.log(m["p_max"] - p)


def fit_space(pz, pt, m):
    """(pz, pt) -> (u, v)."""
    return np.column_stack([logit_p(np.hypot(pz, pt), m), np.log(pt / pz)])


def log_jacobian(pz, pt, m):
    """log |d(u, v) / d(pz, pt)| = log(du/d|p|) + log |d(|p|, v) / d(pz, pt)|, the latter
    being |p| / (pz pt)."""
    p = np.hypot(pz, pt)
    lo, hi = m["p_min"], m["p_max"]
    return (np.log(hi - lo) - np.log(p - lo) - np.log(hi - p)
            + np.log(p) - np.log(pz) - np.log(pt))


def to_momenta(x, m):
    """(u, v) -> (pz, pt)."""
    x = np.asarray(x, dtype=np.float64)
    p = m["p_min"] + (m["p_max"] - m["p_min"]) / (1.0 + np.exp(-x[:, 0]))
    cos = 1.0 / np.sqrt(1.0 + np.exp(2.0 * x[:, 1]))
    return np.column_stack([p * cos, p * cos * np.exp(x[:, 1])])


# --- used by both training scripts ---------------------------------------------------

def add_args(parser):
    parser.add_argument("--p_max", type=float, default=400.0,
                        help="Upper bound on |p|: the proton beam momentum [GeV]")
    parser.add_argument("--p_min", type=float, default=None,
                        help="Lower bound on |p| [GeV], below the data minimum; if omitted, "
                             "0.9 times the smallest |p| in the data")
    # the logit stretches the high-|p| end: the 376 GeV tail sits at ~8.6 sigma
    parser.set_defaults(tail_bound=9.0)


def read_momentum(pz, pt, p_max):
    """[|p|, v] for one chunk of the file. u needs p_min, which can depend on the whole
    file, so set_p_min turns the first column into u once everything is loaded."""
    pz, pt = pz.astype(np.float64), pt.astype(np.float64)
    p = np.hypot(pz, pt)
    if not (p.max() < p_max and pz.min() > 0 and pt.min() > 0):
        raise SystemExit(f"max |p| {p.max():.4g}, min pz {pz.min():.4g}, min pt {pt.min():.4g} "
                         "in the data -- needs |p| < p_max and pz, pt > 0")
    return np.column_stack([p, np.log(pt / pz)])


def set_p_min(data, p_min, p_max):
    """Turn column 0 of every group from |p| into u, in place. Returns p_min, which
    defaults to 0.9 times the smallest |p| in the data."""
    p_data = float(min(A[:, 0].min() for A in data.values()))
    p_min = 0.9 * p_data if p_min is None else p_min
    if not p_min < p_data:
        raise SystemExit(f"--p_min {p_min:g} must be below the smallest |p| in the data, {p_data:.4g}")
    print(f"Smallest |p| in the data {p_data:.4g} GeV -> p_min {p_min:.4g} GeV")
    for A in data.values():
        for c in ft.chunks(A):
            c[:, 0] = logit_p(c[:, 0].astype(np.float64), {"p_min": p_min, "p_max": p_max})
    return p_min


def print_box(mean, std, bound, m):
    """The spline box +-bound (standardized) in physical |p| and theta: the flow can
    essentially not generate outside it."""
    u = mean[0] + np.array([-bound, bound]) * std[0]
    p = m["p_min"] + (m["p_max"] - m["p_min"]) / (1.0 + np.exp(-u))
    theta = np.arctan(np.exp(mean[1] + np.array([-bound, bound]) * std[1]))
    print(f"Spline box: |p| in [{p[0]:.4g}, {p[1]:.4g}] GeV, "
          f"theta in [{theta[0]:.2e}, {theta[1]:.3f}] rad")


if __name__ == "__main__":
    parser = ft.make_parser(__doc__, "outputs/flow_model_ploss.pt")
    add_args(parser)
    args = parser.parse_args()
    device = ft.setup(args)

    def read(g, sl):
        return read_momentum(g["pz"][sl], np.hypot(g["px"][sl], g["py"][sl]), args.p_max)

    data = ft.load(args.data, read, len(COLUMNS), args.num_samples)
    meta = {"columns": list(COLUMNS), "p_max": args.p_max,
            "p_min": set_p_min(data, args.p_min, args.p_max)}
    mean, std, wsum = ft.standardize(data)
    print_box(mean, std, args.tail_bound, meta)
    ft.report_coverage(data, args.tail_bound, COLUMNS)
    train, val = ft.split_val(data, args, device)
    del data

    flow = ft.make_flow(len(COLUMNS), args, device)
    state, val_nll, history = ft.fit(flow, train, val, args, device)
    print(f"best val NLL {val_nll:.4f}")
    ft.finish(flow, state, history, args, "flow_ploss", mean, std, weight=wsum, **meta)
