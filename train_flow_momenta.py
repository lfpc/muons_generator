"""Train a 2D Neural Spline Flow on the muon momentum alone, fit as

    (log pz, log(pt + log_eps))

The baseline momentum model (train_flow_ploss.py is the bounded alternative); test it
with test_momenta.py. Position and charge are left to simulation_test.py to fake. Data,
batching and the fit itself: flow_training.py.

Run: python train_flow_momenta.py --hi_frac 0.5
"""
import numpy as np

import flow_training as ft

COLUMNS = ("log_pz", "log_pt")


def fit_space(pz, pt, m):
    """(pz, pt) -> fit space; `m` holds the fit-space parameters (a checkpoint does)."""
    return np.column_stack([np.log(pz), np.log(pt + m["log_eps"])])


def log_jacobian(pz, pt, m):
    """log |d fit_space / d(pz, pt)|."""
    return -np.log(pz) - np.log(pt + m["log_eps"])


def to_momenta(x, m):
    """fit space -> (pz, pt)."""
    return np.column_stack([np.exp(x[:, 0]), np.maximum(np.exp(x[:, 1]) - m["log_eps"], 0.0)])


if __name__ == "__main__":
    parser = ft.make_parser(__doc__, "outputs/flow_model_momenta.pt")
    parser.add_argument("--log_eps", type=float, default=1e-9,
                        help="Added to pt before the log, which would diverge at pt = 0")
    args = parser.parse_args()
    device = ft.setup(args)
    meta = {"columns": list(COLUMNS), "log_eps": args.log_eps}

    def read(g, sl):
        return fit_space(g["pz"][sl], np.hypot(g["px"][sl], g["py"][sl]), meta)

    data = ft.load(args.data, read, len(COLUMNS), args.num_samples)
    mean, std, wsum = ft.standardize(data)
    ft.report_coverage(data, args.tail_bound, COLUMNS)
    train, val = ft.split_val(data, args, device)
    del data

    flow = ft.make_flow(len(COLUMNS), args, device)
    state, val_nll, history = ft.fit(flow, train, val, args, device)
    print(f"best val NLL {val_nll:.4f}")
    ft.finish(flow, state, history, args, "flow_momenta", mean, std, weight=wsum, **meta)
