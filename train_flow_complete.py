import h5py
import numpy as np

import flow_training as ft
import train_flow_ploss as momentum
from utils import CHUNK

COLS = ('logit_p', 'log_tan_theta', 'dphi', 'r', 'z', 'charge')
DIM = len(COLS)


def reader(m, rng):
    """read(g, sl) for flow_training.load: the fit space of rows sl of g (the momentum
    still as |p|, see train_flow_ploss.read_momentum)."""
    def read(g, sl):
        px, py, pz, x, y = (g[c][sl] for c in ('px', 'py', 'pz', 'x', 'y'))
        dphi = np.arctan2(py, px) - np.arctan2(y, x)
        return np.column_stack([
            momentum.read_momentum(pz, np.hypot(px, py), m['p_max']),
            np.abs((dphi + np.pi) % (2 * np.pi) - np.pi),
            np.arcsinh(np.hypot(x, y) / m['r_scale']),
            np.arcsinh((g['z'][sl] - m['z0']) / m['z_scale']),
            -np.sign(g['pdg'][sl]) + rng.uniform(-1.0, 1.0, px.shape[0]),
        ])
    return read


def to_physical(s, m, rng=None):
    """Fit-space samples (destandardized, as models.sample_flow gives them) -> physical
    (px, py, pz, x, y, z, pdg) float32; `m` holds the fit-space parameters (a checkpoint
    does). The global azimuth is uniform (the beam is azimuthally symmetric) and |dphi|
    places the momentum relative to it, with a random sign. r is clipped to the largest
    radius in the training data: a muon generated metres past it would be nonsense for
    the simulation downstream."""
    rng = np.random.default_rng() if rng is None else rng
    n = s.shape[0]
    if m['columns'][0] == 'logit_p':
        pz, pt = momentum.to_momenta(s[:, :2], m).T
    else:   # checkpoints from before the bounded momentum: (log pz, log(pt + log_eps))
        pz, pt = np.exp(s[:, 0]), np.maximum(np.exp(s[:, 1]) - m['log_eps'], 0.0)
    dphi = np.abs(s[:, 2]) % (2 * np.pi)
    dphi = np.minimum(dphi, 2 * np.pi - dphi) * rng.choice([-1.0, 1.0], size=n)
    r = np.clip(m['r_scale'] * np.sinh(s[:, 3]), 0.0, m['r_max'])
    z = m['z0'] + m['z_scale'] * np.sinh(s[:, 4])
    pdg = np.where(s[:, 5] >= 0.0, -13.0, 13.0)            # pdg = -13 * charge
    phi_r = rng.uniform(0.0, 2 * np.pi, size=n)
    phi_p = phi_r + dphi
    return np.column_stack([pt * np.cos(phi_p), pt * np.sin(phi_p), pz,
                            r * np.cos(phi_r), r * np.sin(phi_r), z, pdg]).astype(np.float32)


if __name__ == "__main__":
    parser = ft.make_parser(__doc__, "outputs/flow_model_complete_ploss.pt")
    momentum.add_args(parser)
    parser.add_argument("--r_scale", type=float, default=2.0,
                        help="Crossover of the asinh on r [m]: 2 brings the whole column "
                             "inside +-6 sigma at a core width within 2%% of log r's")
    parser.add_argument("--z_scale", type=float, default=20.0,
                        help="Crossover of the asinh on z [m], about the half-width of z's core")
    args = parser.parse_args()
    device = ft.setup(args)

    with h5py.File(args.data, 'r') as f:     # z0 only has to sit inside the body of z,
        z = f['z'] if 'z' in f else f[ft.LO]['z']   # so one chunk's median is plenty
        z0 = float(np.median(z[:CHUNK]))
    meta = {"columns": list(COLS), "p_max": args.p_max, "r_scale": args.r_scale,
            "z0": z0, "z_scale": args.z_scale}

    data = ft.load(args.data, reader(meta, np.random.default_rng(args.seed)), DIM,
                   args.num_samples)
    meta["p_min"] = momentum.set_p_min(data, args.p_min, args.p_max)
    meta["r_max"] = float(args.r_scale * np.sinh(max(A[:, 3].max() for A in data.values())))
    print(f"z0 {z0:.4g} m, largest radius {meta['r_max']:.4g} m")
    mean, std, wsum = ft.standardize(data)
    momentum.print_box(mean, std, args.tail_bound, meta)
    ft.report_coverage(data, args.tail_bound, COLS)
    train, val = ft.split_val(data, args, device)
    del data

    flow = ft.make_flow(DIM, args, device)
    state, val_nll, history = ft.fit(flow, train, val, args, device)
    print(f"best val NLL {val_nll:.4f}")
    ft.finish(flow, state, history, args, "flow_complete", mean, std, weight=wsum, **meta)
