"""Train the second stage of a two-stage muon generator: a conditional Neural Spline Flow
for the production position given the momentum and charge,

    p(|dphi|, r, z | u, v, q),   (u, v) = (logit of the bounded |p|, log tan theta),

with q = +-1 the charge (mu+ = +1) and dphi = phi_p - phi_r the azimuth of the position
relative to the momentum's; r and z in train_flow_complete.py's asinh fit space.

The first stage draws what the position is conditioned on:
  - (pz, pt) from a momentum-only flow (train_flow_ploss.py), optionally tempered by its
    latent temperature -- that flow models nothing else, so this acts on the momentum only;
  - the charge from a Bernoulli with the rate-weighted mu+ fraction of the data (stored
    in the checkpoint);
  - phi_p uniform in [0, 2 pi), which splits pt into px, py.
generate() then draws (|dphi|, r, z) from this flow, gives dphi a random sign (the data's
sign of dphi is independent of the charge: P(dphi > 0) = 0.501 for either) and places the
muon at phi_r = phi_p - dphi. Conditioning on phi_p itself would add nothing beyond this:
the beam is azimuthally symmetric, so only the position's azimuth relative to phi_p is
physical.

The two flows must share the momentum bounds (p_min, p_max): train this one with the
momentum flow's --p_min. Data, batching and the fit are flow_training.py's.
Test the pair with  python test_flow.py --model outputs/flow_model_conditional.pt

Run:  python train_flow_conditional.py --p_min 3.0
"""
import h5py
import numpy as np
import torch

import flow_training as ft
import train_flow_complete as complete
import train_flow_ploss as momentum
from models import NeuralSplineFlow
from utils import CHUNK

TARGET = ("dphi", "r", "z")
CONTEXT = momentum.COLUMNS + ("charge",)
DIM, CONTEXT_DIM = len(TARGET), len(CONTEXT)
TWO_PI = 2.0 * np.pi


def reader(m, rng):
    """read(g, sl) for flow_training.load: [|dphi|, asinh r, asinh z, |p|, v, q] --
    complete.reader's columns, target first, the charge undone from its dequantization
    (sign of q + U(-1, 1) is q). |p| becomes u once p_min is known (set_p_min)."""
    read = complete.reader(m, rng)

    def fn(g, sl):
        s = read(g, sl)                            # [|p|, v, dphi, r, z, q + U(-1, 1)]
        return np.column_stack([s[:, 2:5], s[:, :2], np.sign(s[:, 5])])
    return fn


def set_p_min(data, p_min, p_max):
    """Turn the |p| column (DIM) of every group into u, in place; returns p_min, by
    default 0.9 x the smallest |p| in the data (as train_flow_ploss.set_p_min)."""
    p_data = float(min(A[:, DIM].min() for A in data.values()))
    p_min = 0.9 * p_data if p_min is None else p_min
    if not p_min < p_data:
        raise SystemExit(f"--p_min {p_min:g} must be below the smallest |p| in the data, {p_data:.4g}")
    print(f"Smallest |p| in the data {p_data:.4g} GeV -> p_min {p_min:.4g} GeV")
    for A in data.values():
        for c in ft.chunks(A):
            c[:, DIM] = momentum.logit_p(c[:, DIM].astype(np.float64), {"p_min": p_min, "p_max": p_max})
    return p_min


def mu_plus_fraction(data):
    """Rate-weighted fraction of mu+ over all groups: the first stage's Bernoulli."""
    num = den = 0.0
    for A in data.values():
        for c in ft.chunks(A):
            w = c[:, -1].astype(np.float64)
            num += w @ (c[:, DIM + 2] > 0)
            den += w.sum()
    return num / den


# --------------------------------------------------------------------------- #
# generation: first stage (momentum flow, charge, phi_p), then the position
# --------------------------------------------------------------------------- #
def check_pair(momentum_ckpt, ckpt):
    """The momentum flow must generate inside the momentum range this flow was fit on."""
    for k in ("p_min", "p_max"):
        if not np.isclose(momentum_ckpt[k], ckpt[k], rtol=1e-6):
            raise ValueError(f"momentum flow {k} = {momentum_ckpt[k]:g} GeV but the conditional "
                             f"flow's is {ckpt[k]:g} GeV: retrain it with --{k} {momentum_ckpt[k]:g}")


@torch.no_grad()
def sample_position(flow, ckpt, pz, pt, q, phi_p, rng=None, batch=1 << 21):
    """Physical muons (px, py, pz, x, y, z, pdg), float32, one per given first-stage
    state (pz, pt, charge q = +-1, momentum azimuth phi_p): the position drawn from the
    conditional flow."""
    rng = np.random.default_rng() if rng is None else rng
    mean, std = np.asarray(ckpt["mean"]), np.asarray(ckpt["std"])
    dev = next(flow.parameters()).device
    # float rounding can put a |p| a hair outside (p_min, p_max): keep the context finite
    eps = 1e-9 * (ckpt["p_max"] - ckpt["p_min"])
    out = np.empty((len(pz), 7), dtype=np.float32)
    for i in range(0, len(pz), batch):
        sl = slice(i, min(i + batch, len(pz)))
        p = np.clip(np.hypot(pz[sl], pt[sl]), ckpt["p_min"] + eps, ckpt["p_max"] - eps)
        ctx = np.column_stack([momentum.logit_p(p, ckpt), np.log(pt[sl] / pz[sl]), q[sl]])
        ctx = torch.as_tensor((ctx - mean[DIM:]) / std[DIM:], dtype=torch.float32, device=dev)
        s = flow.sample(0, mean=mean[:DIM], std=std[:DIM], context=ctx).double().cpu().numpy()
        n = s.shape[0]
        dphi = np.abs(s[:, 0]) % TWO_PI                    # fold as complete.to_physical
        dphi = np.minimum(dphi, TWO_PI - dphi) * rng.choice([-1.0, 1.0], size=n)
        r = np.clip(ckpt["r_scale"] * np.sinh(s[:, 1]), 0.0, ckpt["r_max"])
        z = ckpt["z0"] + ckpt["z_scale"] * np.sinh(s[:, 2])
        phi_r = phi_p[sl] - dphi
        out[sl] = np.column_stack([pt[sl] * np.cos(phi_p[sl]), pt[sl] * np.sin(phi_p[sl]), pz[sl],
                                   r * np.cos(phi_r), r * np.sin(phi_r), z, -13.0 * q[sl]])
    return out


@torch.no_grad()
def sample_first_stage(momentum_flow, momentum_ckpt, ckpt, n, momentum_temperature=1.0,
                       rng=None, batch=1 << 21):
    """(pz, pt, q, phi_p), float64: the momentum from the momentum flow at the given
    latent temperature, the charge from the Bernoulli, phi_p uniform."""
    rng = np.random.default_rng() if rng is None else rng
    uv = np.concatenate([momentum_flow.sample(min(batch, n - i), temperature=momentum_temperature,
                                              mean=momentum_ckpt["mean"], std=momentum_ckpt["std"])
                         .double().cpu().numpy() for i in range(0, n, batch)])
    pz, pt = momentum.to_momenta(uv, momentum_ckpt).T
    q = np.where(rng.random(n) < ckpt["mu_plus_fraction"], 1.0, -1.0)
    return pz, pt, q, rng.uniform(0.0, TWO_PI, n)


def generate(momentum_flow, momentum_ckpt, flow, ckpt, n, momentum_temperature=1.0, rng=None):
    """n complete muons (px, py, pz, x, y, z, pdg): first stage, then the position."""
    check_pair(momentum_ckpt, ckpt)
    rng = np.random.default_rng() if rng is None else rng
    pz, pt, q, phi_p = sample_first_stage(momentum_flow, momentum_ckpt, ckpt, n,
                                          momentum_temperature, rng)
    return sample_position(flow, ckpt, pz, pt, q, phi_p, rng)


if __name__ == "__main__":
    parser = ft.make_parser(__doc__, "outputs/flow_model_conditional.pt")
    momentum.add_args(parser)
    parser.add_argument("--r_scale", type=float, default=2.0, help="Crossover of the asinh on r [m]")
    parser.add_argument("--z_scale", type=float, default=20.0, help="Crossover of the asinh on z [m]")
    # size sweep (outputs/cond_sweep): depth matters, width hardly -- 2 layers stall ~0.015
    # nats/muon behind, 4 layers ~0.002, 6 layers reach the best level; 6 x 16 (7.8k
    # parameters) is the smallest there
    parser.set_defaults(n_layers=6, hidden=[16, 16])
    args = parser.parse_args()
    device = ft.setup(args)

    with h5py.File(args.data, 'r') as f:           # as train_flow_complete.py
        z = f['z'] if 'z' in f else f[ft.LO]['z']
        z0 = float(np.median(z[:CHUNK]))
    meta = {"columns": list(TARGET), "context_columns": list(CONTEXT), "p_max": args.p_max,
            "r_scale": args.r_scale, "z0": z0, "z_scale": args.z_scale}

    data = ft.load(args.data, reader(meta, np.random.default_rng(args.seed)), DIM + CONTEXT_DIM,
                   args.num_samples)
    meta["p_min"] = set_p_min(data, args.p_min, args.p_max)
    meta["r_max"] = float(args.r_scale * np.sinh(max(A[:, 1].max() for A in data.values())))
    meta["mu_plus_fraction"] = mu_plus_fraction(data)
    print(f"z0 {z0:.4g} m, largest radius {meta['r_max']:.4g} m, "
          f"mu+ fraction {meta['mu_plus_fraction']:.4f}")
    mean, std, wsum = ft.standardize(data)          # target and context columns alike
    ft.report_coverage(data, args.tail_bound, TARGET + CONTEXT)
    train, val = ft.split_val(data, args, device)
    del data

    flow = NeuralSplineFlow(dim=DIM, n_layers=args.n_layers, hidden=tuple(args.hidden),
                            num_bins=args.num_bins, tail_bound=args.tail_bound,
                            context_dim=CONTEXT_DIM).to(device)
    print(f"Conditional flow: {sum(p.numel() for p in flow.parameters()):,} parameters")
    state, val_nll, history = ft.fit(flow, train, val, args, device)
    print(f"best val NLL {val_nll:.4f} (of the position given momentum and charge, fit space)")
    ft.finish(flow, state, history, args, "flow_conditional", mean, std, weight=wsum, **meta)
