"""Neural Spline Flow (rational-quadratic spline coupling, Durkan et al. 2019), pure PyTorch.

A flow maps data x to a standard-normal base z through a stack of invertible coupling
layers: each leaves half the dimensions unchanged and transforms the other half with a
monotonic rational-quadratic spline whose knots an MLP predicts from the unchanged half
(and from a context, for a conditional flow). It gives exact log_prob and sampling.
The flow works in its standardized fit space only; fit_space.py maps to physics.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_MIN_BIN_WIDTH = 1e-3
_MIN_BIN_HEIGHT = 1e-3
_MIN_DERIVATIVE = 1e-3


def _searchsorted(bin_locations, inputs):
    """Index of the bin each input falls into (binary search; an input exactly on the
    right edge comes back as num_bins and is clamped by the caller)."""
    return torch.searchsorted(bin_locations.contiguous(),
                              inputs[..., None].contiguous(), right=True).squeeze(-1) - 1


def _rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights,
                               unnormalized_derivatives, inverse, left, right, bottom, top):
    """RQ spline on the interval, applied elementwise over arbitrary leading dims.
    inputs: (M,); the unnormalized_* tensors: (M, num_bins) / (M, num_bins+1)."""
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = _MIN_BIN_WIDTH + (1 - _MIN_BIN_WIDTH * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = _MIN_DERIVATIVE + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = _MIN_BIN_HEIGHT + (1 - _MIN_BIN_HEIGHT * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_locations = cumheights if inverse else cumwidths
    bin_idx = _searchsorted(bin_locations, inputs).clamp(0, num_bins - 1)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    d_sum = input_derivatives + input_derivatives_plus_one - 2 * input_delta

    if inverse:
        a = (inputs - input_cumheights) * d_sum + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * d_sum
        c = -input_delta * (inputs - input_cumheights)
        discriminant = (b.pow(2) - 4 * a * c).clamp_min(0.0)
        theta = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = theta * input_bin_widths + input_cumwidths
        theta_1mt = theta * (1 - theta)
        denominator = input_delta + d_sum * theta_1mt
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_1mt
            + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_1mt = theta * (1 - theta)
        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_1mt)
        denominator = input_delta + d_sum * theta_1mt
        outputs = input_cumheights + numerator / denominator
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_1mt
            + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


def unconstrained_rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights,
                                            unnormalized_derivatives, inverse=False, tail_bound=1.0):
    """RQ spline inside [-B, B], identity (slope 1) linear tails outside.

    Branch-free on purpose. The natural way to write this -- run the spline on
    inputs[inside] and scatter the result back -- costs a nonzero() for every boolean
    mask and an `if inside.any()`, i.e. six device-to-host syncs per coupling layer, so
    ~150 pipeline stalls per training step; that, not the arithmetic, was what made a
    step take seconds. Here the spline is evaluated on every row (clamped into the
    interval, which is exact for the rows that were inside anyway) and torch.where picks
    spline or identity. The waste is the arithmetic on the out-of-range rows, which are
    a fraction of a percent of a batch, and the clamp keeps their gradient on the
    identity branch where it belongs."""
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # Pad derivatives so the boundary slope is exactly 1, matching the linear tails.
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = math.log(math.expm1(1 - _MIN_DERIVATIVE))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    out_in, logdet_in = _rational_quadratic_spline(
        inputs=inputs.clamp(-tail_bound, tail_bound),
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound)
    outputs = torch.where(inside, out_in, inputs)
    logabsdet = torch.where(inside, logdet_in, torch.zeros_like(logdet_in))
    return outputs, logabsdet


def _make_mlp(in_dim, out_dim, hidden):
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    last = nn.Linear(prev, out_dim)
    nn.init.zeros_(last.weight)          # start each layer near identity for stability
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class _SplineCoupling(nn.Module):
    """One coupling layer: pass through masked dims, spline-transform the rest with
    knot parameters predicted from the passed-through dims (and the context, if any)."""

    def __init__(self, dim, mask, hidden, num_bins, tail_bound, context_dim=0):
        super().__init__()
        self.register_buffer("mask", mask)
        # integer indices, not the boolean mask: x[:, mask] would cost a device-host
        # sync per call; not persistent, so they stay out of the state_dict
        self.register_buffer("id_idx", torch.nonzero(mask, as_tuple=True)[0],
                             persistent=False)
        self.register_buffer("tr_idx", torch.nonzero(~mask, as_tuple=True)[0],
                             persistent=False)
        self.num_bins = num_bins
        self.tail_bound = float(tail_bound)
        self.id_dim = int(mask.sum().item())
        self.tr_dim = dim - self.id_dim
        self.net = _make_mlp(self.id_dim + context_dim, self.tr_dim * (3 * num_bins + 1), hidden)

    def _params(self, id_part, n, context=None):
        inp = id_part if context is None else torch.cat([id_part, context], 1)
        p = self.net(inp).view(n, self.tr_dim, 3 * self.num_bins + 1)
        return (p[..., :self.num_bins],
                p[..., self.num_bins:2 * self.num_bins],
                p[..., 2 * self.num_bins:])

    def forward(self, x, context=None):  # data -> base; returns (z, logabsdet)
        n = x.shape[0]
        id_part = x.index_select(1, self.id_idx)
        tr_part = x.index_select(1, self.tr_idx)
        w, h, d = self._params(id_part, n, context)
        out, logabsdet = unconstrained_rational_quadratic_spline(
            tr_part, w, h, d, inverse=False, tail_bound=self.tail_bound)
        z = torch.empty_like(x)
        z.index_copy_(1, self.id_idx, id_part)
        z.index_copy_(1, self.tr_idx, out)
        return z, logabsdet.sum(-1)

    def inverse(self, z, context=None):  # base -> data
        n = z.shape[0]
        id_part = z.index_select(1, self.id_idx)
        tr_part = z.index_select(1, self.tr_idx)
        w, h, d = self._params(id_part, n, context)
        out, _ = unconstrained_rational_quadratic_spline(
            tr_part, w, h, d, inverse=True, tail_bound=self.tail_bound)
        x = torch.empty_like(z)
        x.index_copy_(1, self.id_idx, id_part)
        x.index_copy_(1, self.tr_idx, out)
        return x


class NeuralSplineFlow(nn.Module):
    """RQ-spline coupling flow over `dim`-D data with a standard-normal base. It lives
    entirely in its (standardized) fit space: mapping that space to and from physical
    quantities is the job of the script that defines it.

    With context_dim > 0 it is a conditional flow p(x | c): every coupling MLP also sees
    the context c, which is not transformed (train_position_flow.py). log_prob takes
    the context separately or as the last context_dim columns of x, so rows [x, c] run
    through the training loop of flow_training.py unchanged."""

    def __init__(self, dim=2, n_layers=6, hidden=(128, 128), num_bins=12, tail_bound=14.0,
                 context_dim=0):
        super().__init__()
        self.dim = dim
        self.context_dim = int(context_dim)
        self.n_layers = n_layers
        self.hidden = tuple(hidden)
        self.num_bins = num_bins
        self.tail_bound = float(tail_bound)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = (torch.arange(dim) % 2 == (i % 2))     # alternate which dims pass through
            self.layers.append(_SplineCoupling(dim, mask, self.hidden, num_bins, self.tail_bound,
                                               self.context_dim))

    @property
    def config(self):
        cfg = dict(dim=self.dim, n_layers=self.n_layers, hidden=list(self.hidden),
                   num_bins=self.num_bins, tail_bound=self.tail_bound)
        if self.context_dim:                 # unconditional configs stay as they were
            cfg["context_dim"] = self.context_dim
        return cfg

    def _split_context(self, x, context):
        if context is None and self.context_dim:
            return x[:, :self.dim], x[:, self.dim:]
        return x, context

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

    def forward(self, x, context=None):  # data -> base, total logabsdet
        x, context = self._split_context(x, context)
        logdet = x.new_zeros(x.shape[0])
        z = x
        for layer in self.layers:
            z, ld = layer(z, context)
            logdet = logdet + ld
        return z, logdet

    def log_prob(self, x, context=None):
        """log p(x) (log p(x | context) for a conditional flow), x standardized."""
        z, logdet = self.forward(x, context)
        base = (-0.5 * (z.pow(2) + math.log(2.0 * math.pi))).sum(-1)
        return base + logdet

    @torch.no_grad()
    def sample(self, n_samples, temperature=1.0, mean=None, std=None, context=None):
        """Samples in the fit space; destandardized if `mean`/`std` are given (the dim
        sampled columns only). A conditional flow draws one sample per row of the
        (standardized) `context`, which then sets n_samples."""
        p = next(self.parameters())
        if self.context_dim:
            if context is None:
                raise ValueError("a conditional flow needs a context to sample")
            context = torch.as_tensor(context, device=p.device, dtype=p.dtype)
            n_samples = context.shape[0]
        z = torch.randn(int(n_samples), self.dim, device=p.device, dtype=p.dtype) * temperature
        for layer in reversed(self.layers):
            z = layer.inverse(z, context)
        if mean is not None and std is not None:
            mean = torch.as_tensor(mean, device=z.device, dtype=z.dtype).view(1, self.dim)
            std = torch.as_tensor(std, device=z.device, dtype=z.dtype).view(1, self.dim)
            z = z * std + mean
        return z


# ---------------------------------------------------------------------------
# Checkpoints: one flat dict -- the flow's config and weights, the mean/std that
# standardize its fit space, and the constants of that space (fit_space.py).
# ---------------------------------------------------------------------------

def save_flow(path, flow, state_dict, mean, std, **meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": flow.config,
                "state_dict": {k: v.cpu() for k, v in state_dict.items()},
                "mean": np.ravel(mean), "std": np.ravel(std), **meta}, path)


def load_flow(path, device="cpu"):
    """(flow in eval mode on `device`, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    flow = NeuralSplineFlow.from_config(ckpt["config"])
    flow.load_state_dict(ckpt["state_dict"])
    return flow.to(device).eval(), ckpt
