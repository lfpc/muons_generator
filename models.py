import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GMM(torch.nn.Module):
    def __init__(self, n_components, n_features, reg_covar=1e-6):
        super(GMM, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.reg_covar = float(reg_covar)
        
        # Initialize the parameters of the GMM
        self.means = torch.nn.Parameter(torch.randn(n_components, n_features))
        self.covariances = torch.nn.Parameter(torch.eye(n_features).repeat(n_components, 1, 1))
        self.weights = torch.nn.Parameter(torch.ones(n_components) / n_components)

    def _regularize_covariance(self, covariance: torch.Tensor) -> torch.Tensor:
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        eye = torch.eye(self.n_features, device=covariance.device, dtype=covariance.dtype)
        return covariance + self.reg_covar * eye

    def _log_gaussian(self, x: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
        covariance = self._regularize_covariance(covariance)
        diff = x - mean

        # Cholesky-based quadratic form and logdet (avoids explicit inverse)
        chol = torch.linalg.cholesky(covariance)
        sol = torch.cholesky_solve(diff.unsqueeze(-1), chol).squeeze(-1)
        quad = (diff * sol).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)

        return -0.5 * (quad + logdet + self.n_features * math.log(2.0 * math.pi))

    def _cholesky_with_jitter(self, covariance: torch.Tensor, max_tries: int = 5) -> torch.Tensor:
        covariance = self._regularize_covariance(covariance)
        eye = torch.eye(self.n_features, device=covariance.device, dtype=covariance.dtype)
        jitter = self.reg_covar

        for _ in range(max_tries):
            chol, info = torch.linalg.cholesky_ex(covariance)
            if info == 0:
                return chol
            jitter = jitter * 10.0
            covariance = covariance + jitter * eye

        return torch.linalg.cholesky(covariance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns log p(x, component=k) for each k
        weights = torch.clamp(self.weights, min=1e-12)
        weights = weights / weights.sum()

        log_probs = []
        for i in range(self.n_components):
            log_p_x_given_k = self._log_gaussian(x, self.means[i], self.covariances[i])
            log_probs.append(log_p_x_given_k + torch.log(weights[i]))
        return torch.stack(log_probs, dim=1)

    def fit(self, x: torch.Tensor, n_iterations: int = 100) -> "GMM":
        x = torch.as_tensor(x)

        for _ in range(int(n_iterations)):
            # E-step
            log_joint = self.forward(x)
            log_norm = torch.logsumexp(log_joint, dim=1, keepdim=True)
            responsibilities = torch.exp(log_joint - log_norm)

            # M-step
            Nk = responsibilities.sum(dim=0).clamp_min(1e-12)
            with torch.no_grad():
                self.weights.copy_(Nk / x.size(0))
                self.means.copy_((responsibilities.t() @ x) / Nk.unsqueeze(1))

                for i in range(self.n_components):
                    diff = x - self.means[i]
                    weighted_diff = responsibilities[:, i].unsqueeze(1) * diff
                    cov = (weighted_diff.t() @ diff) / Nk[i]
                    self.covariances[i].copy_(self._regularize_covariance(cov))

        return self

    @torch.no_grad()
    def sample(self, n_samples: int, return_components: bool = False):
        """Sample from the fitted GMM.

        Returns:
            samples: (n_samples, n_features)
            components (optional): (n_samples,) component indices
        """
        n_samples = int(n_samples)
        weights = torch.clamp(self.weights, min=1e-12)
        weights = weights / weights.sum()

        component_ids = torch.distributions.Categorical(probs=weights).sample((n_samples,))
        samples = torch.empty(
            n_samples,
            self.n_features,
            device=self.means.device,
            dtype=self.means.dtype,
        )

        for k in range(self.n_components):
            mask = component_ids == k
            count = int(mask.sum().item())
            if count == 0:
                continue

            chol = self._cholesky_with_jitter(self.covariances[k])
            eps = torch.randn(count, self.n_features, device=samples.device, dtype=samples.dtype)
            samples[mask] = self.means[k] + eps @ chol.transpose(0, 1)

        if return_components:
            return samples, component_ids
        return samples


# ---------------------------------------------------------------------------
# Neural Spline Flow (rational-quadratic spline coupling), pure PyTorch.
#
# A normalizing flow maps data x to a standard-normal base z through a stack of
# invertible coupling layers. Each layer leaves some dimensions unchanged and
# transforms the rest with a monotonic rational-quadratic spline whose knots are
# predicted by an MLP from the unchanged dimensions (Durkan et al. 2019). Splines
# are far more expressive per layer than affine (RealNVP) coupling, so a few
# layers model sharp, peaked 2D densities well. The flow gives exact log_prob and
# exact sampling. Trained by maximizing log_prob (Adam), not EM like GMM above.
# ---------------------------------------------------------------------------

_MIN_BIN_WIDTH = 1e-3
_MIN_BIN_HEIGHT = 1e-3
_MIN_DERIVATIVE = 1e-3


def _searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations = bin_locations.clone()
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


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
    """RQ spline inside [-B, B], identity (slope 1) linear tails outside."""
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside = ~inside

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # Pad derivatives so the boundary slope is exactly 1, matching the linear tails.
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = math.log(math.expm1(1 - _MIN_DERIVATIVE))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside] = inputs[outside]
    if inside.any():
        out_in, logdet_in = _rational_quadratic_spline(
            inputs=inputs[inside],
            unnormalized_widths=unnormalized_widths[inside, :],
            unnormalized_heights=unnormalized_heights[inside, :],
            unnormalized_derivatives=unnormalized_derivatives[inside, :],
            inverse=inverse,
            left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound)
        outputs[inside] = out_in
        logabsdet[inside] = logdet_in
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
    knot parameters predicted from the passed-through dims."""

    def __init__(self, dim, mask, hidden, num_bins, tail_bound):
        super().__init__()
        self.register_buffer("mask", mask)
        self.num_bins = num_bins
        self.tail_bound = float(tail_bound)
        self.id_dim = int(mask.sum().item())
        self.tr_dim = dim - self.id_dim
        self.net = _make_mlp(self.id_dim, self.tr_dim * (3 * num_bins + 1), hidden)

    def _params(self, id_part, n):
        p = self.net(id_part).view(n, self.tr_dim, 3 * self.num_bins + 1)
        return (p[..., :self.num_bins],
                p[..., self.num_bins:2 * self.num_bins],
                p[..., 2 * self.num_bins:])

    def forward(self, x):                # data -> base; returns (z, logabsdet)
        n = x.shape[0]
        id_part, tr_part = x[:, self.mask], x[:, ~self.mask]
        w, h, d = self._params(id_part, n)
        out, logabsdet = unconstrained_rational_quadratic_spline(
            tr_part, w, h, d, inverse=False, tail_bound=self.tail_bound)
        z = torch.empty_like(x)
        z[:, self.mask] = id_part
        z[:, ~self.mask] = out
        return z, logabsdet.sum(-1)

    def inverse(self, z):                # base -> data
        n = z.shape[0]
        id_part, tr_part = z[:, self.mask], z[:, ~self.mask]
        w, h, d = self._params(id_part, n)
        out, _ = unconstrained_rational_quadratic_spline(
            tr_part, w, h, d, inverse=True, tail_bound=self.tail_bound)
        x = torch.empty_like(z)
        x[:, self.mask] = id_part
        x[:, ~self.mask] = out
        return x


class NeuralSplineFlow(nn.Module):
    """RQ-spline coupling flow over `dim`-D data with a standard-normal base."""

    def __init__(self, dim=2, n_layers=6, hidden=(128, 128), num_bins=12, tail_bound=14.0):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.hidden = tuple(hidden)
        self.num_bins = num_bins
        self.tail_bound = float(tail_bound)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = (torch.arange(dim) % 2 == (i % 2))     # alternate which dims pass through
            self.layers.append(_SplineCoupling(dim, mask, self.hidden, num_bins, self.tail_bound))

    @property
    def config(self):
        return dict(dim=self.dim, n_layers=self.n_layers, hidden=list(self.hidden),
                    num_bins=self.num_bins, tail_bound=self.tail_bound)

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

    def forward(self, x):                # data -> base, total logabsdet
        logdet = x.new_zeros(x.shape[0])
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld
        return z, logdet

    def log_prob(self, x):
        z, logdet = self.forward(x)
        base = (-0.5 * (z.pow(2) + math.log(2.0 * math.pi))).sum(-1)
        return base + logdet

    @torch.no_grad()
    def sample(self, n_samples, temperature=1.0):
        p = next(self.parameters())
        z = torch.randn(int(n_samples), self.dim, device=p.device, dtype=p.dtype) * temperature
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

