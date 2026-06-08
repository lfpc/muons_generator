import math
import torch

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

