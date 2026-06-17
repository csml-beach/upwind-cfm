import math

import torch


class GaussianMixtureOracle:
    """Closed-form marginal quantities of the linear interpolant x_t = (1-t) x0 + t x1
    for x0 ~ N(mu0, sigma0^2 I) independent of x1 ~ sum_k pi_k N(mu_k, sigma_m^2 I).

    Provides the marginal velocity v = E[u | x_t], the conditional velocity covariance
    Sigma = Cov[u | x_t], the score grad log p_t, and the exact material acceleration
    of the marginal flow,

        Dv/Dt = -(1/p) div(p Sigma) = -div(Sigma) - Sigma grad(log p),

    the momentum-balance law verified in scripts/check_momentum_identity.py. This is
    estimator E0 of docs/ideas.md. All methods are batched: x is (B, d), t is (B, 1).
    Internals run in the oracle's own dtype (float64 by default); results are cast
    back to the input dtype.
    """

    def __init__(self, centers, sigma_mode, source_mean, source_std, weights=None, dtype=torch.float64):
        self.mu = torch.as_tensor(centers, dtype=dtype)
        if self.mu.ndim != 2:
            raise ValueError("centers must have shape [n_modes, dim].")
        self.n_modes, self.dim = self.mu.shape
        self.mu0 = torch.as_tensor(source_mean, dtype=dtype).reshape(self.dim)
        self.sigma_m_sq = float(sigma_mode) ** 2
        self.sigma_0_sq = float(source_std) ** 2
        if self.sigma_m_sq <= 0.0 or self.sigma_0_sq <= 0.0:
            raise ValueError("sigma_mode and source_std must be positive.")
        if weights is None:
            self.log_pi = torch.full((self.n_modes,), -math.log(self.n_modes), dtype=dtype)
        else:
            pi = torch.as_tensor(weights, dtype=dtype).reshape(self.n_modes)
            self.log_pi = torch.log(pi / pi.sum())

    @classmethod
    def from_problem(cls, problem):
        for attr in ("mode_centers", "sigma_mode", "source_mean", "source_std"):
            if not hasattr(problem, attr):
                raise ValueError(
                    f"Problem '{getattr(problem, 'name', '?')}' has no '{attr}'; "
                    "the oracle supports Gaussian-mixture problems (five_modes, fan_modes) only."
                )
        source_std = problem.source_std
        if not isinstance(source_std, (int, float)):
            raise ValueError("GaussianMixtureOracle supports scalar source_std only.")
        return cls(problem.mode_centers, problem.sigma_mode, problem.source_mean, source_std)

    def to(self, device):
        self.mu = self.mu.to(device)
        self.mu0 = self.mu0.to(device)
        self.log_pi = self.log_pi.to(device)
        return self

    def _cast(self, x, t):
        return x.to(self.mu.dtype), t.to(self.mu.dtype)

    def _moments(self, x, t):
        gamma_sq = (1 - t) ** 2 * self.sigma_0_sq + t**2 * self.sigma_m_sq  # (B, 1)
        c = t * self.sigma_m_sq - (1 - t) * self.sigma_0_sq  # (B, 1)
        component_means = t.unsqueeze(-1) * self.mu + (1 - t).unsqueeze(-1) * self.mu0  # (B, K, d)
        diffs = x.unsqueeze(1) - component_means  # (B, K, d)
        log_comp = (
            self.log_pi
            - 0.5 * (diffs**2).sum(dim=2) / gamma_sq
            - 0.5 * self.dim * torch.log(2 * math.pi * gamma_sq)
        )  # (B, K)
        log_p = torch.logsumexp(log_comp, dim=1, keepdim=True)  # (B, 1)
        w = torch.softmax(log_comp, dim=1)  # (B, K)
        cond_mean_k = (self.mu - self.mu0) + (c / gamma_sq).unsqueeze(-1) * diffs  # E[u | x, k]
        return gamma_sq, c, diffs, log_p, w, cond_mean_k

    def velocity(self, x, t):
        dtype = x.dtype
        x, t = self._cast(x, t)
        _, _, _, _, w, cond_mean_k = self._moments(x, t)
        return torch.einsum("bk,bkd->bd", w, cond_mean_k).to(dtype)

    def sigma(self, x, t):
        dtype = x.dtype
        x, t = self._cast(x, t)
        gamma_sq, c, _, _, w, cond_mean_k = self._moments(x, t)
        within = self.sigma_0_sq + self.sigma_m_sq - c**2 / gamma_sq  # (B, 1)
        second_moment = torch.einsum("bk,bki,bkj->bij", w, cond_mean_k, cond_mean_k)
        second_moment = second_moment + within.unsqueeze(-1) * torch.eye(
            self.dim, dtype=x.dtype, device=x.device
        )
        v = torch.einsum("bk,bkd->bd", w, cond_mean_k)
        return (second_moment - torch.einsum("bi,bj->bij", v, v)).to(dtype)

    def dispersion(self, x, t):
        """Scalar dispersion tr(Sigma)/d — the quantity the E2 head regresses."""
        dtype = x.dtype
        x, t = self._cast(x, t)
        sigma = self.sigma(x, t)
        return (torch.einsum("bii->b", sigma) / self.dim).unsqueeze(1).to(dtype)

    def log_p(self, x, t):
        dtype = x.dtype
        x, t = self._cast(x, t)
        _, _, _, log_p, _, _ = self._moments(x, t)
        return log_p.to(dtype)

    def score(self, x, t):
        dtype = x.dtype
        x, t = self._cast(x, t)
        gamma_sq, _, diffs, _, w, _ = self._moments(x, t)
        return (-torch.einsum("bk,bkd->bd", w, diffs) / gamma_sq).to(dtype)

    def divergence_sigma(self, x, t):
        dtype = x.dtype
        x, t = self._cast(x, t)

        def sigma_single(x_single, t_single):
            return self.sigma(x_single.unsqueeze(0), t_single.unsqueeze(0)).squeeze(0)

        jac = torch.func.vmap(torch.func.jacrev(sigma_single, argnums=0))(x, t)  # (B, d, d, d)
        return torch.einsum("bijj->bi", jac).to(dtype)

    def acceleration_target(self, x, t):
        """Exact Dv/Dt of the marginal flow: -(div Sigma + Sigma grad log p)."""
        dtype = x.dtype
        x, t = self._cast(x, t)
        div_sigma = self.divergence_sigma(x, t)
        sigma_score = torch.einsum("bij,bj->bi", self.sigma(x, t), self.score(x, t))
        return (-(div_sigma + sigma_score)).to(dtype)
