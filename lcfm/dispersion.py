"""E2: the dispersion head r(x, t) — a learned map of target disagreement.

The second regression of docs/ideas.md: fit a scalar field to the squared
misses of the (frozen) velocity network on ordinary CFM training pairs,

    label = (1/d) ||u - v_theta(x_t, t)||^2,   u = x1 - x0,

whose conditional mean is tr(Sigma)/d, the local variance of the targets
("residual" mode; converges to the dispersion when v_theta is converged).
Alternative "second_moment" mode regresses (1/d)||u||^2 — pure data labels,
independent of any velocity model — and recovers the dispersion at use time
as m - ||v||^2/d. The second mode is what the audit (X2) uses, so the
dispersion estimate never trusts the model being audited.

Training is post-hoc: the velocity model is untouched (Phase-1 doctrine).
The momentum law turns r into the sampler's error density: with the isotropic
closure Sigma ~ r I and the Gaussian-source score identity,

    a(x,t) = -( grad r + r * score ),
    score  = -(x - t v - mu0) / (sigma0^2 (1 - t)).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pairing import apply_pairing


LOG_EPS = 1e-3


class DispersionMLP(nn.Module):
    def __init__(self, dim, hidden=128, depth=3, positive=True):
        super().__init__()
        self.positive = positive
        layers = []
        in_dim = dim + 1
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden), nn.SiLU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_expand = t.expand(x.shape[0], 1)
        out = self.net(torch.cat([x, t_expand], dim=-1))
        return F.softplus(out) if self.positive else out


def head_is_positive(mode):
    return mode != "residual_log"


def dispersion_labels(x0, x1, xt, t, velocity_model, mode):
    u = x1 - x0
    d = x0.shape[1]
    if mode in ("residual", "residual_log"):
        with torch.no_grad():
            v = velocity_model(xt, t)
        residual = (u - v).pow(2).sum(dim=1, keepdim=True) / d
        return torch.log(residual + LOG_EPS) if mode == "residual_log" else residual
    if mode == "second_moment":
        return u.pow(2).sum(dim=1, keepdim=True) / d
    raise ValueError(f"Unknown dispersion label mode: {mode}")


def train_dispersion_head(problem, velocity_model, config, device, mode="residual", epochs=1500, batch_size=256, lr=1e-3, log_every=500):
    head = DispersionMLP(problem.dim, positive=head_is_positive(mode)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    if velocity_model is not None:
        velocity_model.eval()
    history = []
    for epoch in range(epochs):
        x0, x1 = problem.sample_train_batch(batch_size, device)
        x0, x1 = apply_pairing(x0, x1, config)
        t = torch.rand(batch_size, 1, device=device)
        xt = (1 - t) * x0 + t * x1
        labels = dispersion_labels(x0, x1, xt, t, velocity_model, mode)
        loss = F.mse_loss(head(xt, t), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % log_every == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
            print({"head_epoch": epoch, "loss": float(loss.detach().cpu()), "mode": mode})
    head.eval()
    return head, history


def source_params(problem, device, dtype=torch.float32):
    """Gaussian-source parameters (mu0, sigma0^2); defaults to N(0, I) when the
    problem does not declare them (spiral). Raises for per-dimension stds."""
    mean = getattr(problem, "source_mean", None)
    if mean is None:
        mean = torch.zeros(problem.dim)
    mu0 = torch.as_tensor(mean, dtype=dtype, device=device).reshape(problem.dim)
    std = getattr(problem, "source_std", 1.0)
    if not isinstance(std, (int, float)):
        raise ValueError("head acceleration requires a scalar Gaussian source_std.")
    return mu0, float(std) ** 2


def dispersion_value(head, velocity_model, x, t, mode):
    """r(x,t): direct in residual mode; m - ||v||^2/d (clamped) in second-moment mode.

    In second-moment mode the velocity term participates in autograd so that
    grad r includes -grad ||v||^2/d; the caller controls the grad context."""
    r = head(x, t)
    if mode == "residual_log":
        return (torch.exp(r) - LOG_EPS).clamp_min(0.0)
    if mode == "second_moment":
        v = velocity_model(x, t)
        r = (r - v.pow(2).sum(dim=1, keepdim=True) / x.shape[1]).clamp_min(0.0)
    return r


def make_head_accel_fn(head, velocity_model, problem, device, mode="residual", t_max=0.98):
    """Pointwise a(x,t) = -(grad r + r * score) from the head, the velocity model
    (for the score identity), and the problem's Gaussian source parameters."""
    mu0, sigma0_sq = source_params(problem, device)

    def accel(x, t):
        t = t.clamp(max=t_max)
        x_req = x.detach().clone().requires_grad_(True)
        r = dispersion_value(head, velocity_model, x_req, t, mode)
        (grad_r,) = torch.autograd.grad(r.sum(), x_req)
        with torch.no_grad():
            v = velocity_model(x, t)
        score = -(x - t * v - mu0) / (sigma0_sq * (1.0 - t))
        return (-(grad_r + r.detach() * score)).detach()

    return accel
