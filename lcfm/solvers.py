import torch

from .registry import SOLVERS, register


@register(SOLVERS, "euler")
@torch.no_grad()
def euler(model, x0, steps, noise=0.0, **kwargs):
    del kwargs
    dt = 1.0 / steps
    x = x0.clone()
    traj = [x.clone()]
    for i in range(steps):
        t = torch.full((x.shape[0], 1), i * dt, device=x.device)
        v = model(x, t)
        if noise > 0:
            v = v + torch.randn_like(v) * noise
        x = x + dt * v
        traj.append(x.clone())
    return torch.stack(traj)


@register(SOLVERS, "velocity_smoothed_euler")
@torch.no_grad()
def velocity_smoothed_euler(model, x0, steps, noise=0.0, alpha=0.8, **kwargs):
    del kwargs
    dt = 1.0 / steps
    x = x0.clone()
    traj = [x.clone()]
    v_prev = None
    for i in range(steps):
        t = torch.full((x.shape[0], 1), i * dt, device=x.device)
        v = model(x, t)
        if noise > 0:
            v = v + torch.randn_like(v) * noise
        if v_prev is not None:
            denom = torch.sum(v_prev * v_prev, dim=-1, keepdim=True) + 1e-6
            v_proj = (torch.sum(v * v_prev, dim=-1, keepdim=True) / denom) * v_prev
            v = (1 - alpha) * v + alpha * v_proj
        x = x + dt * v
        v_prev = v.clone()
        traj.append(x.clone())
    return torch.stack(traj)


@register(SOLVERS, "heun")
@torch.no_grad()
def heun(model, x0, steps, noise=0.0, **kwargs):
    del kwargs
    dt = 1.0 / steps
    x = x0.clone()
    traj = [x.clone()]
    for i in range(steps):
        t0 = torch.full((x.shape[0], 1), i * dt, device=x.device)
        t1 = torch.full((x.shape[0], 1), min((i + 1) * dt, 1.0), device=x.device)
        v0 = model(x, t0)
        v1 = model(x + dt * v0, t1)
        v = 0.5 * (v0 + v1)
        if noise > 0:
            v = v + torch.randn_like(v) * noise
        x = x + dt * v
        traj.append(x.clone())
    return torch.stack(traj)


def solve(name, model, x0, config):
    return SOLVERS[name](model, x0, **config)
