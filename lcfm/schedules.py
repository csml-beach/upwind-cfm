"""Sampler-side machinery of the pressure-term program (docs/ideas.md, Phase 1).

Estimator outputs are time profiles (ts, err) of the local error density
||Dv/Dt|| along the flow; grids equidistribute the implied Euler truncation
error; kappa is the predicted step-efficiency gain of warping over uniform.
"""
import torch


@torch.no_grad()
def euler_on_grid(velocity, x0, t_grid):
    """Explicit Euler on an arbitrary (possibly non-uniform) time grid."""
    x = x0.clone()
    traj = [x.clone()]
    for i in range(len(t_grid) - 1):
        t = torch.full((x.shape[0], 1), float(t_grid[i]), device=x.device, dtype=x.dtype)
        x = x + (float(t_grid[i + 1]) - float(t_grid[i])) * velocity(x, t)
        traj.append(x.clone())
    return torch.stack(traj)


@torch.no_grad()
def heun_on_grid(velocity, x0, t_grid):
    """Heun (explicit trapezoid) on an arbitrary time grid: 2 evaluations per interval."""
    x = x0.clone()
    traj = [x.clone()]
    for i in range(len(t_grid) - 1):
        t0 = float(t_grid[i])
        t1 = float(t_grid[i + 1])
        dt = t1 - t0
        tt0 = torch.full((x.shape[0], 1), t0, device=x.device, dtype=x.dtype)
        tt1 = torch.full((x.shape[0], 1), t1, device=x.device, dtype=x.dtype)
        v0 = velocity(x, tt0)
        v1 = velocity(x + dt * v0, tt1)
        x = x + 0.5 * dt * (v0 + v1)
        traj.append(x.clone())
    return torch.stack(traj)


def equal_error_grid(ts, err, steps, power=0.5, floor=1e-3, end=None):
    """Time grid with knots at quantiles of the local-error density (S1).

    Euler local error per step scales like dt^2 * ||Dv/Dt||, so equalizing
    per-step error gives dt proportional to err^(-1/2): knots at quantiles of
    the cumulative integral of err^power with power=0.5. When the profile is
    deliberately truncated before t=1 (for example to avoid a score singularity),
    pass end=1.0 so the resulting sampler grid still reaches the terminal time.
    """
    ts = torch.as_tensor(ts, dtype=torch.float64)
    grid_end = float(ts[-1] if end is None else end)
    if grid_end < float(ts[-1]) - 1e-12:
        raise ValueError("end must be greater than or equal to the final profile time.")
    rho = (torch.as_tensor(err, dtype=torch.float64) + floor).pow(power)
    cdf = torch.cumsum(0.5 * (rho[1:] + rho[:-1]) * (ts[1:] - ts[:-1]), dim=0)
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64), cdf]) / cdf[-1]
    grid = [float(ts[0])]
    for q in torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)[1:-1]:
        idx = int(torch.searchsorted(cdf, q).clamp(1, len(ts) - 1))
        c0, c1 = cdf[idx - 1], cdf[idx]
        frac = float((q - c0) / (c1 - c0 + 1e-12))
        grid.append(float(ts[idx - 1] + frac * (ts[idx] - ts[idx - 1])))
    grid.append(grid_end)
    return grid


def kappa(ts, err, floor=1e-3):
    """Stiffness concentration: integral(e) / integral(sqrt(e))^2 >= 1.

    Equals the predicted step-efficiency gain of the equal-error grid over the
    uniform grid under the first-order (no-amplification) error model; 1 iff
    the profile is flat.
    """
    ts = torch.as_tensor(ts, dtype=torch.float64)
    e = torch.as_tensor(err, dtype=torch.float64) + floor
    int_e = torch.trapezoid(e, ts)
    int_sqrt = torch.trapezoid(e.sqrt(), ts)
    span = float(ts[-1] - ts[0])
    return float(span * int_e / int_sqrt**2)


@torch.no_grad()
def rollout_error_profile(model, x0, fine_steps=50):
    """E1, the self-probe: mean material-derivative magnitude along the model's
    own flow, from one fine uniform-Euler rollout: ||v(x_{i+1},t_{i+1}) - v(x_i,t_i)||/dt.

    Oracle-free and data-free, but reports the *model's* curvature: an
    over-straightened field self-reports smooth (see the audit, X2)."""
    dt = 1.0 / fine_steps
    x = x0.clone()
    ts, err = [], []
    t = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    v_prev = model(x, t)
    for i in range(fine_steps):
        x = x + dt * v_prev
        t_next = torch.full_like(t, (i + 1) * dt)
        v_next = model(x, t_next)
        err.append((v_next - v_prev).norm(dim=1).mean().item() / dt)
        ts.append((i + 0.5) * dt)
        v_prev = v_next
    return torch.tensor([0.0] + ts + [1.0]), torch.tensor([err[0]] + err + [err[-1]])


def interpolant_error_profile(accel_fn, x0, x1, grid_size=101, t_max=0.98):
    """Mean ||a(x_t, t)|| over interpolant samples of a paired probe batch.

    accel_fn(x, t) -> (B, d) pointwise acceleration; used with the oracle (E0)
    or the dispersion head (E2). Pairing must already be applied to (x0, x1)."""
    ts = torch.linspace(0.0, float(t_max), grid_size)
    err = []
    for t_val in ts:
        t = torch.full((x0.shape[0], 1), float(t_val), device=x0.device, dtype=x0.dtype)
        xt = (1 - t) * x0 + t * x1
        err.append(accel_fn(xt, t).norm(dim=1).mean().item())
    return ts, torch.tensor(err)


@torch.no_grad()
def euler_per_sample(velocity, x0, base_grid, base_err_ts, base_err, accel_fn, mod_clip=3.0, floor=1e-3):
    """S2: per-sample budgeted Euler. Same step count for every sample (batch
    stays rectangular); each sample modulates the population grid by the square
    root of (population error density / its own pointwise error density), with
    a running normalization that forces landing at t=1 in exactly k steps.
    Reduces to the global warp when the sample's density matches the population's.
    """
    base_grid = torch.as_tensor(base_grid, dtype=x0.dtype)
    k = len(base_grid) - 1
    pop_frac = (base_grid[1:] - base_grid[:-1]).to(x0.device)
    base_err_ts = torch.as_tensor(base_err_ts, dtype=x0.dtype)
    base_err = torch.as_tensor(base_err, dtype=x0.dtype)

    def pop_density(t_scalar):
        idx = int(torch.searchsorted(base_err_ts, t_scalar.clamp(base_err_ts[0], base_err_ts[-1])).clamp(1, len(base_err_ts) - 1))
        t0, t1 = base_err_ts[idx - 1], base_err_ts[idx]
        w = ((t_scalar - t0) / (t1 - t0 + 1e-12)).clamp(0, 1)
        return (1 - w) * base_err[idx - 1] + w * base_err[idx]

    x = x0.clone()
    t = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    traj = [x.clone()]
    times = [t.clone()]
    for i in range(k):
        remaining = 1.0 - t
        if i == k - 1:
            dt = remaining
        else:
            with torch.enable_grad():
                e_sample = accel_fn(x, t).norm(dim=1, keepdim=True) + floor
            e_pop = pop_density(base_grid[i]) + floor
            mod = (e_pop / e_sample).sqrt().clamp(1.0 / mod_clip, mod_clip)
            u = pop_frac[i] * mod
            future = pop_frac[i + 1 :].sum()
            dt = remaining * u / (u + future)
        x = x + dt * velocity(x, t)
        t = t + dt
        traj.append(x.clone())
        times.append(t.clone())
    return torch.stack(traj), torch.stack(times)
