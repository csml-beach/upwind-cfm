"""Time-mesh calibration and sampler-side numerical utilities.

Estimator outputs are local or terminally transported material-acceleration
profiles along the flow. Grids equidistribute the implied Euler defect;
kappa is the predicted step-efficiency gain of warping over uniform.
"""
import resource
import sys
import time

import torch


def power_time_grid(steps, rho=2.0, kind="early"):
    """Deterministic hand-designed non-uniform grids on [0, 1].

    kind='early' puts more points near t=0 for rho>1; kind='late' puts more
    points near t=1; kind='symmetric' puts more points near both endpoints.
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")
    rho = float(rho)
    if rho <= 0:
        raise ValueError("rho must be positive.")
    u = torch.linspace(0.0, 1.0, int(steps) + 1, dtype=torch.float64)
    if kind == "early":
        grid = u.pow(rho)
    elif kind == "late":
        grid = 1.0 - (1.0 - u).pow(rho)
    elif kind == "symmetric":
        left = 0.5 * (2.0 * u).pow(rho)
        right = 1.0 - 0.5 * (2.0 * (1.0 - u)).pow(rho)
        grid = torch.where(u <= 0.5, left, right)
    else:
        raise ValueError("kind must be one of: early, late, symmetric.")
    grid[0] = 0.0
    grid[-1] = 1.0
    return [float(x) for x in grid.tolist()]


def vsfm_time_grid(steps, profile="coasting"):
    """Polynomial velocity-schedule grids from Velocity Scheduled FM.

    The returned values are the original linear-FM times tau=alpha(s) on a
    uniform auxiliary grid s. Both profiles spend progressively more steps
    near the data endpoint.
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")
    s = torch.linspace(0.0, 1.0, int(steps) + 1, dtype=torch.float64)
    if profile == "coasting":
        grid = 1.5 * s - 0.5 * s.pow(3)
    elif profile in {"uniform_deceleration", "uniform_decel"}:
        grid = 2.0 * s - s.pow(2)
    else:
        raise ValueError("profile must be one of: coasting, uniform_deceleration.")
    grid[0] = 0.0
    grid[-1] = 1.0
    return [float(x) for x in grid.tolist()]


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


def optimal_defect_grid(edge_costs, anchor_grid, steps):
    """Choose an exact-step path through anchor times by dynamic programming.

    ``edge_costs[i, j]`` is the estimated defect of replacing the reference
    trajectory from anchor i to j with one Euler step. This is the schedule
    optimization phase used by BOSS-style methods; it does not alter the
    velocity model.
    """
    costs = torch.as_tensor(edge_costs, dtype=torch.float64, device="cpu")
    anchors = torch.as_tensor(anchor_grid, dtype=torch.float64, device="cpu")
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError("edge_costs must be a square matrix.")
    if anchors.ndim != 1 or anchors.numel() != costs.shape[0]:
        raise ValueError("anchor_grid must match edge_costs.")
    if steps <= 0 or steps >= anchors.numel():
        raise ValueError("steps must be between 1 and len(anchor_grid)-1.")
    if not bool(torch.all(anchors[1:] > anchors[:-1])):
        raise ValueError("anchor_grid must be strictly increasing.")

    n = anchors.numel()
    dp = torch.full((steps + 1, n), torch.inf, dtype=torch.float64)
    parent = torch.full((steps + 1, n), -1, dtype=torch.long)
    dp[0, 0] = 0.0
    for used in range(1, steps + 1):
        for end in range(used, n):
            candidates = dp[used - 1, :end] + costs[:end, end]
            value, start = torch.min(candidates, dim=0)
            dp[used, end] = value
            parent[used, end] = start

    if not bool(torch.isfinite(dp[steps, -1])):
        raise ValueError("No finite path reaches the final anchor with the requested steps.")
    indices = [n - 1]
    end = n - 1
    for used in range(steps, 0, -1):
        end = int(parent[used, end].item())
        if end < 0:
            raise RuntimeError("Dynamic-programming backtracking failed.")
        indices.append(end)
    indices.reverse()
    return [float(anchors[index].item()) for index in indices], float(dp[steps, -1].item())


@torch.no_grad()
def euler_defect_cost_matrix(model, x0, anchor_steps=100, reference_solver="heun"):
    """Estimate all finite-interval Euler defects on a pilot batch.

    A fine uniform pilot trajectory supplies states at the anchor times. For
    every i < j, the cost is the mean squared L2 discrepancy between the pilot
    state at j and a single Euler jump from the pilot state at i. Heun is the
    default pilot solver so even adjacent-anchor defects remain informative.
    """
    if anchor_steps <= 0:
        raise ValueError("anchor_steps must be positive.")
    if reference_solver not in {"euler", "heun"}:
        raise ValueError("reference_solver must be 'euler' or 'heun'.")

    anchors = torch.linspace(0.0, 1.0, int(anchor_steps) + 1, dtype=torch.float64)
    x = x0.clone()
    states = [x.clone()]
    velocities = []
    for index in range(anchor_steps):
        t0 = float(anchors[index])
        t1 = float(anchors[index + 1])
        dt = t1 - t0
        tt0 = torch.full((x.shape[0], 1), t0, device=x.device, dtype=x.dtype)
        v0 = model(x, tt0)
        velocities.append(v0.clone())
        if reference_solver == "euler":
            x = x + dt * v0
        else:
            tt1 = torch.full((x.shape[0], 1), t1, device=x.device, dtype=x.dtype)
            v1 = model(x + dt * v0, tt1)
            x = x + 0.5 * dt * (v0 + v1)
        states.append(x.clone())

    state_tensor = torch.stack(states)
    costs = torch.full((anchor_steps + 1, anchor_steps + 1), torch.inf, dtype=torch.float64)
    for start in range(anchor_steps):
        dts = (anchors[start + 1 :] - anchors[start]).to(device=x.device, dtype=x.dtype)
        predictions = state_tensor[start].unsqueeze(0) + dts[:, None, None] * velocities[start].unsqueeze(0)
        defects = predictions - state_tensor[start + 1 :]
        costs[start, start + 1 :] = defects.pow(2).sum(dim=-1).mean(dim=-1).detach().cpu().to(torch.float64)
    return anchors, costs


@torch.no_grad()
def transported_defect_profile(model, x0, fine_steps=50, anchor_count=16, perturbation=1e-3):
    """Estimate terminally transported Euler-defect magnitude along a pilot flow.

    A fine Heun trajectory estimates the local material acceleration from the
    Heun--Euler defect. At selected intervals, a centered finite difference
    then transports perturbations in that acceleration direction to t=1. The
    returned profile is approximately ``||D Phi_{t->1} a_t||``, where Phi is
    the learned flow map. This retains SCTW's local defect signal while also
    accounting for downstream amplification or damping.

    Returns ``(ts, transported, local_acceleration, terminal_gain)``. Endpoint
    values duplicate the nearest measured anchor so the profiles cover [0, 1].
    """
    if fine_steps <= 0:
        raise ValueError("fine_steps must be positive.")
    if anchor_count <= 0:
        raise ValueError("anchor_count must be positive.")
    if perturbation <= 0:
        raise ValueError("perturbation must be positive.")

    fine_steps = int(fine_steps)
    dt = 1.0 / fine_steps
    x = x0.clone()
    states = [x.clone()]
    accelerations = []
    for index in range(fine_steps):
        t0 = index * dt
        t1 = (index + 1) * dt
        tt0 = torch.full((x.shape[0], 1), t0, device=x.device, dtype=x.dtype)
        tt1 = torch.full((x.shape[0], 1), t1, device=x.device, dtype=x.dtype)
        v0 = model(x, tt0)
        euler_state = x + dt * v0
        v1 = model(euler_state, tt1)
        x = x + 0.5 * dt * (v0 + v1)
        states.append(x.clone())
        accelerations.append(2.0 * (x - euler_state) / (dt * dt))

    count = min(int(anchor_count), fine_steps)
    anchor_indices = torch.linspace(0, fine_steps - 1, count, dtype=torch.float64).round().to(torch.long).unique()
    measured_ts = []
    transported_values = []
    local_values = []
    gain_values = []
    eps = float(perturbation)

    for anchor_tensor in anchor_indices:
        anchor = int(anchor_tensor.item())
        acceleration = accelerations[anchor]
        flat_acceleration = acceleration.flatten(start_dim=1)
        acceleration_norm = flat_acceleration.norm(dim=1)
        direction = acceleration / acceleration_norm.clamp_min(1e-12).view(
            acceleration.shape[0], *([1] * (acceleration.ndim - 1))
        )
        plus = states[anchor + 1] + eps * direction
        minus = states[anchor + 1] - eps * direction

        for index in range(anchor + 1, fine_steps):
            t0 = index * dt
            t1 = (index + 1) * dt
            tt0 = torch.full((plus.shape[0], 1), t0, device=plus.device, dtype=plus.dtype)
            tt1 = torch.full((plus.shape[0], 1), t1, device=plus.device, dtype=plus.dtype)
            plus_v0 = model(plus, tt0)
            minus_v0 = model(minus, tt0)
            plus_v1 = model(plus + dt * plus_v0, tt1)
            minus_v1 = model(minus + dt * minus_v0, tt1)
            plus = plus + 0.5 * dt * (plus_v0 + plus_v1)
            minus = minus + 0.5 * dt * (minus_v0 + minus_v1)

        terminal_gain = (plus - minus).flatten(start_dim=1).norm(dim=1) / (2.0 * eps)
        measured_ts.append((anchor + 0.5) * dt)
        local_values.append(float(acceleration_norm.mean().item()))
        gain_values.append(float(terminal_gain.mean().item()))
        transported_values.append(float((terminal_gain * acceleration_norm).mean().item()))

    def cover_unit_interval(values):
        return torch.tensor([values[0], *values, values[-1]], dtype=torch.float64)

    ts = torch.tensor([0.0, *measured_ts, 1.0], dtype=torch.float64)
    return (
        ts,
        cover_unit_interval(transported_values),
        cover_unit_interval(local_values),
        cover_unit_interval(gain_values),
    )


@torch.no_grad()
def tangent_defect_profile(model, x0, fine_steps=50, anchor_count=16):
    """Transport local acceleration with the exact tangent of the Heun pilot.

    This is the differentiable counterpart of :func:`transported_defect_profile`:
    each acceleration direction is propagated with JVPs of the discrete Heun
    steps instead of centered state perturbations.
    """
    if fine_steps <= 0:
        raise ValueError("fine_steps must be positive.")
    if anchor_count <= 0:
        raise ValueError("anchor_count must be positive.")

    fine_steps = int(fine_steps)
    dt = 1.0 / fine_steps
    x = x0.clone()
    states = [x.clone()]
    accelerations = []
    for index in range(fine_steps):
        t0 = index * dt
        t1 = (index + 1) * dt
        tt0 = torch.full((x.shape[0], 1), t0, device=x.device, dtype=x.dtype)
        tt1 = torch.full((x.shape[0], 1), t1, device=x.device, dtype=x.dtype)
        v0 = model(x, tt0)
        euler_state = x + dt * v0
        v1 = model(euler_state, tt1)
        x = x + 0.5 * dt * (v0 + v1)
        states.append(x.clone())
        accelerations.append(2.0 * (x - euler_state) / (dt * dt))

    count = min(int(anchor_count), fine_steps)
    anchor_indices = torch.linspace(0, fine_steps - 1, count, dtype=torch.float64).round().to(torch.long).unique()
    measured_ts = []
    transported_values = []
    local_values = []
    gain_values = []

    for anchor_tensor in anchor_indices:
        anchor = int(anchor_tensor.item())
        tangent = accelerations[anchor]
        state = states[anchor + 1]
        acceleration_norm = tangent.flatten(start_dim=1).norm(dim=1)

        for index in range(anchor + 1, fine_steps):
            t0 = index * dt
            t1 = (index + 1) * dt

            def heun_step(step_state):
                tt0 = torch.full((step_state.shape[0], 1), t0, device=step_state.device, dtype=step_state.dtype)
                tt1 = torch.full((step_state.shape[0], 1), t1, device=step_state.device, dtype=step_state.dtype)
                v0 = model(step_state, tt0)
                v1 = model(step_state + dt * v0, tt1)
                return step_state + 0.5 * dt * (v0 + v1)

            state, tangent = torch.func.jvp(heun_step, (state,), (tangent,))

        transported_norm = tangent.flatten(start_dim=1).norm(dim=1)
        terminal_gain = transported_norm / acceleration_norm.clamp_min(1e-12)
        measured_ts.append((anchor + 0.5) * dt)
        local_values.append(float(acceleration_norm.mean().item()))
        gain_values.append(float(terminal_gain.mean().item()))
        transported_values.append(float(transported_norm.mean().item()))

    def cover_unit_interval(values):
        return torch.tensor([values[0], *values, values[-1]], dtype=torch.float64)

    ts = torch.tensor([0.0, *measured_ts, 1.0], dtype=torch.float64)
    return (
        ts,
        cover_unit_interval(transported_values),
        cover_unit_interval(local_values),
        cover_unit_interval(gain_values),
    )


def randomized_adjoint_defect_profiles(
    model,
    x0,
    fine_steps=50,
    anchor_count=16,
    probe_counts=(2, 4, 8),
    seed=0,
):
    """Estimate transported-defect norms with randomized discrete adjoints.

    Gaussian terminal probes z give
    ``E[(a^T D Phi^T z)^2] = ||D Phi a||^2``. One reverse sweep per probe
    supplies the projected sensitivity at every requested anchor. Profiles for
    all requested prefix counts are returned from a single set of sweeps.
    """
    if fine_steps <= 0:
        raise ValueError("fine_steps must be positive.")
    if anchor_count <= 0:
        raise ValueError("anchor_count must be positive.")
    counts = sorted({int(value) for value in probe_counts})
    if not counts or counts[0] <= 0:
        raise ValueError("probe_counts must contain positive integers.")

    fine_steps = int(fine_steps)
    dt = 1.0 / fine_steps
    with torch.no_grad():
        x = x0.detach().clone()
        states = [x.clone()]
        accelerations = []
        for index in range(fine_steps):
            t0 = index * dt
            t1 = (index + 1) * dt
            tt0 = torch.full((x.shape[0], 1), t0, device=x.device, dtype=x.dtype)
            tt1 = torch.full((x.shape[0], 1), t1, device=x.device, dtype=x.dtype)
            v0 = model(x, tt0)
            euler_state = x + dt * v0
            v1 = model(euler_state, tt1)
            x = x + 0.5 * dt * (v0 + v1)
            states.append(x.clone())
            accelerations.append(2.0 * (x - euler_state) / (dt * dt))

    count = min(int(anchor_count), fine_steps)
    anchor_indices = torch.linspace(0, fine_steps - 1, count, dtype=torch.float64).round().to(torch.long).unique()
    anchors = [int(value.item()) for value in anchor_indices]
    anchor_accelerations = [accelerations[index] for index in anchors]
    local_norms = [value.flatten(start_dim=1).norm(dim=1) for value in anchor_accelerations]
    projected_squares = [torch.zeros_like(value) for value in local_norms]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    profiles = {}

    for probe_index in range(counts[-1]):
        adjoint = torch.randn(x.shape, generator=generator, dtype=x.dtype, device="cpu").to(x.device)
        anchor_position = {anchor: index for index, anchor in enumerate(anchors)}
        for step in range(fine_steps - 1, -1, -1):
            if step in anchor_position:
                position = anchor_position[step]
                projection = (adjoint * anchor_accelerations[position]).flatten(start_dim=1).sum(dim=1)
                projected_squares[position] = projected_squares[position] + projection.square()

            with torch.enable_grad():
                state = states[step].detach().requires_grad_(True)
                t0 = step * dt
                t1 = (step + 1) * dt
                tt0 = torch.full((state.shape[0], 1), t0, device=state.device, dtype=state.dtype)
                tt1 = torch.full((state.shape[0], 1), t1, device=state.device, dtype=state.dtype)
                v0 = model(state, tt0)
                v1 = model(state + dt * v0, tt1)
                next_state = state + 0.5 * dt * (v0 + v1)
                adjoint = torch.autograd.grad(
                    next_state,
                    state,
                    grad_outputs=adjoint,
                    retain_graph=False,
                    create_graph=False,
                )[0].detach()

        used = probe_index + 1
        if used not in counts:
            continue
        transported_norms = [(value / used).sqrt() for value in projected_squares]
        transported_values = [float(value.mean().item()) for value in transported_norms]
        local_values = [float(value.mean().item()) for value in local_norms]
        gain_values = [
            float((transported / local.clamp_min(1e-12)).mean().item())
            for transported, local in zip(transported_norms, local_norms)
        ]

        def cover_unit_interval(values):
            return torch.tensor([values[0], *values, values[-1]], dtype=torch.float64)

        measured_ts = [(index + 0.5) * dt for index in anchors]
        profiles[used] = (
            torch.tensor([0.0, *measured_ts, 1.0], dtype=torch.float64),
            cover_unit_interval(transported_values),
            cover_unit_interval(local_values),
            cover_unit_interval(gain_values),
        )

    return profiles


def _profile_payload(ts, err, local, gain, floor, **metadata):
    err_tensor = torch.as_tensor(err, dtype=torch.float64)
    return {
        "ts": [float(value) for value in torch.as_tensor(ts, dtype=torch.float64).tolist()],
        "err": [float(value) for value in err_tensor.tolist()],
        "local_acceleration": [float(value) for value in torch.as_tensor(local, dtype=torch.float64).tolist()],
        "terminal_gain": [float(value) for value in torch.as_tensor(gain, dtype=torch.float64).tolist()],
        "floor": float(floor),
        "kappa": kappa(ts, err, floor=floor),
        "peak_over_mean": float((err_tensor.max() / err_tensor.mean().clamp_min(1e-12)).item()),
        "estimator_version": 1,
        **metadata,
    }


def _aggregate_profile_batches(profiles, weights):
    if len(profiles) != len(weights) or not profiles:
        raise ValueError("profiles and weights must be non-empty and have the same length.")
    ts = torch.as_tensor(profiles[0][0], dtype=torch.float64)
    tensors = []
    for profile in profiles:
        if not torch.allclose(ts, torch.as_tensor(profile[0], dtype=torch.float64), rtol=0.0, atol=1e-12):
            raise ValueError("All profile batches must use identical anchor times.")
        tensors.append([torch.as_tensor(value, dtype=torch.float64) for value in profile[1:]])
    normalized = torch.as_tensor(weights, dtype=torch.float64)
    normalized = normalized / normalized.sum()
    averaged = []
    for component in range(3):
        averaged.append(sum(weight * values[component] for weight, values in zip(normalized, tensors)))
    return ts, *averaged


def _peak_memory_mb(device):
    device = torch.device(device)
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024**2)
    return peak / 1024.0


def _reset_peak_memory(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def calibrate_schedule_profiles(
    pilot_batches,
    estimators=("local", "transport_adjoint"),
    fine_steps=50,
    anchor_count=16,
    adjoint_probe_counts=(4,),
    adjoint_seed=2718,
    perturbations=(1e-3,),
    floor=1e-3,
    powers=(0.25, 0.5),
    nfe_values=(5, 10, 20, 50),
    pilot_seed=1311,
    context=None,
):
    """Calibrate population time profiles from weighted pilot batches.

    ``pilot_batches`` is an iterable of ``(velocity_callable, x0)`` pairs. A
    separate callable permits fixed conditioning labels for each image batch.
    The returned JSON-compatible bundle contains profiles, calibration cost,
    and deterministic grids for every requested NFE and mesh exponent.
    """
    batches = list(pilot_batches)
    if not batches:
        raise ValueError("pilot_batches must not be empty.")
    estimators = tuple(dict.fromkeys(estimators))
    allowed = {"local", "transport_fd", "transport_jvp", "transport_adjoint"}
    unknown = set(estimators).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown profile estimators: {sorted(unknown)}")
    weights = [int(x0.shape[0]) for _, x0 in batches]
    device = batches[0][1].device
    profiles = {}
    timings = {}

    def run_profile(name, callback, metadata):
        _reset_peak_memory(device)
        started = time.perf_counter()
        values = [callback(batch_index, velocity, x0) for batch_index, (velocity, x0) in enumerate(batches)]
        elapsed = time.perf_counter() - started
        ts, err, local, gain = _aggregate_profile_batches(values, weights)
        profiles[name] = _profile_payload(
            ts,
            err,
            local,
            gain,
            floor,
            fine_steps=int(fine_steps),
            anchor_count=int(anchor_count),
            **metadata,
        )
        timings[name] = {"wall_seconds": elapsed, "peak_memory_mb": _peak_memory_mb(device)}

    if "local" in estimators:
        def local_callback(_batch_index, velocity, x0):
            ts, err = rollout_error_profile(velocity, x0, fine_steps=fine_steps)
            ones = torch.ones_like(err)
            return ts, err, err, ones

        run_profile("local", local_callback, {"estimator": "velocity_finite_difference"})

    if "transport_fd" in estimators:
        for perturbation in perturbations:
            tag = f"{float(perturbation):g}".replace("-", "m").replace(".", "p")
            run_profile(
                f"transport_fd_eps{tag}",
                lambda _index, velocity, x0, eps=perturbation: transported_defect_profile(
                    velocity, x0, fine_steps=fine_steps, anchor_count=anchor_count, perturbation=eps
                ),
                {"estimator": "centered_finite_difference", "perturbation": float(perturbation)},
            )

    if "transport_jvp" in estimators:
        run_profile(
            "transport_jvp",
            lambda _index, velocity, x0: tangent_defect_profile(
                velocity, x0, fine_steps=fine_steps, anchor_count=anchor_count
            ),
            {"estimator": "discrete_heun_jvp"},
        )

    if "transport_adjoint" in estimators:
        for probes in sorted({int(value) for value in adjoint_probe_counts}):
            values = []
            _reset_peak_memory(device)
            started = time.perf_counter()
            for batch_index, (velocity, x0) in enumerate(batches):
                profile = randomized_adjoint_defect_profiles(
                    velocity,
                    x0,
                    fine_steps=fine_steps,
                    anchor_count=anchor_count,
                    probe_counts=(probes,),
                    seed=int(adjoint_seed) + 1009 * batch_index,
                )[probes]
                values.append(profile)
            elapsed = time.perf_counter() - started
            peak_memory = _peak_memory_mb(device)
            ts, err, local, gain = _aggregate_profile_batches(values, weights)
            name = f"transport_adjoint_r{probes}"
            profiles[name] = _profile_payload(
                ts,
                err,
                local,
                gain,
                floor,
                estimator="randomized_discrete_adjoint",
                fine_steps=int(fine_steps),
                anchor_count=int(anchor_count),
                probes=int(probes),
                probe_seed=int(adjoint_seed),
            )
            timings[name] = {"wall_seconds": elapsed, "peak_memory_mb": peak_memory}

    bundle = {
        "schema_version": 1,
        "context": dict(context or {}),
        "calibration": {
            "pilot_seed": int(pilot_seed),
            "pilot_samples": int(sum(weights)),
            "pilot_batches": len(batches),
            "fine_steps": int(fine_steps),
            "anchor_count": int(anchor_count),
            "adjoint_probe_counts": [int(value) for value in adjoint_probe_counts],
            "adjoint_seed": int(adjoint_seed),
            "perturbations": [float(value) for value in perturbations],
            "floor": float(floor),
            "powers": [float(value) for value in powers],
            "nfe_values": [int(value) for value in nfe_values],
            "device": str(device),
        },
        "timings": timings,
        "profiles": profiles,
        "grids": {},
    }
    for profile_name, profile in profiles.items():
        bundle["grids"][profile_name] = {}
        for power in powers:
            power_name = f"p{float(power):g}".replace(".", "p")
            bundle["grids"][profile_name][power_name] = {
                str(int(nfe)): equal_error_grid(
                    profile["ts"], profile["err"], int(nfe), power=power, floor=floor, end=1.0
                )
                for nfe in nfe_values
            }
    return bundle


def validate_profile_bundle(bundle, expected_context, expected_calibration=None):
    """Reject a cache calibrated for a different model or protocol."""
    if int(bundle.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported profile bundle schema: {bundle.get('schema_version')!r}.")
    actual = bundle.get("context", {})
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected_context.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Profile cache context mismatch: {mismatches}")
    expected_calibration = expected_calibration or {}
    calibration = bundle.get("calibration", {})
    calibration_mismatches = {
        key: {"expected": list(value) if isinstance(value, tuple) else value, "actual": calibration.get(key)}
        for key, value in expected_calibration.items()
        if calibration.get(key) != (list(value) if isinstance(value, tuple) else value)
    }
    if calibration_mismatches:
        raise ValueError(f"Profile cache calibration mismatch: {calibration_mismatches}")


def schedule_specs_from_bundle(
    bundle,
    nfe,
    methods=("uniform", "sctw", "adjoint_sctw", "power", "vsfm", "heun"),
    powers=(0.25, 0.5),
    power_rhos=(2.0, 3.0),
    power_kinds=("early", "late", "symmetric"),
    adjoint_probes=4,
):
    """Build the shared evaluator schedule vocabulary for one NFE budget."""
    methods = tuple(methods)
    specs = []
    if "uniform" in methods:
        specs.append({"schedule": "uniform", "grid": [float(x) for x in torch.linspace(0, 1, nfe + 1)], "solver": "euler", "nfe_used": nfe})
    profile_mapping = {
        "sctw": "local",
        "adjoint_sctw": f"transport_adjoint_r{int(adjoint_probes)}",
    }
    for method, profile_name in profile_mapping.items():
        if method not in methods:
            continue
        if profile_name not in bundle.get("profiles", {}):
            raise ValueError(f"Profile bundle does not contain {profile_name!r} required by {method!r}.")
        for power in powers:
            power_name = f"p{float(power):g}".replace(".", "p")
            grid = bundle["grids"][profile_name][power_name][str(int(nfe))]
            specs.append(
                {
                    "schedule": f"{method}_{power_name}",
                    "grid": grid,
                    "solver": "euler",
                    "nfe_used": nfe,
                    "warp_power": float(power),
                    "profile": profile_name,
                }
            )
    if "power" in methods:
        for kind in power_kinds:
            for rho in power_rhos:
                tag = f"{float(rho):g}".replace(".", "p")
                specs.append(
                    {
                        "schedule": f"power_{kind}_rho{tag}",
                        "grid": power_time_grid(nfe, rho=rho, kind=kind),
                        "solver": "euler",
                        "nfe_used": nfe,
                    }
                )
    if "vsfm" in methods:
        for profile in ("coasting", "uniform_deceleration"):
            specs.append(
                {
                    "schedule": f"vsfm_{profile}",
                    "grid": vsfm_time_grid(nfe, profile=profile),
                    "solver": "euler",
                    "nfe_used": nfe,
                }
            )
    if "heun" in methods:
        intervals = max(1, int(nfe) // 2)
        specs.append(
            {
                "schedule": "heun_uniform_matched",
                "grid": [float(x) for x in torch.linspace(0, 1, intervals + 1)],
                "solver": "heun",
                "nfe_used": 2 * intervals,
            }
        )
    return specs


def estimator_schedule_specs_from_bundle(bundle, nfe, profile_names, powers=(0.5,)):
    """Build Euler schedules for FD/JVP/adjoint equivalence-only profiles."""
    specs = []
    for profile_name in profile_names:
        if profile_name not in bundle.get("profiles", {}):
            raise ValueError(f"Profile bundle does not contain equivalence profile {profile_name!r}.")
        for power in powers:
            power_name = f"p{float(power):g}".replace(".", "p")
            specs.append(
                {
                    "schedule": f"{profile_name}_{power_name}",
                    "grid": bundle["grids"][profile_name][power_name][str(int(nfe))],
                    "solver": "euler",
                    "nfe_used": int(nfe),
                    "warp_power": float(power),
                    "profile": profile_name,
                }
            )
    return specs


def profile_equivalence_rows(bundle, reference_profile="transport_jvp"):
    """Compare transported profiles and their cached knots to one reference.

    Profile vectors are L2-normalized before comparison because mesh placement
    depends on temporal shape, not the arbitrary scale of the estimator.
    """
    profiles = bundle.get("profiles", {})
    if reference_profile not in profiles:
        raise ValueError(f"Profile bundle does not contain reference {reference_profile!r}.")
    reference = profiles[reference_profile]
    reference_ts = torch.as_tensor(reference["ts"], dtype=torch.float64)
    reference_err = torch.as_tensor(reference["err"], dtype=torch.float64)
    reference_normalized = reference_err / reference_err.norm().clamp_min(1e-12)
    rows = []
    for name, profile in profiles.items():
        if name == "local":
            continue
        ts = torch.as_tensor(profile["ts"], dtype=torch.float64)
        err = torch.as_tensor(profile["err"], dtype=torch.float64)
        if not torch.allclose(ts, reference_ts, rtol=0.0, atol=1e-12):
            raise ValueError(f"Profile {name!r} does not use the reference anchor times.")
        normalized = err / err.norm().clamp_min(1e-12)
        profile_error = float((normalized - reference_normalized).norm().item())
        timing = bundle.get("timings", {}).get(name, {})
        for power_name, grids_by_nfe in bundle.get("grids", {}).get(name, {}).items():
            reference_grids = bundle["grids"][reference_profile][power_name]
            for nfe, grid in grids_by_nfe.items():
                reference_grid = torch.as_tensor(reference_grids[nfe], dtype=torch.float64)
                candidate_grid = torch.as_tensor(grid, dtype=torch.float64)
                rows.append(
                    {
                        "profile": name,
                        "reference_profile": reference_profile,
                        "normalized_profile_l2": profile_error,
                        "power": float(power_name[1:].replace("p", ".")),
                        "nfe": int(nfe),
                        "max_knot_displacement": float((candidate_grid - reference_grid).abs().max().item()),
                        "wall_seconds": timing.get("wall_seconds"),
                        "peak_memory_mb": timing.get("peak_memory_mb"),
                    }
                )
    return rows


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
