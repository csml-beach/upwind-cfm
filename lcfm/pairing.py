import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def pairing_features(x, config):
    kwargs = config.get("pairing_kwargs", {}) if config else {}
    feature = kwargs.get("cost_feature", "raw")
    if feature == "raw":
        return x
    if feature == "downsampled_pixels":
        image_shape = tuple(kwargs.get("image_shape", (3, 32, 32)))
        downsample_size = int(kwargs.get("downsample_size", 8))
        if x.shape[1] != image_shape[0] * image_shape[1] * image_shape[2]:
            raise ValueError("downsampled_pixels pairing requires flat vectors matching image_shape.")
        images = x.reshape(x.shape[0], *image_shape)
        pooled = F.adaptive_avg_pool2d(images, (downsample_size, downsample_size))
        return pooled.flatten(1)
    raise ValueError(f"Unknown pairing cost_feature: {feature}")


def minibatch_ot_pair(x0, x1, config=None):
    order = minibatch_ot_order(x0, x1, config)
    return x0, x1[order]


def minibatch_ot_order(x0, x1, config=None):
    x0_feat = pairing_features(x0.detach(), config)
    x1_feat = pairing_features(x1.detach(), config)
    cost = torch.cdist(x0_feat, x1_feat).pow(2).cpu().numpy()
    _, col = linear_sum_assignment(cost)
    return torch.as_tensor(col, device=x1.device)


def _pairing_cost(x0_feat, x1_feat):
    return torch.cdist(x0_feat, x1_feat).pow(2)


def _sinkhorn_epsilon(cost, kwargs):
    eps = kwargs.get("sinkhorn_epsilon")
    if eps is not None:
        return torch.as_tensor(float(eps), device=cost.device, dtype=cost.dtype).clamp_min(1e-12)
    scale = float(kwargs.get("sinkhorn_epsilon_scale", 0.05))
    return (scale * _positive_median(cost, 1e-12)).clamp_min(1e-12)


def sinkhorn_plan_from_cost(cost, config=None):
    """Balanced entropic OT plan for a square minibatch cost matrix."""
    kwargs = config.get("pairing_kwargs", {}) if config else {}
    iterations = int(kwargs.get("sinkhorn_iterations", 50))
    if iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive.")
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("Sinkhorn pairing expects a square minibatch cost matrix.")

    epsilon = _sinkhorn_epsilon(cost, kwargs)
    n = cost.shape[0]
    log_kernel = -cost / epsilon
    log_mu = -torch.log(torch.as_tensor(float(n), device=cost.device, dtype=cost.dtype))
    log_nu = log_mu
    u = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    v = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(log_kernel + u[:, None], dim=0)
    return torch.exp(log_kernel + u[:, None] + v[None, :])


def _greedy_plan_order(plan):
    n = plan.shape[0]
    flat_order = torch.argsort(plan.flatten(), descending=True).detach().cpu().tolist()
    row_used = torch.zeros(n, dtype=torch.bool)
    col_used = torch.zeros(n, dtype=torch.bool)
    order = torch.empty(n, dtype=torch.long)
    filled = 0
    for flat_index in flat_order:
        row = flat_index // n
        col = flat_index % n
        if row_used[row] or col_used[col]:
            continue
        row_used[row] = True
        col_used[col] = True
        order[row] = col
        filled += 1
        if filled == n:
            break
    if filled != n:
        raise RuntimeError("Failed to project Sinkhorn plan to a complete greedy assignment.")
    return order.to(plan.device)


def sinkhorn_order_from_cost(cost, config=None):
    kwargs = config.get("pairing_kwargs", {}) if config else {}
    plan = sinkhorn_plan_from_cost(cost, config)
    projection = kwargs.get("sinkhorn_projection", "greedy")
    if projection == "greedy":
        return _greedy_plan_order(plan)
    if projection == "argmax":
        return torch.argmax(plan, dim=1)
    if projection == "sample":
        probs = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(probs, num_samples=1).flatten()
    raise ValueError("sinkhorn_projection must be 'greedy', 'argmax', or 'sample'.")


@torch.no_grad()
def sinkhorn_ot_order(x0, x1, config=None):
    x0_feat = pairing_features(x0.detach(), config)
    x1_feat = pairing_features(x1.detach(), config)
    return sinkhorn_order_from_cost(_pairing_cost(x0_feat, x1_feat), config).to(x1.device)


def sinkhorn_ot_pair(x0, x1, config=None):
    order = sinkhorn_ot_order(x0, x1, config)
    return x0, x1[order]


def _positive_median(values, eps):
    positive = values[values > eps]
    if positive.numel() == 0:
        return torch.ones((), device=values.device, dtype=values.dtype)
    return torch.median(positive).clamp_min(eps)


def _pressure_reference_order(x0_feat, x1_feat, reference_pairing, config):
    if reference_pairing == "independent":
        return torch.arange(x1_feat.shape[0], device=x1_feat.device)
    if reference_pairing == "minibatch_ot":
        cost = _pairing_cost(x0_feat, x1_feat).cpu().numpy()
        _, col = linear_sum_assignment(cost)
        return torch.as_tensor(col, device=x1_feat.device)
    if reference_pairing == "sinkhorn_ot":
        return sinkhorn_order_from_cost(_pairing_cost(x0_feat, x1_feat), config)
    raise ValueError("reference_pairing must be 'independent', 'minibatch_ot', or 'sinkhorn_ot'.")


@torch.no_grad()
def pressure_aware_cost(x0, x1, config):
    """Minibatch OT with a local conditional-velocity variance cost.

    For each candidate pair (x0_i, x1_j), we form an interpolant point x_t,ij
    and estimate how variable nearby conditional velocities are under a reference
    minibatch coupling. High local variance is a scalar pressure proxy: the path
    passes through a region where multiple transport directions overlap.
    """
    kwargs = config.get("pairing_kwargs", {})
    beta = float(kwargs.get("pressure_beta", 0.1))
    pressure_t = kwargs.get("pressure_t", 0.5)
    bandwidth = kwargs.get("pressure_bandwidth")
    reference_pairing = kwargs.get("reference_pairing", "independent")
    eps = float(kwargs.get("eps", 1e-8))

    if isinstance(pressure_t, str):
        if pressure_t != "random":
            raise ValueError("pressure_t must be a float or 'random'.")
        t_value = float(torch.rand((), device=x0.device).item())
    else:
        t_value = float(pressure_t)
    if not 0.0 <= t_value <= 1.0:
        raise ValueError("pressure_t must be in [0, 1].")
    if beta < 0.0:
        raise ValueError("pressure_beta must be non-negative.")

    x0_detached = x0.detach()
    x1_detached = x1.detach()
    x0_feat = pairing_features(x0_detached, config)
    x1_feat = pairing_features(x1_detached, config)
    base_cost = _pairing_cost(x0_feat, x1_feat)
    if beta == 0.0:
        return base_cost
    base_scale = _positive_median(base_cost, eps)

    ref_order = _pressure_reference_order(x0_feat, x1_feat, reference_pairing, config)
    x1_ref_feat = x1_feat[ref_order]
    ref_xt = (1.0 - t_value) * x0_feat + t_value * x1_ref_feat
    ref_u = x1_ref_feat - x0_feat

    ref_dist_sq = torch.cdist(ref_xt, ref_xt).pow(2)
    if bandwidth is None:
        bandwidth_sq = _positive_median(ref_dist_sq, eps)
    else:
        bandwidth_sq = torch.as_tensor(float(bandwidth) ** 2, device=x0.device, dtype=x0.dtype).clamp_min(eps)

    candidate_xt = (1.0 - t_value) * x0_feat[:, None, :] + t_value * x1_feat[None, :, :]
    flat_xt = candidate_xt.reshape(-1, x0_feat.shape[1])
    dist_sq = torch.cdist(flat_xt, ref_xt).pow(2)
    kernel = torch.exp(-0.5 * dist_sq / bandwidth_sq)
    normalizer = kernel.sum(dim=1, keepdim=True).clamp_min(eps)
    local_mean = (kernel @ ref_u) / normalizer
    ref_u_sq = ref_u.pow(2).sum(dim=1, keepdim=True)
    local_second = (kernel @ ref_u_sq) / normalizer
    local_variance = (local_second - local_mean.pow(2).sum(dim=1, keepdim=True)).clamp_min(0.0)
    variance_scale = _positive_median(local_variance, eps)
    pressure_cost = (local_variance / variance_scale).reshape_as(base_cost)

    return base_cost + beta * base_scale * pressure_cost


@torch.no_grad()
def pressure_aware_minibatch_ot_order(x0, x1, config):
    cost = pressure_aware_cost(x0, x1, config)
    _, col = linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(col, device=x1.device)


@torch.no_grad()
def pressure_aware_sinkhorn_ot_order(x0, x1, config):
    cost = pressure_aware_cost(x0, x1, config)
    return sinkhorn_order_from_cost(cost, config).to(x1.device)


@torch.no_grad()
def pressure_aware_minibatch_ot_pair(x0, x1, config):
    order = pressure_aware_minibatch_ot_order(x0, x1, config)
    return x0, x1[order]


@torch.no_grad()
def pressure_aware_sinkhorn_ot_pair(x0, x1, config):
    order = pressure_aware_sinkhorn_ot_order(x0, x1, config)
    return x0, x1[order]


def apply_pairing(x0, x1, config, *extras):
    pairing = config.get("pairing", "independent")
    if pairing == "independent":
        return (x0, x1, *extras) if extras else (x0, x1)
    if pairing == "minibatch_ot":
        order = minibatch_ot_order(x0, x1, config)
        paired = (x0, x1[order])
        return (*paired, *(extra[order] for extra in extras)) if extras else paired
    if pairing == "sinkhorn_ot":
        order = sinkhorn_ot_order(x0, x1, config)
        paired = (x0, x1[order])
        return (*paired, *(extra[order] for extra in extras)) if extras else paired
    if pairing == "pressure_aware_minibatch_ot":
        order = pressure_aware_minibatch_ot_order(x0, x1, config)
        paired_x0, paired_x1 = x0, x1[order]
        if not extras:
            return paired_x0, paired_x1
        return paired_x0, paired_x1, *(extra[order] for extra in extras)
    if pairing == "pressure_aware_sinkhorn_ot":
        order = pressure_aware_sinkhorn_ot_order(x0, x1, config)
        paired_x0, paired_x1 = x0, x1[order]
        if not extras:
            return paired_x0, paired_x1
        return paired_x0, paired_x1, *(extra[order] for extra in extras)
    raise ValueError(f"Unknown pairing: {pairing}")
