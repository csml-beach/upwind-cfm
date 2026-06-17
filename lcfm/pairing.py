import torch
from scipy.optimize import linear_sum_assignment


def minibatch_ot_pair(x0, x1):
    cost = torch.cdist(x0.detach(), x1.detach()).pow(2).cpu().numpy()
    _, col = linear_sum_assignment(cost)
    order = torch.as_tensor(col, device=x1.device)
    return x0, x1[order]


def _positive_median(values, eps):
    positive = values[values > eps]
    if positive.numel() == 0:
        return torch.ones((), device=values.device, dtype=values.dtype)
    return torch.median(positive).clamp_min(eps)


def _pressure_reference_pairs(x0, x1, reference_pairing):
    if reference_pairing == "independent":
        return x1
    if reference_pairing == "minibatch_ot":
        _, x1_ref = minibatch_ot_pair(x0, x1)
        return x1_ref
    raise ValueError("reference_pairing must be 'independent' or 'minibatch_ot'.")


@torch.no_grad()
def pressure_aware_minibatch_ot_pair(x0, x1, config):
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
    if beta == 0.0:
        return minibatch_ot_pair(x0, x1)

    x0_detached = x0.detach()
    x1_detached = x1.detach()
    base_cost = torch.cdist(x0_detached, x1_detached).pow(2)
    base_scale = _positive_median(base_cost, eps)

    x1_ref = _pressure_reference_pairs(x0_detached, x1_detached, reference_pairing)
    ref_xt = (1.0 - t_value) * x0_detached + t_value * x1_ref
    ref_u = x1_ref - x0_detached

    ref_dist_sq = torch.cdist(ref_xt, ref_xt).pow(2)
    if bandwidth is None:
        bandwidth_sq = _positive_median(ref_dist_sq, eps)
    else:
        bandwidth_sq = torch.as_tensor(float(bandwidth) ** 2, device=x0.device, dtype=x0.dtype).clamp_min(eps)

    candidate_xt = (1.0 - t_value) * x0_detached[:, None, :] + t_value * x1_detached[None, :, :]
    flat_xt = candidate_xt.reshape(-1, x0.shape[1])
    dist_sq = torch.cdist(flat_xt, ref_xt).pow(2)
    kernel = torch.exp(-0.5 * dist_sq / bandwidth_sq)
    normalizer = kernel.sum(dim=1, keepdim=True).clamp_min(eps)
    local_mean = (kernel @ ref_u) / normalizer
    ref_u_sq = ref_u.pow(2).sum(dim=1, keepdim=True)
    local_second = (kernel @ ref_u_sq) / normalizer
    local_variance = (local_second - local_mean.pow(2).sum(dim=1, keepdim=True)).clamp_min(0.0)
    variance_scale = _positive_median(local_variance, eps)
    pressure_cost = (local_variance / variance_scale).reshape_as(base_cost)

    cost = base_cost + beta * base_scale * pressure_cost
    _, col = linear_sum_assignment(cost.cpu().numpy())
    order = torch.as_tensor(col, device=x1.device)
    return x0, x1[order]


def apply_pairing(x0, x1, config):
    pairing = config.get("pairing", "independent")
    if pairing == "independent":
        return x0, x1
    if pairing == "minibatch_ot":
        return minibatch_ot_pair(x0, x1)
    if pairing == "pressure_aware_minibatch_ot":
        return pressure_aware_minibatch_ot_pair(x0, x1, config)
    raise ValueError(f"Unknown pairing: {pairing}")
