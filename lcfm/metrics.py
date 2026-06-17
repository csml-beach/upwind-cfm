import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def wasserstein_match(x, y, p=1):
    """Exact empirical Wasserstein distance by optimal bipartite matching.

    The samples are treated as equally weighted empirical measures. For p=1 this
    returns the mean matched Euclidean distance. For p=2 this minimizes squared
    Euclidean cost and returns sqrt(mean squared matched distance). This is exact
    for the finite sample sets; remaining uncertainty is statistical sample noise,
    not assignment approximation.
    """
    if p < 1:
        raise ValueError("p must be >= 1.")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("wasserstein_match requires non-empty sample sets.")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("wasserstein_match expects two 2D tensors with the same feature dimension.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Exact equally weighted matching requires the same number of samples.")

    x_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
    y_np = y.detach().cpu().numpy().astype(np.float64, copy=False)
    d_matrix = cdist(x_np, y_np, metric="euclidean")
    cost = d_matrix if p == 1 else d_matrix**p
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = d_matrix[row_ind, col_ind]
    if p == 1:
        return float(matched.mean())
    return float(np.power(np.mean(matched**p), 1.0 / p))


def path_length_ratio(traj):
    diffs = traj[1:] - traj[:-1]
    path_lengths = torch.norm(diffs, dim=-1).sum(dim=0)
    straight = torch.norm(traj[-1] - traj[0], dim=-1)
    return float((path_lengths / (straight + 1e-6)).mean().item())


def mean_path_length(traj):
    diffs = traj[1:] - traj[:-1]
    path_lengths = torch.norm(diffs, dim=-1).sum(dim=0)
    return float(path_lengths.mean().item())


def mean_endpoint_displacement(traj):
    displacement = torch.norm(traj[-1] - traj[0], dim=-1)
    return float(displacement.mean().item())


def trajectory_acceleration(traj):
    if traj.shape[0] < 3:
        return 0.0
    vel = traj[1:] - traj[:-1]
    acc = vel[1:] - vel[:-1]
    return float(torch.norm(acc, dim=-1).mean().item())


def mode_statistics(samples, centers, p_min=0.05, hit_radius=None):
    if hit_radius is None:
        raise ValueError("mode_statistics requires hit_radius for hit-based mode metrics.")
    distances = torch.cdist(samples, centers)
    assignments = torch.argmin(distances, dim=1)
    nearest_distances = distances[torch.arange(samples.shape[0], device=samples.device), assignments]
    hits = nearest_distances <= hit_radius
    hit_counts = torch.bincount(assignments[hits], minlength=centers.shape[0]).float()
    hit_probs = hit_counts / samples.shape[0]
    return {
        "mode_hit_coverage": int((hit_probs > p_min).sum().item()),
        "target_hit_rate": float(hits.float().mean().item()),
        "mode_hit_probs": [float(p.item()) for p in hit_probs],
        "hit_radius": float(hit_radius),
    }


def temporal_tv(video):
    diffs = video[:, 1:, :] - video[:, :-1, :]
    return float(torch.norm(diffs, dim=-1).pow(2).mean().item())


def rmse(x, y):
    return float(torch.sqrt(torch.mean((x - y) ** 2)).item())


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
