import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def wasserstein_match(x, y):
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    d_matrix = cdist(x_np, y_np, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(d_matrix)
    return float(d_matrix[row_ind, col_ind].mean())


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
