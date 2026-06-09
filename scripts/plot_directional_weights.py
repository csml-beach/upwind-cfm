#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm.losses import build_method, cfm_batch
from lcfm.plotting import MODE_COLORS, load_run
from lcfm.utils import set_seed


def parse_t_values(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def mode_problem_bounds(points_by_panel, centers, sigma_mode):
    all_points = [centers.detach().cpu()]
    all_points.extend(point.detach().cpu() for point in points_by_panel)
    stacked = torch.cat(all_points, dim=0)
    mins = stacked.min(dim=0).values
    maxs = stacked.max(dim=0).values
    spans = (maxs - mins).clamp_min(1.0)
    pad = torch.clamp(0.08 * spans, min=4.0 * sigma_mode)
    return (
        float(mins[0] - pad[0]),
        float(maxs[0] + pad[0]),
        float(mins[1] - pad[1]),
        float(maxs[1] + pad[1]),
    )


def compute_weights(method, model, x0, x1, t_value, include_temporal=True, include_global=True):
    t = torch.full((x0.shape[0], 1), t_value, device=x0.device)
    xt, target, vt = cfm_batch(model, x0, x1, t)
    solver_weight = method.directional_weight_fd(model, xt, t, vt).detach().flatten()
    uncertainty_gate = method.local_velocity_variance_gate(xt, target).detach().flatten()
    temporal_weight = ((1.0 - t).pow(method.alpha) / method.epsilon).detach().flatten()
    weight = solver_weight
    if getattr(method, "uncertainty_gate", "none") != "none":
        weight = weight * uncertainty_gate
    if include_temporal:
        weight = weight * temporal_weight
    if include_global:
        weight = weight * method.weight
    return xt.detach(), weight.detach(), solver_weight.detach(), temporal_weight.detach(), uncertainty_gate.detach()


def draw_centers(ax, centers, sigma_mode):
    colors = MODE_COLORS[: centers.shape[0]]
    for center, color in zip(centers.cpu(), colors):
        ax.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                radius=3.0 * sigma_mode,
                facecolor="none",
                edgecolor=color,
                linewidth=0.8,
                alpha=0.55,
                zorder=3,
            )
        )
    ax.scatter(
        centers[:, 0].cpu(),
        centers[:, 1].cpu(),
        s=24,
        c=colors,
        edgecolors="#111827",
        linewidths=0.5,
        zorder=4,
    )


def plot_directional_weights(
    run_dir,
    output=None,
    t_values=None,
    n_samples=2000,
    seed=1234,
    include_temporal=True,
    include_global=True,
):
    device = torch.device("cpu")
    config, problem, model = load_run(run_dir, device)
    method = build_method(config["method"], config.get("method_kwargs", {}))
    if config["method"] != "directional_regularization_cfm":
        raise ValueError(f"Expected directional_regularization_cfm run, got {config['method']}")
    if not hasattr(method, "directional_weight_fd"):
        raise ValueError("Directional weight diagnostic currently supports FD directional weights only.")
    if t_values is None:
        t_values = [0.25, 0.5, 0.75]

    set_seed(seed)
    x0, x1 = problem.sample_train_batch(n_samples, device)
    panels = [compute_weights(method, model, x0, x1, t, include_temporal, include_global) for t in t_values]
    xt_panels = [item[0] for item in panels]
    centers = problem.centers(device)
    x_min, x_max, y_min, y_max = mode_problem_bounds(xt_panels + [x0, x1], centers, problem.sigma_mode)

    all_weights = torch.cat([item[1] for item in panels]).cpu()
    positive = all_weights[all_weights > 0]
    if positive.numel() == 0:
        vmin, vmax = 1e-12, 1.0
    else:
        vmin = max(float(torch.quantile(positive, 0.01).item()), 1e-12)
        vmax = max(float(torch.quantile(positive, 0.99).item()), vmin * 10.0)

    fig, axes = plt.subplots(
        2,
        len(t_values),
        figsize=(4.2 * len(t_values), 7.2),
        squeeze=False,
        constrained_layout=True,
    )
    for col, (t_value, (xt, weight, solver_weight, temporal_weight, uncertainty_gate)) in enumerate(zip(t_values, panels)):
        ax = axes[0, col]
        ax.set_facecolor("#fbfbfa")
        draw_centers(ax, centers, problem.sigma_mode)
        sc = ax.scatter(
            xt[:, 0].cpu(),
            xt[:, 1].cpu(),
            c=weight.clamp_min(vmin).cpu(),
            s=8,
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            alpha=0.75,
            linewidths=0,
            zorder=2,
        )
        ax.set_title(f"t={t_value:g}", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8, colors="#4b5563")
        for spine in ax.spines.values():
            spine.set_color("#d1d5db")

        hist_ax = axes[1, col]
        hist_ax.hist(weight.cpu().numpy(), bins=60, color="#374151", alpha=0.85)
        hist_ax.set_title(
            f"mean={weight.mean().item():.3g}, p95={torch.quantile(weight, 0.95).item():.3g}",
            fontsize=9,
        )
        hist_ax.set_xlabel("effective weight", fontsize=8)
        hist_ax.set_ylabel("count", fontsize=8)
        hist_ax.tick_params(labelsize=8)
        hist_ax.grid(axis="y", alpha=0.2)
        hist_ax.text(
            0.98,
            0.95,
            f"raw mean={solver_weight.mean().item():.3g}\n"
            f"time mean={temporal_weight.mean().item():.3g}\n"
            f"gate mean={uncertainty_gate.mean().item():.3g}",
            transform=hist_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#e5e7eb", "alpha": 0.88},
        )
    cbar = fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.85)
    cbar.set_label("effective directional weight", fontsize=9)
    fig.suptitle(Path(run_dir).name, fontsize=12)
    fig.suptitle(Path(run_dir).name, fontsize=12)

    output = Path(output) if output else Path(run_dir) / "directional_weight_diagnostic.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--t-values", default="0.25,0.5,0.75")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--raw-solver-weight", action="store_true")
    args = parser.parse_args()
    output = plot_directional_weights(
        args.run_dir,
        args.output,
        parse_t_values(args.t_values),
        args.n_samples,
        args.seed,
        include_temporal=not args.raw_solver_weight,
        include_global=not args.raw_solver_weight,
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
