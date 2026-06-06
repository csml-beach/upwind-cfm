from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import torch

from . import datasets  # noqa: F401
from . import losses  # noqa: F401
from . import models  # noqa: F401
from . import solvers  # noqa: F401
from .models import build_model
from .registry import DATASETS, get
from .solvers import solve
from .utils import read_json, set_seed


MODE_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f59e0b"]


def is_mode_problem(problem):
    return hasattr(problem, "centers") and hasattr(problem, "sigma_mode")


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    dataset_cls = get(DATASETS, config["dataset"])
    set_seed(config.get("seed", 42))
    problem = dataset_cls(config.get("dataset_kwargs", {}))
    model = build_model(config.get("model", "mlp"), problem.dim, config).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return config, problem, model


def spiral_eval_inputs(problem, config, device, eval_seed=None):
    eval_cfg = config.get("eval", {})
    n_eval = eval_cfg.get("n_eval", 1000)
    if eval_seed is None:
        eval_seed = eval_cfg.get("plot_seed", 1234)
    set_seed(eval_seed)
    x0 = problem.eval_initial(n_eval, device)
    target = problem.target_eval(n_eval, device)
    return x0, target


def eval_inputs(problem, config, device, eval_seed=None):
    eval_cfg = config.get("eval", {})
    n_eval = eval_cfg.get("n_eval", 1000)
    if eval_seed is None:
        eval_seed = eval_cfg.get("plot_seed", 1234)
    set_seed(eval_seed)
    x0 = problem.eval_initial(n_eval, device)
    target = problem.target_eval(n_eval, device)
    return x0, target


def solver_config(config, steps=None, noise=None):
    cfg = dict(config.get("solver_kwargs", {"steps": 15}))
    if steps is not None:
        cfg["steps"] = steps
    if noise is not None:
        cfg["noise"] = noise
    return cfg


@torch.no_grad()
def spiral_trajectory(model, x0, config, eval_seed=None, steps=None, noise=None):
    if eval_seed is None:
        eval_seed = config.get("eval", {}).get("plot_seed", 1234)
    set_seed(eval_seed)
    return solve(config.get("solver", "euler"), model, x0, solver_config(config, steps, noise))


def plot_spiral_run(run_dir, output=None, eval_seed=None, n_traj=24, n_final=None, steps=None, noise=None):
    device = torch.device("cpu")
    config, problem, model = load_run(run_dir, device)
    if problem.name != "spiral":
        raise ValueError(f"plot_spiral_run only supports spiral runs, got {problem.name}")
    x0, target = spiral_eval_inputs(problem, config, device, eval_seed)
    traj = spiral_trajectory(model, x0, config, eval_seed, steps, noise)
    final = traj[-1]
    if n_final is not None:
        final = final[:n_final]

    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / "plot.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target[:, 0].cpu(), target[:, 1].cpu(), s=8, c="0.75", alpha=0.55, label="target")
    ax.scatter(final[:, 0].cpu(), final[:, 1].cpu(), s=10, c="#2563eb", alpha=0.75, label="generated")
    for i in range(min(n_traj, traj.shape[1])):
        ax.plot(traj[:, i, 0].cpu(), traj[:, i, 1].cpu(), color="#2563eb", alpha=0.25, linewidth=0.8)
    ax.set_title(run_dir.name)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_spiral_comparison(run_dirs, output, eval_seed=1234, n_traj=16, n_final=None, steps=None, noise=None):
    device = torch.device("cpu")
    loaded = [load_run(run_dir, device) for run_dir in run_dirs]
    first_config, first_problem, _ = loaded[0]
    if first_problem.name != "spiral":
        raise ValueError("plot_spiral_comparison currently only supports spiral runs")
    x0, target = spiral_eval_inputs(first_problem, first_config, device, eval_seed)

    fig, axes = plt.subplots(1, len(loaded), figsize=(5 * len(loaded), 5), squeeze=False)
    colors = ["#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    for ax, run_dir, loaded_item, color in zip(axes[0], run_dirs, loaded, colors):
        config, problem, model = loaded_item
        if problem.name != "spiral":
            raise ValueError(f"comparison only supports spiral runs, got {problem.name}")
        traj = spiral_trajectory(model, x0, config, eval_seed, steps, noise)
        final = traj[-1]
        if n_final is not None:
            final = final[:n_final]
        ax.scatter(target[:, 0].cpu(), target[:, 1].cpu(), s=8, c="0.75", alpha=0.55)
        ax.scatter(final[:, 0].cpu(), final[:, 1].cpu(), s=10, c=color, alpha=0.75)
        for i in range(min(n_traj, traj.shape[1])):
            ax.plot(traj[:, i, 0].cpu(), traj[:, i, 1].cpu(), color=color, alpha=0.25, linewidth=0.8)
        ax.set_title(Path(run_dir).name)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _mode_assignments(points, centers):
    return torch.argmin(torch.cdist(points, centers), dim=1)


def _trajectory_indices(assignments, n_traj, n_modes):
    if n_traj is None:
        return list(range(assignments.shape[0]))
    if n_traj <= 0:
        return []
    selected = []
    per_mode = max(1, n_traj // max(1, n_modes))
    for mode in range(n_modes):
        idx = torch.nonzero(assignments == mode, as_tuple=False).flatten()
        if idx.numel() > 0:
            selected.extend(idx[:per_mode].tolist())
    if len(selected) < n_traj:
        existing = set(selected)
        for idx in range(assignments.shape[0]):
            if idx not in existing:
                selected.append(idx)
            if len(selected) >= n_traj:
                break
    return selected[:n_traj]


def _draw_five_mode_panel(
    ax,
    target,
    traj,
    centers,
    sigma_mode,
    title,
    metrics=None,
    n_traj=None,
    n_final=None,
    trajectory_alpha=0.10,
    trajectory_width=0.55,
):
    final = traj[-1]
    plotted_final = final[:n_final] if n_final is not None else final
    assignments = _mode_assignments(plotted_final, centers)
    colors = MODE_COLORS[: centers.shape[0]]

    ax.set_facecolor("#fbfbfa")
    for center, color in zip(centers.cpu(), colors):
        ax.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                radius=2.0 * sigma_mode,
                facecolor=color,
                edgecolor="none",
                alpha=0.12,
                zorder=0,
            )
        )
        ax.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                radius=3.0 * sigma_mode,
                facecolor="none",
                edgecolor=color,
                linewidth=0.8,
                alpha=0.30,
                zorder=1,
            )
        )

    ax.scatter(
        target[:, 0].cpu(),
        target[:, 1].cpu(),
        s=9,
        c="#9ca3af",
        alpha=0.28,
        linewidths=0,
        zorder=2,
    )

    selected = _trajectory_indices(assignments.cpu(), n_traj, centers.shape[0])
    for idx in selected:
        points = traj[:, idx, :].cpu().numpy()
        segments = [[points[i], points[i + 1]] for i in range(points.shape[0] - 1)]
        lc = LineCollection(segments, colors="#111827", linewidths=trajectory_width, alpha=trajectory_alpha, zorder=3)
        ax.add_collection(lc)

    for mode, color in enumerate(colors):
        mask = assignments == mode
        if mask.any():
            pts = plotted_final[mask]
            ax.scatter(
                pts[:, 0].cpu(),
                pts[:, 1].cpu(),
                s=13,
                c=color,
                alpha=0.72,
                linewidths=0,
                zorder=4,
            )

    ax.scatter(
        centers[:, 0].cpu(),
        centers[:, 1].cpu(),
        s=28,
        c=colors,
        edgecolors="#111827",
        linewidths=0.5,
        zorder=5,
    )

    ax.set_title(title, fontsize=11, pad=8)
    if metrics:
        coverage = metrics.get("mode_hit_coverage", float("nan"))
        hit_rate = metrics.get("target_hit_rate", float("nan"))
        accel = metrics.get("trajectory_acceleration", float("nan"))
        w_dist = metrics.get("wasserstein", float("nan"))
        text = (
            f"W={w_dist:.3f} | hit={hit_rate*100:.1f}% | acc={accel:.2f}\n"
            f"coverage: {coverage}/5 modes"
        )
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#111827",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#e5e7eb", "alpha": 0.86},
        )
    ax.set_aspect("equal", adjustable="box")
    all_points = torch.cat(
        [
            target.detach().cpu(),
            plotted_final.detach().cpu(),
            traj.detach().cpu().reshape(-1, traj.shape[-1]),
            centers.detach().cpu(),
        ],
        dim=0,
    )
    mins = all_points.min(dim=0).values
    maxs = all_points.max(dim=0).values
    spans = (maxs - mins).clamp_min(1.0)
    pad = torch.clamp(0.08 * spans, min=4.0 * sigma_mode)
    ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    ax.tick_params(labelsize=8, colors="#4b5563", length=3)
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")
        spine.set_linewidth(0.8)


@torch.no_grad()
def plot_five_modes_run(
    run_dir,
    output=None,
    eval_seed=None,
    n_traj=None,
    n_final=600,
    steps=None,
    noise=None,
    trajectory_alpha=0.10,
    trajectory_width=0.55,
):
    device = torch.device("cpu")
    config, problem, model = load_run(run_dir, device)
    if not is_mode_problem(problem):
        raise ValueError(f"plot_five_modes_run only supports mode-mixture runs, got {problem.name}")
    x0, target = eval_inputs(problem, config, device, eval_seed)
    traj = spiral_trajectory(model, x0, config, eval_seed, steps, noise)

    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / "five_modes_plot.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else None

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _draw_five_mode_panel(
        ax,
        target,
        traj,
        problem.centers(device),
        problem.sigma_mode,
        run_dir.name,
        metrics=metrics,
        n_traj=n_traj,
        n_final=n_final,
        trajectory_alpha=trajectory_alpha,
        trajectory_width=trajectory_width,
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return output


def plot_five_modes_comparison(
    run_dirs,
    output,
    eval_seed=1234,
    n_traj=None,
    n_final=600,
    steps=None,
    noise=None,
    trajectory_alpha=0.08,
    trajectory_width=0.45,
):
    device = torch.device("cpu")
    loaded = [load_run(run_dir, device) for run_dir in run_dirs]
    first_config, first_problem, _ = loaded[0]
    if not is_mode_problem(first_problem):
        raise ValueError("plot_five_modes_comparison currently only supports mode-mixture runs")
    x0, target = eval_inputs(first_problem, first_config, device, eval_seed)

    fig, axes = plt.subplots(1, len(loaded), figsize=(4.8 * len(loaded), 4.9), squeeze=False)
    axis_bounds = []
    for ax, run_dir, loaded_item in zip(axes[0], run_dirs, loaded):
        config, problem, model = loaded_item
        if problem.name != first_problem.name:
            raise ValueError(f"comparison mixes {first_problem.name} and {problem.name} runs")
        if not is_mode_problem(problem):
            raise ValueError(f"comparison only supports mode-mixture runs, got {problem.name}")
        traj = spiral_trajectory(model, x0, config, eval_seed, steps, noise)
        metrics_path = Path(run_dir) / "metrics.json"
        metrics = read_json(metrics_path) if metrics_path.exists() else None
        _draw_five_mode_panel(
            ax,
            target,
            traj,
            problem.centers(device),
            problem.sigma_mode,
            Path(run_dir).name,
            metrics=metrics,
            n_traj=n_traj,
            n_final=n_final,
            trajectory_alpha=trajectory_alpha,
            trajectory_width=trajectory_width,
        )
        axis_bounds.append((*ax.get_xlim(), *ax.get_ylim()))
    x_min = min(bounds[0] for bounds in axis_bounds)
    x_max = max(bounds[1] for bounds in axis_bounds)
    y_min = min(bounds[2] for bounds in axis_bounds)
    y_max = max(bounds[3] for bounds in axis_bounds)
    for ax in axes[0]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.6, w_pad=0.7)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return output
