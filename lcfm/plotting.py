from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from . import datasets  # noqa: F401
from . import losses  # noqa: F401
from . import models  # noqa: F401
from . import solvers  # noqa: F401
from .models import build_model
from .registry import DATASETS, get
from .solvers import solve
from .utils import read_json, set_seed


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


@torch.no_grad()
def spiral_trajectory(model, x0, config, eval_seed=None):
    if eval_seed is None:
        eval_seed = config.get("eval", {}).get("plot_seed", 1234)
    set_seed(eval_seed)
    return solve(config.get("solver", "euler"), model, x0, config.get("solver_kwargs", {"steps": 15}))


def plot_spiral_run(run_dir, output=None, eval_seed=None, n_traj=24):
    device = torch.device("cpu")
    config, problem, model = load_run(run_dir, device)
    if problem.name != "spiral":
        raise ValueError(f"plot_spiral_run only supports spiral runs, got {problem.name}")
    x0, target = spiral_eval_inputs(problem, config, device, eval_seed)
    traj = spiral_trajectory(model, x0, config, eval_seed)

    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / "plot.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target[:, 0].cpu(), target[:, 1].cpu(), s=8, c="0.75", alpha=0.55, label="target")
    ax.scatter(traj[-1, :, 0].cpu(), traj[-1, :, 1].cpu(), s=10, c="#2563eb", alpha=0.75, label="generated")
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


def plot_spiral_comparison(run_dirs, output, eval_seed=1234, n_traj=16):
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
        traj = spiral_trajectory(model, x0, config, eval_seed)
        ax.scatter(target[:, 0].cpu(), target[:, 1].cpu(), s=8, c="0.75", alpha=0.55)
        ax.scatter(traj[-1, :, 0].cpu(), traj[-1, :, 1].cpu(), s=10, c=color, alpha=0.75)
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
