import time
from pathlib import Path

import torch

from . import datasets  # noqa: F401
from . import losses  # noqa: F401
from . import models  # noqa: F401
from . import solvers  # noqa: F401
from .losses import build_method
from .metrics import (
    mode_statistics,
    path_length_ratio,
    rmse,
    temporal_tv,
    trajectory_acceleration,
    wasserstein_match,
)
from .models import build_model
from .registry import DATASETS, get
from .solvers import solve
from .utils import device_from_config, environment_info, repo_root, set_seed, write_json


def train_model(problem, config, device):
    method = build_method(config["method"], config.get("method_kwargs", {}))
    model = build_model(config.get("model", "mlp"), problem.dim, config).to(device)
    train_cfg = config.get("train", {})
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3))
    epochs = train_cfg.get("epochs", 1000)
    batch_size = train_cfg.get("batch_size", 256)
    log_every = train_cfg.get("log_every", 100)
    history = []

    model.train()
    for epoch in range(epochs):
        x0, x1 = problem.sample_train_batch(batch_size, device)
        terms = method.loss(model, x0, x1)
        loss = sum(terms.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % log_every == 0 or epoch == epochs - 1:
            row = {"epoch": epoch, "loss": float(loss.detach().cpu())}
            row.update({k: float(v.detach().cpu()) for k, v in terms.items()})
            history.append(row)
            print(row)
    return model, history


@torch.no_grad()
def eval_spiral(problem, model, config, device):
    eval_cfg = config.get("eval", {})
    n_eval = eval_cfg.get("n_eval", 1000)
    eval_seed = eval_cfg.get("eval_seed", 1234)
    set_seed(eval_seed)
    x0 = problem.eval_initial(n_eval, device)
    target = problem.target_eval(n_eval, device)
    traj = solve(config.get("solver", "euler"), model, x0, config.get("solver_kwargs", {"steps": 15}))
    return {
        "wasserstein": wasserstein_match(traj[-1], target),
        "path_length_ratio": path_length_ratio(traj),
        "trajectory_acceleration": trajectory_acceleration(traj),
    }


@torch.no_grad()
def eval_five_modes(problem, model, config, device):
    eval_cfg = config.get("eval", {})
    n_eval = eval_cfg.get("n_eval", 1000)
    eval_seed = eval_cfg.get("eval_seed", 1234)
    set_seed(eval_seed)
    x0 = problem.eval_initial(n_eval, device)
    target = problem.target_eval(n_eval, device)
    traj = solve(config.get("solver", "euler"), model, x0, config.get("solver_kwargs", {"steps": 5}))
    metrics = {
        "wasserstein": wasserstein_match(traj[-1], target),
        "path_length_ratio": path_length_ratio(traj),
        "trajectory_acceleration": trajectory_acceleration(traj),
    }
    metrics.update(
        mode_statistics(
            traj[-1],
            problem.centers(device),
            p_min=eval_cfg.get("mode_p_min", 0.05),
            hit_radius=eval_cfg.get("hit_radius", 3.0 * problem.sigma_mode),
        )
    )
    return metrics


@torch.no_grad()
def eval_burgers_autoregressive(problem, model, config, device):
    eval_cfg = config.get("eval", {})
    n_eval = eval_cfg.get("n_eval", min(32, problem.n_test))
    eval_seed = eval_cfg.get("eval_seed", 1234)
    set_seed(eval_seed)
    solver_cfg = dict(config.get("solver_kwargs", {"steps": 5}))
    x = problem.eval_initial(n_eval, device)
    frames = [x.detach().cpu()]
    for _ in range(problem.nt - 1):
        traj = solve(config.get("solver", "euler"), model, x, solver_cfg)
        x = traj[-1]
        frames.append(x.detach().cpu())
    video = torch.stack(frames, dim=1)
    target = problem.target_eval(n_eval, torch.device("cpu"))
    return {
        "rmse": rmse(video, target),
        "temporal_tv": temporal_tv(video),
    }


def evaluate(problem, model, config, device):
    if problem.name == "spiral":
        return eval_spiral(problem, model, config, device)
    if problem.name == "five_modes":
        return eval_five_modes(problem, model, config, device)
    if problem.name == "burgers_autoregressive":
        return eval_burgers_autoregressive(problem, model, config, device)
    raise ValueError(f"No evaluator for problem: {problem.name}")


def run(config):
    seed = config.get("seed", 42)
    set_seed(seed)
    device = device_from_config(config)
    dataset_cls = get(DATASETS, config["dataset"])
    problem = dataset_cls(config.get("dataset_kwargs", {}))
    model, history = train_model(problem, config, device)
    metrics = evaluate(problem, model, config, device)

    run_name = config.get("run_name") or f"{config['dataset']}_{config['method']}_{int(time.time())}"
    out_dir = Path(config.get("out_dir", repo_root() / "runs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "history.json", history)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "environment.json", environment_info())
    print(f"Saved run to {out_dir}")
    print(metrics)
    return metrics
