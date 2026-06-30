import time
import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from . import datasets  # noqa: F401
from . import losses  # noqa: F401
from . import models  # noqa: F401
from . import solvers  # noqa: F401
from .losses import build_method
from .metrics import (
    mean_endpoint_displacement,
    mean_path_length,
    mode_statistics,
    path_length_ratio,
    rmse,
    temporal_tv,
    trajectory_acceleration,
    wasserstein_match,
)
from .models import build_model
from .pairing import apply_pairing
from .registry import DATASETS, get
from .solvers import solve
from .utils import device_from_config, environment_info, repo_root, set_seed, write_json


def _split_batch(batch):
    if len(batch) == 2:
        return batch[0], batch[1], None
    if len(batch) == 3:
        return batch
    raise ValueError("Expected training batch to be (x0, x1) or (x0, x1, labels).")


def _condition_model(model, labels):
    if labels is None:
        return model
    return lambda x, t: model(x, t, labels)


def _image_batch(x, image_shape):
    return x.reshape(x.shape[0], *image_shape).clamp(-1.0, 1.0)


def save_image_grid(samples, path, image_shape, nrow=8):
    cache_dir = Path(tempfile.gettempdir()) / "lcfm_image_grid_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = (_image_batch(samples.detach().cpu(), image_shape) + 1.0) * 0.5
    n_images = images.shape[0]
    nrow = max(1, min(nrow, n_images))
    ncol = (n_images + nrow - 1) // nrow
    channels, height, width = image_shape
    grid = torch.ones(channels, ncol * height, nrow * width)
    for index, image in enumerate(images):
        row = index // nrow
        col = index % nrow
        grid[:, row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    array = grid.permute(1, 2, 0).numpy()
    if channels == 1:
        array = array[:, :, 0]
    plt.imsave(path, array)
    return str(path)


def _image_stats(samples, prefix):
    samples = samples.detach()
    return {
        f"{prefix}_mean": float(samples.mean().cpu()),
        f"{prefix}_std": float(samples.std(unbiased=False).cpu()),
        f"{prefix}_min": float(samples.min().cpu()),
        f"{prefix}_max": float(samples.max().cpu()),
    }


def _to_uint8_images(samples, image_shape):
    images = _image_batch(samples, image_shape)
    return ((images + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)


def _fid_kid_metrics(samples, target, image_shape, device, prefix, eval_cfg):
    if not eval_cfg.get("compute_fid_kid", False):
        return {}
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception as exc:
        return {f"{prefix}_fid_kid_error": f"torchmetrics unavailable: {exc}"}

    try:
        real = _to_uint8_images(target, image_shape).to(device)
        fake = _to_uint8_images(samples, image_shape).to(device)
        fid = FrechetInceptionDistance(feature=eval_cfg.get("fid_feature", 2048), normalize=False).to(device)
        fid.update(real, real=True)
        fid.update(fake, real=False)
        subset_size = min(int(eval_cfg.get("kid_subset_size", 50)), real.shape[0], fake.shape[0])
        kid = KernelInceptionDistance(subset_size=subset_size, normalize=False).to(device)
        kid.update(real, real=True)
        kid.update(fake, real=False)
        kid_mean, kid_std = kid.compute()
        return {
            f"{prefix}_fid": float(fid.compute().detach().cpu()),
            f"{prefix}_kid_mean": float(kid_mean.detach().cpu()),
            f"{prefix}_kid_std": float(kid_std.detach().cpu()),
        }
    except Exception as exc:
        return {f"{prefix}_fid_kid_error": str(exc)}


@torch.no_grad()
def _save_training_sample_grid(problem, model, config, device, out_dir, step):
    if not hasattr(problem, "image_shape"):
        return
    eval_cfg = config.get("eval", {})
    n_samples = int(eval_cfg.get("sample_grid_n", 16))
    nrow = int(eval_cfg.get("sample_grid_nrow", 4))
    steps = int(eval_cfg.get("sample_grid_steps", 5))
    set_seed(int(eval_cfg.get("plot_seed", 1234)) + int(step))
    x0 = problem.eval_initial(n_samples, device)
    labels = problem.eval_labels(n_samples, device) if getattr(problem, "class_conditional", False) else None
    traj = solve(config.get("solver", "euler"), _condition_model(model, labels), x0, {"steps": steps})
    save_image_grid(traj[-1], out_dir / "samples" / f"step_{step:08d}.png", problem.image_shape, nrow=nrow)


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for name, shadow_param in self.shadow.items():
            shadow_param.mul_(self.decay).add_(state[name].detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        self.shadow = {name: value.detach().clone() for name, value in state["shadow"].items()}

    def model_state_dict(self, model):
        state = model.state_dict()
        return {name: self.shadow.get(name, value).detach().clone() for name, value in state.items()}


def _save_checkpoint(path, model, optimizer, scaler, step, history, ema=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "history": history,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
        payload["ema_model"] = ema.model_state_dict(model)
    torch.save(payload, path)


def _append_history_row(history, step, loss, terms):
    row = {"step": step, "epoch": step, "loss": float(loss.detach().cpu())}
    row.update({k: float(v.detach().cpu()) for k, v in terms.items()})
    history.append(row)
    print(row, flush=True)
    return row


def train_model(problem, config, device, out_dir=None):
    method = build_method(config["method"], config.get("method_kwargs", {}))
    model = build_model(config.get("model", "mlp"), problem.dim, config).to(device)
    train_cfg = config.get("train", {})
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3))
    total_steps = int(train_cfg.get("max_steps", train_cfg.get("epochs", 1000)))
    batch_size = train_cfg.get("batch_size", 256)
    log_every = train_cfg.get("log_every", 100)
    save_checkpoint = bool(train_cfg.get("save_checkpoint", True))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 0))
    keep_checkpoints = bool(train_cfg.get("keep_checkpoints", False))
    stop_on_nonfinite = bool(train_cfg.get("stop_on_nonfinite", True))
    save_failed_checkpoint = bool(train_cfg.get("save_failed_checkpoint", False))
    sample_every = int(train_cfg.get("sample_every", 0))
    grad_clip = train_cfg.get("grad_clip")
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None
    ema = ModelEMA(model, train_cfg.get("ema_decay", 0.9999)) if train_cfg.get("ema", False) else None
    checkpoint_path = Path(out_dir) / "checkpoint_latest.pt" if out_dir and save_checkpoint else None
    start_step = 0
    history = []
    if train_cfg.get("resume", False) and checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        start_step = int(checkpoint.get("step", 0))
        history = checkpoint.get("history", [])
        print(f"Resumed {checkpoint_path} from step {start_step}", flush=True)

    model.train()
    final_step = start_step
    for step in range(start_step, total_steps):
        x0, x1, labels = _split_batch(problem.sample_train_batch(batch_size, device))
        if labels is None:
            x0, x1 = apply_pairing(x0, x1, config)
        else:
            x0, x1, labels = apply_pairing(x0, x1, config, labels)
        loss_model = _condition_model(model, labels)
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                terms = method.loss(loss_model, x0, x1)
                loss = sum(terms.values())
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            terms = method.loss(loss_model, x0, x1)
            loss = sum(terms.values())
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
        if ema is not None:
            ema.update(model)

        current_step = step + 1
        final_step = current_step
        if current_step % log_every == 0 or current_step == 1 or current_step == total_steps:
            _append_history_row(history, current_step, loss, terms)
        if stop_on_nonfinite and not torch.isfinite(loss.detach()):
            if current_step % log_every != 0 and current_step != 1 and current_step != total_steps:
                _append_history_row(history, current_step, loss, terms)
            if checkpoint_path and save_failed_checkpoint:
                failed_path = checkpoint_path.with_name(f"checkpoint_failed_step_{current_step:08d}.pt")
                _save_checkpoint(failed_path, model, optimizer, scaler, current_step, history, ema=ema)
            raise FloatingPointError(f"Non-finite training loss at step {current_step}.")
        if checkpoint_path and checkpoint_every > 0 and current_step % checkpoint_every == 0:
            _save_checkpoint(checkpoint_path, model, optimizer, scaler, current_step, history, ema=ema)
            if keep_checkpoints:
                step_path = checkpoint_path.with_name(f"checkpoint_step_{current_step:08d}.pt")
                _save_checkpoint(step_path, model, optimizer, scaler, current_step, history, ema=ema)
        if out_dir and sample_every > 0 and current_step % sample_every == 0:
            model.eval()
            _save_training_sample_grid(problem, model, config, device, Path(out_dir), current_step)
            model.train()
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, model, optimizer, scaler, final_step, history, ema=ema)
    return model, history, ema


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
        "wasserstein2": wasserstein_match(traj[-1], target, p=2),
        "mean_path_length": mean_path_length(traj),
        "mean_endpoint_displacement": mean_endpoint_displacement(traj),
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
        "wasserstein2": wasserstein_match(traj[-1], target, p=2),
        "mean_path_length": mean_path_length(traj),
        "mean_endpoint_displacement": mean_endpoint_displacement(traj),
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


def _periodic_spatial_gradient(x):
    return torch.roll(x, shifts=-1, dims=-1) - x


@torch.no_grad()
def eval_burgers_solution_map(problem, model, config, device):
    eval_cfg = config.get("eval", {})
    n_eval = int(eval_cfg.get("n_eval", min(64, problem.n_test)))
    nfe_values = eval_cfg.get("nfe_values", [5, 10, 20, 50])
    eval_seed = int(eval_cfg.get("eval_seed", 1234))
    set_seed(eval_seed)
    x0 = problem.eval_initial(n_eval, device)
    target = problem.target_eval(n_eval, device)
    target_grad = _periodic_spatial_gradient(target)
    metrics = {"n_eval": n_eval}
    for steps in nfe_values:
        steps = int(steps)
        traj = solve(config.get("solver", "euler"), model, x0, {"steps": steps})
        samples = traj[-1]
        sample_grad = _periodic_spatial_gradient(samples)
        prefix = f"nfe_{steps}"
        metrics[f"{prefix}_rmse"] = rmse(samples, target)
        metrics[f"{prefix}_spatial_grad_rmse"] = rmse(sample_grad, target_grad)
        metrics[f"{prefix}_mean_path_length"] = mean_path_length(traj)
        metrics[f"{prefix}_path_length_ratio"] = path_length_ratio(traj)
        metrics[f"{prefix}_trajectory_acceleration"] = trajectory_acceleration(traj)
    return metrics


@torch.no_grad()
def eval_checkerboard_refinement(problem, model, config, device):
    eval_cfg = config.get("eval", {})
    n_eval = int(eval_cfg.get("n_eval", 1000))
    nfe_values = eval_cfg.get("nfe_values", [5, 10, 20, 50])
    eval_seed = int(eval_cfg.get("eval_seed", 1234))
    set_seed(eval_seed)
    x0 = problem.eval_initial(n_eval, device)
    target = problem.target_eval(n_eval, device)
    metrics = {"n_eval": n_eval}
    for steps in nfe_values:
        steps = int(steps)
        traj = solve(config.get("solver", "euler"), model, x0, {"steps": steps})
        prefix = f"nfe_{steps}"
        metrics[f"{prefix}_wasserstein"] = wasserstein_match(traj[-1], target)
        metrics[f"{prefix}_wasserstein2"] = wasserstein_match(traj[-1], target, p=2)
        metrics[f"{prefix}_mean_path_length"] = mean_path_length(traj)
        metrics[f"{prefix}_path_length_ratio"] = path_length_ratio(traj)
        metrics[f"{prefix}_trajectory_acceleration"] = trajectory_acceleration(traj)
    return metrics


@torch.no_grad()
def eval_cifar10(problem, model, config, device, out_dir=None):
    eval_cfg = config.get("eval", {})
    n_eval = int(eval_cfg.get("n_eval", 64))
    n_metric = int(eval_cfg.get("fid_n_eval", n_eval))
    nfe_values = eval_cfg.get("nfe_values", [5, 10, 20, 50])
    eval_seed = int(eval_cfg.get("eval_seed", 1234))
    nrow = int(eval_cfg.get("sample_grid_nrow", 8))
    solver_name = eval_cfg.get("solver", config.get("solver", "euler"))
    labels = problem.eval_labels(n_eval, device) if getattr(problem, "class_conditional", False) else None
    metric_labels = problem.eval_labels(n_metric, device) if getattr(problem, "class_conditional", False) else None
    target = problem.target_eval(n_eval, device, labels=labels) if labels is not None else problem.target_eval(n_eval, device)
    real_reference = (
        problem.metric_reference(n_metric, device, labels=metric_labels)
        if eval_cfg.get("compute_fid_kid", False) and metric_labels is not None
        else problem.metric_reference(n_metric, device)
        if eval_cfg.get("compute_fid_kid", False)
        else target
    )
    set_seed(eval_seed)
    x0 = problem.eval_initial(n_eval, device)
    eval_model = _condition_model(model, labels)

    metrics = {"n_eval": n_eval}
    for steps in nfe_values:
        steps = int(steps)
        traj = solve(solver_name, eval_model, x0, {"steps": steps})
        samples = traj[-1].clamp(-3.0, 3.0)
        prefix = f"nfe_{steps}"
        metrics.update(_image_stats(samples, prefix))
        metrics[f"{prefix}_pixel_mse_to_eval_target"] = float(F.mse_loss(samples.clamp(-1.0, 1.0), target).detach().cpu())
        metrics.update(_fid_kid_metrics(samples, real_reference, problem.image_shape, device, prefix, eval_cfg))
        if out_dir:
            save_image_grid(samples, Path(out_dir) / "samples" / f"{prefix}.png", problem.image_shape, nrow=nrow)
    return metrics


def evaluate(problem, model, config, device, out_dir=None):
    if problem.name == "spiral":
        return eval_spiral(problem, model, config, device)
    if hasattr(problem, "centers") and hasattr(problem, "sigma_mode"):
        return eval_five_modes(problem, model, config, device)
    if problem.name == "burgers_autoregressive":
        return eval_burgers_autoregressive(problem, model, config, device)
    if problem.name == "burgers_solution_map":
        return eval_burgers_solution_map(problem, model, config, device)
    if problem.name == "checkerboard_refinement":
        return eval_checkerboard_refinement(problem, model, config, device)
    if hasattr(problem, "image_shape"):
        return eval_cifar10(problem, model, config, device, out_dir=out_dir)
    raise ValueError(f"No evaluator for problem: {problem.name}")


def run(config):
    seed = config.get("seed", 42)
    set_seed(seed)
    device = device_from_config(config)
    dataset_cls = get(DATASETS, config["dataset"])
    problem = dataset_cls(config.get("dataset_kwargs", {}))
    run_name = config.get("run_name") or f"{config['dataset']}_{config['method']}_{int(time.time())}"
    out_dir = Path(config.get("out_dir", repo_root() / "runs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "environment.json", environment_info())
    model, history, ema = train_model(problem, config, device, out_dir=out_dir)
    model.eval()
    metrics = evaluate(problem, model, config, device, out_dir=out_dir)

    torch.save(model.state_dict(), out_dir / "model.pt")
    if ema is not None:
        torch.save(ema.model_state_dict(model), out_dir / "model_ema.pt")
    write_json(out_dir / "history.json", history)
    write_json(out_dir / "metrics.json", metrics)
    print(f"Saved run to {out_dir}")
    print(metrics)
    return metrics
