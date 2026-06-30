#!/usr/bin/env python3
"""Evaluate sampler time grids on a trained Burgers autoregressive CFM.

This is sampler-only: it loads an existing frame-to-frame velocity model and
rolls it out autoregressively with uniform Euler, SCTW/E1-warped Euler, and
hand power grids. No training is performed.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lcfm.datasets import BurgersAutoregressiveProblem
from lcfm.metrics import rmse, temporal_tv
from lcfm.models import build_model
from lcfm.schedules import (
    equal_error_grid,
    euler_on_grid,
    heun_on_grid,
    kappa,
    power_time_grid,
    rollout_error_profile,
)
from lcfm.utils import read_json, set_seed, write_json


def parse_csv_numbers(text, cast=float):
    return [cast(part.strip()) for part in text.replace(" ", ",").split(",") if part.strip()]


def rho_tag(value):
    return str(float(value)).replace(".", "p")


def spatial_gradients(video):
    return torch.roll(video, shifts=-1, dims=-1) - video


def metrics_for_video(video, target):
    frame_rmse = torch.sqrt(torch.mean((video - target) ** 2, dim=(0, 2)))
    final_frame_rmse = torch.sqrt(torch.mean((video[:, -1, :] - target[:, -1, :]) ** 2))
    grad_rmse = torch.sqrt(torch.mean((spatial_gradients(video) - spatial_gradients(target)) ** 2))
    return {
        "rmse": rmse(video, target),
        "final_rmse": float(final_frame_rmse.item()),
        "max_frame_rmse": float(frame_rmse.max().item()),
        "mean_frame_rmse": float(frame_rmse.mean().item()),
        "temporal_tv": temporal_tv(video),
        "target_temporal_tv": temporal_tv(target),
        "spatial_grad_rmse": float(grad_rmse.item()),
    }


@torch.no_grad()
def rollout_autoregressive(model, x0, nt, schedule_kind, nfe, args, cache=None):
    x = x0.clone()
    frames = [x.detach().cpu()]
    kappas = []
    peak_mean = []
    grids = []

    for _ in range(nt - 1):
        grid = None
        if schedule_kind == "uniform":
            grid = [float(v) for v in torch.linspace(0.0, 1.0, nfe + 1).tolist()]
            traj = euler_on_grid(model, x, grid)
        elif schedule_kind == "heun_uniform":
            grid = [float(v) for v in torch.linspace(0.0, 1.0, nfe + 1).tolist()]
            traj = heun_on_grid(model, x, grid)
        elif schedule_kind.startswith("e1_warped_p"):
            power = float(schedule_kind.split("_p", 1)[1].replace("p", "."))
            ts, err = rollout_error_profile(model, x, fine_steps=args.profile_fine_steps)
            grid = equal_error_grid(ts, err, nfe, power=power, floor=args.warp_floor, end=1.0)
            kappas.append(kappa(ts, err, floor=args.warp_floor))
            peak_mean.append(float((err.max() / (err.mean() + 1e-12)).item()))
            traj = euler_on_grid(model, x, grid)
        elif schedule_kind.startswith("power_"):
            _, kind, rho_text = schedule_kind.split("_", 2)
            grid = power_time_grid(nfe, rho=float(rho_text.replace("rho", "").replace("p", ".")), kind=kind)
            traj = euler_on_grid(model, x, grid)
        else:
            raise ValueError(f"Unknown schedule kind: {schedule_kind}")

        if cache is not None:
            grids.append(grid)
        x = traj[-1]
        frames.append(x.detach().cpu())

    video = torch.stack(frames, dim=1)
    diagnostics = {}
    if kappas:
        diagnostics["mean_kappa"] = float(sum(kappas) / len(kappas))
        diagnostics["max_kappa"] = float(max(kappas))
        diagnostics["mean_peak_over_mean_err"] = float(sum(peak_mean) / len(peak_mean))
    if cache is not None:
        cache["grids"] = grids
    return video, diagnostics


def plot_results(rows, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"plot unavailable: {exc}"

    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = sorted({row["schedule"] for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    metrics = [("rmse", "Rollout RMSE"), ("final_rmse", "Final-frame RMSE"), ("spatial_grad_rmse", "Gradient RMSE")]
    for ax, (metric, title) in zip(axes, metrics):
        for schedule in schedules:
            items = sorted([row for row in rows if row["schedule"] == schedule], key=lambda row: int(row["nfe"]))
            ax.plot([int(row["nfe"]) for row in items], [float(row[metric]) for row in items], marker="o", label=schedule)
        ax.set_xscale("log")
        ax.set_xlabel("NFE per frame transition")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("lower is better")
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = out_dir / "burgers_schedule_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_videos(videos, target, out_dir, nfe, max_rows=4):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"video plot unavailable: {exc}"

    selected = {name: video for name, video in videos.items() if f"nfe_{nfe}" in name}
    if not selected:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["target"] + sorted(selected)
    n_rows = min(max_rows, target.shape[0])
    fig, axes = plt.subplots(len(names), n_rows, figsize=(3.0 * n_rows, 2.15 * len(names)), squeeze=False)
    vmin = float(target[:n_rows].min().item())
    vmax = float(target[:n_rows].max().item())
    for row_idx, name in enumerate(names):
        video = target if name == "target" else selected[name]
        for col in range(n_rows):
            ax = axes[row_idx, col]
            ax.imshow(video[col].detach().cpu().numpy(), aspect="auto", origin="lower", cmap="coolwarm", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, fontsize=8)
            if row_idx == 0:
                ax.set_title(f"sample {col}", fontsize=9)
    fig.tight_layout()
    path = out_dir / f"burgers_videos_nfe{nfe}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--nfe-values", default="5,10,20,50,100")
    parser.add_argument("--warp-powers", default="0.25,0.5")
    parser.add_argument("--power-rhos", default="2,3")
    parser.add_argument("--power-kinds", default="early,late,symmetric")
    parser.add_argument("--profile-fine-steps", type=int, default=50)
    parser.add_argument("--warp-floor", type=float, default=1e-3)
    parser.add_argument("--include-heun", action="store_true")
    parser.add_argument("--plot-nfe", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval_sctw_schedules"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(run_dir / "config.json")
    dataset_kwargs = dict(config.get("dataset_kwargs", {}))
    set_seed(int(config.get("seed", 42)))
    problem = BurgersAutoregressiveProblem(dataset_kwargs)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = build_model(config.get("model", "unet1d"), problem.dim, config).to(device)
    state = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_eval = int(args.n_eval or config.get("eval", {}).get("n_eval", min(32, problem.n_test)))
    x0 = problem.eval_initial(n_eval, device)
    target = problem.target_eval(n_eval, torch.device("cpu"))
    nfe_values = parse_csv_numbers(args.nfe_values, int)
    warp_powers = parse_csv_numbers(args.warp_powers, float)
    power_rhos = parse_csv_numbers(args.power_rhos, float)
    power_kinds = [part.strip() for part in args.power_kinds.replace(" ", ",").split(",") if part.strip()]

    schedules = ["uniform"]
    schedules += [f"e1_warped_p{rho_tag(power)}" for power in warp_powers]
    for kind in power_kinds:
        for rho in power_rhos:
            schedules.append(f"power_{kind}_rho{rho_tag(rho)}")
    if args.include_heun:
        schedules.append("heun_uniform")

    rows = []
    videos_for_plot = {}
    for nfe in nfe_values:
        for schedule in schedules:
            video, diagnostics = rollout_autoregressive(model, x0, problem.nt, schedule, nfe, args)
            row = {"schedule": schedule, "nfe": int(nfe), **metrics_for_video(video, target), **diagnostics}
            rows.append(row)
            if int(nfe) == int(args.plot_nfe) and schedule in {"uniform", "e1_warped_p0p25", "e1_warped_p0p5", "power_early_rho2p0"}:
                videos_for_plot[f"{schedule}_nfe_{nfe}"] = video
            print(
                f"{schedule:>24} nfe={nfe:>3} rmse={row['rmse']:.6f} "
                f"final={row['final_rmse']:.6f} grad={row['spatial_grad_rmse']:.6f}"
            )

    csv_path = out_dir / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        out_dir / "eval_config.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "n_eval": n_eval,
            "nfe_values": nfe_values,
            "schedules": schedules,
            "profile_fine_steps": args.profile_fine_steps,
            "warp_floor": args.warp_floor,
        },
    )
    plot_path = plot_results(rows, out_dir)
    video_plot_path = plot_videos(videos_for_plot, target, out_dir, args.plot_nfe)
    print(f"Saved Burgers schedule evaluation to {out_dir}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary plot: {plot_path}")
    if video_plot_path:
        print(f"Video plot: {video_plot_path}")


if __name__ == "__main__":
    main()
