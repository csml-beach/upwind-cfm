#!/usr/bin/env python3
"""Evaluate sampler time grids on a trained Burgers solution-map CFM."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lcfm.datasets import BurgersSolutionMapProblem
from lcfm.metrics import mean_path_length, path_length_ratio, rmse, trajectory_acceleration
from lcfm.models import build_model
from lcfm.schedules import equal_error_grid, euler_on_grid, heun_on_grid, kappa, power_time_grid, rollout_error_profile
from lcfm.utils import read_json, set_seed, write_json


def parse_csv_numbers(text, cast=float):
    return [cast(part.strip()) for part in text.replace(" ", ",").split(",") if part.strip()]


def rho_tag(value):
    return str(float(value)).replace(".", "p")


def periodic_spatial_gradient(x):
    return torch.roll(x, shifts=-1, dims=-1) - x


@torch.no_grad()
def solve_schedule(model, x0, schedule, nfe, args):
    if schedule == "uniform":
        grid = [float(value) for value in torch.linspace(0.0, 1.0, nfe + 1).tolist()]
        return euler_on_grid(model, x0, grid), {}
    if schedule == "heun_uniform":
        grid = [float(value) for value in torch.linspace(0.0, 1.0, nfe + 1).tolist()]
        return heun_on_grid(model, x0, grid), {}
    if schedule.startswith("e1_warped_p"):
        power = float(schedule.split("_p", 1)[1].replace("p", "."))
        ts, err = rollout_error_profile(model, x0, fine_steps=args.profile_fine_steps)
        grid = equal_error_grid(ts, err, nfe, power=power, floor=args.warp_floor, end=1.0)
        return euler_on_grid(model, x0, grid), {
            "kappa": kappa(ts, err, floor=args.warp_floor),
            "peak_over_mean_err": float((err.max() / (err.mean() + 1e-12)).item()),
        }
    if schedule.startswith("power_"):
        _, kind, rho_text = schedule.split("_", 2)
        rho = float(rho_text.replace("rho", "").replace("p", "."))
        grid = power_time_grid(nfe, rho=rho, kind=kind)
        return euler_on_grid(model, x0, grid), {}
    raise ValueError(f"Unknown schedule: {schedule}")


def evaluate_samples(traj, target, reference_endpoint=None):
    samples = traj[-1]
    metrics = {
        "rmse": rmse(samples, target),
        "spatial_grad_rmse": rmse(periodic_spatial_gradient(samples), periodic_spatial_gradient(target)),
        "mean_path_length": mean_path_length(traj),
        "path_length_ratio": path_length_ratio(traj),
        "trajectory_acceleration": trajectory_acceleration(traj),
    }
    if reference_endpoint is not None:
        metrics["integration_error"] = float((samples - reference_endpoint).norm(dim=1).mean().item())
    return metrics


def plot_summary(rows, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"plot unavailable: {exc}"

    schedules = sorted({row["schedule"] for row in rows})
    metrics = [("rmse", "Endpoint RMSE"), ("spatial_grad_rmse", "Gradient RMSE"), ("trajectory_acceleration", "Trajectory Accel.")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, (metric, title) in zip(axes, metrics):
        for schedule in schedules:
            items = sorted([row for row in rows if row["schedule"] == schedule], key=lambda row: int(row["nfe"]))
            ax.plot([int(row["nfe"]) for row in items], [float(row[metric]) for row in items], marker="o", label=schedule)
        ax.set_xscale("log")
        ax.set_xlabel("NFE")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("lower is better")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "burgers_solution_map_schedule_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_examples(model, x0, target, rows, out_dir, args):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"example plot unavailable: {exc}"

    nfe = int(args.plot_nfe)
    schedules = ["uniform", "e1_warped_p0p25", "e1_warped_p0p5"]
    x_axis = torch.linspace(0, 2 * torch.pi, x0.shape[-1] + 1)[:-1].cpu()
    fig, axes = plt.subplots(min(4, x0.shape[0]), 1, figsize=(9.5, 8.0), sharex=True)
    if not isinstance(axes, (list, tuple)):
        axes = axes.reshape(-1)
    for sample_idx, ax in enumerate(axes):
        ax.plot(x_axis, x0[sample_idx].detach().cpu(), color="#777777", lw=1.2, label="initial" if sample_idx == 0 else None)
        ax.plot(x_axis, target[sample_idx].detach().cpu(), color="#111111", lw=1.8, label="target" if sample_idx == 0 else None)
        for schedule in schedules:
            traj, _ = solve_schedule(model, x0, schedule, nfe, args)
            label = schedule if sample_idx == 0 else None
            ax.plot(x_axis, traj[-1, sample_idx].detach().cpu(), lw=1.2, alpha=0.85, label=label)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(f"sample {sample_idx}")
    axes[0].legend(fontsize=8, ncol=5)
    axes[-1].set_xlabel("x")
    fig.suptitle(f"Burgers solution-map samples at NFE {nfe}", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"burgers_solution_map_examples_nfe{nfe}.png"
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
    parser.add_argument("--nfe-values", default="5,10,20,50")
    parser.add_argument("--warp-powers", default="0.25,0.5")
    parser.add_argument("--power-rhos", default="2,3")
    parser.add_argument("--power-kinds", default="early,late,symmetric")
    parser.add_argument("--profile-fine-steps", type=int, default=50)
    parser.add_argument("--warp-floor", type=float, default=1e-3)
    parser.add_argument("--include-heun", action="store_true")
    parser.add_argument("--reference-intervals", type=int, default=0)
    parser.add_argument("--reference-check-intervals", type=int, default=0)
    parser.add_argument("--plot-nfe", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval_sctw_schedules"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(run_dir / "config.json")
    set_seed(int(config.get("seed", 42)))
    problem = BurgersSolutionMapProblem(config.get("dataset_kwargs", {}))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = build_model(config.get("model", "unet1d"), problem.dim, config).to(device)
    model.load_state_dict(torch.load(run_dir / args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    n_eval = int(args.n_eval or config.get("eval", {}).get("n_eval", min(64, problem.n_test)))
    x0 = problem.eval_initial(n_eval, device)
    target = problem.target_eval(n_eval, device)
    reference_endpoint = None
    reference_self_error = None
    if args.reference_intervals > 0:
        ref_grid = [float(value) for value in torch.linspace(0.0, 1.0, args.reference_intervals + 1).tolist()]
        reference_endpoint = heun_on_grid(model, x0, ref_grid)[-1]
        if args.reference_check_intervals > 0:
            if args.reference_check_intervals <= args.reference_intervals:
                raise ValueError("--reference-check-intervals must exceed --reference-intervals.")
            check_grid = [
                float(value) for value in torch.linspace(0.0, 1.0, args.reference_check_intervals + 1).tolist()
            ]
            reference_check = heun_on_grid(model, x0, check_grid)[-1]
            reference_self_error = float((reference_endpoint - reference_check).norm(dim=1).mean().item())
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
    for nfe in nfe_values:
        for schedule in schedules:
            traj, diagnostics = solve_schedule(model, x0, schedule, int(nfe), args)
            row = {
                "schedule": schedule,
                "nfe": int(nfe),
                **evaluate_samples(traj, target, reference_endpoint=reference_endpoint),
                **diagnostics,
            }
            if reference_self_error is not None:
                row["reference_self_error"] = reference_self_error
            rows.append(row)
            print(
                f"{schedule:>24} nfe={int(nfe):>3} rmse={row['rmse']:.6f} "
                f"grad={row['spatial_grad_rmse']:.6f} accel={row['trajectory_acceleration']:.6f}",
                flush=True,
            )

    fieldnames = sorted({key for row in rows for key in row})
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
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
            "reference_intervals": args.reference_intervals,
            "reference_check_intervals": args.reference_check_intervals,
            "reference_self_error": reference_self_error,
        },
    )
    print(f"Summary CSV: {summary_path}")
    print(f"Summary plot: {plot_summary(rows, out_dir)}")
    print(f"Example plot: {plot_examples(model, x0[:4], target[:4], rows, out_dir, args)}")


if __name__ == "__main__":
    main()
