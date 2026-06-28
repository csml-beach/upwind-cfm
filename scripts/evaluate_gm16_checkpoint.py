#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm import datasets  # noqa: F401
from lcfm.metrics import mean_path_length, mode_statistics, path_length_ratio, trajectory_acceleration, wasserstein_match
from lcfm.models import build_model
from lcfm.registry import DATASETS, get
from lcfm.schedules import equal_error_grid, euler_on_grid, kappa, power_time_grid, rollout_error_profile
from lcfm.solvers import solve
from lcfm.utils import read_json, set_seed, write_json


def parse_ints(text):
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def parse_floats(text):
    return [float(part) for part in text.replace(",", " ").split() if part.strip()]


def parse_names(text):
    return [part.strip() for part in text.replace(",", " ").split() if part.strip()]


def rho_tag(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def uniform_grid(nfe):
    return [float(x) for x in torch.linspace(0.0, 1.0, int(nfe) + 1).tolist()]


def evaluate_samples(model, problem, x0, target, schedule, nfe, grid, hit_radius):
    if grid is None:
        traj = solve("euler", model, x0, {"steps": int(nfe)})
    else:
        traj = euler_on_grid(model, x0, grid)
    samples = traj[-1]
    metrics = {
        "schedule": schedule,
        "nfe": int(nfe),
        "wasserstein": wasserstein_match(samples, target),
        "wasserstein2": wasserstein_match(samples, target, p=2),
        "mean_path_length": mean_path_length(traj),
        "path_length_ratio": path_length_ratio(traj),
        "trajectory_acceleration": trajectory_acceleration(traj),
        "time_grid": grid,
    }
    metrics.update(mode_statistics(samples, problem.centers(samples.device), hit_radius=hit_radius))
    return {"samples": samples, "metrics": metrics}


def plot_summary(out_path, profile, rows, samples_by_name, x0, target, nfe_for_points):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    ts = torch.tensor(profile["ts"])
    err = torch.tensor(profile["err"])
    err_norm = err / err.max().clamp_min(1e-12)
    rho025 = (err + 1e-3).pow(0.25)
    rho05 = (err + 1e-3).pow(0.5)
    axes[0, 0].plot(ts, err_norm, color="#222222", lw=2.2, label="E1 profile")
    axes[0, 0].plot(ts, rho025 / rho025.max().clamp_min(1e-12), color="#2b7bba", ls="--", label="density p=0.25")
    axes[0, 0].plot(ts, rho05 / rho05.max().clamp_min(1e-12), color="#1b9e77", ls=":", label="density p=0.5")
    axes[0, 0].set_title(f"E1 profile, kappa={profile['kappa']:.2f}")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("normalized value")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)

    available_nfes = sorted({row["nfe"] for row in rows})
    if nfe_for_points not in available_nfes:
        nfe_for_points = available_nfes[0]
    subset = [row for row in rows if row["nfe"] == nfe_for_points]
    labels = [row["sample_name"] for row in subset]
    y_positions = list(range(len(labels)))[::-1]
    for y, row in zip(y_positions, subset):
        grid = row["time_grid"] or uniform_grid(nfe_for_points)
        axes[0, 1].hlines(y, 0, 1, color="#dddddd", lw=1)
        axes[0, 1].vlines(grid, y - 0.25, y + 0.25, lw=2)
    axes[0, 1].set_xlim(-0.01, 1.01)
    axes[0, 1].set_yticks(y_positions)
    axes[0, 1].set_yticklabels(labels, fontsize=7)
    axes[0, 1].set_title(f"NFE {nfe_for_points} time grids")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].grid(True, axis="x", alpha=0.25)

    metric_rows = sorted(subset, key=lambda row: row["wasserstein"])
    axes[0, 2].barh([row["sample_name"] for row in metric_rows], [row["wasserstein"] for row in metric_rows], color="#777777")
    axes[0, 2].set_title(f"NFE {nfe_for_points} endpoint W1")
    axes[0, 2].set_xlabel("W1")
    axes[0, 2].tick_params(axis="y", labelsize=7)

    axes[1, 0].scatter(x0[:, 0], x0[:, 1], s=5, alpha=0.35, c="#3b6ea8", linewidths=0)
    axes[1, 0].set_title("source projection")
    axes[1, 1].scatter(target[:, 0], target[:, 1], s=5, alpha=0.35, c="#d95f02", linewidths=0)
    axes[1, 1].set_title("target projection")
    best_name = metric_rows[0]["sample_name"]
    samples = samples_by_name[best_name]
    axes[1, 2].scatter(target[:, 0], target[:, 1], s=5, alpha=0.18, c="#d95f02", linewidths=0, label="target")
    axes[1, 2].scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.45, c="#2b7bba", linewidths=0, label=best_name)
    axes[1, 2].set_title(f"best NFE {nfe_for_points}: {best_name}")
    axes[1, 2].legend(frameon=False, fontsize=8)
    for ax in axes[1, :]:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("x0")
    axes[1, 0].set_ylabel("x1")
    fig.suptitle("GM16 schedule diagnostic, first two coordinates", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SCTW and hand time grids on a trained GM16 CFM.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--nfe-values", default="5,10,20,50")
    parser.add_argument("--profile-samples", type=int, default=1000)
    parser.add_argument("--profile-fine-steps", type=int, default=50)
    parser.add_argument("--warp-powers", default="0.25,0.5")
    parser.add_argument("--warp-floor", type=float, default=1e-3)
    parser.add_argument("--power-rhos", default="2,3")
    parser.add_argument("--power-kinds", default="early,late,symmetric")
    parser.add_argument("--hit-radius", type=float, default=1.6)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--plot-nfe", type=int, default=20)
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = read_json(run_dir / "config.json")
    if config["dataset"] != "gaussian_mixture_nd":
        raise SystemExit("This evaluator expects dataset='gaussian_mixture_nd'.")

    device = torch.device(args.device)
    set_seed(int(config.get("seed", 0)))
    problem = get(DATASETS, config["dataset"])(config.get("dataset_kwargs", {}))
    model = build_model(config.get("model", "mlp"), problem.dim, config).to(device)
    state = torch.load(run_dir / args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval_sctw_schedules"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "eval_config.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "n_samples": args.n_samples,
            "nfe_values": parse_ints(args.nfe_values),
            "profile_samples": args.profile_samples,
            "profile_fine_steps": args.profile_fine_steps,
            "warp_powers": parse_floats(args.warp_powers),
            "warp_floor": args.warp_floor,
            "power_rhos": parse_floats(args.power_rhos),
            "power_kinds": parse_names(args.power_kinds),
            "hit_radius": args.hit_radius,
            "eval_seed": args.eval_seed,
        },
    )

    set_seed(args.eval_seed)
    x0 = problem.eval_initial(args.n_samples, device)
    target = problem.target_eval(args.n_samples, device)
    profile_x0 = problem.eval_initial(args.profile_samples, device)
    ts, err = rollout_error_profile(model, profile_x0, fine_steps=args.profile_fine_steps)
    profile = {
        "ts": [float(x) for x in ts.tolist()],
        "err": [float(x) for x in err.tolist()],
        "kappa": kappa(ts, err, floor=args.warp_floor),
    }
    write_json(out_dir / "e1_profile.json", profile)

    rows = []
    results = {}
    samples_by_name = {}
    for nfe in parse_ints(args.nfe_values):
        specs = [{"sample_name": f"uniform_nfe_{nfe}", "schedule": "uniform", "grid": None}]
        for power in parse_floats(args.warp_powers):
            specs.append(
                {
                    "sample_name": f"e1_warped_p{rho_tag(power)}_nfe_{nfe}",
                    "schedule": "e1_warped",
                    "grid": equal_error_grid(profile["ts"], profile["err"], nfe, power=power, floor=args.warp_floor, end=1.0),
                    "warp_power": power,
                }
            )
        for kind in parse_names(args.power_kinds):
            for rho in parse_floats(args.power_rhos):
                specs.append(
                    {
                        "sample_name": f"power_{kind}_rho{rho_tag(rho)}_nfe_{nfe}",
                        "schedule": f"power_{kind}",
                        "grid": power_time_grid(nfe, rho=rho, kind=kind),
                        "power_kind": kind,
                        "power_rho": rho,
                    }
                )
        for spec in specs:
            eval_result = evaluate_samples(model, problem, x0, target, spec["schedule"], nfe, spec["grid"], args.hit_radius)
            metrics = eval_result["metrics"]
            metrics["sample_name"] = spec["sample_name"]
            for key in ["warp_power", "power_kind", "power_rho"]:
                if key in spec:
                    metrics[key] = spec[key]
            results[spec["sample_name"]] = metrics
            samples_by_name[spec["sample_name"]] = eval_result["samples"].detach().cpu()
            rows.append(metrics)
            print({k: v for k, v in metrics.items() if k not in {"time_grid", "mode_hit_probs"}}, flush=True)

    fieldnames = sorted({key for row in rows for key in row if key not in {"time_grid", "mode_hit_probs"}})
    with (out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key not in {"time_grid", "mode_hit_probs"}})
    write_json(
        out_dir / "metrics.json",
        {"run_dir": str(run_dir), "checkpoint": args.checkpoint, "e1_profile": profile, "results": results},
    )
    plot_summary(
        out_dir / "schedule_diagnostic.png",
        profile,
        rows,
        samples_by_name,
        x0.detach().cpu(),
        target.detach().cpu(),
        args.plot_nfe,
    )
    print(f"Saved GM16 schedule evaluation to {out_dir}")


if __name__ == "__main__":
    main()
