#!/usr/bin/env python3
"""Audit timestep-schedule headroom on a trained Burgers solution-map CFM."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lcfm.datasets import BurgersSolutionMapProblem
from lcfm.metrics import mean_path_length, path_length_ratio, rmse, trajectory_acceleration
from lcfm.models import build_model
from lcfm.schedules import (
    calibrate_schedule_profiles,
    equal_error_grid,
    euler_defect_cost_matrix,
    euler_on_grid,
    heun_on_grid,
    optimal_defect_grid,
    schedule_specs_from_bundle,
    validate_profile_bundle,
)
from lcfm.utils import file_sha256, read_json, set_seed, write_json


def parse_ints(text):
    return [int(part.strip()) for part in text.replace(" ", ",").split(",") if part.strip()]


def parse_floats(text):
    return [float(part.strip()) for part in text.replace(" ", ",").split(",") if part.strip()]


def value_tag(value):
    return f"{float(value):g}".replace(".", "p")


def uniform_grid(intervals):
    return [float(value) for value in torch.linspace(0.0, 1.0, int(intervals) + 1).tolist()]


def periodic_spatial_gradient(x):
    return torch.roll(x, shifts=-1, dims=-1) - x


def endpoint_metrics(traj, target, reference_endpoint):
    endpoint = traj[-1]
    return {
        "integration_error": float((endpoint - reference_endpoint).norm(dim=1).mean().item()),
        "rmse": rmse(endpoint, target),
        "spatial_grad_rmse": rmse(periodic_spatial_gradient(endpoint), periodic_spatial_gradient(target)),
        "mean_path_length": mean_path_length(traj),
        "path_length_ratio": path_length_ratio(traj),
        "trajectory_acceleration": trajectory_acceleration(traj),
    }


def resolve_cache_path(run_dir, config, override):
    if override:
        return str(Path(override).expanduser().resolve())
    configured = Path(config.get("dataset_kwargs", {}).get("cache_path", "")).expanduser()
    if configured.is_file():
        return str(configured)
    local = Path("data/burgers") / configured.name
    if local.is_file():
        return str(local.resolve())
    raise FileNotFoundError(f"Burgers cache not found: {configured}. Pass --data-cache.")


def build_schedule_specs(
    nfe,
    bundle,
    boss_grid,
    boss_cost,
    warp_powers,
    transported_powers,
    power_rhos,
    power_kinds,
):
    specs = schedule_specs_from_bundle(
        bundle,
        nfe,
        methods=("uniform", "sctw", "adjoint_sctw", "power", "vsfm", "heun"),
        powers=warp_powers,
        power_rhos=power_rhos,
        power_kinds=power_kinds,
        adjoint_probes=4,
    )
    specs.append({"schedule": "boss_dp", "grid": boss_grid, "nfe_used": nfe, "solver": "euler", "pilot_objective": boss_cost})
    equivalence_profiles = [name for name in bundle["profiles"] if name.startswith("transport_fd_") or name == "transport_jvp"]
    equivalence_profiles += [name for name in bundle["profiles"] if name.startswith("transport_adjoint_r") and name != "transport_adjoint_r4"]
    for estimator in equivalence_profiles:
        transported_profile = bundle["profiles"][estimator]
        for power in transported_powers:
            specs.append(
                {
                    "schedule": f"{estimator}_p{value_tag(power)}",
                    "grid": equal_error_grid(
                        transported_profile["ts"],
                        transported_profile["err"],
                        nfe,
                        power=power,
                        floor=transported_profile["floor"],
                    ),
                    "nfe_used": nfe,
                    "solver": "euler",
                    "warp_power": power,
                    "kappa": transported_profile["kappa"],
                }
            )
    return specs


def plot_audit(rows, grids, out_dir, plot_nfe):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"plot unavailable: {exc}"

    preferred = [
        "uniform",
        "sctw_p0p25",
        "sctw_p0p5",
        "adjoint_sctw_p0p25",
        "adjoint_sctw_p0p5",
        "transport_fd_eps0p001_p0p5",
        "transport_jvp_p0p5",
        "transport_adjoint_r2_p0p5",
        "transport_adjoint_r8_p0p5",
        "vsfm_coasting",
        "vsfm_uniform_deceleration",
        "boss_dp",
        "heun_uniform_matched",
        "power_early_rho2",
        "power_early_rho3",
    ]
    schedules = [name for name in preferred if any(row["schedule"] == name for row in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    for ax, metric, title in [
        (axes[0, 0], "integration_error", "Learned-ODE integration error"),
        (axes[0, 1], "rmse", "Endpoint PDE RMSE"),
        (axes[1, 0], "spatial_grad_rmse", "Endpoint gradient RMSE"),
    ]:
        for schedule in schedules:
            items = sorted((row for row in rows if row["schedule"] == schedule), key=lambda row: row["nfe_budget"])
            ax.plot([row["nfe_budget"] for row in items], [row[metric] for row in items], marker="o", label=schedule)
        ax.set_xscale("log")
        ax.set_xlabel("NFE budget")
        ax.set_ylabel("lower is better")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    for level, schedule in enumerate(schedules):
        key = f"{schedule}_nfe_{plot_nfe}"
        if key not in grids:
            continue
        grid = grids[key]
        ax.hlines(level, 0.0, 1.0, color="#dddddd", lw=1)
        ax.vlines(grid, level - 0.3, level + 0.3, lw=1.6)
    ax.set_yticks(range(len(schedules)))
    ax.set_yticklabels(schedules, fontsize=8)
    ax.set_xlim(-0.01, 1.01)
    ax.set_xlabel("flow time")
    ax.set_title(f"NFE {plot_nfe} grids")
    ax.grid(True, axis="x", alpha=0.25)
    axes[0, 1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = out_dir / "schedule_ceiling_audit.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", default="model_ema.pt")
    parser.add_argument("--data-cache")
    parser.add_argument("--out-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-eval", type=int, default=32)
    parser.add_argument("--n-pilot", type=int, default=32)
    parser.add_argument("--pilot-seed", type=int, default=1311)
    parser.add_argument("--profile-cache")
    parser.add_argument("--nfe-values", default="5,10,20,50")
    parser.add_argument("--profile-fine-steps", type=int, default=50)
    parser.add_argument("--warp-powers", default="0.25,0.5")
    parser.add_argument("--warp-floor", type=float, default=1e-3)
    parser.add_argument("--transport-fine-steps", type=int, default=50)
    parser.add_argument("--transport-anchor-count", type=int, default=16)
    parser.add_argument("--transport-perturbations", default="0.001,0.003")
    parser.add_argument("--transport-powers", default="0.5")
    parser.add_argument("--adjoint-probes", default="2,4,8")
    parser.add_argument("--adjoint-seed", type=int, default=2718)
    parser.add_argument("--boss-anchor-steps", type=int, default=100)
    parser.add_argument("--boss-reference-solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--power-rhos", default="2,3")
    parser.add_argument("--power-kinds", default="early,late,symmetric")
    parser.add_argument("--reference-intervals", type=int, default=1000)
    parser.add_argument("--reference-check-intervals", type=int, default=2000)
    parser.add_argument("--plot-nfe", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "schedule_ceiling_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(run_dir / "config.json")
    config.setdefault("dataset_kwargs", {})["cache_path"] = resolve_cache_path(run_dir, config, args.data_cache)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    set_seed(int(config.get("seed", 42)))
    problem = BurgersSolutionMapProblem(config.get("dataset_kwargs", {}))
    model = build_model(config.get("model", "unet1d"), problem.dim, config).to(device)
    model.load_state_dict(torch.load(run_dir / args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    if args.n_eval > problem.n_test:
        raise ValueError(f"n_eval={args.n_eval} exceeds held-out test size {problem.n_test}.")
    if args.n_pilot > problem.n_train:
        raise ValueError(f"n_pilot={args.n_pilot} exceeds training size {problem.n_train}.")
    eval_x0 = problem.test_x0[: args.n_eval].to(device)
    target = problem.test_x1[: args.n_eval].to(device)
    if args.profile_fine_steps != args.transport_fine_steps:
        raise ValueError("Shared calibration requires --profile-fine-steps and --transport-fine-steps to match.")
    set_seed(args.pilot_seed)
    pilot_indices = torch.randperm(problem.n_train)[: args.n_pilot]
    pilot_x0 = problem.train_x0[pilot_indices].to(device)
    nfe_values = parse_ints(args.nfe_values)
    warp_powers = parse_floats(args.warp_powers)
    profile_cache = Path(args.profile_cache) if args.profile_cache else out_dir / "schedule_profiles.json"
    profile_context = {
        "dataset": config["dataset"],
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(run_dir / args.checkpoint),
    }
    if profile_cache.is_file():
        bundle = read_json(profile_cache)
        validate_profile_bundle(
            bundle,
            profile_context,
            {
                "pilot_seed": args.pilot_seed,
                "pilot_samples": args.n_pilot,
                "fine_steps": args.transport_fine_steps,
                "anchor_count": args.transport_anchor_count,
                "adjoint_probe_counts": parse_ints(args.adjoint_probes),
                "adjoint_seed": args.adjoint_seed,
                "perturbations": parse_floats(args.transport_perturbations),
                "floor": args.warp_floor,
                "powers": warp_powers,
                "nfe_values": nfe_values,
            },
        )
    else:
        bundle = calibrate_schedule_profiles(
            [(model, pilot_x0)],
            estimators=("local", "transport_fd", "transport_jvp", "transport_adjoint"),
            fine_steps=args.transport_fine_steps,
            anchor_count=args.transport_anchor_count,
            adjoint_probe_counts=parse_ints(args.adjoint_probes),
            adjoint_seed=args.adjoint_seed,
            perturbations=parse_floats(args.transport_perturbations),
            floor=args.warp_floor,
            powers=warp_powers,
            nfe_values=nfe_values,
            pilot_seed=args.pilot_seed,
            context=profile_context,
        )
        bundle["calibration"]["pilot_indices"] = [int(value) for value in pilot_indices]
        write_json(profile_cache, bundle)
    profile = bundle["profiles"]["local"]
    transported_profiles = {name: value for name, value in bundle["profiles"].items() if name != "local"}
    calibration_seconds = {name: value["wall_seconds"] for name, value in bundle["timings"].items()}
    anchor_grid, edge_costs = euler_defect_cost_matrix(
        model,
        pilot_x0,
        anchor_steps=args.boss_anchor_steps,
        reference_solver=args.boss_reference_solver,
    )

    boss = {}
    for nfe in nfe_values:
        grid, objective = optimal_defect_grid(edge_costs, anchor_grid, nfe)
        boss[nfe] = {"grid": grid, "objective": objective}

    reference_grid = uniform_grid(args.reference_intervals)
    reference_endpoint = heun_on_grid(model, eval_x0, reference_grid)[-1]
    reference_self_error = None
    if args.reference_check_intervals > args.reference_intervals:
        check_endpoint = heun_on_grid(model, eval_x0, uniform_grid(args.reference_check_intervals))[-1]
        reference_self_error = float((reference_endpoint - check_endpoint).norm(dim=1).mean().item())

    transported_powers = parse_floats(args.transport_powers)
    power_rhos = parse_floats(args.power_rhos)
    power_kinds = [part.strip() for part in args.power_kinds.replace(" ", ",").split(",") if part.strip()]
    rows = []
    grids = {}
    for nfe in nfe_values:
        specs = build_schedule_specs(
            nfe,
            bundle,
            boss[nfe]["grid"],
            boss[nfe]["objective"],
            warp_powers,
            transported_powers,
            power_rhos,
            power_kinds,
        )
        for spec in specs:
            solver = spec.get("solver", "euler")
            traj = heun_on_grid(model, eval_x0, spec["grid"]) if solver == "heun" else euler_on_grid(model, eval_x0, spec["grid"])
            row = {
                "schedule": spec["schedule"],
                "nfe_budget": nfe,
                "nfe_used": spec["nfe_used"],
                **endpoint_metrics(traj, target, reference_endpoint),
            }
            for key in ("warp_power", "pilot_objective", "kappa"):
                if key in spec:
                    row[key] = spec[key]
            if spec["schedule"].startswith("sctw_"):
                row["kappa"] = profile["kappa"]
            if reference_self_error is not None:
                row["reference_self_error"] = reference_self_error
            rows.append(row)
            grids[f"{spec['schedule']}_nfe_{nfe}"] = spec["grid"]
            print(
                f"{spec['schedule']:>28} budget={nfe:>3} used={spec['nfe_used']:>3} "
                f"integration={row['integration_error']:.6f} rmse={row['rmse']:.6f}",
                flush=True,
            )

    fieldnames = sorted({key for row in rows for key in row})
    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        out_dir / "audit.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "device": str(device),
            "pilot_split": "train",
            "evaluation_split": "test",
            "n_pilot": args.n_pilot,
            "n_eval": args.n_eval,
            "profile_bundle": bundle,
            "profile_cache": str(profile_cache),
            "profile": profile,
            "transported_profiles": transported_profiles,
            "calibration_seconds": calibration_seconds,
            "boss_anchor_steps": args.boss_anchor_steps,
            "boss_reference_solver": args.boss_reference_solver,
            "boss": {str(key): value for key, value in boss.items()},
            "reference_intervals": args.reference_intervals,
            "reference_check_intervals": args.reference_check_intervals,
            "reference_self_error": reference_self_error,
            "grids": grids,
            "rows": rows,
        },
    )
    print(f"Saved audit to {out_dir}")
    print(f"Diagnostic plot: {plot_audit(rows, grids, out_dir, args.plot_nfe)}")


if __name__ == "__main__":
    main()
