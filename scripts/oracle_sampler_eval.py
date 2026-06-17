#!/usr/bin/env python3
"""Exact-field sampler audit for Gaussian-mixture CFM oracles.

This answers the first benchmark-ladder question in docs/ideas.md:
if the pressure curvature is actually present in the exact marginal field,
do pressure-warped time grids help at fixed low NFE?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv

import torch

import lcfm.datasets  # noqa: F401 - registers datasets
from lcfm.metrics import mode_statistics, wasserstein_match
from lcfm.oracle import GaussianMixtureOracle
from lcfm.registry import DATASETS, get
from lcfm.schedules import equal_error_grid, euler_on_grid, heun_on_grid, interpolant_error_profile, kappa
from lcfm.utils import set_seed


GEOMETRIES = {
    "clumped015": {
        "dataset": "five_modes",
        "dataset_kwargs": {
            "n_train": 5000,
            "n_test": 2000,
            "radius": 4.0,
            "sigma_mode": 0.2,
            "source_std": 0.15,
        },
    },
    "ring": {
        "dataset": "five_modes",
        "dataset_kwargs": {"n_train": 5000, "n_test": 2000, "radius": 4.0, "sigma_mode": 0.2},
    },
    "fan": {
        "dataset": "fan_modes",
        "dataset_kwargs": {"n_train": 5000, "n_test": 2000, "sigma_mode": 0.2},
    },
    "staged": {
        "dataset": "staged_modes",
        "dataset_kwargs": {"n_train": 5000, "n_test": 2000, "sigma_mode": 0.2, "source_std": 0.15},
        "hit_radius": 0.6,
    },
    "gm16": {
        "dataset": "gaussian_mixture_nd",
        "dataset_kwargs": {
            "dim": 16,
            "n_modes": 8,
            "n_train": 5000,
            "n_test": 2000,
            "radius": 4.0,
            "sigma_mode": 0.2,
            "source_std": 0.15,
        },
        "hit_radius": 1.6,
    },
}


def evaluate_endpoint(problem, x_end, target, x_ref, hit_radius=None, reference_self_error=None):
    metrics = {
        "integration_error": float((x_end - x_ref).norm(dim=1).mean().item()),
        "wasserstein": wasserstein_match(x_end, target),
        "wasserstein2": wasserstein_match(x_end, target, p=2),
        "reference_self_error": reference_self_error,
    }
    if hasattr(problem, "centers") and hasattr(problem, "sigma_mode"):
        metrics.update(
            {
                k: v
                for k, v in mode_statistics(
                    x_end,
                    problem.centers(x_end.device),
                    p_min=0.05,
                    hit_radius=hit_radius or 3.0 * problem.sigma_mode,
                ).items()
                if k in {"mode_hit_coverage", "target_hit_rate"}
            }
        )
    return metrics


def run_geometry(name, args):
    spec = GEOMETRIES[name]
    problem_cls = get(DATASETS, spec["dataset"])
    problem = problem_cls(dict(spec["dataset_kwargs"]))
    oracle = GaussianMixtureOracle.from_problem(problem)
    velocity = lambda x, t: oracle.velocity(x, t)

    set_seed(args.seed + 17)
    px0 = problem.eval_initial(args.n_profile, torch.device("cpu"))
    px1 = problem.target_eval(args.n_profile, torch.device("cpu"))
    e0 = interpolant_error_profile(oracle.acceleration_target, px0, px1, grid_size=args.profile_grid, t_max=1.0)
    grid_e0 = equal_error_grid(*e0, args.steps)

    set_seed(args.seed)
    x0 = problem.eval_initial(args.n_eval, torch.device("cpu"))
    target = problem.target_eval(args.n_eval, torch.device("cpu"))
    x_ref = heun_on_grid(velocity, x0, torch.linspace(0, 1, args.ref_intervals + 1).tolist())[-1]
    reference_self_error = None
    if args.ref_check_intervals:
        if args.ref_check_intervals <= args.ref_intervals:
            raise ValueError("--ref-check-intervals must be larger than --ref-intervals.")
        x_ref_check = heun_on_grid(velocity, x0, torch.linspace(0, 1, args.ref_check_intervals + 1).tolist())[-1]
        reference_self_error = float((x_ref - x_ref_check).norm(dim=1).mean().item())

    heun_intervals = (args.steps + 1) // 2
    schedules = {
        f"euler{args.steps}_uniform": ("euler", torch.linspace(0, 1, args.steps + 1).tolist()),
        f"euler{args.steps}_warp_e0": ("euler", grid_e0),
        f"euler{2 * args.steps}_uniform": ("euler", torch.linspace(0, 1, 2 * args.steps + 1).tolist()),
        f"heun{heun_intervals}_uniform": ("heun", torch.linspace(0, 1, heun_intervals + 1).tolist()),
    }

    rows = []
    for schedule, (kind, grid) in schedules.items():
        if kind == "euler":
            x_end = euler_on_grid(velocity, x0, grid)[-1]
            nfe = len(grid) - 1
        else:
            x_end = heun_on_grid(velocity, x0, grid)[-1]
            nfe = 2 * (len(grid) - 1)
        rows.append(
            {
                "geometry": name,
                "schedule": schedule,
                "nfe": nfe,
                "kappa_e0": kappa(*e0),
                "grid_e0": " ".join(f"{t:.6f}" for t in grid_e0),
            **evaluate_endpoint(
                problem,
                x_end,
                target,
                x_ref,
                hit_radius=spec.get("hit_radius"),
                reference_self_error=reference_self_error,
            ),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/phase1/oracle_sampler_eval.csv")
    parser.add_argument("--geometries", nargs="+", default=list(GEOMETRIES), choices=list(GEOMETRIES))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--n-profile", type=int, default=2000)
    parser.add_argument("--profile-grid", type=int, default=101)
    parser.add_argument("--ref-intervals", type=int, default=1000)
    parser.add_argument(
        "--ref-check-intervals",
        type=int,
        default=0,
        help="Optional finer Heun rollout used only to estimate reference error.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rows = []
    for geometry in args.geometries:
        rows.extend(run_geometry(geometry, args))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output}")
    print(f"{'geometry':>12} {'schedule':>20} {'nfe':>4} {'kappa':>7} {'int_err':>9} {'W':>9} {'hit':>7} {'cov':>4}")
    for row in rows:
        print(
            f"{row['geometry']:>12} {row['schedule']:>20} {row['nfe']:>4} "
            f"{row['kappa_e0']:7.3f} {row['integration_error']:9.4f} "
            f"{row['wasserstein']:9.4f} {row.get('target_hit_rate', float('nan')):7.3f} "
            f"{row.get('mode_hit_coverage', '-')!s:>4}"
        )


if __name__ == "__main__":
    main()
