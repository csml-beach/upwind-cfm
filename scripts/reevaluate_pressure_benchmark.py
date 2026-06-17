#!/usr/bin/env python3
"""Reevaluate saved pressure-training checkpoints with current metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from statistics import mean, pstdev

_cache_dir = Path(tempfile.gettempdir()) / "lcfm_pressure_eval_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir / "xdg"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from lcfm.metrics import (
    mean_endpoint_displacement,
    mean_path_length,
    mode_statistics,
    path_length_ratio,
    trajectory_acceleration,
    wasserstein_match,
)
from lcfm.plotting import load_run
from lcfm.utils import set_seed


def euler_traj(velocity, x0, intervals):
    x = x0.clone()
    traj = [x.clone()]
    dt = 1.0 / intervals
    for i in range(intervals):
        t = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
        x = x + dt * velocity(x, t)
        traj.append(x.clone())
    return torch.stack(traj)


def heun_endpoint(velocity, x0, intervals):
    x = x0.clone()
    dt = 1.0 / intervals
    for i in range(intervals):
        t0 = i * dt
        t1 = (i + 1) * dt
        tt0 = torch.full((x.shape[0], 1), t0, device=x.device, dtype=x.dtype)
        tt1 = torch.full((x.shape[0], 1), t1, device=x.device, dtype=x.dtype)
        v0 = velocity(x, tt0)
        v1 = velocity(x + dt * v0, tt1)
        x = x + 0.5 * dt * (v0 + v1)
    return x


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_run_name(path):
    name = Path(path).name
    if "_seed" not in name:
        return name, "", ""
    stem, seed = name.rsplit("_seed", 1)
    geometry = stem.split("_", 1)[0]
    variant = stem[len(geometry) + 1 :]
    return geometry, variant, int(seed)


def discover_runs(root, geometries, variants):
    root = Path(root)
    runs = []
    for geometry in geometries:
        for variant in variants:
            for run_dir in sorted(root.glob(f"{geometry}_{variant}_seed*")):
                if (run_dir / "model.pt").exists() and (run_dir / "config.json").exists():
                    runs.append(run_dir)
    return runs


@torch.no_grad()
def evaluate_run(run_dir, args):
    device = torch.device(args.device)
    config, problem, model = load_run(run_dir, device)
    velocity = lambda x, t: model(x, t)
    geometry, run_variant, run_seed = parse_run_name(run_dir)
    variant = config.get("variant", run_variant)
    seed = config.get("seed", run_seed)

    set_seed(args.eval_seed)
    x0 = problem.eval_initial(args.n_eval, device)
    target = problem.target_eval(args.n_eval, device)
    traj = euler_traj(velocity, x0, args.steps)
    x_end = traj[-1]
    x_ref = heun_endpoint(velocity, x0, args.ref_intervals)
    x_ref_check = None
    reference_self_error = None
    if args.ref_check_intervals:
        if args.ref_check_intervals <= args.ref_intervals:
            raise ValueError("--ref-check-intervals must be larger than --ref-intervals.")
        x_ref_check = heun_endpoint(velocity, x0, args.ref_check_intervals)
        reference_self_error = float((x_ref - x_ref_check).norm(dim=1).mean().item())

    row = {
        "run": Path(run_dir).name,
        "run_dir": str(Path(run_dir)),
        "geometry": geometry or problem.name,
        "dataset": config.get("dataset", problem.name),
        "variant": variant,
        "kind": config.get("variant_kwargs", {}).get("kind", ""),
        "weight": config.get("variant_kwargs", {}).get("weight", ""),
        "seed": seed,
        "eval_seed": args.eval_seed,
        "n_eval": args.n_eval,
        "schedule": f"euler{args.steps}_uniform",
        "nfe": args.steps,
        "ref_intervals": args.ref_intervals,
        "reference_self_error": reference_self_error,
        "integration_error": float((x_end - x_ref).norm(dim=1).mean().item()),
        "wasserstein": wasserstein_match(x_end, target),
        "wasserstein2": wasserstein_match(x_end, target, p=2),
        "mean_path_length": mean_path_length(traj),
        "mean_endpoint_displacement": mean_endpoint_displacement(traj),
        "path_length_ratio": path_length_ratio(traj),
        "trajectory_acceleration": trajectory_acceleration(traj),
    }
    if hasattr(problem, "centers") and hasattr(problem, "sigma_mode"):
        row.update(
            mode_statistics(
                x_end,
                problem.centers(device),
                p_min=args.mode_p_min,
                hit_radius=args.hit_radius or args.hit_radius_by_geometry.get(row["geometry"], 3.0 * problem.sigma_mode),
            )
        )
    del x_ref_check
    return row


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["geometry"], row["dataset"], row["variant"], row["schedule"], row["nfe"])
        grouped[key].append(row)
    metrics = [
        "integration_error",
        "wasserstein",
        "wasserstein2",
        "target_hit_rate",
        "mode_hit_coverage",
        "mean_path_length",
        "mean_endpoint_displacement",
        "path_length_ratio",
        "trajectory_acceleration",
        "reference_self_error",
    ]
    aggregates = []
    for key, items in sorted(grouped.items()):
        geometry, dataset, variant, schedule, nfe = key
        out = {
            "geometry": geometry,
            "dataset": dataset,
            "variant": variant,
            "schedule": schedule,
            "nfe": nfe,
            "n_seeds": len(items),
            "seeds": " ".join(str(row["seed"]) for row in items),
        }
        for metric in metrics:
            values = [row.get(metric) for row in items]
            values = [float(value) for value in values if value not in ("", None)]
            if values:
                out[f"{metric}_mean"] = mean(values)
                out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregates.append(out)
    return aggregates


def print_table(aggregate):
    print(
        f"{'geometry':>8} {'variant':>16} {'n':>3} {'W1':>15} {'W2':>15} "
        f"{'hit':>15} {'int_err':>15} {'ref_self':>10}"
    )
    for row in aggregate:
        print(
            f"{row['geometry']:>8} {row['variant']:>16} {row['n_seeds']:>3} "
            f"{row.get('wasserstein_mean', float('nan')):7.3f}+/-{row.get('wasserstein_std', 0.0):.3f} "
            f"{row.get('wasserstein2_mean', float('nan')):7.3f}+/-{row.get('wasserstein2_std', 0.0):.3f} "
            f"{row.get('target_hit_rate_mean', float('nan')):7.3f}+/-{row.get('target_hit_rate_std', 0.0):.3f} "
            f"{row.get('integration_error_mean', float('nan')):7.3f}+/-{row.get('integration_error_std', 0.0):.3f} "
            f"{row.get('reference_self_error_mean', float('nan')):10.5f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/phase1/pressure_training_sweep")
    parser.add_argument("--run-dirs", nargs="+", help="Explicit run dirs. Overrides discovery.")
    parser.add_argument("--geometries", nargs="+", default=["staged", "gm16"])
    parser.add_argument("--variants", nargs="+", default=["standard", "upper_w2", "band_w1_eta05"])
    parser.add_argument("--output-dir", default="results/phase1/decision_benchmark_eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-eval", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--ref-intervals", type=int, default=500)
    parser.add_argument("--ref-check-intervals", type=int, default=1000)
    parser.add_argument("--mode-p-min", type=float, default=0.05)
    parser.add_argument("--hit-radius", type=float)
    args = parser.parse_args()
    args.hit_radius_by_geometry = {"staged": 0.6, "gm16": 1.6, "clumped015": 0.6}

    torch.set_num_threads(args.threads)
    run_dirs = [Path(path) for path in args.run_dirs] if args.run_dirs else discover_runs(args.root, args.geometries, args.variants)
    if not run_dirs:
        raise ValueError("No run dirs found.")

    rows = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[reeval] {index}/{len(run_dirs)} {run_dir}", flush=True)
        rows.append(evaluate_run(run_dir, args))

    aggregate = aggregate_rows(rows)
    output_dir = Path(args.output_dir)
    raw_path = output_dir / "decision_benchmark_raw.csv"
    aggregate_path = output_dir / "decision_benchmark_aggregate.csv"
    write_csv(raw_path, rows)
    write_csv(aggregate_path, aggregate)
    print_table(aggregate)
    print(f"\nwrote {raw_path}")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()
