#!/usr/bin/env python3
"""Validity audit for the 16D Gaussian-mixture benchmark."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
import tempfile

_cache_dir = Path(tempfile.gettempdir()) / "lcfm_gm16_audit_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir / "xdg"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from lcfm.datasets import GaussianMixtureNDProblem
from lcfm.metrics import mode_statistics, wasserstein_match
from lcfm.oracle import GaussianMixtureOracle
from lcfm.plotting import load_run
from lcfm.utils import set_seed


DEFAULT_CONFIG = {
    "dim": 16,
    "n_modes": 8,
    "n_train": 5000,
    "n_test": 5000,
    "radius": 4.0,
    "sigma_mode": 0.2,
    "source_std": 0.15,
}


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


def euler_endpoint(velocity, x0, intervals):
    x = x0.clone()
    dt = 1.0 / intervals
    for i in range(intervals):
        t = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
        x = x + dt * velocity(x, t)
    return x


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


def parse_run_name(path):
    name = Path(path).name
    if "_seed" not in name:
        return name, ""
    stem, seed = name.rsplit("_seed", 1)
    variant = stem.split("_", 1)[1] if "_" in stem else stem
    return variant, int(seed)


def endpoint_metrics(problem, samples, target, hit_radius):
    metrics = {
        "wasserstein": wasserstein_match(samples, target),
        "wasserstein2": wasserstein_match(samples, target, p=2),
    }
    metrics.update(mode_statistics(samples, problem.centers(samples.device), hit_radius=hit_radius))
    return metrics


def geometry_rows(problem, hit_radius):
    centers = problem.centers(torch.device("cpu"))
    pairwise = torch.pdist(centers)
    return [
        {
            "row_type": "geometry",
            "name": "gm16",
            "dim": problem.dim,
            "n_modes": problem.n_modes,
            "radius_mean": float(centers.norm(dim=1).mean().item()),
            "center_distance_min": float(pairwise.min().item()),
            "center_distance_mean": float(pairwise.mean().item()),
            "center_distance_max": float(pairwise.max().item()),
            "sigma_mode": problem.sigma_mode,
            "source_std": problem.source_std,
            "hit_radius": hit_radius,
            "hit_regions_overlap": bool(2.0 * hit_radius >= float(pairwise.min().item())),
            "mode_subspace_rank": int(torch.linalg.matrix_rank(centers - centers.mean(dim=0, keepdim=True)).item()),
        }
    ]


def floor_rows(problem, args, device):
    rows = []
    for seed in args.floor_seeds:
        set_seed(seed)
        target_a = problem.target_eval(args.n_eval, device)
        target_b = problem._sample_modes(args.n_eval).to(device)
        source = problem.eval_initial(args.n_eval, device)
        for name, samples in [("target_target", target_a), ("source_target", source)]:
            metrics = endpoint_metrics(problem, samples, target_b if name == "target_target" else target_a, args.hit_radius)
            rows.append(
                {
                    "row_type": "floor",
                    "name": name,
                    "seed": seed,
                    "n_eval": args.n_eval,
                    **metrics,
                }
            )
    return rows


@torch.no_grad()
def sampler_rows(problem, name, velocity, x0, target, args):
    rows = []
    x_ref = heun_endpoint(velocity, x0, args.ref_intervals)
    x_check = heun_endpoint(velocity, x0, args.ref_check_intervals) if args.ref_check_intervals else None
    reference_self_error = None
    if x_check is not None:
        reference_self_error = float((x_ref - x_check).norm(dim=1).mean().item())
    schedules = {
        f"{name}_heun{args.ref_intervals}": (x_ref, 2 * args.ref_intervals, 0.0),
        f"{name}_euler{args.steps}": (
            euler_endpoint(velocity, x0, args.steps),
            args.steps,
            None,
        ),
    }
    for schedule, (samples, nfe, integration_error) in schedules.items():
        if integration_error is None:
            integration_error = float((samples - x_ref).norm(dim=1).mean().item())
        metrics = endpoint_metrics(problem, samples, target, args.hit_radius)
        rows.append(
            {
                "row_type": "sampler",
                "name": name,
                "schedule": schedule,
                "nfe": nfe,
                "n_eval": args.n_eval,
                "ref_intervals": args.ref_intervals,
                "reference_self_error": reference_self_error,
                "integration_error": integration_error,
                **metrics,
            }
        )
    return rows


def run_model_rows(problem, run_dirs, x0, target, args, device):
    rows = []
    for run_dir in run_dirs:
        config, loaded_problem, model = load_run(run_dir, device)
        if loaded_problem.name != problem.name or loaded_problem.dim != problem.dim:
            raise ValueError(f"{run_dir} is not a compatible gm16 run.")
        variant, seed = parse_run_name(run_dir)
        velocity = lambda x, t, model=model: model(x, t)
        for row in sampler_rows(problem, variant, velocity, x0, target, args):
            row["row_type"] = "model"
            row["run"] = Path(run_dir).name
            row["run_dir"] = str(Path(run_dir))
            row["variant"] = config.get("variant", variant)
            row["seed"] = config.get("seed", seed)
            rows.append(row)
    return rows


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        if row["row_type"] not in {"floor", "sampler", "model"}:
            continue
        key = (row["row_type"], row["name"], row.get("schedule", ""))
        grouped.setdefault(key, []).append(row)
    aggregate = []
    metrics = ["wasserstein", "wasserstein2", "target_hit_rate", "mode_hit_coverage", "integration_error", "reference_self_error"]
    for (row_type, name, schedule), items in sorted(grouped.items()):
        out = {"row_type": row_type, "name": name, "schedule": schedule, "n": len(items)}
        for metric in metrics:
            values = [item.get(metric) for item in items]
            values = [float(value) for value in values if value not in (None, "")]
            if values:
                tensor = torch.tensor(values, dtype=torch.float64)
                out[f"{metric}_mean"] = float(tensor.mean().item())
                out[f"{metric}_std"] = float(tensor.std(unbiased=False).item()) if len(values) > 1 else 0.0
        aggregate.append(out)
    return aggregate


def print_summary(rows, aggregate):
    print("Geometry")
    for row in rows:
        if row["row_type"] == "geometry":
            print(
                f"dim={row['dim']} modes={row['n_modes']} rank={row['mode_subspace_rank']} "
                f"center_dist={row['center_distance_min']:.3f} hit_radius={row['hit_radius']:.3f} "
                f"overlap={row['hit_regions_overlap']}"
            )
    print("\nAudit Summary")
    print(f"{'type':>8} {'name':>18} {'schedule':>24} {'n':>3} {'W1':>14} {'W2':>14} {'hit':>14} {'int':>14} {'ref':>10}")
    for row in aggregate:
        print(
            f"{row['row_type']:>8} {row['name'][-18:]:>18} {row['schedule'][-24:]:>24} {row['n']:>3} "
            f"{row.get('wasserstein_mean', float('nan')):7.3f}+/-{row.get('wasserstein_std', 0.0):.3f} "
            f"{row.get('wasserstein2_mean', float('nan')):7.3f}+/-{row.get('wasserstein2_std', 0.0):.3f} "
            f"{row.get('target_hit_rate_mean', float('nan')):7.3f}+/-{row.get('target_hit_rate_std', 0.0):.3f} "
            f"{row.get('integration_error_mean', float('nan')):7.3f}+/-{row.get('integration_error_std', 0.0):.3f} "
            f"{row.get('reference_self_error_mean', float('nan')):10.5f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/phase1/gm16_validity_audit")
    parser.add_argument("--run-dirs", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-eval", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--ref-intervals", type=int, default=500)
    parser.add_argument("--ref-check-intervals", type=int, default=1000)
    parser.add_argument("--hit-radius", type=float, default=1.6)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--floor-seeds", nargs="+", type=int, default=[101, 202, 303])
    args = parser.parse_args()

    if args.ref_check_intervals and args.ref_check_intervals <= args.ref_intervals:
        raise ValueError("--ref-check-intervals must be larger than --ref-intervals.")

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    set_seed(args.eval_seed)
    problem = GaussianMixtureNDProblem(DEFAULT_CONFIG)
    oracle = GaussianMixtureOracle.from_problem(problem).to(device)
    x0 = problem.eval_initial(args.n_eval, device)
    target = problem.target_eval(args.n_eval, device)

    rows = []
    rows.extend(geometry_rows(problem, args.hit_radius))
    rows.extend(floor_rows(problem, args, device))
    rows.extend(sampler_rows(problem, "oracle", lambda x, t: oracle.velocity(x, t), x0, target, args))
    if args.run_dirs:
        rows.extend(run_model_rows(problem, [Path(path) for path in args.run_dirs], x0, target, args, device))

    aggregate = aggregate_rows(rows)
    output_dir = Path(args.output_dir)
    raw_path = output_dir / "gm16_validity_raw.csv"
    aggregate_path = output_dir / "gm16_validity_aggregate.csv"
    write_csv(raw_path, rows)
    write_csv(aggregate_path, aggregate)
    print_summary(rows, aggregate)
    print(f"\nwrote {raw_path}")
    print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()
