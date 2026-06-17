#!/usr/bin/env python3
"""Audit metric stability for saved CFM runs.

This script is intentionally evaluation-only. It checks two separate issues:

1. Endpoint distribution metrics: exact empirical W1/W2 matching as n_eval and
   eval seed vary. This estimates finite-sample noise in the main endpoint metric.
2. Integration error references: low-NFE Euler endpoints against increasingly
   fine Heun references, plus a self-error against an even finer Heun reference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from statistics import mean, pstdev

_cache_dir = Path(tempfile.gettempdir()) / "lcfm_metric_audit_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir / "xdg"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lcfm.metrics import mode_statistics, wasserstein_match
from lcfm.plotting import load_run
from lcfm.utils import set_seed


def run_label(run_dir):
    path = Path(run_dir)
    if path.parent.name and path.parent.name not in {".", ""}:
        return f"{path.parent.name}/{path.name}"
    return path.name


def summarize(values):
    values = [float(value) for value in values]
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


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


def endpoint_stats(problem, x_end, target):
    metrics = {
        "wasserstein": wasserstein_match(x_end, target),
        "wasserstein2": wasserstein_match(x_end, target, p=2),
    }
    if hasattr(problem, "centers") and hasattr(problem, "sigma_mode"):
        metrics.update(
            {
                k: v
                for k, v in mode_statistics(
                    x_end,
                    problem.centers(x_end.device),
                    p_min=0.05,
                    hit_radius=3.0 * problem.sigma_mode,
                ).items()
                if k in {"mode_hit_coverage", "target_hit_rate"}
            }
        )
    return metrics


@torch.no_grad()
def euler_endpoint(velocity, x0, intervals):
    x = x0.clone()
    dt = 1.0 / intervals
    for i in range(intervals):
        t = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
        x = x + dt * velocity(x, t)
    return x


@torch.no_grad()
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


@torch.no_grad()
def evaluate_run(run_dir, args):
    device = torch.device(args.device)
    config, problem, model = load_run(run_dir, device)
    velocity = lambda x, t: model(x, t)
    label = run_label(run_dir)
    rows = []

    for n_eval in args.n_eval:
        for eval_seed in args.eval_seeds:
            print(f"[metric-audit] {label}: n_eval={n_eval}, eval_seed={eval_seed}", flush=True)
            set_seed(eval_seed)
            x0 = problem.eval_initial(n_eval, device)
            target = problem.target_eval(n_eval, device)
            x_low = euler_endpoint(velocity, x0, args.low_steps)
            endpoint = endpoint_stats(problem, x_low, target)
            rows.append(
                {
                    "row_type": "endpoint",
                    "run": label,
                    "run_dir": str(Path(run_dir)),
                    "dataset": config.get("dataset", problem.name),
                    "method": config.get("method", config.get("variant", "")),
                    "seed": config.get("seed", ""),
                    "eval_seed": eval_seed,
                    "n_eval": n_eval,
                    "low_steps": args.low_steps,
                    "ref_intervals": "",
                    "nfe": args.low_steps,
                    "integration_error": "",
                    "reference_self_error": "",
                    **endpoint,
                }
            )

            ref_cache = {}
            needed_intervals = set(args.ref_intervals)
            needed_intervals.update(interval * args.ref_check_multiplier for interval in args.ref_intervals)
            for intervals in sorted(needed_intervals):
                ref_cache[intervals] = heun_endpoint(velocity, x0, intervals)

            for intervals in args.ref_intervals:
                x_ref = ref_cache[intervals]
                check_intervals = intervals * args.ref_check_multiplier
                x_ref_check = ref_cache[check_intervals]
                rows.append(
                    {
                        "row_type": "reference",
                        "run": label,
                        "run_dir": str(Path(run_dir)),
                        "dataset": config.get("dataset", problem.name),
                        "method": config.get("method", config.get("variant", "")),
                        "seed": config.get("seed", ""),
                        "eval_seed": eval_seed,
                        "n_eval": n_eval,
                        "low_steps": args.low_steps,
                        "ref_intervals": intervals,
                        "nfe": 2 * intervals,
                        "integration_error": float((x_low - x_ref).norm(dim=1).mean().item()),
                        "reference_self_error": float((x_ref - x_ref_check).norm(dim=1).mean().item()),
                        "wasserstein": "",
                        "wasserstein2": "",
                        "target_hit_rate": "",
                        "mode_hit_coverage": "",
                    }
                )
    return rows


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["row_type"],
            row["run"],
            row["dataset"],
            row["method"],
            row["n_eval"],
            row["low_steps"],
            row["ref_intervals"],
        )
        grouped[key].append(row)

    aggregates = []
    for key, items in sorted(grouped.items()):
        row_type, run, dataset, method, n_eval, low_steps, ref_intervals = key
        out = {
            "row_type": row_type,
            "run": run,
            "dataset": dataset,
            "method": method,
            "n_eval": n_eval,
            "low_steps": low_steps,
            "ref_intervals": ref_intervals,
            "n_eval_seeds": len(items),
        }
        metric_names = [
            "wasserstein",
            "wasserstein2",
            "target_hit_rate",
            "mode_hit_coverage",
            "integration_error",
            "reference_self_error",
        ]
        for metric in metric_names:
            values = [item.get(metric) for item in items]
            values = [value for value in values if value not in ("", None)]
            if values:
                metric_mean, metric_std = summarize(values)
                out[f"{metric}_mean"] = metric_mean
                out[f"{metric}_std"] = metric_std
        aggregates.append(out)
    return aggregates


def aggregate_lookup(aggregate):
    lookup = {}
    for row in aggregate:
        lookup[(row["row_type"], row["run"], int(row["n_eval"]), row["ref_intervals"])] = row
    return lookup


def plot_audit(aggregate, args, output):
    lookup = aggregate_lookup(aggregate)
    runs = sorted({row["run"] for row in aggregate})
    colors = plt.cm.tab10.colors
    largest_n = max(args.n_eval)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        xs = []
        w_means = []
        w_stds = []
        w2_means = []
        w2_stds = []
        for n_eval in args.n_eval:
            row = lookup.get(("endpoint", run, n_eval, ""))
            if row is None:
                continue
            xs.append(n_eval)
            w_means.append(row.get("wasserstein_mean", float("nan")))
            w_stds.append(row.get("wasserstein_std", 0.0))
            w2_means.append(row.get("wasserstein2_mean", float("nan")))
            w2_stds.append(row.get("wasserstein2_std", 0.0))
        if xs:
            axes[0].errorbar(xs, w_means, yerr=w_stds, marker="o", color=color, label=run)
            axes[1].errorbar(xs, w2_means, yerr=w2_stds, marker="o", color=color, label=run)

        ref_xs = []
        int_means = []
        int_stds = []
        self_means = []
        self_stds = []
        for intervals in args.ref_intervals:
            row = lookup.get(("reference", run, largest_n, intervals))
            if row is None:
                continue
            ref_xs.append(intervals)
            int_means.append(row.get("integration_error_mean", float("nan")))
            int_stds.append(row.get("integration_error_std", 0.0))
            self_means.append(row.get("reference_self_error_mean", float("nan")))
            self_stds.append(row.get("reference_self_error_std", 0.0))
        if ref_xs:
            axes[2].errorbar(ref_xs, int_means, yerr=int_stds, marker="o", color=color, label=f"{run} int")
            axes[2].errorbar(
                ref_xs,
                self_means,
                yerr=self_stds,
                marker="x",
                linestyle="--",
                color=color,
                alpha=0.75,
                label=f"{run} ref",
            )

    axes[0].set_title("Endpoint W1 vs n_eval")
    axes[0].set_xlabel("n_eval")
    axes[0].set_ylabel("empirical W1")
    axes[1].set_title("Endpoint W2 vs n_eval")
    axes[1].set_xlabel("n_eval")
    axes[1].set_ylabel("empirical W2")
    axes[2].set_title(f"Integration Reference, n_eval={largest_n}")
    axes[2].set_xlabel("Heun reference intervals")
    axes[2].set_ylabel("mean endpoint distance")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    axes[2].legend(fontsize=7, loc="best")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def print_summary(aggregate):
    endpoint = [row for row in aggregate if row["row_type"] == "endpoint"]
    reference = [row for row in aggregate if row["row_type"] == "reference"]
    print("\nEndpoint Wasserstein stability")
    print(f"{'run':>32} {'n':>6} {'W1':>16} {'W2':>16}")
    for row in endpoint:
        print(
            f"{row['run'][-32:]:>32} {int(row['n_eval']):6d} "
            f"{row.get('wasserstein_mean', float('nan')):8.4f}+/-{row.get('wasserstein_std', 0.0):.4f} "
            f"{row.get('wasserstein2_mean', float('nan')):8.4f}+/-{row.get('wasserstein2_std', 0.0):.4f}"
        )
    print("\nReference stability")
    print(f"{'run':>32} {'n':>6} {'ref':>6} {'int_err':>16} {'ref_self':>16}")
    for row in reference:
        print(
            f"{row['run'][-32:]:>32} {int(row['n_eval']):6d} {int(row['ref_intervals']):6d} "
            f"{row.get('integration_error_mean', float('nan')):8.4f}+/-{row.get('integration_error_std', 0.0):.4f} "
            f"{row.get('reference_self_error_mean', float('nan')):8.4f}+/-{row.get('reference_self_error_std', 0.0):.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="Saved run directories containing config.json and model.pt.")
    parser.add_argument("--output-dir", default="results/phase1/metric_convergence_audit")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-eval", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[1234, 2234, 3234])
    parser.add_argument("--low-steps", type=int, default=5)
    parser.add_argument("--ref-intervals", nargs="+", type=int, default=[250, 500, 1000, 2000])
    parser.add_argument("--ref-check-multiplier", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if any(n <= 0 for n in args.n_eval):
        raise ValueError("--n-eval values must be positive.")
    if any(interval <= 0 for interval in args.ref_intervals):
        raise ValueError("--ref-intervals values must be positive.")
    if args.ref_check_multiplier <= 1:
        raise ValueError("--ref-check-multiplier must be greater than 1.")

    torch.set_num_threads(args.threads)
    all_rows = []
    for run_dir in args.run_dirs:
        all_rows.extend(evaluate_run(run_dir, args))

    output_dir = Path(args.output_dir)
    raw_path = output_dir / "metric_convergence_raw.csv"
    aggregate_path = output_dir / "metric_convergence_aggregate.csv"
    plot_path = output_dir / "metric_convergence.png"
    write_csv(raw_path, all_rows)
    aggregate = aggregate_rows(all_rows)
    write_csv(aggregate_path, aggregate)
    plot_audit(aggregate, args, plot_path)
    print_summary(aggregate)
    print(f"\nwrote {raw_path}")
    print(f"wrote {aggregate_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
