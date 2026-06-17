#!/usr/bin/env python3
"""Plot radial and commitment diagnostics for gm16 saved runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
import tempfile
from collections import defaultdict

_cache_dir = Path(tempfile.gettempdir()) / "lcfm_gm16_radial_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir / "xdg"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lcfm.plotting import load_run
from lcfm.utils import set_seed


LABELS = {
    "independent": "Independent",
    "minibatch_ot": "Minibatch OT",
    "pressure_aware_ot": "Pressure-aware OT",
    "iso_fd_w05": "Iso-FD w0.5",
}


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


def discover_runs(root, variants):
    root = Path(root)
    return {
        variant: sorted(root.glob(f"gm16_{variant}_seed*"))
        for variant in variants
    }


def endpoint_diagnostics(samples, centers, hit_radius):
    distances = torch.cdist(samples, centers)
    assignments = torch.argmin(distances, dim=1)
    assigned_centers = centers[assignments]
    center_norm = assigned_centers.norm(dim=1).clamp_min(1e-8)
    center_dirs = assigned_centers / center_norm[:, None]
    projection = (samples * center_dirs).sum(dim=1)
    progress = projection / center_norm
    orthogonal = (samples - projection[:, None] * center_dirs).norm(dim=1)
    nearest_distance = distances[torch.arange(samples.shape[0], device=samples.device), assignments]
    hit = nearest_distance <= hit_radius
    return {
        "endpoint_norm": samples.norm(dim=1),
        "nearest_distance": nearest_distance,
        "radial_projection": projection,
        "radial_progress": progress,
        "orthogonal_distance": orthogonal,
        "hit": hit.float(),
    }


def summarize_tensor(values):
    values = torch.cat(values).float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "p10": float(torch.quantile(values, 0.10).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
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


@torch.no_grad()
def collect(args):
    device = torch.device(args.device)
    runs_by_variant = discover_runs(args.root, args.variants)
    raw = []
    summary_values = defaultdict(list)
    target_values = defaultdict(list)
    target_done = False
    center_radius = None
    for variant, run_dirs in runs_by_variant.items():
        if not run_dirs:
            raise ValueError(f"No runs found for gm16_{variant}_seed* under {args.root}.")
        for run_dir in run_dirs:
            config, problem, model = load_run(run_dir, device)
            if problem.name != "gaussian_mixture_nd":
                raise ValueError(f"{run_dir} is not a gm16 run.")
            centers = problem.centers(device)
            center_radius = float(centers.norm(dim=1).mean().item())
            set_seed(args.eval_seed)
            x0 = problem.eval_initial(args.n_eval, device)
            target = problem.target_eval(args.n_eval, device)
            if not target_done:
                target_diag = endpoint_diagnostics(target, centers, args.hit_radius)
                for metric, values in target_diag.items():
                    target_values[metric].append(values.detach().cpu())
                target_done = True

            velocity = lambda x, t, model=model: model(x, t)
            endpoints = {
                f"Euler-{args.steps}": euler_endpoint(velocity, x0, args.steps),
                f"Heun-{args.heun_intervals}": heun_endpoint(velocity, x0, args.heun_intervals),
            }
            seed = config.get("seed", run_dir.name.rsplit("_seed", 1)[-1])
            for schedule, samples in endpoints.items():
                diag = endpoint_diagnostics(samples, centers, args.hit_radius)
                for metric, values in diag.items():
                    summary_values[(variant, schedule, metric)].append(values.detach().cpu())
                for metric in ("endpoint_norm", "nearest_distance", "radial_progress", "orthogonal_distance", "hit"):
                    values = diag[metric].detach().cpu()
                    raw.append(
                        {
                            "variant": variant,
                            "schedule": schedule,
                            "seed": seed,
                            "metric": metric,
                            "mean": float(values.mean().item()),
                            "std": float(values.std(unbiased=False).item()),
                            "p10": float(torch.quantile(values.float(), 0.10).item()),
                            "p50": float(torch.quantile(values.float(), 0.50).item()),
                            "p90": float(torch.quantile(values.float(), 0.90).item()),
                        }
                    )

    aggregate = []
    for (variant, schedule, metric), values in sorted(summary_values.items()):
        stats = summarize_tensor(values)
        aggregate.append({"variant": variant, "schedule": schedule, "metric": metric, **stats})
    target_aggregate = []
    for metric, values in sorted(target_values.items()):
        stats = summarize_tensor(values)
        target_aggregate.append({"variant": "target", "schedule": "target", "metric": metric, **stats})
    return raw, target_aggregate + aggregate, center_radius


def row_lookup(aggregate):
    return {(row["variant"], row["schedule"], row["metric"]): row for row in aggregate}


def plot(aggregate, variants, center_radius, output):
    lookup = row_lookup(aggregate)
    schedules = ["Euler-5", "Heun-500"]
    metrics = [
        ("radial_progress", "radial progress toward assigned center", (0.0, 1.15)),
        ("nearest_distance", "distance to nearest center", (0.0, 5.5)),
        ("orthogonal_distance", "orthogonal distance", (0.0, 3.0)),
        ("endpoint_norm", "endpoint norm", (0.0, max(8.0, center_radius + 1.0))),
    ]
    fig, axes = plt.subplots(len(metrics), len(schedules), figsize=(11.5, 10.5), squeeze=False)
    x = torch.arange(len(variants)).numpy()
    colors = {"Euler-5": "#2563eb", "Heun-500": "#16a34a"}

    for row_idx, (metric, title, ylim) in enumerate(metrics):
        target = lookup.get(("target", "target", metric))
        for col_idx, schedule in enumerate(schedules):
            ax = axes[row_idx][col_idx]
            means, p10s, p90s = [], [], []
            for variant in variants:
                row = lookup[(variant, schedule, metric)]
                means.append(row["mean"])
                p10s.append(row["p10"])
                p90s.append(row["p90"])
            lower = [m - lo for m, lo in zip(means, p10s)]
            upper = [hi - m for m, hi in zip(means, p90s)]
            ax.bar(x, means, yerr=[lower, upper], capsize=3, color=colors[schedule], alpha=0.80)
            if target is not None:
                ax.axhline(target["mean"], color="#111827", linestyle="--", linewidth=1.1, label="target mean")
            if metric == "endpoint_norm":
                ax.axhline(center_radius, color="#991b1b", linestyle=":", linewidth=1.1, label="center radius")
            ax.set_title(f"{title}\n{schedule}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([LABELS.get(v, v) for v in variants], rotation=22, ha="right", fontsize=8)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.22)
            if col_idx == 0:
                ax.set_ylabel(metric)
            if row_idx == 0 and col_idx == len(schedules) - 1:
                ax.legend(fontsize=8)

    fig.suptitle("gm16 Radial and Mode-Commitment Diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/phase1/pressure_aware_coupling_benchmark/runs")
    parser.add_argument("--variants", nargs="+", default=["independent", "minibatch_ot", "pressure_aware_ot", "iso_fd_w05"])
    parser.add_argument("--output-dir", default="results/phase1/pressure_aware_coupling_benchmark/diagnostics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-eval", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--heun-intervals", type=int, default=500)
    parser.add_argument("--hit-radius", type=float, default=1.6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    raw, aggregate, center_radius = collect(args)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "gm16_radial_diagnostics_raw.csv", raw)
    write_csv(output_dir / "gm16_radial_diagnostics_aggregate.csv", aggregate)
    plot(aggregate, args.variants, center_radius, output_dir / "gm16_radial_diagnostics.png")
    print(f"wrote {output_dir / 'gm16_radial_diagnostics.png'}")
    print(f"wrote {output_dir / 'gm16_radial_diagnostics_aggregate.csv'}")


if __name__ == "__main__":
    main()
