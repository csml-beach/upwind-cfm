#!/usr/bin/env python3
"""Plot mode commitment diagnostics for saved mode-mixture runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
import tempfile
from collections import defaultdict

_cache_dir = Path(tempfile.gettempdir()) / "lcfm_mode_commitment_cache"
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
    "iso_fd_w01": "Iso-FD w0.1",
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


def discover_runs(root, geometry, variants):
    root = Path(root)
    return {variant: sorted(root.glob(f"{geometry}_{variant}_seed*")) for variant in variants}


def source_mean(problem, device):
    if hasattr(problem, "source_mean"):
        return problem.source_mean.to(device)
    return torch.zeros(problem.dim, device=device)


def endpoint_diagnostics(samples, centers, origin, hit_radius):
    distances = torch.cdist(samples, centers)
    assignments = torch.argmin(distances, dim=1)
    assigned_centers = centers[assignments]

    transport = assigned_centers - origin[None, :]
    transport_length = transport.norm(dim=1).clamp_min(1e-8)
    transport_dirs = transport / transport_length[:, None]
    displacement = samples - origin[None, :]
    projection = (displacement * transport_dirs).sum(dim=1)
    progress = projection / transport_length
    orthogonal = (displacement - projection[:, None] * transport_dirs).norm(dim=1)
    nearest_distance = distances[torch.arange(samples.shape[0], device=samples.device), assignments]
    hit = nearest_distance <= hit_radius
    return {
        "endpoint_norm": displacement.norm(dim=1),
        "nearest_distance": nearest_distance,
        "radial_projection": projection,
        "radial_progress": progress,
        "orthogonal_distance": orthogonal,
        "hit": hit.float(),
    }


def mode_probs(samples, centers, hit_radius):
    distances = torch.cdist(samples, centers)
    assignments = torch.argmin(distances, dim=1)
    nearest = distances[torch.arange(samples.shape[0], device=samples.device), assignments]
    hits = nearest <= hit_radius
    n_modes = centers.shape[0]
    assignment_probs = torch.bincount(assignments, minlength=n_modes).float() / samples.shape[0]
    hit_probs = torch.bincount(assignments[hits], minlength=n_modes).float() / samples.shape[0]
    return assignment_probs.cpu(), hit_probs.cpu()


def summarize_tensor(values):
    values = torch.cat(values).float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "p10": float(torch.quantile(values, 0.10).item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
    }


def summarize_stack(values):
    tensor = torch.stack(values)
    if tensor.shape[0] == 1:
        return tensor[0], torch.zeros_like(tensor[0])
    return tensor.mean(dim=0), tensor.std(dim=0, unbiased=False)


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
    runs_by_variant = discover_runs(args.root, args.geometry, args.variants)
    raw = []
    summary_values = defaultdict(list)
    hist_raw = []
    target_values = defaultdict(list)
    target_probs = None
    target_done = False
    center_lengths = None

    for variant, run_dirs in runs_by_variant.items():
        if not run_dirs:
            raise ValueError(f"No runs found for {args.geometry}_{variant}_seed* under {args.root}.")
        for run_dir in run_dirs:
            config, problem, model = load_run(run_dir, device)
            if not hasattr(problem, "centers"):
                raise ValueError(f"{run_dir} does not expose mode centers.")
            centers = problem.centers(device)
            origin = source_mean(problem, device)
            center_lengths = (centers - origin[None, :]).norm(dim=1).detach().cpu()

            set_seed(args.eval_seed)
            x0 = problem.eval_initial(args.n_eval, device)
            target = problem.target_eval(args.n_eval, device)
            if not target_done:
                target_diag = endpoint_diagnostics(target, centers, origin, args.hit_radius)
                for metric, values in target_diag.items():
                    target_values[metric].append(values.detach().cpu())
                target_probs = mode_probs(target, centers, args.hit_radius)[0]
                target_done = True

            velocity = lambda x, t, model=model: model(x, t)
            endpoints = {
                f"Euler-{args.steps}": euler_endpoint(velocity, x0, args.steps),
                f"Heun-{args.heun_intervals}": heun_endpoint(velocity, x0, args.heun_intervals),
            }
            seed = config.get("seed", run_dir.name.rsplit("_seed", 1)[-1])
            for schedule, samples in endpoints.items():
                diag = endpoint_diagnostics(samples, centers, origin, args.hit_radius)
                for metric, values in diag.items():
                    summary_values[(variant, schedule, metric)].append(values.detach().cpu())
                for metric in ("endpoint_norm", "nearest_distance", "radial_progress", "orthogonal_distance", "hit"):
                    values = diag[metric].detach().cpu()
                    raw.append(
                        {
                            "geometry": args.geometry,
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

                assignment, hit = mode_probs(samples, centers, args.hit_radius)
                for mode in range(centers.shape[0]):
                    hist_raw.append(
                        {
                            "geometry": args.geometry,
                            "variant": variant,
                            "schedule": schedule,
                            "seed": seed,
                            "mode": mode,
                            "assignment_prob": float(assignment[mode]),
                            "hit_prob": float(hit[mode]),
                            "target_prob": float(target_probs[mode]),
                        }
                    )

    aggregate = []
    for metric, values in sorted(target_values.items()):
        aggregate.append({"geometry": args.geometry, "variant": "target", "schedule": "target", "metric": metric, **summarize_tensor(values)})
    for (variant, schedule, metric), values in sorted(summary_values.items()):
        aggregate.append({"geometry": args.geometry, "variant": variant, "schedule": schedule, "metric": metric, **summarize_tensor(values)})

    hist_grouped = defaultdict(lambda: {"assignment": [], "hit": [], "target": []})
    for row in hist_raw:
        key = (row["variant"], row["schedule"], int(row["mode"]))
        hist_grouped[key]["assignment"].append(torch.tensor(row["assignment_prob"]))
        hist_grouped[key]["hit"].append(torch.tensor(row["hit_prob"]))
        hist_grouped[key]["target"].append(torch.tensor(row["target_prob"]))
    hist_aggregate = []
    for (variant, schedule, mode), values in sorted(hist_grouped.items()):
        assignment_mean, assignment_std = summarize_stack(values["assignment"])
        hit_mean, hit_std = summarize_stack(values["hit"])
        target_mean, _ = summarize_stack(values["target"])
        hist_aggregate.append(
            {
                "geometry": args.geometry,
                "variant": variant,
                "schedule": schedule,
                "mode": mode,
                "assignment_prob_mean": float(assignment_mean.item()),
                "assignment_prob_std": float(assignment_std.item()),
                "hit_prob_mean": float(hit_mean.item()),
                "hit_prob_std": float(hit_std.item()),
                "target_prob": float(target_mean.item()),
            }
        )
    return raw, aggregate, hist_raw, hist_aggregate, center_lengths


def row_lookup(aggregate):
    return {(row["variant"], row["schedule"], row["metric"]): row for row in aggregate}


def plot_radial(aggregate, variants, center_lengths, output, title):
    lookup = row_lookup(aggregate)
    schedules = ["Euler-5", "Heun-500"]
    metrics = [
        ("radial_progress", "progress toward assigned center", 1.25),
        ("nearest_distance", "distance to nearest center", 1.0),
        ("orthogonal_distance", "orthogonal distance from source-center ray", 1.0),
        ("endpoint_norm", "distance from source mean", 1.0),
    ]
    fig, axes = plt.subplots(len(metrics), len(schedules), figsize=(11.5, 10.5), squeeze=False)
    x = torch.arange(len(variants)).numpy()
    colors = {"Euler-5": "#2563eb", "Heun-500": "#16a34a"}

    for row_idx, (metric, metric_title, min_top) in enumerate(metrics):
        target = lookup.get(("target", "target", metric))
        metric_rows = [row for row in aggregate if row["metric"] == metric]
        max_p90 = max(row["p90"] for row in metric_rows)
        ylim = (0.0, max(min_top, max_p90 * 1.12))
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
            if metric == "endpoint_norm" and center_lengths is not None:
                ax.axhline(float(center_lengths.mean().item()), color="#991b1b", linestyle=":", linewidth=1.1, label="mean center distance")
            ax.set_title(f"{metric_title}\n{schedule}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([LABELS.get(v, v) for v in variants], rotation=22, ha="right", fontsize=8)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.22)
            if col_idx == 0:
                ax.set_ylabel(metric)
            if row_idx == 0 and col_idx == len(schedules) - 1:
                ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_histograms(hist_aggregate, variants, output, title):
    schedules = sorted({row["schedule"] for row in hist_aggregate}, key=lambda name: (name.startswith("Heun"), name))
    n_modes = 1 + max(row["mode"] for row in hist_aggregate)
    fig, axes = plt.subplots(len(schedules), len(variants), figsize=(4.2 * len(variants), 3.2 * len(schedules)), sharey=True)
    if len(schedules) == 1:
        axes = axes[None, :]
    if len(variants) == 1:
        axes = axes[:, None]

    by_key = {(row["variant"], row["schedule"], row["mode"]): row for row in hist_aggregate}
    x = torch.arange(n_modes).numpy()
    for row_idx, schedule in enumerate(schedules):
        for col_idx, variant in enumerate(variants):
            ax = axes[row_idx, col_idx]
            assignment, assignment_err, hit, hit_err, target = [], [], [], [], []
            for mode in range(n_modes):
                row = by_key[(variant, schedule, mode)]
                assignment.append(row["assignment_prob_mean"])
                assignment_err.append(row["assignment_prob_std"])
                hit.append(row["hit_prob_mean"])
                hit_err.append(row["hit_prob_std"])
                target.append(row["target_prob"])
            ax.bar(x, assignment, yerr=assignment_err, color="#9ca3af", alpha=0.35, width=0.78, label="nearest")
            ax.bar(x, hit, yerr=hit_err, color="#2563eb", alpha=0.82, width=0.48, label="hit")
            ax.plot(x, target, color="#111827", linestyle="--", linewidth=1.2, marker=".", markersize=4, label="target")
            ax.set_title(f"{LABELS.get(variant, variant)}\n{schedule}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in range(n_modes)], fontsize=8)
            ax.set_ylim(0.0, max(0.34, max(target + assignment + hit) * 1.2))
            ax.grid(axis="y", alpha=0.20)
            if col_idx == 0:
                ax.set_ylabel("probability")
            if row_idx == len(schedules) - 1:
                ax.set_xlabel("nearest mode")
            if row_idx == 0 and col_idx == len(variants) - 1:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/phase1/pressure_aware_coupling_benchmark/runs")
    parser.add_argument("--geometry", default="staged")
    parser.add_argument("--variants", nargs="+", default=["independent", "minibatch_ot", "pressure_aware_ot", "iso_fd_w05"])
    parser.add_argument("--output-dir", default="results/phase1/pressure_aware_coupling_benchmark/diagnostics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-eval", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--heun-intervals", type=int, default=500)
    parser.add_argument("--hit-radius", type=float, default=0.6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    raw, aggregate, hist_raw, hist_aggregate, center_lengths = collect(args)
    output_dir = Path(args.output_dir)
    prefix = args.geometry
    write_csv(output_dir / f"{prefix}_radial_diagnostics_raw.csv", raw)
    write_csv(output_dir / f"{prefix}_radial_diagnostics_aggregate.csv", aggregate)
    write_csv(output_dir / f"{prefix}_mode_histograms_raw.csv", hist_raw)
    write_csv(output_dir / f"{prefix}_mode_histograms_aggregate.csv", hist_aggregate)
    plot_radial(
        aggregate,
        args.variants,
        center_lengths,
        output_dir / f"{prefix}_radial_diagnostics.png",
        f"{args.geometry} radial and mode-commitment diagnostics",
    )
    plot_histograms(
        hist_aggregate,
        args.variants,
        output_dir / f"{prefix}_mode_histograms.png",
        f"{args.geometry} mode assignment histograms",
    )
    print(f"wrote {output_dir / f'{prefix}_radial_diagnostics.png'}")
    print(f"wrote {output_dir / f'{prefix}_radial_diagnostics_aggregate.csv'}")
    print(f"wrote {output_dir / f'{prefix}_mode_histograms.png'}")
    print(f"wrote {output_dir / f'{prefix}_mode_histograms_aggregate.csv'}")


if __name__ == "__main__":
    main()
