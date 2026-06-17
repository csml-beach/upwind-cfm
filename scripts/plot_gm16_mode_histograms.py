#!/usr/bin/env python3
"""Plot nearest-mode and hit-mode histograms for gm16 saved runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
import tempfile
from collections import defaultdict

_cache_dir = Path(tempfile.gettempdir()) / "lcfm_gm16_hist_cache"
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


def mode_probs(samples, centers, hit_radius):
    distances = torch.cdist(samples, centers)
    assignments = torch.argmin(distances, dim=1)
    nearest = distances[torch.arange(samples.shape[0], device=samples.device), assignments]
    hits = nearest <= hit_radius
    n_modes = centers.shape[0]
    assignment_probs = torch.bincount(assignments, minlength=n_modes).float() / samples.shape[0]
    hit_probs = torch.bincount(assignments[hits], minlength=n_modes).float() / samples.shape[0]
    return assignment_probs.cpu(), hit_probs.cpu()


def discover_runs(root, variants):
    root = Path(root)
    return {
        variant: sorted(root.glob(f"gm16_{variant}_seed*"))
        for variant in variants
    }


def summarize(values):
    tensor = torch.stack(values)
    if tensor.shape[0] == 1:
        return tensor[0], torch.zeros_like(tensor[0])
    return tensor.mean(dim=0), tensor.std(dim=0, unbiased=False)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect(args):
    device = torch.device(args.device)
    runs_by_variant = discover_runs(args.root, args.variants)
    raw = []
    target_probs = None
    for variant, run_dirs in runs_by_variant.items():
        if not run_dirs:
            raise ValueError(f"No runs found for gm16_{variant}_seed* under {args.root}.")
        for run_dir in run_dirs:
            config, problem, model = load_run(run_dir, device)
            if problem.name != "gaussian_mixture_nd":
                raise ValueError(f"{run_dir} is not a gm16 run.")
            centers = problem.centers(device)
            set_seed(args.eval_seed)
            x0 = problem.eval_initial(args.n_eval, device)
            target = problem.target_eval(args.n_eval, device)
            if target_probs is None:
                target_probs = mode_probs(target, centers, args.hit_radius)[0]
            velocity = lambda x, t, model=model: model(x, t)
            endpoints = {
                f"Euler-{args.steps}": euler_endpoint(velocity, x0, args.steps),
                f"Heun-{args.heun_intervals}": heun_endpoint(velocity, x0, args.heun_intervals),
            }
            seed = config.get("seed", run_dir.name.rsplit("_seed", 1)[-1])
            for schedule, samples in endpoints.items():
                assignment, hit = mode_probs(samples, centers, args.hit_radius)
                for mode in range(problem.n_modes):
                    raw.append(
                        {
                            "variant": variant,
                            "schedule": schedule,
                            "seed": seed,
                            "mode": mode,
                            "assignment_prob": float(assignment[mode]),
                            "hit_prob": float(hit[mode]),
                            "target_prob": float(target_probs[mode]),
                        }
                    )
    return raw, target_probs


def aggregate(raw):
    grouped = defaultdict(lambda: {"assignment": [], "hit": [], "target": []})
    for row in raw:
        key = (row["variant"], row["schedule"], int(row["mode"]))
        grouped[key]["assignment"].append(torch.tensor(row["assignment_prob"]))
        grouped[key]["hit"].append(torch.tensor(row["hit_prob"]))
        grouped[key]["target"].append(torch.tensor(row["target_prob"]))
    rows = []
    for (variant, schedule, mode), values in sorted(grouped.items()):
        assignment_mean, assignment_std = summarize(values["assignment"])
        hit_mean, hit_std = summarize(values["hit"])
        target_mean, _ = summarize(values["target"])
        rows.append(
            {
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
    return rows


def plot(agg, variants, output):
    schedules = sorted({row["schedule"] for row in agg}, key=lambda name: (name.startswith("Heun"), name))
    n_modes = 1 + max(row["mode"] for row in agg)
    fig, axes = plt.subplots(len(schedules), len(variants), figsize=(4.2 * len(variants), 3.2 * len(schedules)), sharey=True)
    if len(schedules) == 1:
        axes = axes[None, :]
    if len(variants) == 1:
        axes = axes[:, None]

    colors = {"assignment": "#9ca3af", "hit": "#2563eb", "target": "#111827"}
    by_key = {(row["variant"], row["schedule"], row["mode"]): row for row in agg}
    x = torch.arange(n_modes).numpy()
    for row_idx, schedule in enumerate(schedules):
        for col_idx, variant in enumerate(variants):
            ax = axes[row_idx, col_idx]
            assignment = []
            assignment_err = []
            hit = []
            hit_err = []
            target = []
            for mode in range(n_modes):
                row = by_key[(variant, schedule, mode)]
                assignment.append(row["assignment_prob_mean"])
                assignment_err.append(row["assignment_prob_std"])
                hit.append(row["hit_prob_mean"])
                hit_err.append(row["hit_prob_std"])
                target.append(row["target_prob"])

            ax.bar(x, assignment, yerr=assignment_err, color=colors["assignment"], alpha=0.35, width=0.78, label="nearest")
            ax.bar(x, hit, yerr=hit_err, color=colors["hit"], alpha=0.82, width=0.48, label="hit")
            ax.plot(x, target, color=colors["target"], linestyle="--", linewidth=1.2, marker=".", markersize=4, label="target")
            ax.set_title(f"{LABELS.get(variant, variant)}\n{schedule}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in range(n_modes)], fontsize=8)
            ax.set_ylim(0.0, 0.34)
            ax.grid(axis="y", alpha=0.20)
            if col_idx == 0:
                ax.set_ylabel("probability")
            if row_idx == len(schedules) - 1:
                ax.set_xlabel("nearest mode")
            if row_idx == 0 and col_idx == len(variants) - 1:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("gm16 Mode Assignment Histograms: nearest assignment vs in-radius hit mass", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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
    raw, _ = collect(args)
    agg = aggregate(raw)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "gm16_mode_histograms_raw.csv", raw)
    write_csv(output_dir / "gm16_mode_histograms_aggregate.csv", agg)
    plot(agg, args.variants, output_dir / "gm16_mode_histograms.png")
    print(f"wrote {output_dir / 'gm16_mode_histograms.png'}")
    print(f"wrote {output_dir / 'gm16_mode_histograms_aggregate.csv'}")


if __name__ == "__main__":
    main()
