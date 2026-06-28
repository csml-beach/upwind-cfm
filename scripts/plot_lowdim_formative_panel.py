#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm import datasets  # noqa: F401
from lcfm.models import build_model
from lcfm.registry import DATASETS, get
from lcfm.schedules import equal_error_grid, euler_on_grid, power_time_grid
from lcfm.solvers import solve
from lcfm.utils import read_json, set_seed


PROBLEMS = [
    ("spiral", "spiral_standard_cfm_1782668226", "Spiral"),
    ("five_modes", "five_modes_standard_r6", "Five Modes"),
    ("fan_modes", "fan_modes_standard", "Fan Modes"),
]


def read_metrics(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def best_power_grid(metrics, nfe):
    candidates = []
    for row in metrics:
        if int(float(row["nfe"])) != nfe:
            continue
        if not row["sample_name"].startswith("power_"):
            continue
        candidates.append(row)
    best = min(candidates, key=lambda row: float(row["wasserstein"]))
    kind = best["power_kind"]
    rho = float(best["power_rho"])
    return best["sample_name"].replace(f"_nfe_{nfe}", ""), power_time_grid(nfe, rho=rho, kind=kind)


def load_run(run_dir, device):
    config = read_json(run_dir / "config.json")
    set_seed(int(config.get("seed", 0)))
    problem = get(DATASETS, config["dataset"])(config.get("dataset_kwargs", {}))
    model = build_model(config.get("model", "mlp"), problem.dim, config).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return config, problem, model


def axis_limits(*arrays, pad=0.08):
    data = torch.cat([arr[:, :2].detach().cpu() for arr in arrays], dim=0)
    lo = data.min(dim=0).values
    hi = data.max(dim=0).values
    span = (hi - lo).clamp_min(1e-3)
    return (float(lo[0] - pad * span[0]), float(hi[0] + pad * span[0])), (float(lo[1] - pad * span[1]), float(hi[1] + pad * span[1]))


def scatter_geometry(ax, x0, target, title):
    ax.scatter(x0[:, 0], x0[:, 1], s=5, c="#2b7bba", alpha=0.25, linewidths=0, label="prior")
    ax.scatter(target[:, 0], target[:, 1], s=5, c="#d95f02", alpha=0.35, linewidths=0, label="target")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.tick_params(labelsize=8)


def plot_profile(ax, profile, sctw_grid):
    ts = torch.tensor(profile["ts"], dtype=torch.float64)
    err = torch.tensor(profile["err"], dtype=torch.float64)
    density = (err + 1e-3).pow(0.5)
    ax.plot(ts, err / err.max().clamp_min(1e-12), color="#222222", lw=1.9, label="profile")
    ax.plot(ts, density / density.max().clamp_min(1e-12), color="#1b9e77", lw=1.7, ls="--", label="density")
    for knot in sctw_grid:
        ax.axvline(knot, color="#1b9e77", alpha=0.18, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("norm.", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)


def plot_knots(ax, grids):
    colors = ["#777777", "#2b7bba", "#1b9e77", "#d95f02"]
    labels = list(grids)
    y_positions = list(range(len(labels)))[::-1]
    for y, label, color in zip(y_positions, labels, colors):
        grid = grids[label]
        ax.hlines(y, 0, 1, color="#e3e3e3", lw=1)
        ax.vlines(grid, y - 0.27, y + 0.27, lw=2, color=color)
    ax.set_xlim(-0.01, 1.01)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("t", fontsize=8)
    ax.grid(True, axis="x", alpha=0.2)
    ax.tick_params(axis="x", labelsize=8)


def plot_endpoints(ax, target, uniform, sctw, w_uniform, w_sctw):
    ax.scatter(target[:, 0], target[:, 1], s=5, c="#bdbdbd", alpha=0.25, linewidths=0, label="target")
    ax.scatter(uniform[:, 0], uniform[:, 1], s=5, c="#d95f02", alpha=0.38, linewidths=0, label=f"uniform {w_uniform:.3f}")
    ax.scatter(sctw[:, 0], sctw[:, 1], s=5, c="#1b9e77", alpha=0.38, linewidths=0, label=f"SCTW {w_sctw:.3f}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.tick_params(labelsize=8)


def plot_problem(root, device, problem_key, run_name, title, args):
    run_dir = root / "runs" / run_name
    eval_dir = root / "evals" / run_name
    config, problem, model = load_run(run_dir, device)
    del config
    set_seed(args.eval_seed)
    x0 = problem.eval_initial(args.n_samples, device)
    target = problem.target_eval(args.n_samples, device)
    profile = read_json(eval_dir / "e1_profile.json")
    metrics = read_metrics(eval_dir / "metrics.csv")
    uniform_grid = [float(x) for x in torch.linspace(0.0, 1.0, args.nfe + 1).tolist()]
    sctw025 = equal_error_grid(profile["ts"], profile["err"], args.nfe, power=0.25, floor=1e-3, end=1.0)
    sctw05 = equal_error_grid(profile["ts"], profile["err"], args.nfe, power=0.5, floor=1e-3, end=1.0)
    best_name, best_grid = best_power_grid(metrics, args.nfe)
    uniform_samples = solve("euler", model, x0, {"steps": args.nfe})[-1].detach().cpu()
    sctw_samples = euler_on_grid(model, x0, sctw05)[-1].detach().cpu()
    x0_cpu = x0.detach().cpu()
    target_cpu = target.detach().cpu()
    xlim, ylim = axis_limits(x0_cpu, target_cpu, uniform_samples, sctw_samples)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    scatter_geometry(axes[0, 0], x0_cpu, target_cpu, f"{title}: prior and target")
    axes[0, 0].set_xlim(*xlim)
    axes[0, 0].set_ylim(*ylim)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")

    plot_profile(axes[0, 1], profile, sctw05)
    axes[0, 1].set_title("Self-curvature profile and SCTW p=.5 knots", fontsize=11)
    axes[0, 1].legend(frameon=False, fontsize=8, loc="upper right")

    grids = {
        "uniform": uniform_grid,
        "SCTW p=.25": sctw025,
        "SCTW p=.5": sctw05,
        best_name: best_grid,
    }
    plot_knots(axes[1, 0], grids)
    axes[1, 0].set_title(f"NFE {args.nfe} step allocation", fontsize=11)

    rows_by_name = {row["sample_name"]: row for row in metrics if int(float(row["nfe"])) == args.nfe}
    w_uniform = float(rows_by_name[f"uniform_nfe_{args.nfe}"]["wasserstein"])
    w_sctw = float(rows_by_name[f"e1_warped_p0p5_nfe_{args.nfe}"]["wasserstein"])
    plot_endpoints(axes[1, 1], target_cpu, uniform_samples, sctw_samples, w_uniform, w_sctw)
    axes[1, 1].set_xlim(*xlim)
    axes[1, 1].set_ylim(*ylim)
    axes[1, 1].set_title(f"NFE {args.nfe} endpoints", fontsize=11)

    fig.suptitle(f"{title} SCTW Diagnostic", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.out_dir) / f"{problem_key}_diagnostic_nfe{args.nfe}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Create formative low-dimensional SCTW diagnostic figures.")
    parser.add_argument("--root", default="results/lowdim_sctw_complementary")
    parser.add_argument("--problem", default="all", choices=["all"] + [key for key, _, _ in PROBLEMS])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--nfe", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--out-dir", default="results/lowdim_sctw_complementary/formative_panels")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(args.root)
    device = torch.device(args.device)
    selected = PROBLEMS if args.problem == "all" else [item for item in PROBLEMS if item[0] == args.problem]
    for problem_key, run_name, title in selected:
        print(plot_problem(root, device, problem_key, run_name, title, args))


if __name__ == "__main__":
    main()
