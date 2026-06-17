#!/usr/bin/env python3
"""Audit whether a trained Gaussian-mixture CFM model learns the oracle pressure layer.

For independent Gaussian-mixture problems, the oracle gives the exact marginal
velocity v(x,t) and material acceleration Dv/Dt. This script compares a
trained model against that oracle on interpolant probe points x_t.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-upwind-cfm")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lcfm.oracle import GaussianMixtureOracle
from lcfm.pairing import apply_pairing
from lcfm.plotting import load_run
from lcfm.schedules import kappa, rollout_error_profile
from lcfm.utils import set_seed


def layer_fraction(ts, err, t_layer=0.2):
    ts = torch.as_tensor(ts, dtype=torch.float64)
    err = torch.as_tensor(err, dtype=torch.float64)
    mask = ts <= t_layer
    if mask.sum() < 2:
        return float("nan")
    total = torch.trapezoid(err, ts)
    layer = torch.trapezoid(err[mask], ts[mask])
    return float(layer / (total + 1e-12))


@torch.no_grad()
def model_material_fd(model, x, t, dt=1e-3):
    """Forward finite-difference approximation of Dv_theta/Dt along v_theta."""
    v = model(x, t)
    step = min(float(dt), max(1e-6, 1.0 - float(t[0].item())))
    t_next = torch.full_like(t, float(t[0].item()) + step)
    v_next = model(x + step * v, t_next)
    return (v_next - v) / step


@torch.no_grad()
def audit_run(run_dir, args):
    run_dir = Path(run_dir)
    config, problem, model = load_run(run_dir, torch.device("cpu"))
    oracle = GaussianMixtureOracle.from_problem(problem)

    set_seed(args.seed)
    x0 = problem.eval_initial(args.n_probe, torch.device("cpu"))
    x1 = problem.target_eval(args.n_probe, torch.device("cpu"))
    if config.get("pairing", "independent") != "independent" and not args.include_nonindependent:
        raise ValueError(
            f"{run_dir} uses pairing={config.get('pairing')}; "
            "the GaussianMixtureOracle is exact only for independent coupling. "
            "Pass --include-nonindependent to treat E0 as a reference only."
        )
    x0, x1 = apply_pairing(x0, x1, config)

    ts = torch.linspace(0.0, args.t_max, args.grid_size)
    rows = []
    e0_profile = []
    model_fd_profile = []
    rel_rmse_profile = []
    cosine_profile = []

    for t_val in ts:
        t = torch.full((args.n_probe, 1), float(t_val))
        xt = (1 - t) * x0 + t * x1
        v_oracle = oracle.velocity(xt, t)
        v_model = model(xt, t)
        diff = v_model - v_oracle
        rmse = diff.pow(2).sum(dim=1).mean().sqrt()
        oracle_norm = v_oracle.norm(dim=1).mean()
        rel_rmse = rmse / (oracle_norm + 1e-8)
        cosine = torch.nn.functional.cosine_similarity(v_model, v_oracle, dim=1, eps=1e-8).mean()

        a_oracle = oracle.acceleration_target(xt, t)
        a_model = model_material_fd(model, xt, t, dt=args.fd_dt)
        e0 = a_oracle.norm(dim=1).mean()
        model_fd = a_model.norm(dim=1).mean()
        accel_ratio = model_fd / (e0 + 1e-8)

        row = {
            "run": run_dir.name,
            "group": run_dir.parent.name,
            "seed": config.get("seed"),
            "pairing": config.get("pairing", "independent"),
            "t": float(t_val),
            "velocity_rmse": float(rmse),
            "velocity_relative_rmse": float(rel_rmse),
            "velocity_cosine": float(cosine),
            "oracle_acceleration": float(e0),
            "model_fd_acceleration": float(model_fd),
            "model_to_oracle_acceleration_ratio": float(accel_ratio),
        }
        rows.append(row)
        e0_profile.append(float(e0))
        model_fd_profile.append(float(model_fd))
        rel_rmse_profile.append(float(rel_rmse))
        cosine_profile.append(float(cosine))

    set_seed(args.seed + 1)
    e1_ts, e1_err = rollout_error_profile(model, problem.eval_initial(args.n_probe, torch.device("cpu")))
    ts_f64 = ts.to(torch.float64)
    e0_tensor = torch.tensor(e0_profile, dtype=torch.float64)
    model_tensor = torch.tensor(model_fd_profile, dtype=torch.float64)
    rel_tensor = torch.tensor(rel_rmse_profile, dtype=torch.float64)
    cos_tensor = torch.tensor(cosine_profile, dtype=torch.float64)
    layer_mask = ts <= args.layer_t
    tail_mask = ts >= args.tail_t

    summary = {
        "run": run_dir.name,
        "group": run_dir.parent.name,
        "seed": config.get("seed"),
        "pairing": config.get("pairing", "independent"),
        "kappa_e0_interpolant": kappa(ts_f64, e0_tensor),
        "kappa_model_fd_interpolant": kappa(ts_f64, model_tensor),
        "kappa_e1_rollout": kappa(e1_ts, e1_err),
        "layer_frac_e0": layer_fraction(ts_f64, e0_tensor, args.layer_t),
        "layer_frac_model_fd": layer_fraction(ts_f64, model_tensor, args.layer_t),
        "layer_frac_e1": layer_fraction(e1_ts, e1_err, args.layer_t),
        "mean_rel_rmse": float(rel_tensor.mean()),
        "layer_rel_rmse": float(rel_tensor[layer_mask].mean()),
        "tail_rel_rmse": float(rel_tensor[tail_mask].mean()),
        "mean_cosine": float(cos_tensor.mean()),
        "layer_cosine": float(cos_tensor[layer_mask].mean()),
        "tail_cosine": float(cos_tensor[tail_mask].mean()),
        "mean_accel_ratio": float((model_tensor / (e0_tensor + 1e-8)).mean()),
        "layer_accel_ratio": float((model_tensor[layer_mask] / (e0_tensor[layer_mask] + 1e-8)).mean()),
        "tail_accel_ratio": float((model_tensor[tail_mask] / (e0_tensor[tail_mask] + 1e-8)).mean()),
    }
    return config, rows, summary, (ts, e0_tensor, model_tensor, rel_tensor, cos_tensor, e1_ts, e1_err)


def save_run_plot(output_dir, run_name, profile):
    ts, e0, model_fd, rel_rmse, cosine, e1_ts, e1_err = profile
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=False)
    axes[0].semilogy(ts, e0 + 1e-3, color="#111827", linewidth=2.0, label="E0 oracle on interpolants")
    axes[0].semilogy(ts, model_fd + 1e-3, color="#2563eb", linewidth=1.8, label="model FD on interpolants")
    axes[0].semilogy(e1_ts, e1_err + 1e-3, color="#f59e0b", linewidth=1.4, linestyle="--", label="E1 rollout")
    axes[0].set_ylabel("mean material acceleration")
    axes[0].set_title(run_name)
    axes[0].legend(fontsize=8)

    axes[1].plot(ts, rel_rmse, color="#dc2626", linewidth=1.8, label="relative velocity RMSE")
    axes[1].plot(ts, cosine, color="#16a34a", linewidth=1.8, label="velocity cosine")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("velocity fidelity")
    axes[1].set_ylim(bottom=min(-0.05, float(cosine.min()) - 0.05))
    axes[1].legend(fontsize=8)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{run_name}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output-dir", default="results/phase1/oracle_model_audit")
    parser.add_argument("--n-probe", type=int, default=2000)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--t-max", type=float, default=0.98)
    parser.add_argument("--fd-dt", type=float, default=1e-3)
    parser.add_argument("--layer-t", type=float, default=0.2)
    parser.add_argument("--tail-t", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--include-nonindependent", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_rows = []
    summaries = []
    for run_dir in args.run_dirs:
        config, rows, summary, profile = audit_run(run_dir, args)
        all_rows.extend(rows)
        summaries.append(summary)
        save_run_plot(output_dir, summary["run"], profile)
        print(
            f"{summary['run']}: kappa E0={summary['kappa_e0_interpolant']:.2f} "
            f"modelFD={summary['kappa_model_fd_interpolant']:.2f} "
            f"E1={summary['kappa_e1_rollout']:.2f} "
            f"layer relRMSE={summary['layer_rel_rmse']:.2f} "
            f"layer accel ratio={summary['layer_accel_ratio']:.2f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    profile_path = output_dir / "profiles.csv"
    with profile_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    grouped = defaultdict(list)
    for row in summaries:
        grouped[row["group"]].append(row)

    print(f"\nwrote {summary_path}")
    print(f"wrote {profile_path}")
    print(
        f"\n{'group':>24} {'E0 kappa':>9} {'model kappa':>12} {'E1 kappa':>9} "
        f"{'layer rmse':>11} {'layer accel':>12}"
    )
    for group, items in sorted(grouped.items()):
        def mean(key):
            return sum(float(item[key]) for item in items) / len(items)

        print(
            f"{group:>24} {mean('kappa_e0_interpolant'):9.2f} "
            f"{mean('kappa_model_fd_interpolant'):12.2f} "
            f"{mean('kappa_e1_rollout'):9.2f} "
            f"{mean('layer_rel_rmse'):11.2f} "
            f"{mean('layer_accel_ratio'):12.2f}"
        )


if __name__ == "__main__":
    main()
