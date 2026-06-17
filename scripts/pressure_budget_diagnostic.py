#!/usr/bin/env python3
"""Pressure-budget diagnostics against the Gaussian-mixture oracle.

This is deliberately post-hoc: no training, just asking whether pressure-budget
violations separate good and bad trained models. The oracle is exact only for
independent Gaussian-mixture couplings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from collections import defaultdict

import torch

from lcfm.oracle import GaussianMixtureOracle
from lcfm.pairing import apply_pairing
from lcfm.plotting import load_run
from lcfm.schedules import kappa, rollout_error_profile
from lcfm.utils import set_seed


@torch.no_grad()
def model_material_fd(model, x, t, dt=1e-3):
    v = model(x, t)
    step = min(float(dt), max(1e-6, 1.0 - float(t[0].item())))
    t_next = torch.full_like(t, float(t[0].item()) + step)
    return (model(x + step * v, t_next) - v) / step


def safe_metric(numerator, denominator, eps=1e-12):
    return float(numerator / (denominator + eps))


@torch.no_grad()
def diagnose_run(run_dir, args):
    run_dir = Path(run_dir)
    config, problem, model = load_run(run_dir, torch.device("cpu"))
    pairing = config.get("pairing", "independent")
    if pairing != "independent" and not args.include_nonindependent:
        raise ValueError(f"{run_dir} has pairing={pairing}; pass --include-nonindependent to audit as reference only.")

    oracle = GaussianMixtureOracle.from_problem(problem)
    set_seed(args.seed)
    x0 = problem.eval_initial(args.n_probe, torch.device("cpu"))
    x1 = problem.target_eval(args.n_probe, torch.device("cpu"))
    x0, x1 = apply_pairing(x0, x1, config)

    ts = torch.linspace(0.0, args.t_max, args.grid_size)
    eps = args.eps
    totals = defaultdict(float)
    layer_totals = defaultdict(float)
    profiles = []

    for t_val in ts:
        t = torch.full((args.n_probe, 1), float(t_val))
        xt = (1 - t) * x0 + t * x1
        a_p = oracle.acceleration_target(xt, t)
        a_m = model_material_fd(model, xt, t, dt=args.fd_dt)

        p_norm = a_p.norm(dim=1, keepdim=True)
        m_norm = a_m.norm(dim=1, keepdim=True)
        p_hat = a_p / (p_norm + eps)
        alpha = (a_m * p_hat).sum(dim=1, keepdim=True)
        a_perp = a_m - alpha * p_hat

        orth = a_perp.pow(2).sum(dim=1, keepdim=True)
        opposite = torch.relu(-alpha).pow(2)
        excess = torch.relu(alpha - args.budget_c * p_norm).pow(2)
        pure_violation = orth + opposite + excess
        pressure_mismatch = (a_m - a_p).pow(2).sum(dim=1, keepdim=True)
        deficit = torch.relu(args.deficit_eta * p_norm - alpha).pow(2)
        model_energy = m_norm.pow(2)
        pressure_energy = p_norm.pow(2)
        pressure_weighted_util_num = alpha * p_norm
        cosine_num = alpha

        batch = {
            "pure_violation": pure_violation.sum().item(),
            "orth": orth.sum().item(),
            "opposite": opposite.sum().item(),
            "excess": excess.sum().item(),
            "pressure_mismatch": pressure_mismatch.sum().item(),
            "deficit": deficit.sum().item(),
            "model_energy": model_energy.sum().item(),
            "pressure_energy": pressure_energy.sum().item(),
            "util_num": pressure_weighted_util_num.sum().item(),
            "cos_num": cosine_num.sum().item(),
            "model_norm": m_norm.sum().item(),
            "pressure_norm": p_norm.sum().item(),
        }
        for key, value in batch.items():
            totals[key] += value
            if float(t_val) <= args.layer_t:
                layer_totals[key] += value

        profiles.append(
            {
                "run": run_dir.name,
                "group": run_dir.parent.name,
                "t": float(t_val),
                "mean_oracle_acceleration": float(p_norm.mean()),
                "mean_model_acceleration": float(m_norm.mean()),
                "pure_violation_pressure_norm": safe_metric(batch["pure_violation"], batch["pressure_energy"]),
                "pure_violation_model_norm": safe_metric(batch["pure_violation"], batch["model_energy"]),
                "pressure_utilization": safe_metric(batch["util_num"], batch["pressure_energy"]),
                "pressure_mismatch_norm": safe_metric(batch["pressure_mismatch"], batch["pressure_energy"]),
                "deficit_norm": safe_metric(batch["deficit"], batch["pressure_energy"]),
            }
        )

    e1_ts, e1_err = rollout_error_profile(model, problem.eval_initial(args.n_probe, torch.device("cpu")))
    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    summary = {
        "run": run_dir.name,
        "group": run_dir.parent.name,
        "method": config.get("method"),
        "pairing": pairing,
        "seed": config.get("seed"),
        "wasserstein": metrics.get("wasserstein"),
        "target_hit_rate": metrics.get("target_hit_rate"),
        "mode_hit_coverage": metrics.get("mode_hit_coverage"),
        "kappa_e1": kappa(e1_ts, e1_err),
        "pure_violation_pressure_norm": safe_metric(totals["pure_violation"], totals["pressure_energy"]),
        "pure_violation_model_norm": safe_metric(totals["pure_violation"], totals["model_energy"]),
        "orth_model_frac": safe_metric(totals["orth"], totals["model_energy"]),
        "opposite_model_frac": safe_metric(totals["opposite"], totals["model_energy"]),
        "excess_pressure_norm": safe_metric(totals["excess"], totals["pressure_energy"]),
        "pressure_utilization": safe_metric(totals["util_num"], totals["pressure_energy"]),
        "pressure_mismatch_norm": safe_metric(totals["pressure_mismatch"], totals["pressure_energy"]),
        "deficit_norm": safe_metric(totals["deficit"], totals["pressure_energy"]),
        "layer_pure_violation_pressure_norm": safe_metric(layer_totals["pure_violation"], layer_totals["pressure_energy"]),
        "layer_pressure_utilization": safe_metric(layer_totals["util_num"], layer_totals["pressure_energy"]),
        "layer_pressure_mismatch_norm": safe_metric(layer_totals["pressure_mismatch"], layer_totals["pressure_energy"]),
        "layer_deficit_norm": safe_metric(layer_totals["deficit"], layer_totals["pressure_energy"]),
    }
    return summary, profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output-dir", default="results/phase1/pressure_budget_diagnostic")
    parser.add_argument("--n-probe", type=int, default=2000)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--t-max", type=float, default=0.98)
    parser.add_argument("--layer-t", type=float, default=0.2)
    parser.add_argument("--fd-dt", type=float, default=1e-3)
    parser.add_argument("--budget-c", type=float, default=1.0)
    parser.add_argument("--deficit-eta", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--include-nonindependent", action="store_true")
    args = parser.parse_args()

    summaries = []
    profiles = []
    for run_dir in args.run_dirs:
        summary, run_profiles = diagnose_run(run_dir, args)
        summaries.append(summary)
        profiles.extend(run_profiles)
        print(
            f"{summary['run']}: W={summary['wasserstein']} hit={summary['target_hit_rate']} "
            f"pure={summary['pure_violation_pressure_norm']:.3f} "
            f"util={summary['pressure_utilization']:.3f} deficit={summary['deficit_norm']:.3f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    profile_path = output_dir / "profiles.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with profile_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(profiles[0].keys()))
        writer.writeheader()
        writer.writerows(profiles)

    grouped = defaultdict(list)
    for row in summaries:
        grouped[row["group"]].append(row)
    print(f"\nwrote {summary_path}")
    print(f"wrote {profile_path}")
    print(f"\n{'group':>28} {'W':>7} {'hit':>6} {'pure':>8} {'util':>8} {'deficit':>8} {'kE1':>6}")
    for group, rows in sorted(grouped.items()):
        def mean(key):
            vals = [row[key] for row in rows if row[key] is not None]
            return sum(float(v) for v in vals) / len(vals) if vals else float("nan")

        print(
            f"{group:>28} {mean('wasserstein'):7.3f} {mean('target_hit_rate'):6.3f} "
            f"{mean('pure_violation_pressure_norm'):8.3f} {mean('pressure_utilization'):8.3f} "
            f"{mean('deficit_norm'):8.3f} {mean('kappa_e1'):6.2f}"
        )


if __name__ == "__main__":
    main()
