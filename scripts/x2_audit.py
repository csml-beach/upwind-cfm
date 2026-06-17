#!/usr/bin/env python3
"""X2b: the audit. Compare each model's self-probed curvature profile (E1, what
the field actually does) against the data-side profile (E2 head / E0 oracle,
what the data says the curvature must be).

A healthy model's E1 tracks the data-side shape; an over-straightened model
self-reports smooth while the data-side profile still shows the commitment
layer — the disagreement flags missing modes without generating a sample.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lcfm.dispersion import DispersionMLP, head_is_positive, make_head_accel_fn
from lcfm.models import build_model
from lcfm.oracle import GaussianMixtureOracle
from lcfm.pairing import apply_pairing
from lcfm.plotting import load_run
from lcfm.schedules import interpolant_error_profile, kappa, rollout_error_profile
from lcfm.utils import read_json, set_seed


def layer_fraction(ts, err, t_layer=0.2):
    ts = torch.as_tensor(ts, dtype=torch.float64)
    err = torch.as_tensor(err, dtype=torch.float64)
    mask = ts <= t_layer
    total = torch.trapezoid(err, ts)
    layer = torch.trapezoid(err[mask], ts[mask])
    return float(layer / total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-run", required=True, help="Phase-1 run providing problem + E2 head.")
    parser.add_argument("--audit-runs", nargs="+", required=True)
    parser.add_argument("--head-mode", default="residual_log")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    config, problem, healthy_model = load_run(args.problem_run, device)
    head = DispersionMLP(problem.dim, positive=head_is_positive(args.head_mode)).to(device)
    head.load_state_dict(
        torch.load(Path(args.problem_run) / f"head_{args.head_mode}.pt", map_location=device, weights_only=True)
    )
    head.eval()

    set_seed(7)
    x0 = problem.eval_initial(2000, device)
    x1 = problem.target_eval(2000, device)
    x0, x1 = apply_pairing(x0, x1, config)
    oracle = GaussianMixtureOracle.from_problem(problem)
    e0 = interpolant_error_profile(oracle.acceleration_target, x0, x1)
    accel = make_head_accel_fn(head, healthy_model, problem, device, mode=args.head_mode)
    e2 = interpolant_error_profile(accel, x0, x1)

    print(f"data-side: kappa_E0={kappa(*e0):.2f} layer_frac_E0={layer_fraction(*e0):.2f} | "
          f"kappa_E2={kappa(*e2):.2f} layer_frac_E2={layer_fraction(*e2):.2f}")
    print(f"\n{'audited run':>42} {'kappa_E1':>8} {'layer_E1':>8} {'coverage':>8} {'hit':>6}")

    groups = {}
    for run_dir in args.audit_runs:
        run_dir = Path(run_dir)
        run_config = read_json(run_dir / "config.json")
        model = build_model(run_config.get("model", "mlp"), problem.dim, run_config).to(device)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
        model.eval()
        set_seed(11)
        probe_x0 = problem.eval_initial(2000, device)
        e1 = rollout_error_profile(model, probe_x0)
        metrics = {}
        if (run_dir / "metrics.json").exists():
            metrics = json.loads((run_dir / "metrics.json").read_text())
        group = run_dir.parent.name
        groups.setdefault(group, []).append(e1)
        print(
            f"{run_dir.name:>42} {kappa(*e1):8.2f} {layer_fraction(*e1):8.2f} "
            f"{metrics.get('mode_hit_coverage', float('nan')):>8} {metrics.get('target_hit_rate', float('nan')):6.2f}"
        )

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.5), squeeze=False)
    for ax, (group, profiles) in zip(axes[0], groups.items()):
        for i, (ts, err) in enumerate(profiles):
            ax.semilogy(ts, err + 1e-3, color="#2563eb", alpha=0.5,
                        label="E1 self-probe (model)" if i == 0 else None)
        ax.semilogy(e0[0], e0[1] + 1e-3, color="#111827", linewidth=2, label="E0 oracle (data)")
        ax.semilogy(e2[0], e2[1] + 1e-3, color="#dc2626", linewidth=2, linestyle="--", label="E2 head (data)")
        ax.set_title(f"{group}: model vs data curvature")
        ax.set_xlabel("t")
        ax.set_ylabel("mean ||Dv/Dt||")
        ax.legend()
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
