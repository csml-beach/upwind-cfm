#!/usr/bin/env python3
"""M3/X0/X1 evaluation: samplers at matched NFE on Phase-1 runs.

Per run: kappa from each available estimator (E0 oracle, E1 self-probe,
E2 head residual_log), then sampling under uniform and warped grids with
Euler (S1 global warp, S2 per-sample) and Heun (S3) at matched NFE.

Primary metric: integration error, mean ||x_end - x_ref_end|| against a
high-NFE uniform Heun reference from the same initial samples — the quantity
kappa actually predicts. Distribution metrics (W1/W2, hit, coverage) secondary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
from collections import defaultdict

import torch

from lcfm.dispersion import DispersionMLP, head_is_positive, make_head_accel_fn
from lcfm.metrics import mode_statistics, wasserstein_match
from lcfm.oracle import GaussianMixtureOracle
from lcfm.pairing import apply_pairing
from lcfm.plotting import load_run
from lcfm.schedules import (
    equal_error_grid,
    euler_on_grid,
    euler_per_sample,
    heun_on_grid,
    interpolant_error_profile,
    kappa,
    rollout_error_profile,
)
from lcfm.utils import set_seed


def load_head(run_dir, dim, mode, device):
    head = DispersionMLP(dim, positive=head_is_positive(mode)).to(device)
    head.load_state_dict(torch.load(Path(run_dir) / f"head_{mode}.pt", map_location=device, weights_only=True))
    head.eval()
    return head


def evaluate_endpoint(problem, config, x_end, target, x_ref_end, reference_self_error=None):
    eval_cfg = config.get("eval", {})
    metrics = {
        "integration_error": float((x_end - x_ref_end).norm(dim=1).mean().item()),
        "wasserstein": wasserstein_match(x_end, target),
        "wasserstein2": wasserstein_match(x_end, target, p=2),
        "reference_self_error": reference_self_error,
    }
    if hasattr(problem, "centers") and hasattr(problem, "sigma_mode"):
        metrics.update(
            {
                k: v
                for k, v in mode_statistics(
                    x_end,
                    problem.centers(x_end.device),
                    p_min=eval_cfg.get("mode_p_min", 0.05),
                    hit_radius=eval_cfg.get("hit_radius", 3.0 * problem.sigma_mode),
                ).items()
                if k in {"mode_hit_coverage", "target_hit_rate"}
            }
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--head-mode", default="residual_log")
    parser.add_argument("--ref-intervals", type=int, default=1000)
    parser.add_argument(
        "--ref-check-intervals",
        type=int,
        default=0,
        help="Optional finer Heun rollout used only to estimate reference error.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    k = args.steps
    rows = []
    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        if not (run_dir / "model.pt").exists():
            continue
        config, problem, model = load_run(run_dir, device)
        head = load_head(run_dir, problem.dim, args.head_mode, device)
        accel = make_head_accel_fn(head, model, problem, device, mode=args.head_mode)

        set_seed(7)
        px0 = problem.eval_initial(2000, device)
        px1 = problem.target_eval(2000, device)
        px0, px1 = apply_pairing(px0, px1, config)
        e2 = interpolant_error_profile(accel, px0, px1)
        set_seed(11)
        e1 = rollout_error_profile(model, problem.eval_initial(2000, device))
        kappas = {"kappa_e1": kappa(*e1), "kappa_e2": kappa(*e2)}
        try:
            oracle = GaussianMixtureOracle.from_problem(problem)
            e0 = interpolant_error_profile(oracle.acceleration_target, px0, px1)
            kappas["kappa_e0"] = kappa(*e0)
        except ValueError:
            kappas["kappa_e0"] = None

        eval_cfg = config.get("eval", {})
        set_seed(eval_cfg.get("eval_seed", 1234))
        x0 = problem.eval_initial(eval_cfg.get("n_eval", 1000), device)
        target = problem.target_eval(eval_cfg.get("n_eval", 1000), device)
        ref_grid = torch.linspace(0, 1, args.ref_intervals + 1).tolist()
        x_ref = heun_on_grid(model, x0, ref_grid)[-1]
        reference_self_error = None
        if args.ref_check_intervals:
            if args.ref_check_intervals <= args.ref_intervals:
                raise ValueError("--ref-check-intervals must be larger than --ref-intervals.")
            check_grid = torch.linspace(0, 1, args.ref_check_intervals + 1).tolist()
            x_ref_check = heun_on_grid(model, x0, check_grid)[-1]
            reference_self_error = float((x_ref - x_ref_check).norm(dim=1).mean().item())

        grid_e2 = equal_error_grid(*e2, k, end=1.0)
        grid_e1 = equal_error_grid(*e1, k)
        schedules = {
            f"euler{k}_uniform": ("euler", torch.linspace(0, 1, k + 1).tolist()),
            f"euler{2*k}_uniform": ("euler", torch.linspace(0, 1, 2 * k + 1).tolist()),
            f"euler{k}_warp_e2": ("euler", grid_e2),
            f"euler{k}_warp_e1": ("euler", grid_e1),
            f"euler{k}_persample_e2": ("persample", grid_e2),
            f"heun{k//2}_uniform": ("heun", torch.linspace(0, 1, k // 2 + 1).tolist()),
            f"heun{k//2}_warp_e2": ("heun", equal_error_grid(*e2, k // 2, end=1.0)),
        }
        for name, (kind, grid) in schedules.items():
            if kind == "euler":
                x_end = euler_on_grid(model, x0, grid)[-1]
                nfe = len(grid) - 1
            elif kind == "heun":
                x_end = heun_on_grid(model, x0, grid)[-1]
                nfe = 2 * (len(grid) - 1)
            else:
                traj, _ = euler_per_sample(model, x0, grid, e2[0], e2[1], accel)
                x_end = traj[-1]
                nfe = len(grid) - 1
            metrics = evaluate_endpoint(problem, config, x_end, target, x_ref, reference_self_error)
            rows.append(
                {
                    "run": run_dir.name,
                    "group": run_dir.parent.name,
                    "seed": config.get("seed"),
                    "schedule": name,
                    "nfe": nfe,
                    **kappas,
                    **metrics,
                }
            )
        print(f"{run_dir.name}: kappas " + " ".join(f"{m}={v:.2f}" for m, v in kappas.items() if v))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["schedule"])].append(row)
    print(f"\n{'group':>24} {'schedule':>22} {'nfe':>4} {'int_err':>17} {'W':>15} {'cov':>5}")
    for (group, schedule), items in sorted(grouped.items()):
        ints = torch.tensor([r["integration_error"] for r in items])
        ws = torch.tensor([r["wasserstein"] for r in items])
        covs = [r.get("mode_hit_coverage") for r in items if r.get("mode_hit_coverage") is not None]
        cov_str = f"{sum(covs)/len(covs):4.1f}" if covs else "   -"
        print(
            f"{group:>24} {schedule:>22} {items[0]['nfe']:>4} "
            f"{ints.mean():9.4f}+/-{ints.std(unbiased=False):.4f} "
            f"{ws.mean():8.4f}+/-{ws.std(unbiased=False):.4f} {cov_str}"
        )


if __name__ == "__main__":
    main()
