#!/usr/bin/env python3
"""Oracle-assisted pressure-in-training viability tests.

This script is intentionally separate from the normal method registry. The
losses here use the Gaussian-mixture oracle, so they are research probes rather
than deployable CFM training methods. The goal is to answer a narrow question:
can a pressure-side training signal improve the low-NFE behavior of learned
fields, or does it merely train a stiffer model?

All pressure losses use aggregate pressure-energy normalization. Earlier
per-sample normalization was rejected because it overweights low-pressure
points and should not be used for scientific conclusions.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import lcfm.datasets  # noqa: F401 - registers datasets
from lcfm.metrics import (
    mean_endpoint_displacement,
    mean_path_length,
    mode_statistics,
    path_length_ratio,
    trajectory_acceleration,
    wasserstein_match,
)
from lcfm.models import build_model
from lcfm.oracle import GaussianMixtureOracle
from lcfm.pairing import apply_pairing
from lcfm.registry import DATASETS, get
from lcfm.schedules import euler_on_grid, heun_on_grid
from lcfm.utils import environment_info, set_seed, write_json


PRESSURE_NORMALIZATION = "global_pressure_energy"


GEOMETRIES = {
    "clumped015": {
        "dataset": "five_modes",
        "dataset_kwargs": {
            "n_train": 5000,
            "n_test": 2000,
            "radius": 4.0,
            "sigma_mode": 0.2,
            "source_std": 0.15,
        },
        "hit_radius": 0.6,
    },
    "staged": {
        "dataset": "staged_modes",
        "dataset_kwargs": {
            "n_train": 5000,
            "n_test": 2000,
            "sigma_mode": 0.2,
            "source_std": 0.15,
        },
        "hit_radius": 0.6,
    },
    "gm16": {
        "dataset": "gaussian_mixture_nd",
        "dataset_kwargs": {
            "dim": 16,
            "n_modes": 8,
            "n_train": 5000,
            "n_test": 2000,
            "radius": 4.0,
            "sigma_mode": 0.2,
            "source_std": 0.15,
        },
        "hit_radius": 1.6,
    },
}


VARIANTS = {
    "standard": {"kind": "standard", "weight": 0.0},
    "upper_w01": {"kind": "upper_budget", "weight": 0.1, "budget_c": 1.0},
    "upper_w025": {"kind": "upper_budget", "weight": 0.25, "budget_c": 1.0},
    "upper_w05": {"kind": "upper_budget", "weight": 0.5, "budget_c": 1.0},
    "upper_w1": {"kind": "upper_budget", "weight": 1.0, "budget_c": 1.0},
    "upper_w2": {"kind": "upper_budget", "weight": 2.0, "budget_c": 1.0},
    "band_w01": {"kind": "pressure_band", "weight": 0.1, "budget_c": 1.0, "eta": 0.35},
    "band_w05_eta02": {"kind": "pressure_band", "weight": 0.5, "budget_c": 1.0, "eta": 0.20},
    "band_w05_eta035": {"kind": "pressure_band", "weight": 0.5, "budget_c": 1.0, "eta": 0.35},
    "band_w05_eta05": {"kind": "pressure_band", "weight": 0.5, "budget_c": 1.0, "eta": 0.50},
    "band_w1_eta02": {"kind": "pressure_band", "weight": 1.0, "budget_c": 1.0, "eta": 0.20},
    "band_w1_eta035": {"kind": "pressure_band", "weight": 1.0, "budget_c": 1.0, "eta": 0.35},
    "band_w1_eta05": {"kind": "pressure_band", "weight": 1.0, "budget_c": 1.0, "eta": 0.50},
    "band_w1": {"kind": "pressure_band", "weight": 1.0, "budget_c": 1.0, "eta": 0.35},
    "align_w01": {"kind": "alignment", "weight": 0.1},
    "exact_w001": {"kind": "exact_match", "weight": 0.001},
    "exact_w01": {"kind": "exact_match", "weight": 0.1},
}


AGGREGATE_METRICS = [
    "integration_error",
    "wasserstein",
    "wasserstein2",
    "reference_self_error",
    "target_hit_rate",
    "mode_hit_coverage",
    "mean_path_length",
    "mean_endpoint_displacement",
    "path_length_ratio",
    "trajectory_acceleration",
    "pure_violation",
    "deficit",
    "pressure_utilization",
    "mean_model_acceleration",
    "mean_pressure_acceleration",
]


def material_derivative_jvp_train(model, x, t, velocity, detach_tangent=False):
    tangent = velocity.detach() if detach_tangent else velocity
    _, material = torch.autograd.functional.jvp(
        lambda x_in, t_in: model(x_in, t_in),
        (x, t),
        (tangent, torch.ones_like(t)),
        create_graph=True,
    )
    return material


def pressure_terms(a_theta, a_pressure, variant, eps=1e-8):
    kind = variant["kind"]
    p_norm = torch.linalg.vector_norm(a_pressure, dim=1, keepdim=True)
    theta_norm = torch.linalg.vector_norm(a_theta, dim=1, keepdim=True)
    p_hat = a_pressure / (p_norm + eps)
    alpha = torch.sum(a_theta * p_hat, dim=1, keepdim=True)
    a_perp = a_theta - alpha * p_hat

    pressure_energy = p_norm.pow(2)
    pressure_energy_sum = pressure_energy.sum().clamp_min(eps)
    orth = a_perp.pow(2).sum(dim=1, keepdim=True)
    opposite = torch.relu(-alpha).pow(2)
    excess = torch.relu(alpha - variant.get("budget_c", 1.0) * p_norm).pow(2)
    deficit = torch.relu(variant.get("eta", 0.0) * p_norm - alpha).pow(2)
    pure_violation = orth + opposite + excess

    def pressure_normalized(numerator):
        return numerator.sum() / pressure_energy_sum

    if kind == "upper_budget":
        reg = pressure_normalized(pure_violation)
    elif kind == "pressure_band":
        reg = pressure_normalized(pure_violation + deficit)
    elif kind == "alignment":
        cosine = alpha / (theta_norm * p_norm + eps)
        pressure_weight = (pressure_energy / (pressure_energy.mean().detach() + eps)).clamp(max=5.0)
        reg = (pressure_weight * (1.0 - cosine)).sum() / pressure_weight.sum().clamp_min(eps)
    elif kind == "exact_match":
        reg = pressure_normalized((a_theta - a_pressure).pow(2).sum(dim=1, keepdim=True))
    else:
        raise ValueError(f"Unknown pressure variant kind: {kind}")

    stats = {
        "reg": reg,
        "pure_violation": pressure_normalized(pure_violation),
        "deficit": pressure_normalized(deficit),
        "pressure_utilization": (alpha * p_norm).sum() / pressure_energy_sum,
        "mean_model_acceleration": theta_norm.mean(),
        "mean_pressure_acceleration": p_norm.mean(),
    }
    return stats


def make_problem(geometry):
    spec = GEOMETRIES[geometry]
    problem_cls = get(DATASETS, spec["dataset"])
    return problem_cls(dict(spec["dataset_kwargs"]))


def train_variant(args, variant_name, variant):
    set_seed(args.seed)
    device = torch.device(args.device)
    problem = make_problem(args.geometry)
    oracle = GaussianMixtureOracle.from_problem(problem).to(device)
    model_config = {"model_kwargs": {"hidden": args.hidden, "depth": args.depth}}
    model = build_model("mlp", problem.dim, model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    started = time.time()

    model.train()
    for epoch in range(args.epochs):
        x0, x1 = problem.sample_train_batch(args.batch_size, device)
        x0, x1 = apply_pairing(x0, x1, {"pairing": args.pairing})
        t = torch.rand(args.batch_size, 1, device=device) * args.t_max
        target = x1 - x0

        if variant["kind"] == "standard":
            xt = (1 - t) * x0 + t * x1
            vt = model(xt, t)
            cfm = F.mse_loss(vt, target)
            reg = torch.zeros((), device=device)
            term_stats = {}
        else:
            xt = ((1 - t) * x0 + t * x1).detach().requires_grad_(True)
            t_req = t.detach().requires_grad_(True)
            vt = model(xt, t_req)
            cfm = F.mse_loss(vt, target)
            with torch.no_grad():
                a_pressure = oracle.acceleration_target(xt.detach(), t_req.detach())
            a_theta = material_derivative_jvp_train(
                model,
                xt,
                t_req,
                vt,
                detach_tangent=args.detach_tangent,
            )
            term_stats = pressure_terms(a_theta, a_pressure, variant, eps=args.eps)
            reg = term_stats["reg"]

        loss = cfm + variant.get("weight", 0.0) * reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "cfm": float(cfm.detach().cpu()),
                "pressure_reg": float(reg.detach().cpu()),
                "elapsed_sec": time.time() - started,
            }
            for key, value in term_stats.items():
                row[key] = float(value.detach().cpu())
            history.append(row)
            print(f"{variant_name} {row}", flush=True)

    return problem, model, history


@torch.enable_grad()
def pressure_diagnostic(model, problem, oracle, args):
    device = torch.device(args.device)
    set_seed(args.seed + 2718)
    x0 = problem.eval_initial(args.n_diagnostic, device)
    x1 = problem.target_eval(args.n_diagnostic, device)
    x0, x1 = apply_pairing(x0, x1, {"pairing": args.pairing})
    totals = {
        "pure_violation": 0.0,
        "deficit": 0.0,
        "pressure_utilization": 0.0,
        "mean_model_acceleration": 0.0,
        "mean_pressure_acceleration": 0.0,
    }
    model.eval()
    for t_val in torch.linspace(0.0, args.t_max, args.diag_grid):
        t = torch.full((x0.shape[0], 1), float(t_val), device=device)
        xt = ((1 - t) * x0 + t * x1).detach().requires_grad_(True)
        t_req = t.detach().requires_grad_(True)
        vt = model(xt, t_req)
        with torch.no_grad():
            a_pressure = oracle.acceleration_target(xt.detach(), t_req.detach())
        a_theta = material_derivative_jvp_train(
            model,
            xt,
            t_req,
            vt,
            detach_tangent=args.detach_tangent,
        )
        stats = pressure_terms(a_theta, a_pressure, {"kind": "pressure_band", "budget_c": 1.0, "eta": 0.35})
        for key in totals:
            totals[key] += float(stats[key].detach().cpu())
    return {key: value / args.diag_grid for key, value in totals.items()}


@torch.no_grad()
def endpoint_metrics(problem, traj, target, x_ref, args, reference_self_error=None):
    x_end = traj[-1]
    metrics = {
        "integration_error": float((x_end - x_ref).norm(dim=1).mean().item()),
        "wasserstein": wasserstein_match(x_end, target),
        "wasserstein2": wasserstein_match(x_end, target, p=2),
        "reference_self_error": reference_self_error,
        "mean_path_length": mean_path_length(traj),
        "mean_endpoint_displacement": mean_endpoint_displacement(traj),
        "path_length_ratio": path_length_ratio(traj),
        "trajectory_acceleration": trajectory_acceleration(traj),
    }
    if hasattr(problem, "centers") and hasattr(problem, "sigma_mode"):
        metrics.update(
            mode_statistics(
                x_end,
                problem.centers(x_end.device),
                p_min=args.mode_p_min,
                hit_radius=args.hit_radius or GEOMETRIES[args.geometry].get("hit_radius") or 3.0 * problem.sigma_mode,
            )
        )
    return metrics


@torch.no_grad()
def evaluate_variant(model, problem, args):
    device = torch.device(args.device)
    set_seed(args.eval_seed)
    x0 = problem.eval_initial(args.n_eval, device)
    target = problem.target_eval(args.n_eval, device)
    velocity = lambda x, t: model(x, t)
    ref_grid = torch.linspace(0.0, 1.0, args.ref_intervals + 1).tolist()
    ref_traj = heun_on_grid(velocity, x0, ref_grid)
    x_ref = ref_traj[-1]
    reference_self_error = None
    if args.ref_check_intervals:
        if args.ref_check_intervals <= args.ref_intervals:
            raise ValueError("--ref-check-intervals must be larger than --ref-intervals.")
        check_grid = torch.linspace(0.0, 1.0, args.ref_check_intervals + 1).tolist()
        x_ref_check = heun_on_grid(velocity, x0, check_grid)[-1]
        reference_self_error = float((x_ref - x_ref_check).norm(dim=1).mean().item())

    rows = []
    ref_metrics = endpoint_metrics(problem, ref_traj, target, x_ref, args, reference_self_error)
    rows.append({"schedule": f"heun{args.ref_intervals}_ref", "nfe": 2 * args.ref_intervals, **ref_metrics})
    for steps in args.eval_steps:
        grid = torch.linspace(0.0, 1.0, steps + 1).tolist()
        traj = euler_on_grid(velocity, x0, grid)
        metrics = endpoint_metrics(problem, traj, target, x_ref, args, reference_self_error)
        rows.append({"schedule": f"euler{steps}_uniform", "nfe": steps, **metrics})
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(args, variant_name):
    variant = VARIANTS[variant_name]
    problem, model, history = train_variant(args, variant_name, variant)
    oracle = GaussianMixtureOracle.from_problem(problem).to(torch.device(args.device))
    diagnostics = pressure_diagnostic(model, problem, oracle, args)
    metric_rows = evaluate_variant(model, problem, args)

    run_dir = Path(args.output_dir) / f"{args.geometry}_{variant_name}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    write_json(run_dir / "config.json", make_run_config(args, variant_name, variant))
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "pressure_diagnostic.json", diagnostics)
    write_json(run_dir / "environment.json", environment_info())
    write_csv(run_dir / "metrics.csv", metric_rows)

    summary_rows = []
    for row in metric_rows:
        summary_rows.append(
            {
                "geometry": args.geometry,
                "variant": variant_name,
                "kind": variant["kind"],
                "weight": variant.get("weight", 0.0),
                "seed": args.seed,
                "pressure_normalization": PRESSURE_NORMALIZATION,
                **row,
                **diagnostics,
            }
        )
    return summary_rows


def make_run_config(args, variant_name, variant):
    spec = GEOMETRIES[args.geometry]
    return {
        "dataset": spec["dataset"],
        "dataset_kwargs": spec["dataset_kwargs"],
        "pairing": args.pairing,
        "variant": variant_name,
        "variant_kwargs": variant,
        "pressure_loss": {
            "normalization": PRESSURE_NORMALIZATION,
            "material_derivative": "jvp",
            "oracle": "GaussianMixtureOracle",
        },
        "model": "mlp",
        "model_kwargs": {"hidden": args.hidden, "depth": args.depth},
        "seed": args.seed,
        "train": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "t_max": args.t_max,
            "detach_tangent": args.detach_tangent,
        },
        "eval": {
            "n_eval": args.n_eval,
            "eval_steps": args.eval_steps,
            "ref_intervals": args.ref_intervals,
            "ref_check_intervals": args.ref_check_intervals,
            "eval_seed": args.eval_seed,
        },
    }


def numeric(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance**0.5


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        key = (
            row["geometry"],
            row["variant"],
            row["kind"],
            row["weight"],
            row["schedule"],
            row["nfe"],
            row["pressure_normalization"],
        )
        grouped.setdefault(key, []).append(row)

    aggregate = []
    for key, group_rows in sorted(grouped.items()):
        geometry, variant, kind, weight, schedule, nfe, normalization = key
        row = {
            "geometry": geometry,
            "variant": variant,
            "kind": kind,
            "weight": weight,
            "schedule": schedule,
            "nfe": nfe,
            "pressure_normalization": normalization,
            "n_seeds": len(group_rows),
        }
        for metric in AGGREGATE_METRICS:
            values = [numeric(group_row, metric) for group_row in group_rows]
            values = [value for value in values if value is not None]
            if values:
                mean, std = mean_std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
        aggregate.append(row)
    return aggregate


def print_seed_rows(rows, ref_intervals):
    print(f"{'variant':>12} {'schedule':>16} {'W':>8} {'hit':>7} {'int_err':>8} {'util':>8} {'def':>8}")
    for row in rows:
        if row["schedule"] not in {"euler5_uniform", f"heun{ref_intervals}_ref"}:
            continue
        print(
            f"{row['variant']:>12} {row['schedule']:>16} "
            f"{row['wasserstein']:8.3f} {row.get('target_hit_rate', float('nan')):7.3f} "
            f"{row['integration_error']:8.3f} {row['pressure_utilization']:8.3f} {row['deficit']:8.3f}"
        )


def print_aggregate_rows(rows, ref_intervals):
    print(
        f"{'variant':>12} {'schedule':>16} {'n':>3} {'W':>8} {'hit':>7} "
        f"{'int_err':>8} {'util':>8} {'def':>8}"
    )
    for row in rows:
        if row["schedule"] not in {"euler5_uniform", f"heun{ref_intervals}_ref"}:
            continue
        print(
            f"{row['variant']:>12} {row['schedule']:>16} {row['n_seeds']:>3} "
            f"{row.get('wasserstein_mean', float('nan')):8.3f} "
            f"{row.get('target_hit_rate_mean', float('nan')):7.3f} "
            f"{row.get('integration_error_mean', float('nan')):8.3f} "
            f"{row.get('pressure_utilization_mean', float('nan')):8.3f} "
            f"{row.get('deficit_mean', float('nan')):8.3f}"
        )


def run_seed(args, seed):
    seed_args = argparse.Namespace(**vars(args))
    seed_args.seed = seed
    all_rows = []
    for variant_name in seed_args.variants:
        all_rows.extend(run_one(seed_args, variant_name))

    output_dir = Path(seed_args.output_dir)
    summary_path = output_dir / f"{seed_args.geometry}_seed{seed}_summary.csv"
    write_csv(summary_path, all_rows)
    print(f"\nwrote {summary_path}")
    print_seed_rows(all_rows, seed_args.ref_intervals)
    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default="staged", choices=sorted(GEOMETRIES))
    parser.add_argument("--variants", nargs="+", default=["standard", "upper_w01", "band_w01", "exact_w001"])
    parser.add_argument("--output-dir", default="results/phase1/pressure_training_viability_globalnorm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional multi-seed mode; overrides --seed.")
    parser.add_argument("--pairing", default="independent", choices=["independent", "minibatch_ot"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--t-max", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--detach-tangent", action="store_true")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--n-eval", type=int, default=1024)
    parser.add_argument("--eval-steps", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--ref-intervals", type=int, default=1000)
    parser.add_argument(
        "--ref-check-intervals",
        type=int,
        default=0,
        help="Optional finer Heun rollout used only to estimate reference error.",
    )
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--mode-p-min", type=float, default=0.05)
    parser.add_argument("--hit-radius", type=float)
    parser.add_argument("--n-diagnostic", type=int, default=256)
    parser.add_argument("--diag-grid", type=int, default=21)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    unknown = [name for name in args.variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; available: {sorted(VARIANTS)}")

    torch.set_num_threads(args.threads)
    seeds = args.seeds or [args.seed]
    all_rows = []
    for seed in seeds:
        all_rows.extend(run_seed(args, seed))

    if len(seeds) > 1:
        aggregate = aggregate_rows(all_rows)
        seed_label = "-".join(str(seed) for seed in seeds)
        aggregate_path = Path(args.output_dir) / f"{args.geometry}_seeds_{seed_label}_aggregate.csv"
        write_csv(aggregate_path, aggregate)
        print(f"\nwrote {aggregate_path}")
        print_aggregate_rows(aggregate, args.ref_intervals)


if __name__ == "__main__":
    main()
