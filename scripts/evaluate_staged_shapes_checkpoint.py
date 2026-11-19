#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm import datasets  # noqa: F401
from lcfm import models  # noqa: F401
from lcfm.experiment import save_image_grid
from lcfm.metrics import mean_path_length, path_length_ratio, trajectory_acceleration
from lcfm.models import build_model
from lcfm.pairing import pairing_features
from lcfm.registry import DATASETS, get
from lcfm.schedules import (
    calibrate_schedule_profiles,
    estimator_schedule_specs_from_bundle,
    euler_on_grid,
    heun_on_grid,
    schedule_specs_from_bundle,
    validate_profile_bundle,
)
from lcfm.utils import file_sha256, read_json, set_seed, write_json


def parse_nfe_values(text):
    return [int(part) for part in text.replace(",", " ").split()]


def parse_names(text):
    return [part.strip() for part in text.replace(",", " ").split() if part.strip()]


def parse_floats(text):
    return [float(part) for part in text.replace(",", " ").split() if part.strip()]


def wasserstein_features(x, y, p=1):
    cost = torch.cdist(x, y, p=2)
    if p == 2:
        cost = cost.pow(2)
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    values = cost[row, col]
    if p == 2:
        return float(values.mean().sqrt().item())
    return float(values.mean().item())


def mode_prototypes(problem, device, samples_per_mode=256):
    labels = torch.arange(problem.n_modes, device=device).repeat_interleave(samples_per_mode)
    targets = problem.target_eval(labels.numel(), device, labels=labels)
    prototypes = []
    radii = []
    feature_cfg = {
        "pairing_kwargs": {
            "cost_feature": "downsampled_pixels",
            "image_shape": list(problem.image_shape),
            "downsample_size": 8,
        }
    }
    target_features = pairing_features(targets, feature_cfg)
    for mode in range(problem.n_modes):
        mode_features = target_features[labels == mode]
        prototype = mode_features.mean(dim=0)
        distances = torch.linalg.vector_norm(mode_features - prototype[None, :], dim=1)
        prototypes.append(prototype)
        radii.append(torch.quantile(distances, 0.99))
    return torch.stack(prototypes), torch.stack(radii), feature_cfg


def mode_metrics(samples, prototypes, radii, feature_cfg):
    features = pairing_features(samples, feature_cfg)
    distances = torch.cdist(features, prototypes)
    nearest_dist, nearest = distances.min(dim=1)
    hits = nearest_dist <= radii[nearest]
    hist = torch.bincount(nearest, minlength=prototypes.shape[0]).float() / samples.shape[0]
    hit_hist = torch.bincount(nearest[hits], minlength=prototypes.shape[0]).float() / max(1, int(hits.sum().item()))
    uniform = torch.full_like(hist, 1.0 / hist.numel())
    kl = (hist.clamp_min(1e-12) * (hist.clamp_min(1e-12) / uniform).log()).sum()
    entropy = -(hist.clamp_min(1e-12) * hist.clamp_min(1e-12).log()).sum()
    return {
        "mode_hit_rate": float(hits.float().mean().item()),
        "mode_histogram": [float(x) for x in hist.tolist()],
        "mode_hit_histogram": [float(x) for x in hit_hist.tolist()],
        "mode_kl_to_uniform": float(kl.item()),
        "mode_entropy": float(entropy.item()),
        "mean_nearest_mode_distance": float(nearest_dist.mean().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate staged image-shape CFM checkpoints.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="model_ema.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--nfe-values", default="5,10,20,50")
    parser.add_argument("--profile-samples", type=int, default=100)
    parser.add_argument("--profile-fine-steps", type=int, default=50)
    parser.add_argument("--profile-anchor-count", type=int, default=16)
    parser.add_argument("--profile-batch-size", type=int, default=8)
    parser.add_argument("--profile-estimators", default="local,transport_adjoint")
    parser.add_argument("--equivalence-profiles", default="")
    parser.add_argument("--equivalence-powers", default="0.5")
    parser.add_argument("--profile-cache")
    parser.add_argument("--pilot-seed", type=int, default=1311)
    parser.add_argument("--adjoint-probes", default="4")
    parser.add_argument("--adjoint-seed", type=int, default=2718)
    parser.add_argument("--transport-perturbations", default="0.001,0.003")
    parser.add_argument("--warp-powers", default="0.25,0.5")
    parser.add_argument("--warp-floor", type=float, default=1e-3)
    parser.add_argument("--power-rhos", default="2.0,3.0", help="Rho values for hand-designed power grids.")
    parser.add_argument("--power-kinds", default="early,late,symmetric", help="Kinds for power grids: early,late,symmetric.")
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--methods", default="uniform,sctw,adjoint_sctw,power,vsfm,heun")
    parser.add_argument("--reference-samples", type=int, default=32)
    parser.add_argument("--reference-intervals", type=int, default=200)
    parser.add_argument("--reference-check-intervals", type=int, default=400)
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = read_json(run_dir / "config.json")
    if config["dataset"] != "staged_shapes_easy":
        raise SystemExit("This evaluator expects dataset='staged_shapes_easy'.")

    device = torch.device(args.device)
    set_seed(args.eval_seed)
    problem = get(DATASETS, config["dataset"])(config.get("dataset_kwargs", {}))
    model = build_model(config.get("model", "mlp"), problem.dim, config).to(device)
    state = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval_staged_shapes"
    out_dir.mkdir(parents=True, exist_ok=True)
    prototypes, radii, feature_cfg = mode_prototypes(problem, device)
    labels = torch.arange(args.n_samples, device=device) % problem.n_modes
    target = problem.target_eval(args.n_samples, device, labels=labels)
    target_features = pairing_features(target, feature_cfg)
    x0 = problem.eval_initial(args.n_samples, device)

    nfe_values = parse_nfe_values(args.nfe_values)
    warp_powers = parse_floats(args.warp_powers)
    set_seed(args.pilot_seed)
    profile_x0 = problem.eval_initial(args.profile_samples, device)
    pilot_batches = [
        (model, profile_x0[start : start + args.profile_batch_size])
        for start in range(0, args.profile_samples, args.profile_batch_size)
    ]
    profile_cache = Path(args.profile_cache) if args.profile_cache else out_dir / "schedule_profiles.json"
    profile_context = {
        "dataset": config["dataset"],
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(run_dir / args.checkpoint),
    }
    if profile_cache.is_file():
        bundle = read_json(profile_cache)
        validate_profile_bundle(
            bundle,
            profile_context,
            {
                "pilot_seed": args.pilot_seed,
                "pilot_samples": args.profile_samples,
                "fine_steps": args.profile_fine_steps,
                "anchor_count": args.profile_anchor_count,
                "adjoint_probe_counts": parse_nfe_values(args.adjoint_probes),
                "adjoint_seed": args.adjoint_seed,
                "perturbations": parse_floats(args.transport_perturbations),
                "floor": args.warp_floor,
                "powers": warp_powers,
                "nfe_values": nfe_values,
            },
        )
    else:
        bundle = calibrate_schedule_profiles(
            pilot_batches,
            estimators=parse_names(args.profile_estimators),
            fine_steps=args.profile_fine_steps,
            anchor_count=args.profile_anchor_count,
            adjoint_probe_counts=parse_nfe_values(args.adjoint_probes),
            adjoint_seed=args.adjoint_seed,
            perturbations=parse_floats(args.transport_perturbations),
            floor=args.warp_floor,
            powers=warp_powers,
            nfe_values=nfe_values,
            pilot_seed=args.pilot_seed,
            context=profile_context,
        )
        write_json(profile_cache, bundle)

    write_json(
        out_dir / "eval_config.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "n_samples": args.n_samples,
            "nfe_values": nfe_values,
            "methods": parse_names(args.methods),
            "equivalence_profiles": parse_names(args.equivalence_profiles),
            "equivalence_powers": parse_floats(args.equivalence_powers),
            "profile_cache": str(profile_cache),
            "profile_samples": args.profile_samples,
            "profile_batch_size": args.profile_batch_size,
            "profile_fine_steps": args.profile_fine_steps,
            "profile_anchor_count": args.profile_anchor_count,
            "profile_estimators": parse_names(args.profile_estimators),
            "pilot_seed": args.pilot_seed,
            "adjoint_probes": parse_nfe_values(args.adjoint_probes),
            "adjoint_seed": args.adjoint_seed,
            "transport_perturbations": parse_floats(args.transport_perturbations),
            "warp_powers": warp_powers,
            "warp_floor": args.warp_floor,
            "power_rhos": parse_floats(args.power_rhos),
            "power_kinds": parse_names(args.power_kinds),
            "eval_seed": args.eval_seed,
            "reference_samples": args.reference_samples,
            "reference_intervals": args.reference_intervals,
            "reference_check_intervals": args.reference_check_intervals,
        },
    )

    set_seed(args.eval_seed)
    x0 = problem.eval_initial(args.n_samples, device)
    reference_count = min(args.reference_samples, args.n_samples)
    reference_x0 = x0[:reference_count]
    reference = heun_on_grid(
        model,
        reference_x0,
        [float(value) for value in torch.linspace(0.0, 1.0, args.reference_intervals + 1)],
    )[-1]
    reference_check = heun_on_grid(
        model,
        reference_x0,
        [float(value) for value in torch.linspace(0.0, 1.0, args.reference_check_intervals + 1)],
    )[-1]
    reference_self_error = float((reference - reference_check).norm(dim=1).mean().item())

    rows = []
    all_metrics = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "profile_bundle": bundle,
        "reference_self_error": reference_self_error,
        "results": {},
    }
    for nfe in nfe_values:
        schedule_specs = schedule_specs_from_bundle(
            bundle,
            nfe,
            methods=parse_names(args.methods),
            powers=warp_powers,
            power_rhos=parse_floats(args.power_rhos),
            power_kinds=parse_names(args.power_kinds),
            adjoint_probes=4,
        )
        if parse_names(args.equivalence_profiles):
            schedule_specs += estimator_schedule_specs_from_bundle(
                bundle,
                nfe,
                parse_names(args.equivalence_profiles),
                powers=parse_floats(args.equivalence_powers),
            )

        for spec in schedule_specs:
            schedule = spec["schedule"]
            grid = spec["grid"]
            chunks = []
            for start in range(0, args.n_samples, args.batch_size):
                x_batch = x0[start : start + args.batch_size]
                traj = heun_on_grid(model, x_batch, grid) if spec["solver"] == "heun" else euler_on_grid(model, x_batch, grid)
                chunks.append(traj[-1].detach())
            diagnostic_traj = (
                heun_on_grid(model, reference_x0, grid)
                if spec["solver"] == "heun"
                else euler_on_grid(model, reference_x0, grid)
            )
            samples = torch.cat(chunks, dim=0).clamp(-3.0, 3.0)
            sample_features = pairing_features(samples, feature_cfg)
            metrics = {
                "schedule": schedule,
                "nfe": nfe,
                "nfe_used": spec["nfe_used"],
                "feature_w1": wasserstein_features(sample_features, target_features, p=1),
                "feature_w2": wasserstein_features(sample_features, target_features, p=2),
                "pixel_mse_to_balanced_target": float(F.mse_loss(samples.clamp(-1.0, 1.0), target).item()),
                "sample_mean": float(samples.mean().item()),
                "sample_std": float(samples.std(unbiased=False).item()),
                "time_grid": grid,
                "integration_error": float((diagnostic_traj[-1] - reference).norm(dim=1).mean().item()),
                "reference_self_error": reference_self_error,
                "mean_path_length": mean_path_length(diagnostic_traj),
                "path_length_ratio": path_length_ratio(diagnostic_traj),
                "mean_acceleration": trajectory_acceleration(diagnostic_traj),
            }
            if "warp_power" in spec:
                metrics["warp_power"] = spec["warp_power"]
            if "profile" in spec:
                metrics["profile"] = spec["profile"]
            if "power_kind" in spec:
                metrics["power_kind"] = spec["power_kind"]
                metrics["power_rho"] = spec["power_rho"]
            metrics.update(mode_metrics(samples, prototypes, radii, feature_cfg))
            sample_name = f"{schedule}_nfe_{nfe}"
            all_metrics["results"][sample_name] = metrics
            rows.append({key: value for key, value in metrics.items() if key != "time_grid" and not isinstance(value, list)})
            save_image_grid(samples[:64], out_dir / "samples" / f"{sample_name}.png", problem.image_shape, nrow=8)
            if schedule == "uniform":
                save_image_grid(samples[:64], out_dir / "samples" / f"nfe_{nfe}.png", problem.image_shape, nrow=8)
            print(metrics, flush=True)

    with (out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out_dir / "metrics.json", all_metrics)
    print(f"Saved staged-shapes evaluation to {out_dir}")


if __name__ == "__main__":
    main()
