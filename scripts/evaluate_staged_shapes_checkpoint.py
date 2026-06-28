#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm import datasets  # noqa: F401
from lcfm import models  # noqa: F401
from lcfm.experiment import save_image_grid
from lcfm.models import build_model
from lcfm.pairing import pairing_features
from lcfm.registry import DATASETS, get
from lcfm.schedules import equal_error_grid, euler_on_grid, kappa, power_time_grid, rollout_error_profile
from lcfm.solvers import solve
from lcfm.utils import read_json, set_seed, write_json


def parse_nfe_values(text):
    return [int(part) for part in text.replace(",", " ").split()]


def parse_names(text):
    return [part.strip() for part in text.replace(",", " ").split() if part.strip()]


def parse_floats(text):
    return [float(part) for part in text.replace(",", " ").split() if part.strip()]


def rho_tag(value):
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


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
    parser.add_argument("--schedules", default="uniform", help="Comma/space list: uniform,e1_warped,power.")
    parser.add_argument("--profile-samples", type=int, default=512)
    parser.add_argument("--profile-fine-steps", type=int, default=50)
    parser.add_argument("--warp-power", type=float, default=0.5)
    parser.add_argument("--warp-powers", help="Optional comma/space list of E1 warp powers.")
    parser.add_argument("--warp-floor", type=float, default=1e-3)
    parser.add_argument("--power-rhos", default="2.0", help="Rho values for hand-designed power grids.")
    parser.add_argument("--power-kinds", default="early,late,symmetric", help="Kinds for power grids: early,late,symmetric.")
    parser.add_argument("--eval-seed", type=int, default=1234)
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
    state = torch.load(run_dir / args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval_staged_shapes"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "eval_config.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "n_samples": args.n_samples,
            "nfe_values": parse_nfe_values(args.nfe_values),
            "schedules": parse_names(args.schedules),
            "profile_samples": args.profile_samples,
            "profile_fine_steps": args.profile_fine_steps,
            "warp_power": args.warp_power,
            "warp_powers": parse_floats(args.warp_powers) if args.warp_powers else [args.warp_power],
            "warp_floor": args.warp_floor,
            "power_rhos": parse_floats(args.power_rhos),
            "power_kinds": parse_names(args.power_kinds),
            "eval_seed": args.eval_seed,
        },
    )

    prototypes, radii, feature_cfg = mode_prototypes(problem, device)
    labels = torch.arange(args.n_samples, device=device) % problem.n_modes
    target = problem.target_eval(args.n_samples, device, labels=labels)
    target_features = pairing_features(target, feature_cfg)
    x0 = problem.eval_initial(args.n_samples, device)

    schedules = parse_names(args.schedules)
    unknown = set(schedules).difference({"uniform", "e1_warped", "power"})
    if unknown:
        raise SystemExit(f"Unknown schedules: {sorted(unknown)}")
    e1_profile = None
    if "e1_warped" in schedules:
        set_seed(args.eval_seed + 77)
        profile_x0 = problem.eval_initial(args.profile_samples, device)
        ts, err = rollout_error_profile(model, profile_x0, fine_steps=args.profile_fine_steps)
        e1_profile = {
            "ts": [float(x) for x in ts.tolist()],
            "err": [float(x) for x in err.tolist()],
            "kappa": kappa(ts, err, floor=args.warp_floor),
        }
        write_json(out_dir / "e1_profile.json", e1_profile)

    rows = []
    all_metrics = {"run_dir": str(run_dir), "checkpoint": args.checkpoint, "e1_profile": e1_profile, "results": {}}
    for nfe in parse_nfe_values(args.nfe_values):
        schedule_specs = []
        for schedule in schedules:
            if schedule == "uniform":
                schedule_specs.append({"schedule": "uniform", "sample_name": f"uniform_nfe_{nfe}", "grid": None})
            elif schedule == "e1_warped":
                warp_powers = parse_floats(args.warp_powers) if args.warp_powers else [args.warp_power]
                for power in warp_powers:
                    sample_name = f"e1_warped_nfe_{nfe}" if len(warp_powers) == 1 else f"e1_warped_p{rho_tag(power)}_nfe_{nfe}"
                    schedule_specs.append(
                        {
                            "schedule": "e1_warped",
                            "sample_name": sample_name,
                            "grid": equal_error_grid(
                                e1_profile["ts"],
                                e1_profile["err"],
                                nfe,
                                power=power,
                                floor=args.warp_floor,
                                end=1.0,
                            ),
                            "warp_power": power,
                        }
                    )
            elif schedule == "power":
                for kind in parse_names(args.power_kinds):
                    for rho in parse_floats(args.power_rhos):
                        schedule_specs.append(
                            {
                                "schedule": f"power_{kind}",
                                "sample_name": f"power_{kind}_rho{rho_tag(rho)}_nfe_{nfe}",
                                "grid": power_time_grid(nfe, rho=rho, kind=kind),
                                "power_kind": kind,
                                "power_rho": rho,
                            }
                        )

        for spec in schedule_specs:
            schedule = spec["schedule"]
            grid = spec["grid"]
            chunks = []
            for start in range(0, args.n_samples, args.batch_size):
                x_batch = x0[start : start + args.batch_size]
                if schedule == "uniform":
                    traj = solve(config.get("solver", "euler"), model, x_batch, {"steps": nfe})
                else:
                    traj = euler_on_grid(model, x_batch, grid)
                chunks.append(traj[-1].detach())
            samples = torch.cat(chunks, dim=0).clamp(-3.0, 3.0)
            sample_features = pairing_features(samples, feature_cfg)
            metrics = {
                "schedule": schedule,
                "nfe": nfe,
                "feature_w1": wasserstein_features(sample_features, target_features, p=1),
                "feature_w2": wasserstein_features(sample_features, target_features, p=2),
                "pixel_mse_to_balanced_target": float(F.mse_loss(samples.clamp(-1.0, 1.0), target).item()),
                "sample_mean": float(samples.mean().item()),
                "sample_std": float(samples.std(unbiased=False).item()),
                "time_grid": grid,
            }
            if "warp_power" in spec:
                metrics["warp_power"] = spec["warp_power"]
            if "power_kind" in spec:
                metrics["power_kind"] = spec["power_kind"]
                metrics["power_rho"] = spec["power_rho"]
            metrics.update(mode_metrics(samples, prototypes, radii, feature_cfg))
            all_metrics["results"][spec["sample_name"]] = metrics
            rows.append({key: value for key, value in metrics.items() if key != "time_grid" and not isinstance(value, list)})
            save_image_grid(samples[:64], out_dir / "samples" / f"{spec['sample_name']}.png", problem.image_shape, nrow=8)
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
