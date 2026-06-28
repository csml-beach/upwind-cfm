#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm import datasets  # noqa: F401
from lcfm import models  # noqa: F401
from lcfm.cifar_metrics import (
    CIFAR10_CLASSES,
    classifier_distribution_metrics,
    flat_to_uint8_images,
    generate_cifar_samples,
    load_cifar_classifier,
    load_cifar_generator,
    uint8_to_classifier_float,
)
from lcfm.registry import DATASETS, get
from lcfm.schedules import equal_error_grid, kappa, rollout_error_profile
from lcfm.utils import read_json, set_seed, write_json


def parse_ints(text):
    return [int(part) for part in text.replace(",", " ").split() if part.strip()]


def parse_floats(text):
    return [float(part) for part in text.replace(",", " ").split() if part.strip()]


def parse_optional_floats(text):
    values = []
    for part in text.replace(",", " ").split():
        if part.lower() in {"none", "null", "off"}:
            values.append(None)
        else:
            values.append(float(part))
    return values


def blend_grid(warped_grid, steps, alpha):
    warped = torch.as_tensor(warped_grid, dtype=torch.float64)
    uniform = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    grid = (1.0 - float(alpha)) * uniform + float(alpha) * warped
    grid[0] = 0.0
    grid[-1] = 1.0
    return [float(x) for x in grid.tolist()]


def cap_step_ratio(grid, steps, max_ratio):
    if max_ratio is None:
        return grid
    upper = float(max_ratio) / float(steps)
    if upper < 1.0 / steps - 1e-12:
        raise ValueError("max_step_ratio must be >= 1.")
    dt0 = torch.diff(torch.as_tensor(grid, dtype=torch.float64)).clamp_min(1e-12)
    fixed = torch.zeros_like(dt0, dtype=torch.bool)
    dt = dt0.clone()
    for _ in range(len(dt0) + 1):
        over = (~fixed) & (dt > upper)
        if not bool(over.any()):
            break
        fixed |= over
        remaining = 1.0 - float(fixed.sum().item()) * upper
        free = ~fixed
        if not bool(free.any()):
            break
        weights = dt0[free]
        dt[fixed] = upper
        dt[free] = remaining * weights / weights.sum()
    dt = dt.clamp_max(upper)
    deficit = 1.0 - float(dt.sum().item())
    free = dt < upper - 1e-12
    if abs(deficit) > 1e-12 and bool(free.any()):
        dt[free] += deficit * dt[free] / dt[free].sum()
    capped = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(dt, dim=0)])
    capped[-1] = 1.0
    return [float(x) for x in capped.tolist()]


@torch.no_grad()
def classifier_features(classifier, images_uint8, batch_size, device):
    features = []

    def hook(_module, _inputs, output):
        features.append(torch.flatten(output.detach(), 1).cpu())

    handle = classifier.avgpool.register_forward_hook(hook)
    try:
        for start in range(0, images_uint8.shape[0], batch_size):
            batch = uint8_to_classifier_float(images_uint8[start : start + batch_size].to(device))
            classifier(batch)
    finally:
        handle.remove()
    return torch.cat(features, dim=0)


def feature_mmd_rbf(x, y):
    z = torch.cat([x, y], dim=0)
    sample = z[torch.linspace(0, z.shape[0] - 1, min(512, z.shape[0])).long()]
    distances = torch.cdist(sample, sample).pow(2)
    median = torch.median(distances[distances > 0]).clamp_min(1e-6)
    gamma = 1.0 / (2.0 * median)
    kxx = torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
    kyy = torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
    kxy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
    return float((kxx + kyy - 2.0 * kxy).item())


def main():
    parser = argparse.ArgumentParser(description="Sweep CIFAR-10 curvature-aware Euler time meshes against a fine reference sampler.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="model_ema.pt")
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--nfe-values", default="5,10,20")
    parser.add_argument("--reference-nfe", type=int, default=200)
    parser.add_argument("--profile-samples", type=int, default=512)
    parser.add_argument("--profile-fine-steps", type=int, default=50)
    parser.add_argument("--warp-powers", default="0.25,0.5")
    parser.add_argument("--blend-alphas", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--max-step-ratios", default="none,2,3")
    parser.add_argument("--warp-floor", type=float, default=1e-3)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = read_json(run_dir / "config.json")
    if args.data_root is not None:
        config.setdefault("dataset_kwargs", {})["data_root"] = args.data_root
        config.setdefault("dataset_kwargs", {})["download"] = False
    if config.get("solver", "euler") != "euler":
        raise SystemExit("time-mesh sweep currently expects solver='euler'.")

    device = torch.device(args.device)
    metric_device = torch.device(args.metric_device)
    set_seed(args.eval_seed)

    problem = get(DATASETS, config["dataset"])(config.get("dataset_kwargs", {}))
    model = load_cifar_generator(run_dir, problem, config, args.checkpoint, device)
    classifier, classifier_info = load_cifar_classifier(args.classifier_checkpoint, metric_device)

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "time_mesh_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_chunks = []
    profile_weights = []
    set_seed(args.eval_seed + 77)
    for start in range(0, args.profile_samples, args.batch_size):
        end = min(start + args.batch_size, args.profile_samples)
        size = end - start
        labels = (
            (torch.arange(start, end, dtype=torch.long, device=device) % len(CIFAR10_CLASSES))
            if getattr(problem, "class_conditional", False)
            else None
        )
        profile_x0 = problem.eval_initial(size, device)
        profile_model = (lambda x, t, labels=labels: model(x, t, labels)) if labels is not None else model
        ts, err = rollout_error_profile(profile_model, profile_x0, fine_steps=args.profile_fine_steps)
        profile_chunks.append(err)
        profile_weights.append(float(size))
    weights = torch.tensor(profile_weights, dtype=torch.float32, device=profile_chunks[0].device)
    profile_err = (torch.stack(profile_chunks) * weights[:, None]).sum(dim=0) / weights.sum()
    profile = {
        "ts": [float(x) for x in ts.tolist()],
        "err": [float(x) for x in profile_err.tolist()],
        "kappa": kappa(ts, profile_err, floor=args.warp_floor),
    }

    write_json(
        out_dir / "sweep_config.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "n_samples": args.n_samples,
            "nfe_values": parse_ints(args.nfe_values),
            "reference_nfe": args.reference_nfe,
            "profile_samples": args.profile_samples,
            "profile_fine_steps": args.profile_fine_steps,
            "warp_powers": parse_floats(args.warp_powers),
            "blend_alphas": parse_floats(args.blend_alphas),
            "max_step_ratios": args.max_step_ratios,
            "warp_floor": args.warp_floor,
            "eval_seed": args.eval_seed,
            "classifier_info": classifier_info,
            "e1_profile": profile,
        },
    )
    write_json(out_dir / "e1_profile.json", profile)

    reference_samples, reference_labels = generate_cifar_samples(
        model,
        problem,
        config,
        args.n_samples,
        args.reference_nfe,
        args.batch_size,
        args.eval_seed,
        device,
    )
    reference_uint8 = flat_to_uint8_images(reference_samples, problem.image_shape)
    reference_features = classifier_features(classifier, reference_uint8, args.metric_batch_size, metric_device)

    rows = []
    nfe_values = parse_ints(args.nfe_values)
    for nfe in nfe_values:
        uniform_samples, labels = generate_cifar_samples(
            model,
            problem,
            config,
            args.n_samples,
            nfe,
            args.batch_size,
            args.eval_seed,
            device,
        )
        uniform_uint8 = flat_to_uint8_images(uniform_samples, problem.image_shape)
        uniform_features = classifier_features(classifier, uniform_uint8, args.metric_batch_size, metric_device)
        base_row = {
            "schedule": "uniform",
            "nfe": nfe,
            "warp_power": 0.0,
            "blend_alpha": 0.0,
            "max_step_ratio": None,
            "pixel_mse_to_ref": float(F.mse_loss(uniform_samples, reference_samples).item()),
            "feature_mse_to_ref": float(F.mse_loss(uniform_features, reference_features).item()),
            "feature_mmd_to_ref": feature_mmd_rbf(uniform_features, reference_features),
            "sample_std": float(uniform_samples.std(unbiased=False).item()),
        }
        base_row.update(classifier_distribution_metrics(classifier, uniform_uint8, args.metric_batch_size, metric_device))
        rows.append(base_row)
        print(base_row, flush=True)

        for power in parse_floats(args.warp_powers):
            raw_grid = equal_error_grid(ts, profile_err, nfe, power=power, floor=args.warp_floor, end=1.0)
            for alpha in parse_floats(args.blend_alphas):
                blended = blend_grid(raw_grid, nfe, alpha)
                for max_ratio in parse_optional_floats(args.max_step_ratios):
                    grid = cap_step_ratio(blended, nfe, max_ratio)
                    samples, _ = generate_cifar_samples(
                        model,
                        problem,
                        config,
                        args.n_samples,
                        nfe,
                        args.batch_size,
                        args.eval_seed,
                        device,
                        time_grid=grid,
                    )
                    images = flat_to_uint8_images(samples, problem.image_shape)
                    features = classifier_features(classifier, images, args.metric_batch_size, metric_device)
                    row = {
                        "schedule": "e1_warped",
                        "nfe": nfe,
                        "warp_power": power,
                        "blend_alpha": alpha,
                        "max_step_ratio": max_ratio,
                        "pixel_mse_to_ref": float(F.mse_loss(samples, reference_samples).item()),
                        "feature_mse_to_ref": float(F.mse_loss(features, reference_features).item()),
                        "feature_mmd_to_ref": feature_mmd_rbf(features, reference_features),
                        "sample_std": float(samples.std(unbiased=False).item()),
                        "time_grid": grid,
                    }
                    row.update(classifier_distribution_metrics(classifier, images, args.metric_batch_size, metric_device))
                    rows.append(row)
                    print(row, flush=True)

    csv_rows = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]
    fieldnames = sorted({key for row in csv_rows for key in row})
    with (out_dir / "time_mesh_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(out_dir / "time_mesh_sweep.json", {"profile": profile, "rows": rows})
    print(f"Saved time-mesh sweep to {out_dir}")


if __name__ == "__main__":
    main()
