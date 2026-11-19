#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm import datasets  # noqa: F401
from lcfm import models  # noqa: F401
from lcfm.cifar_metrics import (
    CIFAR10_CLASSES,
    classifier_distribution_metrics,
    classifier_metrics,
    fid_kid_metrics,
    flat_to_uint8_images,
    generate_cifar_samples,
    load_cifar_classifier,
    load_cifar_generator,
    load_or_make_inception_reference_cache,
    make_eval_labels,
    save_eval_outputs,
)
from lcfm.experiment import save_image_grid
from lcfm.metrics import mean_path_length, path_length_ratio, trajectory_acceleration
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


class LabelVelocity:
    """Bind labels to a conditional velocity without changing its call signature."""

    def __init__(self, model, labels):
        self.model = model
        self.labels = labels

    def __call__(self, x, t):
        return self.model(x, t, self.labels)


def make_pilot_batches(model, problem, x0, batch_size):
    labels = problem.eval_labels(x0.shape[0], x0.device) if getattr(problem, "class_conditional", False) else None
    batches = []
    for start in range(0, x0.shape[0], batch_size):
        end = min(start + batch_size, x0.shape[0])
        velocity = LabelVelocity(model, labels[start:end]) if labels is not None else model
        batches.append((velocity, x0[start:end]))
    return batches, labels


def is_cuda_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 CFM checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing config.json and model/checkpoint.")
    parser.add_argument("--checkpoint", default="model.pt", help="model.pt or checkpoint_latest.pt.")
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights from a checkpoint payload.")
    parser.add_argument("--data-root", help="Override CIFAR-10 data root.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric-device", default="cpu", help="Device for FID/KID and classifier inference.")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--nfe-values", default="5,10,20,50")
    parser.add_argument("--methods", default="uniform,sctw,adjoint_sctw,power,vsfm,heun")
    parser.add_argument("--profile-samples", type=int, default=100)
    parser.add_argument("--profile-fine-steps", type=int, default=50)
    parser.add_argument("--profile-anchor-count", type=int, default=16)
    parser.add_argument("--profile-batch-size", type=int, default=4)
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
    parser.add_argument("--reference-samples", type=int, default=32)
    parser.add_argument("--reference-intervals", type=int, default=200)
    parser.add_argument("--reference-check-intervals", type=int, default=400)
    parser.add_argument("--reference-split", choices=["train", "test"], default="train")
    parser.add_argument("--class-conditional", action="store_true", help="Force class-conditional sampling even if the saved config is unconditional.")
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--kid-seed", type=int, default=1234)
    parser.add_argument("--classifier-checkpoint")
    parser.add_argument(
        "--inception-weights",
        default="artifacts/cifar10_metrics/inception/weights-inception-2015-12-05-6726825d.pth",
    )
    parser.add_argument("--skip-fid-kid", action="store_true")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--out-dir")
    parser.add_argument("--save-sample-tensors", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = read_json(run_dir / "config.json")
    if args.data_root is not None:
        config.setdefault("dataset_kwargs", {})["data_root"] = args.data_root
        config.setdefault("dataset_kwargs", {})["download"] = False
    config.setdefault("dataset_kwargs", {})
    if args.class_conditional:
        config["dataset_kwargs"]["class_conditional"] = True

    device = torch.device(args.device)
    metric_device = torch.device(args.metric_device)
    set_seed(args.eval_seed)

    dataset_cls = get(DATASETS, config["dataset"])
    problem = dataset_cls(config.get("dataset_kwargs", {}))
    model = load_cifar_generator(run_dir, problem, config, args.checkpoint, device, use_ema=args.use_ema)

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    nfe_values = parse_nfe_values(args.nfe_values)
    warp_powers = parse_floats(args.warp_powers)
    methods = parse_names(args.methods)
    allowed_methods = {"uniform", "sctw", "adjoint_sctw", "power", "vsfm", "heun"}
    unknown = set(methods).difference(allowed_methods)
    if unknown:
        raise SystemExit(f"Unknown methods: {sorted(unknown)}")
    if getattr(problem, "class_conditional", False) and args.profile_samples != 100:
        raise SystemExit("Conditional CIFAR calibration requires --profile-samples=100 (10 pilots per class).")

    profile_cache = Path(args.profile_cache) if args.profile_cache else out_dir / "schedule_profiles.json"
    profile_context = {
        "dataset": config["dataset"],
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(run_dir / args.checkpoint),
        "class_conditional": bool(getattr(problem, "class_conditional", False)),
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
        set_seed(args.pilot_seed)
        profile_x0 = problem.eval_initial(args.profile_samples, device)
        attempted_batch_sizes = []
        candidate_batch_sizes = [args.profile_batch_size]
        if device.type == "cuda":
            candidate_batch_sizes.extend(size for size in (2, 1) if size < args.profile_batch_size)
        for profile_batch_size in candidate_batch_sizes:
            attempted_batch_sizes.append(profile_batch_size)
            pilot_batches, pilot_labels = make_pilot_batches(model, problem, profile_x0, profile_batch_size)
            try:
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
                break
            except RuntimeError as exc:
                if device.type != "cuda" or not is_cuda_oom(exc) or profile_batch_size == candidate_batch_sizes[-1]:
                    raise
                torch.cuda.empty_cache()
        bundle["calibration"]["requested_batch_size"] = args.profile_batch_size
        bundle["calibration"]["batch_size"] = profile_batch_size
        bundle["calibration"]["attempted_batch_sizes"] = attempted_batch_sizes
        if pilot_labels is not None:
            class_counts = torch.bincount(pilot_labels.cpu(), minlength=len(CIFAR10_CLASSES))
            bundle["calibration"]["conditional_label_counts"] = [int(value) for value in class_counts]
        write_json(profile_cache, bundle)
    if getattr(problem, "class_conditional", False):
        expected_counts = [10] * len(CIFAR10_CLASSES)
        if bundle.get("calibration", {}).get("conditional_label_counts") != expected_counts:
            raise ValueError("Conditional profile cache does not preserve exactly 10 pilot examples per CIFAR-10 class.")

    write_json(
        out_dir / "eval_config.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "use_ema": args.use_ema,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "metric_batch_size": args.metric_batch_size,
            "nfe_values": nfe_values,
            "methods": methods,
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
            "reference_split": args.reference_split,
            "kid_seed": args.kid_seed,
            "class_conditional": bool(config.get("dataset_kwargs", {}).get("class_conditional", False)),
            "skip_fid_kid": args.skip_fid_kid,
            "skip_classifier": args.skip_classifier,
            "classifier_checkpoint": args.classifier_checkpoint,
            "inception_weights": args.inception_weights,
        },
    )

    classifier = None
    classifier_info = None
    if not args.skip_classifier:
        if not args.classifier_checkpoint:
            raise SystemExit("--classifier-checkpoint is required unless --skip-classifier is set.")
        classifier, classifier_info = load_cifar_classifier(args.classifier_checkpoint, metric_device)

    inception_reference = None
    if not args.skip_fid_kid:
        inception_weights = Path(args.inception_weights)
        if not inception_weights.is_file():
            raise SystemExit(f"Inception weights not found: {inception_weights}")
        reference_labels = (
            make_eval_labels(args.n_samples, torch.device("cpu"))
            if getattr(problem, "class_conditional", False)
            else None
        )
        inception_reference = load_or_make_inception_reference_cache(
            problem,
            args.n_samples,
            args.reference_split,
            reference_labels,
            Path(config.get("dataset_kwargs", {}).get("data_root", "data")) / "metric_cache",
            args.metric_batch_size,
            metric_device,
            weights_path=inception_weights,
        )

    reference_count = min(args.reference_samples, args.n_samples)
    set_seed(args.eval_seed)
    diagnostic_x0 = problem.eval_initial(reference_count, device)
    diagnostic_labels = problem.eval_labels(reference_count, device) if getattr(problem, "class_conditional", False) else None
    diagnostic_velocity = LabelVelocity(model, diagnostic_labels) if diagnostic_labels is not None else model
    reference_grid = [float(value) for value in torch.linspace(0.0, 1.0, args.reference_intervals + 1)]
    reference_check_grid = [float(value) for value in torch.linspace(0.0, 1.0, args.reference_check_intervals + 1)]
    ode_reference = heun_on_grid(diagnostic_velocity, diagnostic_x0, reference_grid)[-1]
    ode_reference_check = heun_on_grid(diagnostic_velocity, diagnostic_x0, reference_check_grid)[-1]
    reference_self_error = float((ode_reference - ode_reference_check).norm(dim=1).mean().item())

    rows = []
    all_metrics = {
        "classes": CIFAR10_CLASSES,
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "reference_split": args.reference_split,
        "classifier_info": classifier_info,
        "fid_kid_reference": (
            {key: value for key, value in inception_reference.items() if key != "features"}
            if inception_reference is not None
            else None
        ),
        "profile_bundle": bundle,
        "reference_self_error": reference_self_error,
        "results": {},
        "nfe": {},
    }

    for nfe in nfe_values:
        schedule_specs = schedule_specs_from_bundle(
            bundle,
            nfe,
            methods=methods,
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
            samples, labels = generate_cifar_samples(
                model,
                problem,
                config,
                args.n_samples,
                nfe,
                args.batch_size,
                args.eval_seed,
                device,
                time_grid=grid,
                solver_override=spec["solver"],
            )
            fake_uint8 = flat_to_uint8_images(samples, problem.image_shape)
            sample_name = f"{schedule}_nfe_{nfe}"
            save_image_grid(samples[:64], out_dir / "samples" / f"{sample_name}.png", problem.image_shape, nrow=8)
            if schedule == "uniform":
                save_image_grid(samples[:64], out_dir / "samples" / f"nfe_{nfe}.png", problem.image_shape, nrow=8)
            if args.save_sample_tensors:
                torch.save(
                    {"samples": fake_uint8.cpu(), "labels": labels.cpu() if labels is not None else None},
                    out_dir / f"{sample_name}_samples.pt",
                )

            diagnostic_traj = (
                heun_on_grid(diagnostic_velocity, diagnostic_x0, grid)
                if spec["solver"] == "heun"
                else euler_on_grid(diagnostic_velocity, diagnostic_x0, grid)
            )

            nfe_metrics = {
                "schedule": schedule,
                "nfe": nfe,
                "nfe_used": spec["nfe_used"],
                "sample_mean": float(samples.mean().item()),
                "sample_std": float(samples.std(unbiased=False).item()),
                "sample_min": float(samples.min().item()),
                "sample_max": float(samples.max().item()),
                "time_grid": grid,
                "integration_error": float((diagnostic_traj[-1] - ode_reference).norm(dim=1).mean().item()),
                "reference_self_error": reference_self_error,
                "mean_path_length": mean_path_length(diagnostic_traj),
                "path_length_ratio": path_length_ratio(diagnostic_traj),
                "mean_acceleration": trajectory_acceleration(diagnostic_traj),
            }
            if "warp_power" in spec:
                nfe_metrics["warp_power"] = spec["warp_power"]
            if "profile" in spec:
                nfe_metrics["profile"] = spec["profile"]
            if "power_kind" in spec:
                nfe_metrics["power_kind"] = spec["power_kind"]
                nfe_metrics["power_rho"] = spec["power_rho"]

            if not args.skip_fid_kid:
                nfe_metrics.update(
                    fid_kid_metrics(
                        fake_uint8,
                        inception_reference["features"],
                        args.metric_batch_size,
                        metric_device,
                        kid_subset_size=args.kid_subset_size,
                        kid_seed=args.kid_seed,
                        weights_path=inception_weights,
                    )
                )

            if classifier is not None and labels is not None:
                nfe_metrics.update(classifier_metrics(classifier, fake_uint8, labels, args.metric_batch_size, metric_device))
            elif classifier is not None:
                nfe_metrics.update(classifier_distribution_metrics(classifier, fake_uint8, args.metric_batch_size, metric_device))

            all_metrics["results"][sample_name] = nfe_metrics
            if schedule == "uniform":
                all_metrics["nfe"][str(nfe)] = nfe_metrics
            rows.append({key: value for key, value in nfe_metrics.items() if key != "time_grid" and not isinstance(value, (dict, list))})
            save_eval_outputs(out_dir, all_metrics, rows)
            print(nfe_metrics, flush=True)

    print(f"Saved CIFAR-10 evaluation to {out_dir}")


if __name__ == "__main__":
    main()
