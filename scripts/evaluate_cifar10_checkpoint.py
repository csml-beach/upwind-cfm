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
    load_or_make_reference_cache,
    save_eval_outputs,
)
from lcfm.experiment import save_image_grid
from lcfm.registry import DATASETS, get
from lcfm.utils import read_json, set_seed, write_json


def parse_nfe_values(text):
    return [int(part) for part in text.replace(",", " ").split()]


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
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--reference-split", choices=["train", "test"], default="test")
    parser.add_argument("--class-conditional", action="store_true", help="Force class-conditional sampling even if the saved config is unconditional.")
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--classifier-checkpoint")
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
    write_json(
        out_dir / "eval_config.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "use_ema": args.use_ema,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "metric_batch_size": args.metric_batch_size,
            "nfe_values": parse_nfe_values(args.nfe_values),
            "eval_seed": args.eval_seed,
            "reference_split": args.reference_split,
            "class_conditional": bool(config.get("dataset_kwargs", {}).get("class_conditional", False)),
            "skip_fid_kid": args.skip_fid_kid,
            "skip_classifier": args.skip_classifier,
            "classifier_checkpoint": args.classifier_checkpoint,
        },
    )

    classifier = None
    classifier_info = None
    if not args.skip_classifier:
        if not args.classifier_checkpoint:
            raise SystemExit("--classifier-checkpoint is required unless --skip-classifier is set.")
        classifier, classifier_info = load_cifar_classifier(args.classifier_checkpoint, metric_device)

    rows = []
    all_metrics = {
        "classes": CIFAR10_CLASSES,
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "reference_split": args.reference_split,
        "classifier_info": classifier_info,
        "nfe": {},
    }

    for nfe in parse_nfe_values(args.nfe_values):
        samples, labels = generate_cifar_samples(
            model,
            problem,
            config,
            args.n_samples,
            nfe,
            args.batch_size,
            args.eval_seed,
            device,
        )
        fake_uint8 = flat_to_uint8_images(samples, problem.image_shape)
        save_image_grid(samples[:64], out_dir / "samples" / f"nfe_{nfe}.png", problem.image_shape, nrow=8)
        if args.save_sample_tensors:
            torch.save({"samples": fake_uint8.cpu(), "labels": labels.cpu() if labels is not None else None}, out_dir / f"nfe_{nfe}_samples.pt")

        nfe_metrics = {
            "nfe": nfe,
            "sample_mean": float(samples.mean().item()),
            "sample_std": float(samples.std(unbiased=False).item()),
            "sample_min": float(samples.min().item()),
            "sample_max": float(samples.max().item()),
        }

        if not args.skip_fid_kid:
            reference = load_or_make_reference_cache(
                problem,
                args.n_samples,
                args.reference_split,
                labels,
                Path(config.get("dataset_kwargs", {}).get("data_root", "data")) / "metric_cache",
            )
            nfe_metrics.update(
                fid_kid_metrics(
                    fake_uint8,
                    reference["images"],
                    args.metric_batch_size,
                    metric_device,
                    kid_subset_size=args.kid_subset_size,
                )
            )

        if classifier is not None and labels is not None:
            nfe_metrics.update(classifier_metrics(classifier, fake_uint8, labels, args.metric_batch_size, metric_device))
        elif classifier is not None:
            nfe_metrics.update(classifier_distribution_metrics(classifier, fake_uint8, args.metric_batch_size, metric_device))

        all_metrics["nfe"][str(nfe)] = nfe_metrics
        rows.append({key: value for key, value in nfe_metrics.items() if not isinstance(value, (dict, list))})
        save_eval_outputs(out_dir, all_metrics, rows)
        print(nfe_metrics, flush=True)

    print(f"Saved CIFAR-10 evaluation to {out_dir}")


if __name__ == "__main__":
    main()
