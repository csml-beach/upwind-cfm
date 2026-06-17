#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm.experiment import run
from lcfm.utils import read_json


def main():
    parser = argparse.ArgumentParser(description="Run one CIFAR-10 benchmark config with job-friendly overrides.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-group", default="manual")
    parser.add_argument("--out-root", default="results/cifar10_low_nfe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()

    config = read_json(args.config)
    variant = config.get("variant") or Path(args.config).stem.replace("cifar10_", "")
    config["seed"] = args.seed
    config["device"] = args.device
    config["run_name"] = args.run_name or f"cifar10_{variant}_seed{args.seed}"
    config["out_dir"] = str(Path(args.out_root) / args.run_group / "runs")
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    if args.data_root is not None:
        config.setdefault("dataset_kwargs", {})["data_root"] = args.data_root
    run(config)


if __name__ == "__main__":
    main()
