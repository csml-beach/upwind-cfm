#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm.experiment import run
from lcfm.utils import read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a JSON experiment config.")
    parser.add_argument("--out-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.out_dir is not None:
        config["out_dir"] = args.out_dir
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_steps is not None:
        config.setdefault("train", {})["max_steps"] = args.max_steps
    run(config)


if __name__ == "__main__":
    main()
