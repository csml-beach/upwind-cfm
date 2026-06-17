#!/usr/bin/env python3
"""Train capacity/training-time variants for the Gaussian-mixture oracle audit.

The default target is the clumped five-mode independent problem, where the
oracle pressure layer is strong. This script only trains standard CFM models;
use scripts/oracle_model_audit.py afterwards to compare each model to E0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch

from lcfm.experiment import run


VARIANTS = {
    "small_h64_d2_e2000": {"hidden": 64, "depth": 2, "epochs": 2000},
    "base_h128_d3_e2000": {"hidden": 128, "depth": 3, "epochs": 2000},
    "large_h256_d4_e2000": {"hidden": 256, "depth": 4, "epochs": 2000},
    "base_h128_d3_e8000": {"hidden": 128, "depth": 3, "epochs": 8000},
    "large_h256_d4_e8000": {"hidden": 256, "depth": 4, "epochs": 8000},
}


def make_config(variant, seed, out_dir):
    spec = VARIANTS[variant]
    return {
        "dataset": "five_modes",
        "dataset_kwargs": {
            "n_train": 5000,
            "n_test": 2000,
            "radius": 4.0,
            "sigma_mode": 0.2,
            "source_std": 0.15,
        },
        "method": "standard_cfm",
        "method_kwargs": {},
        "pairing": "independent",
        "model": "mlp",
        "model_kwargs": {"hidden": spec["hidden"], "depth": spec["depth"]},
        "seed": seed,
        "solver": "euler",
        "solver_kwargs": {"steps": 5, "noise": 0.0},
        "train": {
            "batch_size": 256,
            "epochs": spec["epochs"],
            "log_every": spec["epochs"],
            "lr": 0.001,
        },
        "eval": {"n_eval": 1000, "mode_p_min": 0.05, "hit_radius": 0.6, "eval_seed": 1234, "plot_seed": 1234},
        "out_dir": str(Path(out_dir) / variant),
        "run_name": f"{variant}_seed{seed}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/phase1/capacity_audit/runs")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    trained = []
    for variant in args.variants:
        for seed in args.seeds:
            config = make_config(variant, seed, args.out)
            run_dir = Path(config["out_dir"]) / config["run_name"]
            if (run_dir / "model.pt").exists():
                print(f"skip existing {run_dir}")
            else:
                print(f"train {run_dir}")
                run(config)
            trained.append(run_dir)

    print("\nrun dirs:")
    for run_dir in trained:
        print(run_dir)


if __name__ == "__main__":
    main()
