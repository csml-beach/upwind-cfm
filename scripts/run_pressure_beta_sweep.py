#!/usr/bin/env python3
"""Coarse robustness sweep for pressure-aware minibatch OT beta."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from lcfm.experiment import run


GEOMETRIES = {
    "staged": {
        "dataset": "staged_modes",
        "dataset_kwargs": {"n_train": 5000, "n_test": 2000, "sigma_mode": 0.2, "source_std": 0.15},
        "batch_size": 128,
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
        "batch_size": 64,
        "hit_radius": 1.6,
    },
}


def beta_label(beta):
    text = f"{beta:g}".replace(".", "p")
    return f"pressure_beta_{text}"


def make_config(args, geometry_name, beta, seed):
    geometry = GEOMETRIES[geometry_name]
    variant = beta_label(beta)
    return {
        "dataset": geometry["dataset"],
        "dataset_kwargs": geometry["dataset_kwargs"],
        "method": "standard_cfm",
        "method_kwargs": {},
        "pairing": "pressure_aware_minibatch_ot",
        "pairing_kwargs": {
            "pressure_beta": beta,
            "pressure_t": "random",
            "reference_pairing": "minibatch_ot",
        },
        "variant": variant,
        "model": "mlp",
        "model_kwargs": {"hidden": args.hidden, "depth": args.depth},
        "seed": seed,
        "device": args.device,
        "solver": "euler",
        "solver_kwargs": {"steps": args.eval_steps},
        "train": {
            "batch_size": args.batch_size or geometry["batch_size"],
            "epochs": args.epochs,
            "log_every": args.log_every,
            "lr": args.lr,
        },
        "eval": {
            "n_eval": args.n_eval,
            "mode_p_min": 0.05,
            "hit_radius": geometry["hit_radius"],
            "eval_seed": args.eval_seed,
            "plot_seed": args.eval_seed,
        },
        "out_dir": str(Path(args.output_dir) / "runs"),
        "run_name": f"{geometry_name}_{variant}_seed{seed}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/phase1/pressure_beta_sweep")
    parser.add_argument("--geometries", nargs="+", default=["staged", "gm16"], choices=sorted(GEOMETRIES))
    parser.add_argument("--betas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if any(beta < 0.0 for beta in args.betas):
        raise ValueError("All beta values must be non-negative.")

    total = len(args.geometries) * len(args.betas) * len(args.seeds)
    index = 0
    for geometry in args.geometries:
        for beta in args.betas:
            for seed in args.seeds:
                index += 1
                config = make_config(args, geometry, beta, seed)
                run_dir = Path(config["out_dir"]) / config["run_name"]
                if args.skip_existing and (run_dir / "model.pt").exists():
                    print(f"[pressure-beta-sweep] {index}/{total} skip {run_dir}", flush=True)
                    continue
                print(f"[pressure-beta-sweep] {index}/{total} train {run_dir}", flush=True)
                run(config)


if __name__ == "__main__":
    main()
