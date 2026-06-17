#!/usr/bin/env python3
"""Train coupling and cooling baselines around pressure-aware minibatch pairing."""
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

METHODS = {
    "independent": {
        "method": "standard_cfm",
        "method_kwargs": {},
        "pairing": "independent",
        "pairing_kwargs": {},
    },
    "minibatch_ot": {
        "method": "standard_cfm",
        "method_kwargs": {},
        "pairing": "minibatch_ot",
        "pairing_kwargs": {},
    },
    "pressure_aware_ot": {
        "method": "standard_cfm",
        "method_kwargs": {},
        "pairing": "pressure_aware_minibatch_ot",
        "pairing_kwargs": {
            "pressure_beta": 0.2,
            "pressure_t": "random",
            "reference_pairing": "minibatch_ot",
        },
    },
    "iso_fd_w05": {
        "method": "iso_fm_finite_difference",
        "method_kwargs": {"weight": 0.5},
        "pairing": "independent",
        "pairing_kwargs": {},
    },
    "iso_fd_w01": {
        "method": "iso_fm_finite_difference",
        "method_kwargs": {"weight": 0.1},
        "pairing": "independent",
        "pairing_kwargs": {},
    },
}


def make_config(args, geometry_name, method_name, seed):
    geometry = GEOMETRIES[geometry_name]
    method = METHODS[method_name]
    return {
        "dataset": geometry["dataset"],
        "dataset_kwargs": geometry["dataset_kwargs"],
        "method": method["method"],
        "method_kwargs": method["method_kwargs"],
        "pairing": method["pairing"],
        "pairing_kwargs": method["pairing_kwargs"],
        "variant": method_name,
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
        "run_name": f"{geometry_name}_{method_name}_seed{seed}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/phase1/pressure_aware_coupling_benchmark")
    parser.add_argument("--geometries", nargs="+", default=["staged", "gm16"], choices=sorted(GEOMETRIES))
    parser.add_argument("--pairings", nargs="+", default=list(METHODS), choices=list(METHODS))
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

    total = len(args.geometries) * len(args.pairings) * len(args.seeds)
    index = 0
    for geometry in args.geometries:
        for method_name in args.pairings:
            for seed in args.seeds:
                index += 1
                config = make_config(args, geometry, method_name, seed)
                run_dir = Path(config["out_dir"]) / config["run_name"]
                if args.skip_existing and (run_dir / "model.pt").exists():
                    print(f"[coupling-benchmark] {index}/{total} skip {run_dir}", flush=True)
                    continue
                print(f"[coupling-benchmark] {index}/{total} train {run_dir}", flush=True)
                run(config)


if __name__ == "__main__":
    main()
