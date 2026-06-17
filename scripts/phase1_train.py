#!/usr/bin/env python3
"""Phase-1 training sweep: standard CFM (training untouched) across geometries
and couplings, then post-hoc dispersion heads (E2) per run.

Geometries: clumped015, ring, fan, spiral. Couplings: independent, minibatch OT.
Each run dir gets model.pt plus head_residual.pt and head_second_moment.pt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import ProcessPoolExecutor

GEOMETRIES = {
    "clumped015": {
        "dataset": "five_modes",
        "dataset_kwargs": {"n_train": 5000, "n_test": 2000, "radius": 4.0, "sigma_mode": 0.2, "source_std": 0.15},
        "eval": {"n_eval": 1000, "mode_p_min": 0.05, "hit_radius": 0.6, "eval_seed": 1234, "plot_seed": 1234},
    },
    "ring": {
        "dataset": "five_modes",
        "dataset_kwargs": {"n_train": 5000, "n_test": 2000, "radius": 4.0, "sigma_mode": 0.2},
        "eval": {"n_eval": 1000, "mode_p_min": 0.05, "hit_radius": 0.6, "eval_seed": 1234, "plot_seed": 1234},
    },
    "fan": {
        "dataset": "fan_modes",
        "dataset_kwargs": {"n_train": 5000, "n_test": 2000, "sigma_mode": 0.2},
        "eval": {"n_eval": 1000, "mode_p_min": 0.05, "hit_radius": 0.6, "eval_seed": 1234, "plot_seed": 1234},
    },
    "staged": {
        "dataset": "staged_modes",
        "dataset_kwargs": {"n_train": 5000, "n_test": 2000, "sigma_mode": 0.2, "source_std": 0.15},
        "eval": {"n_eval": 1000, "mode_p_min": 0.05, "hit_radius": 0.6, "eval_seed": 1234, "plot_seed": 1234},
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
        "eval": {"n_eval": 1000, "mode_p_min": 0.05, "hit_radius": 1.6, "eval_seed": 1234, "plot_seed": 1234},
    },
    "spiral": {
        "dataset": "spiral",
        "dataset_kwargs": {"n_train": 2000, "n_test": 1000, "noise": 0.15},
        "eval": {"n_eval": 1000, "eval_seed": 1234, "plot_seed": 1234},
    },
}
COUPLINGS = ["independent", "minibatch_ot"]


def make_config(geometry, coupling, seed, out_dir):
    base = GEOMETRIES[geometry]
    group = f"{geometry}_{coupling}"
    return {
        "dataset": base["dataset"],
        "dataset_kwargs": dict(base["dataset_kwargs"]),
        "method": "standard_cfm",
        "method_kwargs": {},
        "pairing": coupling,
        "model": "mlp",
        "seed": seed,
        "solver": "euler",
        "solver_kwargs": {"steps": 5, "noise": 0.0},
        "train": {"batch_size": 256, "epochs": 2000, "log_every": 2000, "lr": 0.001},
        "eval": dict(base["eval"]),
        "out_dir": str(Path(out_dir) / group),
        "run_name": f"{group}_seed{seed}",
    }


def execute(config):
    import torch

    torch.set_num_threads(2)
    from lcfm.dispersion import train_dispersion_head
    from lcfm.experiment import run
    from lcfm.plotting import load_run

    run_dir = Path(config["out_dir"]) / config["run_name"]
    if not (run_dir / "model.pt").exists():
        run(config)
    device = torch.device("cpu")
    _, problem, model = load_run(run_dir, device)
    for mode in ("residual", "second_moment", "residual_log"):
        head_path = run_dir / f"head_{mode}.pt"
        if head_path.exists():
            continue
        head, _ = train_dispersion_head(problem, model, config, device, mode=mode)
        torch.save(head.state_dict(), head_path)
    return config["run_name"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/phase1/runs")
    parser.add_argument("--geometries", nargs="+", default=list(GEOMETRIES), choices=list(GEOMETRIES))
    parser.add_argument("--couplings", nargs="+", default=COUPLINGS, choices=COUPLINGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    configs = [
        make_config(geometry, coupling, seed, args.out)
        for geometry in args.geometries
        for coupling in args.couplings
        for seed in args.seeds
    ]
    print(f"{len(configs)} runs with {args.jobs} workers")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for name in pool.map(execute, configs):
            print(f"done {name}")


if __name__ == "__main__":
    main()
