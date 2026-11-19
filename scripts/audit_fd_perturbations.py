#!/usr/bin/env python3
"""Audit centered-FD convergence to the exact discrete-Heun JVP profile."""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm import datasets  # noqa: F401
from lcfm import models  # noqa: F401
from lcfm.models import build_model
from lcfm.registry import DATASETS, get
from lcfm.schedules import equal_error_grid, tangent_defect_profile, transported_defect_profile
from lcfm.utils import read_json, set_seed, write_json


def parse_floats(text):
    return [float(part) for part in text.replace(",", " ").split() if part]


def parse_ints(text):
    return [int(part) for part in text.replace(",", " ").split() if part]


def normalized_l2(candidate, reference):
    candidate = candidate.double() / candidate.double().norm().clamp_min(1e-12)
    reference = reference.double() / reference.double().norm().clamp_min(1e-12)
    return float((candidate - reference).norm().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pilot-samples", type=int, default=512)
    parser.add_argument("--pilot-seed", type=int, default=1311)
    parser.add_argument("--fine-steps", type=int, default=50)
    parser.add_argument("--anchor-count", type=int, default=16)
    parser.add_argument("--perturbations", default="0.01,0.003,0.001,0.0003,0.0001,0.00003,0.00001")
    parser.add_argument("--nfe-values", default="5,10,20,50")
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument("--floor", type=float, default=1e-3)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(run_dir / "config.json")
    device = torch.device(args.device)
    problem = get(DATASETS, config["dataset"])(config.get("dataset_kwargs", {}))
    model = build_model(config.get("model", "mlp"), problem.dim, config).to(device)
    state = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model_state", state) if isinstance(state, dict) else state)
    model.eval()
    set_seed(args.pilot_seed)
    pilot = problem.eval_initial(args.pilot_samples, device)

    ts, jvp, local, gain = tangent_defect_profile(
        model, pilot, fine_steps=args.fine_steps, anchor_count=args.anchor_count
    )
    jvp_grids = {
        nfe: equal_error_grid(ts, jvp, nfe, power=args.power, floor=args.floor, end=1.0)
        for nfe in parse_ints(args.nfe_values)
    }
    rows = []
    profiles = {"jvp": [float(value) for value in jvp]}
    for perturbation in parse_floats(args.perturbations):
        fd_ts, fd, _, _ = transported_defect_profile(
            model,
            pilot,
            fine_steps=args.fine_steps,
            anchor_count=args.anchor_count,
            perturbation=perturbation,
        )
        if not torch.allclose(ts, fd_ts, rtol=0.0, atol=1e-12):
            raise ValueError("FD and JVP anchor times differ.")
        profile_error = normalized_l2(fd, jvp)
        profiles[f"fd_{perturbation:g}"] = [float(value) for value in fd]
        for nfe, jvp_grid in jvp_grids.items():
            fd_grid = equal_error_grid(ts, fd, nfe, power=args.power, floor=args.floor, end=1.0)
            displacement = max(abs(left - right) for left, right in zip(fd_grid, jvp_grid))
            rows.append(
                {
                    "perturbation": perturbation,
                    "nfe": nfe,
                    "normalized_profile_l2": profile_error,
                    "max_knot_displacement": displacement,
                }
            )

    with (out_dir / "fd_perturbation_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        out_dir / "fd_perturbation_audit.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "pilot_samples": args.pilot_samples,
            "pilot_seed": args.pilot_seed,
            "fine_steps": args.fine_steps,
            "anchor_count": args.anchor_count,
            "power": args.power,
            "ts": [float(value) for value in ts],
            "local_acceleration": [float(value) for value in local],
            "jvp_terminal_gain": [float(value) for value in gain],
            "profiles": profiles,
            "rows": rows,
        },
    )
    print(f"Saved FD perturbation audit to {out_dir}")


if __name__ == "__main__":
    main()
