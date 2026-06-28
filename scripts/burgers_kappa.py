#!/usr/bin/env python3
"""X0 entry for burgers_autoregressive: self-probe (E1) stiffness profiles of the
frame-to-frame flow at several rollout depths.

The coupling is paired (frame_k -> frame_{k+1}): a nearly deterministic, cold
coupling, so the law predicts kappa near 1 and little to gain from warping.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch

from lcfm.datasets import BurgersAutoregressiveProblem
from lcfm.models import build_model
from lcfm.schedules import euler_on_grid, kappa, rollout_error_profile
from lcfm.utils import read_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--transitions", nargs="+", type=int, default=[0, 8, 16, 24])
    args = parser.parse_args()

    device = torch.device("cpu")
    config = read_json(Path(args.run_dir) / "config.json")
    kwargs = dict(config.get("dataset_kwargs", {}))
    kwargs["n_train"] = 8  # probe needs only test frames; skip regenerating training sims
    set_seed(config.get("seed", 42))
    problem = BurgersAutoregressiveProblem(kwargs)
    model = build_model(config.get("model", "unet1d"), problem.dim, config).to(device)
    model.load_state_dict(torch.load(Path(args.run_dir) / "model.pt", map_location=device, weights_only=True))
    model.eval()

    x = problem.eval_initial(min(32, problem.n_test), device)
    print(f"{'transition':>10} {'kappa_E1':>9} {'peak/mean err':>14}")
    kappas = []
    for step in range(max(args.transitions) + 1):
        if step in args.transitions:
            ts, err = rollout_error_profile(model, x, fine_steps=40)
            kap = kappa(ts, err)
            kappas.append(kap)
            print(f"{step:>10} {kap:9.3f} {err.max().item() / err.mean().item():14.2f}")
        x = euler_on_grid(model, x, torch.linspace(0, 1, 21).tolist())[-1]
    print(f"\nmean kappa_E1 over transitions: {sum(kappas)/len(kappas):.3f}")


if __name__ == "__main__":
    main()
