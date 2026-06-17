#!/usr/bin/env python3
"""X2a: validate the dispersion head (E2) against the closed-form oracle (E0)
on a Gaussian-mixture run, and compare all three estimator profiles.

Checks, on interpolant probe points:
  1. r(x,t) vs oracle tr(Sigma)/d (relative error, correlation), both label modes.
  2. ||a_head(x,t)|| profile vs oracle ||a*|| profile vs self-probe (E1) profile,
     with the kappa each implies.
Saves an overlay figure next to the run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lcfm.dispersion import DispersionMLP, dispersion_value, head_is_positive, make_head_accel_fn
from lcfm.oracle import GaussianMixtureOracle
from lcfm.pairing import apply_pairing
from lcfm.plotting import load_run
from lcfm.schedules import interpolant_error_profile, kappa, rollout_error_profile
from lcfm.utils import set_seed


def load_head(run_dir, dim, mode, device):
    head = DispersionMLP(dim, positive=head_is_positive(mode)).to(device)
    head.load_state_dict(torch.load(Path(run_dir) / f"head_{mode}.pt", map_location=device, weights_only=True))
    head.eval()
    return head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--n-probe", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cpu")
    config, problem, model = load_run(args.run_dir, device)
    oracle = GaussianMixtureOracle.from_problem(problem)

    set_seed(7)
    x0 = problem.eval_initial(args.n_probe, device)
    x1 = problem.target_eval(args.n_probe, device)
    x0, x1 = apply_pairing(x0, x1, config)

    print(f"{'mode':>15} {'t':>5} {'mean r_head':>12} {'mean r_oracle':>13} {'rel err':>9} {'corr':>7}")
    for mode in ("residual", "second_moment", "residual_log"):
        head = load_head(args.run_dir, problem.dim, mode, device)
        for t_val in [0.05, 0.2, 0.5, 0.8]:
            t = torch.full((args.n_probe, 1), t_val)
            xt = (1 - t) * x0 + t * x1
            with torch.no_grad():
                r_head = dispersion_value(head, model, xt, t, mode)
            r_oracle = oracle.dispersion(xt, t)
            rel = ((r_head - r_oracle).abs().mean() / r_oracle.abs().mean()).item()
            stacked = torch.stack([r_head.flatten(), r_oracle.flatten()])
            corr = torch.corrcoef(stacked)[0, 1].item()
            print(
                f"{mode:>15} {t_val:5.2f} {r_head.mean().item():12.4f} "
                f"{r_oracle.mean().item():13.4f} {rel:9.3f} {corr:7.3f}"
            )

    ts_oracle, err_oracle = interpolant_error_profile(oracle.acceleration_target, x0, x1)
    set_seed(11)
    probe_x0 = problem.eval_initial(args.n_probe, device)
    ts_e1, err_e1 = rollout_error_profile(model, probe_x0)
    profiles = {"E0 oracle": (ts_oracle, err_oracle), "E1 self-probe": (ts_e1, err_e1)}
    for mode in ("residual", "second_moment", "residual_log"):
        head = load_head(args.run_dir, problem.dim, mode, device)
        accel = make_head_accel_fn(head, model, problem, device, mode=mode)
        profiles[f"E2 head ({mode})"] = interpolant_error_profile(accel, x0, x1)

    print(f"\n{'estimator':>22} {'kappa':>7} {'peak err':>9}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (ts, err) in profiles.items():
        print(f"{name:>22} {kappa(ts, err):7.3f} {err.max().item():9.2f}")
        ax.semilogy(ts, err + 1e-3, label=f"{name} (kappa={kappa(ts, err):.2f})")
    ax.set_xlabel("t")
    ax.set_ylabel("mean ||Dv/Dt||")
    ax.set_title(f"error-density profiles: {Path(args.run_dir).name}")
    ax.legend()
    fig.tight_layout()
    out = Path(args.run_dir) / "estimator_profiles.png"
    fig.savefig(out, dpi=180)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
