#!/usr/bin/env python3
"""Numerically verify the exact momentum-balance law of linear-interpolant CFM:

    Dv/Dt = dv/dt + (v . grad) v = -(1/p) div(p Sigma) = -div(Sigma) - Sigma grad(log p),

with v = E[u|x_t], Sigma = Cov[u|x_t], on the clumped five-mode geometry where all
fields are closed form (lcfm.oracle, estimator E0). Also cross-checks the fast
batched acceleration_target (torch.func + analytic score) against slow pointwise
autograd, and its behavior under no_grad.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import torch

from lcfm.oracle import GaussianMixtureOracle

torch.set_default_dtype(torch.float64)


def pointwise(fn, x, t):
    return fn(x.unsqueeze(0), t.reshape(1, 1)).squeeze(0)


def slow_material_and_source(oracle, x, t):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    jac_v_x = torch.autograd.functional.jacobian(lambda xx: pointwise(oracle.velocity, xx, t), x)
    dv_dt = torch.autograd.functional.jacobian(lambda tt: pointwise(oracle.velocity, x, tt), t)
    jac_sigma_x = torch.autograd.functional.jacobian(lambda xx: pointwise(oracle.sigma, xx, t), x)
    score = torch.autograd.grad(pointwise(oracle.log_p, x, t), x)[0]
    v = pointwise(oracle.velocity, x, t)
    sigma = pointwise(oracle.sigma, x, t)
    material = dv_dt.squeeze(-1) + jac_v_x @ v
    source = -(torch.einsum("ijj->i", jac_sigma_x) + sigma @ score)
    return material, source, score


def main():
    torch.manual_seed(0)
    angles = torch.arange(5) * (2 * math.pi / 5)
    centers = 4.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    oracle = GaussianMixtureOracle(centers, sigma_mode=0.20, source_mean=[0.0, 0.0], source_std=0.15)

    points, times = [], []
    worst_identity, worst_score = 0.0, 0.0
    for t_val in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        for _ in range(4):
            x = 3.0 * torch.randn(2)
            t = torch.tensor(t_val)
            material, source, score = slow_material_and_source(oracle, x, t)
            rel = (material - source).norm().item() / (material.norm().item() + 1e-12)
            worst_identity = max(worst_identity, rel)
            worst_score = max(worst_score, (pointwise(oracle.score, x, t) - score).norm().item())
            points.append(x)
            times.append(t.reshape(1))

    print(f"max relative identity residual: {worst_identity:.3e}")
    print(f"max analytic-vs-autograd score error: {worst_score:.3e}")
    if worst_identity > 1e-8 or worst_score > 1e-8:
        raise SystemExit("momentum identity check FAILED")

    x_batch = torch.stack(points)
    t_batch = torch.stack(times)
    fast = oracle.acceleration_target(x_batch, t_batch)
    slow = torch.stack(
        [slow_material_and_source(oracle, x, t.squeeze())[1] for x, t in zip(points, times)]
    )
    batched_err = (fast - slow).abs().max().item()
    print(f"max batched-target vs pointwise error: {batched_err:.3e}")
    if batched_err > 1e-8:
        raise SystemExit("batched acceleration_target check FAILED")

    with torch.no_grad():
        no_grad_err = (oracle.acceleration_target(x_batch, t_batch) - fast).abs().max().item()
    print(f"max no_grad-context discrepancy: {no_grad_err:.3e}")
    if no_grad_err > 0.0:
        raise SystemExit("acceleration_target changes under no_grad FAILED")
    print("momentum identity check passed: Dv/Dt = -(1/p) div(p Sigma) exactly")


if __name__ == "__main__":
    main()
