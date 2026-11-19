#!/usr/bin/env python3
"""Small deterministic checks for schedule calibration utilities."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lcfm.schedules import (
    calibrate_schedule_profiles,
    euler_defect_cost_matrix,
    optimal_defect_grid,
    profile_equivalence_rows,
    randomized_adjoint_defect_profiles,
    schedule_specs_from_bundle,
    tangent_defect_profile,
    transported_defect_profile,
    validate_profile_bundle,
    vsfm_time_grid,
)


class QuadraticVelocity(torch.nn.Module):
    def forward(self, x, t):
        return (1.0 + t) * x


class QuadraticStateVelocity(torch.nn.Module):
    def forward(self, x, t):
        return x.square()


class LabelVelocity(torch.nn.Module):
    def forward(self, x, t, labels):
        return (1.0 + t + labels[:, None].to(x.dtype)) * x


def assert_grid(grid, steps):
    assert len(grid) == steps + 1
    assert grid[0] == 0.0
    assert grid[-1] == 1.0
    assert all(right > left for left, right in zip(grid, grid[1:]))


def main():
    for profile in ("coasting", "uniform_deceleration"):
        assert_grid(vsfm_time_grid(8, profile), 8)

    anchors = torch.linspace(0.0, 1.0, 6)
    costs = torch.full((6, 6), torch.inf)
    for start in range(5):
        for end in range(start + 1, 6):
            costs[start, end] = (end - start) ** 2
    grid, objective = optimal_defect_grid(costs, anchors, steps=3)
    assert_grid(grid, 3)
    assert objective == 9.0

    model = QuadraticVelocity()
    pilot = torch.tensor([[0.5], [1.0]], dtype=torch.float32)
    anchor_grid, edge_costs = euler_defect_cost_matrix(model, pilot, anchor_steps=12, reference_solver="heun")
    assert anchor_grid.shape == (13,)
    assert edge_costs.shape == (13, 13)
    assert torch.isfinite(edge_costs[0, 1:]).all()
    assert (edge_costs[0, 1:] >= 0).all()
    assert_grid(optimal_defect_grid(edge_costs, anchor_grid, steps=4)[0], 4)

    ts, transported, local, gain = transported_defect_profile(
        model,
        pilot,
        fine_steps=12,
        anchor_count=4,
        perturbation=1e-3,
    )
    assert ts.shape == transported.shape == local.shape == gain.shape == (6,)
    assert ts[0] == 0.0 and ts[-1] == 1.0
    assert torch.all(ts[1:] > ts[:-1])
    assert torch.isfinite(transported).all() and torch.all(transported >= 0)
    assert torch.isfinite(local).all() and torch.all(local >= 0)
    assert torch.isfinite(gain).all() and torch.all(gain >= 0)

    tangent_ts, tangent, tangent_local, tangent_gain = tangent_defect_profile(
        model,
        pilot,
        fine_steps=12,
        anchor_count=4,
    )
    assert torch.equal(ts, tangent_ts)
    assert torch.allclose(transported, tangent, rtol=2e-3, atol=2e-3)
    assert torch.allclose(local, tangent_local, rtol=1e-6, atol=1e-6)
    assert torch.allclose(gain, tangent_gain, rtol=2e-3, atol=2e-3)

    dt = 1.0 / 12
    step_factors = []
    for index in range(12):
        c0 = 1.0 + index * dt
        c1 = 1.0 + (index + 1) * dt
        step_factors.append(1.0 + 0.5 * dt * (c0 + c1 * (1.0 + dt * c0)))
    anchors = torch.linspace(0, 11, 4, dtype=torch.float64).round().long().unique().tolist()
    analytic_gain = []
    for anchor in anchors:
        value = 1.0
        for factor in step_factors[anchor + 1 :]:
            value *= factor
        analytic_gain.append(value)
    expected_gain = torch.tensor([analytic_gain[0], *analytic_gain, analytic_gain[-1]], dtype=torch.float64)
    assert torch.allclose(tangent_gain, expected_gain, rtol=2e-6, atol=2e-6)

    quadratic = QuadraticStateVelocity()
    quadratic_pilot = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
    q_ts, q_jvp, _, q_gain = tangent_defect_profile(quadratic, quadratic_pilot, fine_steps=6, anchor_count=3)
    _, q_fd, _, _ = transported_defect_profile(
        quadratic,
        quadratic_pilot,
        fine_steps=6,
        anchor_count=3,
        perturbation=1e-5,
    )
    assert torch.allclose(q_jvp, q_fd, rtol=2e-4, atol=2e-5)
    h = 1.0 / 6
    states = quadratic_pilot.clone()
    derivatives = []
    for _ in range(6):
        predictor = states + h * states.square()
        derivatives.append(1.0 + h * states + h * predictor * (1.0 + 2.0 * h * states))
        states = states + 0.5 * h * (states.square() + predictor.square())
    q_anchors = torch.linspace(0, 5, 3, dtype=torch.float64).round().long().unique().tolist()
    q_expected = []
    for anchor in q_anchors:
        propagated = torch.ones_like(quadratic_pilot)
        for derivative in derivatives[anchor + 1 :]:
            propagated = propagated * derivative
        q_expected.append(float(propagated.abs().mean().item()))
    q_expected = torch.tensor([q_expected[0], *q_expected, q_expected[-1]], dtype=torch.float64)
    assert torch.allclose(q_ts, torch.tensor([0.0, 1 / 12, 5 / 12, 11 / 12, 1.0], dtype=torch.float64))
    assert torch.allclose(q_gain, q_expected, rtol=2e-6, atol=2e-6)

    adjoint_profiles = randomized_adjoint_defect_profiles(
        model,
        pilot,
        fine_steps=12,
        anchor_count=4,
        probe_counts=(2, 4, 8),
        seed=7,
    )
    assert set(adjoint_profiles) == {2, 4, 8}
    for adjoint_ts, adjoint, adjoint_local, adjoint_gain in adjoint_profiles.values():
        assert torch.equal(ts, adjoint_ts)
        assert torch.isfinite(adjoint).all() and torch.all(adjoint >= 0)
        assert torch.allclose(local, adjoint_local, rtol=1e-6, atol=1e-6)
        assert torch.isfinite(adjoint_gain).all() and torch.all(adjoint_gain >= 0)

    labels = torch.tensor([0, 1], dtype=torch.long)
    conditional_model = LabelVelocity()
    conditional_velocity = lambda x, t: conditional_model(x, t, labels)
    bundle = calibrate_schedule_profiles(
        [(conditional_velocity, pilot)],
        estimators=("local", "transport_fd", "transport_jvp", "transport_adjoint"),
        fine_steps=12,
        anchor_count=4,
        adjoint_probe_counts=(4,),
        adjoint_seed=7,
        perturbations=(1e-3, 3e-3),
        powers=(0.25, 0.5),
        nfe_values=(5, 10),
        context={"dataset": "toy", "checkpoint_sha256": "abc"},
    )
    replayed_bundle = json.loads(json.dumps(bundle))
    assert replayed_bundle == bundle
    validate_profile_bundle(
        bundle,
        {"dataset": "toy", "checkpoint_sha256": "abc"},
        {"pilot_seed": 1311, "fine_steps": 12, "nfe_values": [5, 10]},
    )
    try:
        validate_profile_bundle(bundle, {"dataset": "toy", "checkpoint_sha256": "wrong"})
    except ValueError:
        pass
    else:
        raise AssertionError("Profile cache context mismatch was not rejected.")
    try:
        validate_profile_bundle(
            bundle,
            {"dataset": "toy", "checkpoint_sha256": "abc"},
            {"pilot_seed": 999},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Profile cache calibration mismatch was not rejected.")
    assert set(bundle["profiles"]) == {
        "local",
        "transport_fd_eps0p001",
        "transport_fd_eps0p003",
        "transport_jvp",
        "transport_adjoint_r4",
    }
    equivalence = profile_equivalence_rows(bundle)
    fd_rows = [row for row in equivalence if row["profile"].startswith("transport_fd_")]
    assert fd_rows
    assert max(row["normalized_profile_l2"] for row in fd_rows) < 0.02
    assert max(row["max_knot_displacement"] for row in fd_rows) < 0.02
    specs = schedule_specs_from_bundle(
        bundle,
        nfe=5,
        methods=("uniform", "sctw", "adjoint_sctw", "power", "vsfm", "heun"),
        power_rhos=(2.0,),
        power_kinds=("early",),
    )
    names = {spec["schedule"] for spec in specs}
    assert {"uniform", "sctw_p0p25", "sctw_p0p5", "adjoint_sctw_p0p25", "adjoint_sctw_p0p5"} <= names
    for spec in specs:
        assert_grid(spec["grid"], len(spec["grid"]) - 1)
        multiplier = 1 if spec["solver"] == "euler" else 2
        assert spec["nfe_used"] == multiplier * (len(spec["grid"]) - 1)
    replayed_specs = schedule_specs_from_bundle(
        replayed_bundle,
        nfe=5,
        methods=("sctw", "adjoint_sctw"),
    )
    original_specs = schedule_specs_from_bundle(
        bundle,
        nfe=5,
        methods=("sctw", "adjoint_sctw"),
    )
    assert replayed_specs == original_specs
    for original, replayed in zip(original_specs, replayed_specs):
        original_sample = pilot.clone()
        replayed_sample = pilot.clone()
        for left, right in zip(original["grid"], original["grid"][1:]):
            t = torch.full_like(original_sample, left)
            original_sample = original_sample + (right - left) * model(original_sample, t)
        for left, right in zip(replayed["grid"], replayed["grid"][1:]):
            t = torch.full_like(replayed_sample, left)
            replayed_sample = replayed_sample + (right - left) * model(replayed_sample, t)
        assert torch.equal(original_sample, replayed_sample)
    print("Schedule calibration checks passed.")


if __name__ == "__main__":
    main()
