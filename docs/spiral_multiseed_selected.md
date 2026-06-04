# Experiment Log: spiral_multiseed_selected

## Purpose

The `spiral_multiseed_selected` experiment tests whether the promising single-seed spiral results are reliable across random seeds.

The earlier weight sweep suggested:

- finite-difference Lagrangian consistency improves both Wasserstein distance and trajectory smoothness at stronger weights,
- JVP material-derivative regularization provides a strong smoothness control knob, with a clearer accuracy/smoothness tradeoff.

Because those results came from one seed, this experiment reruns selected methods across multiple seeds.

## Methods

We compare:

- `standard_cfm`
- `lc_finite_difference`, weight `2.0`
- `lc_finite_difference`, weight `4.0`
- `lc_jvp_material_derivative`, weight `0.001`
- `lc_jvp_material_derivative`, weight `0.0025`

Each method is run with seeds `0, 1, 2, 3, 4`.

## Stress Setting

Evaluation uses the same low-NFE noisy setting as the challenge sweep:

- Euler steps: `5`
- inference velocity noise: `0.25`
- epochs: `2000`
- batch size: `512`

## What We Are Testing

The key question is whether regularized training improves robustness and trajectory smoothness consistently, not just for seed `42`.

If finite-difference regularization keeps improving mean Wasserstein and acceleration across seeds, it becomes the strongest practical candidate. If JVP reliably lowers acceleration while trading off distribution quality, it remains valuable as an analytic acceleration-control ablation.
