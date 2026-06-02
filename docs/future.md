# Future Work: Toward a True Upwind Flow-Matching Scheme

## Motivation

The current method regularizes Conditional Flow Matching by comparing the learned velocity at a sampled point with the velocity predicted at a nearby upstream point. This is best understood as a finite-difference velocity-consistency or material-derivative regularizer, not yet as a full numerical upwind scheme.

For a stronger scientific contribution, we should formulate a genuinely upwind-inspired training objective from the continuous transport geometry of flow matching.

## Core Task

Derive and test a direction-aware discretization of the convective/material derivative of the learned flow-matching velocity field:

$$
\frac{Dv}{Dt}
=
\partial_t v + (v \cdot \nabla_x)v
$$

The goal is to define a stable, characteristic-aware regularizer that controls how the learned velocity changes along the generated transport path.

## Candidate Formulation

Given a learned velocity field $v_\theta(x,t)$, define an upstream point along the local characteristic:

$$
x^- = x - \Delta t \, v_\theta(x,t)
$$

Then approximate the material derivative with an upwind/semi-Lagrangian finite difference:

$$
\frac{Dv_\theta}{Dt}(x,t)
\approx
\frac{
v_\theta(x,t) - v_\theta(x^-, t-\Delta t)
}{
\Delta t
}
$$

The current method penalizes the numerator of this expression. The future work is to make this approximation principled, compare it to alternative discretizations, and determine whether the target should be zero acceleration, bounded acceleration, or an analytically derived acceleration from the chosen probability path.

## Research Questions

1. What is the correct continuous quantity to regularize: material derivative, curvature, acceleration magnitude, or deviation from an analytically known path acceleration?
2. Is zero material derivative appropriate, or does it over-constrain expressive generative flows?
3. Can the upwind discretization be made consistent as $\Delta t \to 0$?
4. Does the method reduce solver error at low NFE without degrading distributional accuracy?
5. Does it provide advantages over existing velocity-consistency, curvature, rectification, and acceleration-regularization methods?
6. Can the method be extended to physical or sequential data where characteristic directions have a clearer meaning?

## Required Baselines

Any true-upwind claim should be compared against:

- Standard Conditional Flow Matching
- OT-CFM
- Rectified Flow / Reflow
- Consistency Flow Matching
- Isokinetic or acceleration-regularized Flow Matching
- Trajectory-curvature regularization
- Temporal pair consistency
- Jacobian or Lipschitz regularization
- Simple inference-time velocity smoothing
- Higher-order ODE solvers such as Heun, RK4, and adaptive solvers

## Minimum Experiments

- 2D toy distributions with multi-seed statistics
- NFE-quality curves without artificial solver noise
- Robustness sweeps with controlled perturbations
- Ablations over $\Delta t$, regularization weight, and discretization direction
- Comparisons against equal-compute baselines
- Sequential or PDE data with physics-aware metrics such as residual error, conservation error, rollout RMSE, and spectral error

## Success Criteria

This direction is worth developing into a paper only if it can show at least one of the following:

- Better few-step generation than strong rectification/consistency baselines
- Lower trajectory curvature or acceleration without loss of sample quality
- More stable rollouts on physical/sequential data
- A clear theoretical stability or consistency result
- A genuinely useful discretization principle that existing flow-matching regularizers do not already capture

## Positioning

The current method should be presented as a preliminary semi-Lagrangian velocity-consistency regularizer. The future method should only be called an upwind scheme if we can derive and validate a discretization that meaningfully matches the numerical PDE concept of upwinding.
