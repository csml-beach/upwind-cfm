# Working Plan: Uncertainty-Aware Streamline Stabilization

## Current Position

The previous finite-difference Lagrangian consistency loss should no longer be treated as the main novelty. It is best understood as a semi-Lagrangian / characteristic finite-difference approximation of the material derivative,

$$
\frac{Dv_\theta}{Dt}
=
\partial_t v_\theta + (v_\theta \cdot \nabla_x)v_\theta,
$$

and is conceptually very close to Isokinetic Flow Matching's Jacobian-free material-derivative regularizer. It remains valuable as a baseline and implementation scaffold, but it is not enough by itself for a strong paper claim.

The working research idea is now:

> Flow matching should be stabilized where coarse ODE integration is numerically fragile, but acceleration should not be suppressed uniformly in regions where multimodal uncertainty makes acceleration necessary.

## Proposed Method Direction

Define the material residual

$$
R_\theta(x,t)
=
\partial_t v_\theta(x,t)
+
(v_\theta(x,t)\cdot\nabla_x)v_\theta(x,t).
$$

Instead of penalizing $\|R_\theta\|^2$ uniformly, use a solver-aware and uncertainty-aware stabilization weight:

$$
\mathcal{L}_{stab}
=
\tau(x,t)\|R_\theta(x,t)\|^2.
$$

The intended structure is

$$
\tau(x,t)
=
\tau_{\mathrm{CFL}}(x,t)
\,
g_{\mathrm{uncertainty}}(x,t).
$$

The CFL/SUPG-style part should grow when the learned field is likely to cause coarse-solver error. A simple first form is

$$
\tau_{\mathrm{CFL}}(x,t)
\approx
\frac{\Delta t_{\mathrm{infer}}}
{1 + \Delta t_{\mathrm{infer}} L_\theta(x,t)},
$$

where $L_\theta$ is a local stiffness proxy such as a directional Jacobian norm, finite-difference velocity change, or local Lipschitz estimate.

The uncertainty gate should reduce regularization where the conditional transport is ambiguous:

$$
g_{\mathrm{uncertainty}}(x,t)
=
\frac{1}
{1+\kappa \widehat{\mathrm{Var}}[u\mid x_t,t]}.
$$

As a first ablation, a time-only gate such as $g(t)=t^\beta$ is acceptable, but the paper should not stop there. The more scientific version needs a measurable uncertainty proxy.

## Paper Narrative

When we overhaul the writeup, the derivation should proceed in this order:

1. Start from the Lagrangian view of a sample moving under the learned flow-matching ODE.
2. Introduce the material derivative as pathwise acceleration.
3. Explain why large material residuals cause low-NFE solver error.
4. Explain why zero acceleration is not globally appropriate for multimodal generative transport.
5. Introduce a stabilized residual penalty with a CFL/SUPG-style weight.
6. Add uncertainty-aware gating so the method regularizes unnecessary acceleration more than necessary mode-commitment acceleration.
7. Treat semi-Lagrangian finite-difference LC and Iso-FM as close baselines, not as the main novelty.

## Immediate Benchmark Needs

The first benchmark should expose both sides of the idea:

- solver fragility under coarse inference,
- and early-time/multimodal ambiguity where uniform acceleration suppression may over-constrain.

Useful metrics:

- low-NFE Wasserstein or task error,
- NFE-quality curves,
- trajectory acceleration / material residual,
- path length ratio,
- mode coverage or mode assignment accuracy on multimodal toy data,
- sensitivity to regularization strength and uncertainty-gate strength.

## Baselines

Required initial baselines:

- Standard CFM
- LC finite difference as our semi-Lagrangian/Iso-FM-style variant
- Iso-FM-faithful finite-difference loss if its weighting/normalization differs materially
- JVP material-derivative penalty
- CFL/SUPG-style stabilization without uncertainty gate
- uncertainty-gated stabilization without CFL/stiffness weighting
- full proposed CFL/SUPG plus uncertainty gate

Later baselines:

- OT-CFM
- Rectified Flow / Reflow
- Consistency Flow Matching
- Temporal Pair Consistency
- higher-order ODE solvers such as Heun, RK4, and adaptive solvers

## Success Criteria

This direction is worth developing only if it can show at least one of the following:

- better few-step generation than strong acceleration-regularization baselines,
- less over-smoothing or better mode coverage than uniform material-derivative suppression,
- improved stability on autoregressive or PDE-like tasks,
- a clear solver-aware explanation that predicts when the regularizer should help,
- a meaningful ablation showing that CFL weighting and uncertainty gating contribute differently.
