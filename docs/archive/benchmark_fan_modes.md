# Fan-Mode Asymmetric Transport Benchmark

## Purpose

The centered five-mode benchmark is useful, but its radial symmetry makes many methods look visually similar. The fan-mode benchmark keeps the same independent-pairing multimodal target idea while moving the source distribution far to the left of the target modes.

This creates a clearer left-to-right transport direction before the flow branches into modes. It is meant to stress the directional-regularization hypothesis more directly:

> If coarse-step error is governed by local motion along the learned velocity field, a method that regularizes directionally should help most when the transport has a coherent advective direction and then must branch.

## Dataset

Source distribution:

$$
x_0 \sim \mathcal{N}((-8,0), I).
$$

Target distribution:

$$
x_1 \sim \frac{1}{5}\sum_{k=1}^{5}
\mathcal{N}(\mu_k,\sigma_{\mathrm{mode}}^2 I),
$$

with default fan centers:

$$
\mu_k \in
\{(4,-4),(5,-2),(5.5,0),(5,2),(4,4)\}.
$$

The default mode scale is `sigma_mode = 0.2`, and train pairs are independently sampled. There is no OT pairing in this benchmark.

## Initial Comparisons

This branch currently has runnable configs for:

- Standard CFM
- Iso-FM-style finite-difference regularization
- older backward LC finite-difference regularization
- JVP material-derivative regularization
- Directional-Regularization CFM with finite-difference residual and finite-difference directional weight

The first Directional-Regularization CFM variant uses the Iso-FD-style forward material residual, then multiplies it by a detached solver-risk weight estimated from a finite difference along the current model velocity.

## Evaluation

Use the same low-NFE regime as the centered five-mode benchmark:

- Euler solver
- `steps = 5`
- fixed evaluation seed across methods
- same number of generated samples per method

Report:

- Wasserstein distance to the target sample cloud
- mode hit coverage, using the 3-sigma hit radius
- target hit rate
- path length ratio
- trajectory acceleration

The plot should show all transported prior samples when possible. This matters here because the paper claim concerns acceleration and transport trajectories, not only final clouds.

## Scientific Caveat

This benchmark is only helpful if it separates behaviors. If Standard CFM, Iso-FD, and directional regularization all produce nearly identical trajectories and metrics, the geometry is still too easy or the regularizer is not active in a meaningful way.

In that case, the next controlled changes should be:

- increase the source-target distance,
- narrow the target modes,
- reduce inference steps below five,
- or make the fan more curved so branch commitment is delayed.

The goal is not to manufacture a flattering toy problem. The goal is to find a simple diagnostic where directional regularization has a principled opportunity to matter and where failures are visible.
