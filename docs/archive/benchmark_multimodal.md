# First Benchmark: Five-Mode Independent Transport

## Purpose

This benchmark is designed to test the central claim of uncertainty-aware streamline stabilization:

> Uniform material-residual suppression may improve smoothness, but it can over-constrain multimodal transport. A useful method should improve coarse-step sampling while preserving mode coverage.

The benchmark should be simple enough to run quickly across seeds, but sharp enough to expose failure modes that a smoothness-only metric would miss.

## Dataset

Use a two-dimensional independent-pairing flow-matching task.

Source distribution:

$$
x_0 \sim \mathcal{N}(0,I).
$$

Target distribution: five isotropic Gaussian modes arranged on a circle:

$$
x_1 \sim \frac{1}{5}\sum_{k=1}^{5}
\mathcal{N}(\mu_k, \sigma_{\mathrm{mode}}^2 I),
$$

where

$$
\mu_k = r
\begin{bmatrix}
\cos(2\pi k/5) \\
\sin(2\pi k/5)
\end{bmatrix}.
$$

Initial defaults:

- `r = 4.0`
- `sigma_mode = 0.20`
- independent source-target pairing
- no minibatch OT pairing in the first benchmark

Independent pairing is important because it creates genuine multimodal ambiguity in the marginal velocity field. The benchmark should not be simplified by using OT pairing before we understand whether the uncertainty gate is useful.

## Training Path

Use the standard linear conditional flow-matching path:

$$
x_t = (1-t)x_0 + t x_1,
$$

with target velocity

$$
u = x_1 - x_0.
$$

The learned model predicts

$$
v_\theta(x_t,t) \approx u.
$$

## Methods To Compare

Start with a deliberately small set:

1. **Standard CFM**
2. **Iso-FM-style finite difference**
3. **Uniform JVP material residual**
4. **Time-gated material residual**
5. **Solver-risk material residual**
6. **Solver-risk plus uncertainty-gated material residual**

Do not include WENO, Hessian/jerk regularization, Burgers, Reflow, or Consistency Flow Matching in this first benchmark. Those are later-stage comparisons once this diagnostic is understood.

The Iso-FM-style finite-difference baseline should be kept as close as possible to the paper's practical regularizer:

$$
x_{t+\epsilon} = x_t + \epsilon\,\mathrm{sg}(v_\theta(x_t,t)),
$$

$$
\mathcal{L}_{Iso}
=
\mathbb{E}
\left[
\frac{(1-t)^\alpha}{\epsilon}
\left\|
\frac{
v_\theta(x_t,t) -
\mathrm{sg}(v_\theta(x_{t+\epsilon},t+\epsilon))
}{
\|\mathrm{sg}(v_\theta(x_t,t))\|_2+\zeta
}
\right\|_1
\right].
$$

This is different from the older backward LC finite-difference loss in the repository. The older loss can remain useful as an ablation, but the literature baseline should use the forward lookahead, stop-gradient target, velocity normalization, and temporal weighting above.

## Inference Setting

Evaluate at one deliberately coarse inference setting first:

- Euler solver
- `steps = 5`
- fixed evaluation seeds across methods
- same number of generated samples per method

No NFE-quality curve is required for the first pass. The first question is whether the method changes behavior in a fixed low-NFE regime.

After the benchmark is informative, we can add a small NFE sweep later if needed.

## Metrics

Report:

- Wasserstein or sliced Wasserstein distance to the target sample cloud
- mode hit coverage: number of target modes receiving enough generated samples inside the hit radius
- target hit rate: fraction of generated samples that land inside any target basin
- trajectory acceleration or material residual magnitude
- path length ratio

Do not use assignment entropy in the first benchmark.

## Mode Hit Metrics

Assign each generated final point to the nearest target mode center:

$$
\hat{k}(x) = \arg\min_k \|x-\mu_k\|_2.
$$

A generated sample counts as a mode hit only if it lands inside the outer plotted circle. We intentionally avoid reporting nearest-mode-only coverage or balance, because samples near the origin are still assigned to some closest mode even when they have not reached the target distribution.

Use

$$
r_{\mathrm{hit}} = 3\sigma_{\mathrm{mode}}.
$$

Hit assignment:

$$
h_i =
\mathbf{1}
\left[
\|x_i-\mu_{\hat{k}(x_i)}\|_2
\le
r_{\mathrm{hit}}
\right].
$$

Mode hit probability:

$$
\hat{p}^{hit}_k
=
\frac{1}{N}
\sum_i
\mathbf{1}
\left[
\hat{k}(x_i)=k
\right]
h_i.
$$

Mode hit coverage:

$$
\mathrm{hit\_coverage}
=
\sum_{k=1}^{5}
\mathbf{1}
\left[
\hat{p}^{hit}_k > p_{\min}
\right],
$$

where $p_{\min}$ is the minimum generated mass required for a mode to count as covered.

Target hit rate:

$$
\mathrm{hit\_rate}
=
\frac{1}{N}
\sum_i h_i.
$$

Initial default:

- `p_min = 0.05`

## Evidence That Supports The Idea

The uncertainty-aware method is promising if it:

- improves or matches final distribution quality against Standard CFM,
- improves low-NFE behavior against uniform material-residual regularization,
- preserves better hit coverage and target hit rate than uniform residual suppression,
- reduces trajectory acceleration enough to explain improved coarse-step sampling,
- does not only win by making trajectories smooth while collapsing modes.

## Evidence That Weakens The Idea

The idea is in trouble if:

- uniform residual regularization dominates all uncertainty-aware variants,
- the uncertainty gate improves mode coverage only by removing useful stabilization,
- target hit rate remains poor despite lower acceleration,
- the method is highly sensitive to gate strength or regularization weight,
- gains disappear across random seeds.

## First Implementation Target

The first implementation should add:

- a `five_modes` dataset,
- hit-based mode metrics,
- configs for the six methods above,
- one comparison plotting script/view that shows final samples and a few trajectories.

Keep this benchmark small, fast, and reproducible. It is a diagnostic, not the final paper experiment.

## Initial Experimental Observations

**Iso-FM Overregularization (2026-06-05):** Initial experiments with Iso-FM finite difference using hyperparameters from the literature (weight=4.0, epsilon=0.05, alpha=2.0) resulted in severe mode collapse. Standard CFM achieved 5/5 mode coverage with 72.6% hit rate, while Iso-FM w=4.0 achieved 0/5 coverage with 0.1% hit rate (W=10.2 vs 0.5). 

Reducing the regularization weight revealed a sharp transition: w=1.0 improved to W=2.0 but still failed mode coverage (9% hit rate). **w=0.5 recovered full 5/5 coverage** with improved smoothness metrics: trajectory acceleration reduced 76% (0.079 vs 0.331), path straightness improved (1.004 vs 1.135), at the cost of slightly lower hit rate (52% vs 73%) and modest Wasserstein increase (0.73 vs 0.51).

This demonstrates the central tension: uniform residual suppression can improve trajectory smoothness but requires careful weight tuning to avoid over-constraining mode commitment in independent-pairing multimodal transport. The sharp weight sensitivity (collapse at w=1.0, success at w=0.5) suggests this benchmark successfully exposes the fragility predicted by the hypothesis.

## Asymmetric Follow-Up

The centered five-mode setup is intentionally simple, but its radial symmetry can make distinct regularizers look too similar. The next diagnostic benchmark is `fan_modes`, documented in `docs/archive/benchmark_fan_modes.md`.

`fan_modes` moves the source prior to the left of a right-side five-mode fan. This creates a coherent transport direction before branching, making it a sharper test for directional or CFL-motivated regularization than the centered case.
