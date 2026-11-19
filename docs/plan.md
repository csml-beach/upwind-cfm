# Active Plan: SCTW and Adjoint-SCTW

## Paper Thesis

The paper studies population-calibrated time meshes for sampling a frozen flow-matching ODE at a
fixed neural-function-evaluation (NFE) budget. No checkpoint is retrained.

- **SCTW** allocates Euler knots from the local material-acceleration profile.
- **Adjoint-SCTW** weights each local Euler defect by its sensitivity to the terminal state.

The narrow novelty claim is population-calibrated terminal sensitivity for native flow-matching
meshes. Adjoint models, adaptive timesteps, randomized projections, and dual-weighted residuals
are established ideas and must not be presented as new.

## Derivation

For the frozen learned ODE

$$
\dot x=v_\theta(x,t),
$$

the material acceleration is

$$
a=\frac{D v_\theta}{D t}=\partial_t v_\theta+D_xv_\theta\,v_\theta.
$$

An Euler step of length $h_i$ has leading local defect

$$
d_i\simeq \frac{h_i^2}{2}a_i.
$$

Let $M(1,t)=D_x\Phi_{1,t}$ be the state tangent of the learned flow. Since the defect is injected
at the end of step $i$,

$$
d_i^{\mathrm{terminal}}\simeq M(1,t_{i+1})d_i.
$$

This gives two profiles:

$$
e_i=\|a_i\|,\qquad q_i=\|M(1,t_{i+1})a_i\|.
$$

SCTW uses $e_i$. Adjoint-SCTW uses $q_i$.

### Three Transport Estimators

1. **Centered finite difference:** perturb the state at $t_{i+1}$ along $a_i/\|a_i\|$, roll both
   perturbations to the endpoint, and estimate the directional gain.
2. **Exact discrete JVP:** initialize the tangent with $a_i$ and propagate it through JVPs of the
   discrete Heun pilot steps.
3. **Randomized discrete adjoint:** draw terminal probes $z_r$, reverse-propagate
   $\lambda_{i+1}^{(r)}=M(1,t_{i+1})^Tz_r$, and estimate

   $$
   \widehat q_i=\left[\frac{1}{R}\sum_{r=1}^R
   (a_i^T\lambda_{i+1}^{(r)})^2\right]^{1/2}.
   $$

FD and JVP are verification methods. Randomized adjoint with declared default `R=4` is the
scalable method. The implementation recomputes and differentiates one Heun step at a time during
the reverse pass, so activation memory does not grow with the number of fine steps.
The squared estimator is unbiased, while taking its square root introduces finite-`R` bias; the
declared `R` and probe-seed audits measure that approximation rather than tuning it away.

## Mesh Rule

For either $r(t)=e(t)$ or $r(t)=q(t)$, define

$$
\rho_p(t)=(r(t)+\epsilon)^p,
\qquad p\in\{0.25,0.5\}.
$$

The knots are quantiles of the cumulative density. `p=0.5` is the Euler-derived equal-defect rule;
`p=0.25` is a declared tempering ablation. Both values are reported, never selected post hoc.

## Calibration Protocol

- fine Heun steps: `50`
- transport anchors: `16`
- adjoint probes: `R=4`
- pilot seed: `1311`
- adjoint seed: `2718`
- evaluation seed: `1234`
- low-dimensional and GM16 pilots: `512`
- Burgers pilots: `32` training cases
- image pilots: `100`; conditional CIFAR uses exactly 10 examples per class
- CIFAR chunk size: `4`, with deterministic recorded fallback to `2` then `1` on OOM
- staged-shapes chunk size: `8`
- CIFAR reference split: train; Inception weights and real-feature caches are SHA-versioned
- KID subset seed: `1234`, shared by every method

Calibration priors are disjoint from evaluation priors. The JSON profile bundle records estimator
version, seeds, pilot counts, batching, fine steps, anchor times, local acceleration, terminal
gain, transported profile, powers, perturbations, runtime, peak memory, and all generated grids.

## Full Endpoint Suite

Existing checkpoints only:

| Benchmark | Checkpoints | Evaluation |
| --- | ---: | ---: |
| Five modes, fan modes, spiral | 1 each | 1,000 samples |
| GM16 | seeds 0, 1, 2 | 1,000 samples each |
| Staged-shapes easy | Sinkhorn EMA | 1,000 samples |
| Hard Burgers solution map | h256 EMA | 32 held-out cases |
| Conditional CIFAR-10 | large 16x16 minibatch-OT EMA | 5,000 samples |
| Unconditional CIFAR-10 | large 16x16 Sinkhorn EMA | 5,000 samples |

At NFE `{5,10,20,50}`, compare uniform Euler; SCTW and Adjoint-SCTW at both powers; early, late,
and symmetric power meshes with rho `{2,3}`; VSFM coasting and uniform-deceleration grids; and
uniform Heun at matched actual NFE. A BOSS-style dynamic program is retained on low-dimensional,
GM16, and Burgers cases as an offline optimizer of an additive one-step-defect surrogate. It is
not a mathematical ceiling on endpoint error.

## Equivalence Tier

On all three visual problems, GM16 seed 0, Burgers, and small image pilot subsets, compare FD,
JVP, and adjoint profiles using:

- normalized profile L2 error against JVP;
- maximum knot displacement;
- learned-ODE integration error on common priors;
- calibration runtime and peak memory;
- FD epsilon `{1e-3,3e-3}`;
- adjoint `R={2,4,8}` and probe seeds `{7,2718,31415}`.

Only `R=4` proceeds to full endpoint metrics.

## Local Status

The local endpoint tier is complete for five modes, fan modes, spiral, GM16 seeds 0--2, and hard
Burgers. At NFE 5, Adjoint-SCTW with `p=0.5` improves learned-ODE integration error over SCTW on
fan modes, spiral, GM16, and Burgers, while SCTW remains better on five modes. On GM16, the
three-seed mean falls from about `0.527` for uniform Euler to `0.263` for SCTW and `0.237` for
Adjoint-SCTW. On Burgers, the corresponding values are `1.756`, `1.673`, and `1.250`.

The equivalence tier supports the estimator but also exposes one necessary caveat. Adjoint `R=4`
stays below the declared `10%` normalized-profile error and within `5%` of JVP schedule integration
error on every completed local case; probe-seed endpoint variation is also below `5%`. The declared
FD perturbations pass on fan, spiral, GM16, and Burgers, but fail on five modes because `1e-3` is
outside its local linear regime. A separate convergence audit reaches `0.23%` profile error at
`1e-4`, then shows float32 cancellation at still smaller epsilon. The failure remains in the
reported artifact rather than being replaced post hoc.

Burgers also illustrates the method boundary: time-mesh improvements materially reduce error to
the learned ODE's fine Heun trajectory, while endpoint PDE RMSE remains almost unchanged. This is
evidence for improved numerical integration, not corrected vector-field bias.

The remote endpoint tier is also complete. Staged-shapes is the strongest positive result:
Adjoint-SCTW with p=0.5 beats uniform, SCTW, and every tested fixed Euler mesh in feature W1 at
NFE 5, 10, 20, and 50; at NFE 5 the values are 1.524, 1.017, and 0.870 for
uniform/SCTW/Adjoint-SCTW. Conditional and unconditional CIFAR are more cautious. SCTW improves
FID over uniform at NFE 10, 20, and 50, while Adjoint-SCTW lowers learned-ODE integration error
but generally does not beat SCTW in FID. Across schedules, integration error and feature W1 are
nearly rank-aligned on staged-shapes but weakly or negatively aligned with CIFAR FID at moderate
and high NFE. This supports terminal sensitivity when the learned-ODE endpoint metric is aligned,
not as a correction for learned-field bias.

The staged-shapes equivalence pilot is complete for adjoint seeds `7`, `2718`, and `31415`.
At the default seed, `R=4` differs from JVP by `0.020%` in normalized profile L2 and by at most
`1.47%` in integration error, but its maximum knot displacement is `0.034`, above the declared
`0.02` gate. The profile is strongly concentrated near `t=0`, so cumulative-density inversion
amplifies small normalized profile differences. Probe-seed integration variation is below `5%`
at NFE 5, 10, and 50, but reaches `5.06%` at NFE 20. These near-failures are retained as evidence
that profile agreement alone does not guarantee perfectly stable knots.

The conditional and unconditional CIFAR equivalence pilots are complete for the same three seeds.
Both pass every declared image-scale gate. At `R=4`, normalized profile L2 error is `1.56%` and
`1.17%`, maximum knot displacement is `0.0066` and `0.0082`, and integration error differs from
the exact JVP schedule by at most `0.30%`. Probe-seed variation is below `0.70%`. Exact JVP
calibration takes about `1772--1779 s`, versus `245--247 s` for the four-probe adjoint. Therefore
the weaker CIFAR FID outcome reflects endpoint-metric alignment rather than estimator failure.

## Scientific Decision Rule

Report both methods regardless of outcome. Call terminal sensitivity successful only if it
improves or statistically matches SCTW on most benchmark families and materially closes a hand-grid
gap on at least one non-Burgers problem. If gains are Burgers-only, present Adjoint-SCTW as a
sensitivity-analysis extension. If integration error improves but FID or Wasserstein does not,
attribute the disconnect to learned-field bias rather than claiming better generation.

## Negative Results Kept In Scope

Pressure-aware training and pressure-aware coupling were investigated and often degraded endpoint
quality in higher dimension. The pressure identity remains useful theory, but no oracle pressure
field is used by either deployed sampler. Checkerboard, obsolete pressure-training runs, and
autoregressive Burgers are excluded from the endpoint suite.
