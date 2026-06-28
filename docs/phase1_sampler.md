# Phase 1 Results: A Priori Error-Controlled Sampling

## Current Status

This file is the detailed experiment record for the sampler-side direction. The active paper
candidate extracted from it is **Self-Curvature Time Warping (SCTW)**. Later pressure-training
and pressure-coupling notes in this file should be read as history/diagnostics, not as the current
main method.

For the current plan, see `docs/plan.md`. For novelty and nearby work, see
`docs/literature.md`. For the active checklist, see `docs/todo.md`.

Implementation of the Phase-1 strategy now archived in
`docs/archive/pressure_law_and_phase1_history.md`. Training is untouched everywhere:
all velocity models are standard CFM; everything below is post-hoc (a dispersion head trained
on frozen models, and sampler schedules).

Toolkit: `lcfm/oracle.py` (E0, law re-verified to 1e-13 by
`scripts/check_momentum_identity.py`), `lcfm/schedules.py` (grids, kappa, Euler/Heun on
arbitrary grids, per-sample controller), `lcfm/dispersion.py` (E2 head). Runs:
`results/phase1/runs/{geometry}_{coupling}/` — 4 geometries (clumped015, ring, fan, spiral)
x 2 couplings (independent, minibatch OT) x 3 seeds, plus one Burgers autoregressive model.

## X0: kappa survey

kappa = predicted step-efficiency gain of error-equidistributed steps over uniform
(integral of e over the square of the integral of sqrt(e); 1 = flat profile, nothing to gain).

| problem | kappa_E1 (model) | kappa_E2 (data, head) | kappa_E0 (oracle) |
| --- | ---: | ---: | ---: |
| clumped015 independent | 1.69 | 2.82 | 2.79 |
| clumped015 minibatch-OT | 1.70 | 1.44 | (1.27)* |
| ring independent | 1.13 | 1.36 | 1.15 |
| ring minibatch-OT | 1.19 | 1.62 | (1.27)* |
| fan independent | 1.11 | 1.06 | 1.06 |
| fan minibatch-OT | 1.05 | 1.44 | (—)* |
| spiral independent | 1.03 | 1.07 | n/a |
| spiral minibatch-OT | 1.04 | 1.43 | n/a |
| burgers autoregressive (paired) | **1.001** | n/a | n/a |

*the closed-form oracle assumes the independent coupling; for OT-paired runs it is a
reference, not ground truth.

Go/no-go outcomes: Burgers' paired coupling is cold (flat profile, peak/mean ~1.1) — the
survey says skip warping there, at the cost of one 40-step probe. The only strongly stiff
configuration among these is the clumped source.

## X2a: estimator validation (head vs oracle, clumped independent)

Three label modes for the head, validated against closed-form tr(Sigma)/d:

| mode | layer (t=0.05) | tail (t=0.2–0.8) | kappa vs oracle 2.79 |
| --- | --- | --- | ---: |
| residual (raw MSE) | rel err 0.23, corr 0.85 | collapses to ~0 | 8.20 |
| second_moment | rel err 0.34, corr 0.96 | 3–8x overestimate (cancellation in m − ‖v‖²/d) | 1.35 |
| **residual_log** | rel err 0.29, corr 0.95 | rel err 0.14–0.33 | **2.88** |

Raw-MSE fitting ignores small labels; the difference-of-large-numbers mode cannot resolve a
small tail. Fitting log(residual + 1e-3) preserves the profile across decades — kappa within
3% of the oracle. `residual_log` is the E2 estimator of record. (Its peak overshoots, 147 vs
68; kappa and grids depend on sqrt-integrals where the tail dominates, so this is tolerable —
but see the scheduling caveat below.)

## X2b: the audit

Data-side profile (E0/E2) vs each model's self-probe (E1) on the clumped benchmark. The data
demands ~90% of its curvature mass in t <= 0.2 (layer fraction 0.93 oracle / 0.88 head).

| models | kappa_E1 | E1 layer fraction | coverage |
| --- | --- | --- | --- |
| standard CFM (5 seeds) | 1.66–1.71 | 0.75–0.78 | 4–5 / 5 |
| Iso-FD w0.5 (5 seeds) | 1.11–1.31 | 0.43–0.62 | 2–4 / 5 |

No overlap between groups. An over-straightened field self-reports smooth while the data-side
profile still shows the layer; the deficit flags missing modes **without generating a single
sample**. Figure: `results/phase1/x2_audit.png`. This is the diagnostic only the law provides —
a generic error estimator has no notion of what curvature should exist.

## M3: samplers at matched NFE (6 evaluations)

Primary metric: integration error against a high-NFE Heun reference from the same initial
samples (the quantity kappa predicts). Current evaluators default to 1000 Heun intervals
and can optionally compare that reference against an even finer Heun rollout via
`--ref-check-intervals`. Full table: `results/phase1/sampler_eval.csv`. Key rows
(mean over 3 seeds):

| group | uniform Euler-6 | warp-E1 Euler-6 | uniform Euler-12 | uniform Heun-3 |
| --- | ---: | ---: | ---: | ---: |
| clumped indep | 0.565 | **0.277** | 0.315 | 0.497 |
| clumped OT | 0.444 | **0.207** | 0.250 | 0.515 |
| ring indep | 0.294 | 0.240 | 0.130 | 0.284 |
| fan indep | 0.540 | 0.428 | 0.273 | **0.313** |
| spiral indep | 0.219 | 0.202 | 0.101 | **0.192** |

Findings:

1. **Schedule with the model's own profile (E1).** On stiff problems, E1-warped Euler-6 beats
   uniform Euler-**12** — half the evaluations, lower error — and endpoint matching agrees
   (clumped: 0.55 vs 0.60 in the original W1-style metric). The E2-derived grid underperforms
   E1's everywhere (clumped 0.469, ring 0.455 — worse
   than uniform on ring): it equidistributes the *data's* curvature, but the integrated object
   is the *model's smoother field*. Division of labor, now empirically forced: **E1 schedules,
   E2 predicts and audits.**
2. **Order vs steps is governed by kappa.** Where the profile is flat (spiral, fan), Heun at
   matched NFE beats everything Euler — curvature is spread out, so order pays. Where the
   profile is a layer (clumped), warped Euler crushes Heun (0.277 vs 0.497): resolution in the
   right place beats order. kappa tells you which regime you are in before sampling.
3. **Per-sample S2: negative result at this stage.** Built on the E2 pointwise density, it
   inherits E2's bias and adds noise; it loses to the global E1 warp everywhere. A per-sample
   *model-side* density (local probes spend NFE; a distilled model-curvature head does not)
   is the open item — P4 unresolved, not refuted.
4. **Cooling (P3) confirmed in absolute terms**: minibatch-OT collapses integration error at
   fixed NFE (fan: 0.038 vs 0.540 uniform Euler-6) — the cold coupling is simply easier to
   integrate; warping then matters little (gain 1.03 on fan-OT).

## Working Sampler Method: Self-Curvature Time Warping

The current sampler-side method of record is **Self-Curvature Time Warping** (SCTW). It is
training-free and oracle-free: a trained CFM velocity field is probed along its own rollout, and
Euler steps are placed according to the model's self-reported material-derivative magnitude.

Default settings for image-scale checks:

- `profile_samples = 512`
- `profile_fine_steps = 50`
- `warp_power = 0.25`
- `warp_floor = 1e-3`

The profile is

$$
e(t_i) \approx \frac{\|v_\theta(x_{i+1}, t_{i+1}) - v_\theta(x_i, t_i)\|}{\Delta t},
$$

estimated on a fine uniform Euler probe. The sampler uses the density

$$
\rho(t) = (e(t) + 10^{-3})^{0.25}
$$

and places knots at equal cumulative mass of $\rho$. The exponent `0.25` is a tempered version of
the raw Euler equal-error exponent `0.5`. The tempering matters on CIFAR-10: raw `0.5` improves
moderate/high NFE but is too aggressive at NFE 5, while `0.25` largely removes that failure and
keeps most of the gains.

Do not call this pressure-aware in the method name. It is motivated by the pressure/material
derivative narrative, but the deployed sampler does not compute oracle pressure, pressure
gradients, or pressure budgets. It is a self-curvature/adaptive-time sampler.

## X1: the gain law

`results/phase1/x1_gain_vs_kappa_e1.png` — realized gain (uniform / E1-warped integration
error, Euler-6) against kappa_E1, 8 problem-coupling pairs, 3 seeds each:

| group | kappa_E1 | realized gain |
| --- | ---: | ---: |
| clumped indep / OT | 1.69 / 1.70 | 2.04 / 2.16 |
| ring indep / OT | 1.13 / 1.19 | 1.23 / 1.29 |
| fan indep / OT | 1.11 / 1.05 | 1.26 / 1.03 |
| spiral indep / OT | 1.03 / 1.04 | 1.09 / 1.07 |

Monotone, rank-perfect, near-diagonal at small kappa, and *conservative* at large kappa
(realized 2.0–2.2 vs predicted 1.7 — equidistribution also tames error amplification through
the layer, which the first-order model ignores). With kappa_E2 on the x-axis the correlation
breaks (`x1_gain_vs_kappa_e2.png`), again because data-side stiffness is not model-side
stiffness. **P1 confirmed for kappa_E1.**

## X2c: capacity and training-time audit

The capacity audit asks whether standard neural CFM can learn the oracle pressure layer on the
clumped five-mode problem, or whether the observed E0/E1 gap is simply unavoidable model
smoothing. Scripts:

- `scripts/run_oracle_capacity_audit.py`
- `scripts/oracle_model_audit.py`

Single-seed results (`results/phase1/capacity_audit`):

| variant | kappa_E0 | model-FD kappa | E1 kappa | layer rel-RMSE |
| --- | ---: | ---: | ---: | ---: |
| small h64 d2, 2k | 2.80 | 1.63 | 1.27 | 0.98 |
| base h128 d3, 2k | 2.80 | 1.91 | 1.64 | 0.69 |
| large h256 d4, 2k | 2.80 | 2.21 | 2.00 | 0.61 |
| base h128 d3, 8k | 2.80 | 2.35 | 2.09 | 0.46 |
| large h256 d4, 8k | 2.80 | 2.69 | 2.33 | 0.17 |

Interpretation: the pressure layer is learnable. Increasing capacity or training time moves the
model's material-derivative profile toward the exact oracle profile. The earlier E0/E1 gap is
therefore not merely a theorem-level impossibility; it is partly a finite-capacity/optimization
smoothing effect.

The cost is numerical stiffness. The more faithful models are worse under uniform Euler-5, but
are rescued by warped grids:

| variant | uniform Euler-5 W | E1-warp Euler-5 W | E0-warp Euler-5 W | uniform Euler-10 W |
| --- | ---: | ---: | ---: | ---: |
| base h128 d3, 2k | 0.893 | 0.623 | 0.638 | 0.675 |
| base h128 d3, 8k | 1.122 | 0.414 | 0.411 | 0.560 |
| large h256 d4, 2k | 1.030 | 0.706 | 0.723 | 0.754 |
| large h256 d4, 8k | 2.341 | 0.428 | 0.443 | 0.669 |

This is the cleanest evidence so far for the central tension:

> CFM fidelity and low-NFE uniform integrability are not the same objective. As the model learns
> the pressure layer more faithfully, uniform coarse Euler fails harder. Pressure-aware stepping
> can recover the learned field without erasing that layer.

This changes the training-side outlook. Naively matching the exact pressure acceleration remains
dangerous if evaluation insists on uniform low-NFE Euler, because it trains a more faithful but
stiffer field. But pressure-aware training may be useful if it is paired with a mesh-aware
objective, a filtered pressure target, or a solver policy that resolves the layer.

## X2d: pressure-budget diagnostic

Before training with a pressure-budget loss, we tested whether a pure post-hoc pressure-budget
violation diagnoses good and bad clumped-source models. Script:

- `scripts/pressure_budget_diagnostic.py`

The pure upper-budget diagnostic decomposes model acceleration $a_\theta$ relative to oracle
pressure acceleration $a_p$ and penalizes orthogonal, opposite, or excessive pressure-aligned
curvature. It does **not** penalize missing pressure-aligned curvature.

Mean results (`results/phase1/pressure_budget_diagnostic_clumped`):

| group | W | hit | pure violation | pressure utilization | deficit |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 0.802 | 0.464 | 0.050 | 0.373 | 0.046 |
| Iso-FD w0.5 | 0.819 | 0.390 | 0.129 | 0.214 | 0.155 |
| directional FD+FD w2 | 0.513 | 0.762 | 0.059 | 0.356 | 0.067 |
| small h64 d2, 2k | 1.360 | 0.160 | 0.046 | 0.219 | 0.094 |
| base h128 d3, 8k | 1.122 | 0.311 | 0.031 | 0.631 | 0.017 |
| large h256 d4, 8k | 2.341 | 0.045 | 0.040 | 0.843 | 0.011 |

Conclusion: pure upper-budget violation does flag Iso-FD as more pressure-inconsistent than
standard/directional models, but it is not a global quality diagnostic. It falsely blesses the
large-long model, which is highly pressure-aligned but too stiff for uniform Euler-5. A perfectly
zero-acceleration field would also have no upper-budget violation while missing all pressure
curvature.

So a pure pressure-budget term is not sufficient as the main training objective. The useful signal
is two-dimensional:

- **wrong/excess pressure curvature**: pure budget violation,
- **missing pressure usage**: pressure utilization or deficit.

Using the second signal as a loss becomes a pressure floor/band method, which is less "pure" than
an upper budget but probably necessary if the method must prevent cold-flow collapse.

## X2e: pressure-in-training viability probe

Canonical short status: `docs/archive/pressure_training_status.md`.

The first training-side pressure test is deliberately oracle-assisted and should not be treated as
a deployable method yet. Script:

- `scripts/train_pressure_viability.py`

It trains staged Gaussian-mixture models with the usual CFM loss plus one oracle pressure term:

- `upper_budget`: penalize pressure-orthogonal, opposite, or excessive material acceleration.
- `pressure_band`: upper budget plus a lower pressure-aligned floor.
- `alignment`: encourage pressure direction only.
- `exact_match`: match the exact oracle pressure acceleration, included as a risky reference.

Implementation note: the useful version normalizes losses by aggregate pressure energy over the
batch/grid. A per-sample normalization was rejected because it overweights near-zero-pressure
points and turns the loss into a low-pressure artifact penalty.

Corrected staged-mode results, three seeds, 1000 epochs, independent pairing, uniform Euler
evaluation (`results/phase1/pressure_training_viability_globalnorm/staged_aggregate.csv`):

| variant | Euler-5 W | Euler-5 hit | Euler-5 integration error | Heun-ref W | pressure utilization | deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 1.251 | 0.120 | 0.652 | 1.166 | 0.117 | 1.193 |
| upper w1 | 1.197 | 0.126 | 0.598 | 1.106 | 0.103 | 0.514 |
| band w1 | 1.147 | 0.124 | 0.679 | 1.121 | 0.121 | 0.445 |
| exact w0.1 | 1.199 | 0.131 | 0.716 | 1.182 | 0.115 | 0.890 |
| low-weight variants | roughly standard | roughly standard | roughly standard | roughly standard | - | - |

Interpretation:

- Pressure-in-training is viable enough to keep testing: global-normalized upper/band losses
  improve endpoint W over standard on this small staged benchmark.
- The signal is not yet a paper method. The best endpoint variant (`band w1`) has worse
  integration error than standard, so it may be changing the learned field rather than making it
  easier to integrate.
- The cleanest low-NFE integrability hint is `upper w1`: modest W improvement and lower
  integration error. This is surprising enough to follow up, but not strong enough to trust from
  three toy seeds.
- Exact pressure matching is not the winner. This supports the earlier warning that exact
  pressure fidelity is not the same thing as low-NFE usefulness.

Next pressure-training test: tune `upper_budget` and `pressure_band` on staged modes with a small
weight/eta grid, then repeat the winner on clumped and 16D mixtures. A method only graduates into
the main registry if it improves both endpoint quality and the relevant low-NFE/integration
tradeoff across geometries.

## X2f: pressure-training sweep across geometries

The stronger sweep tested a compact upper/band grid on `staged`, then carried the most informative
candidates to `clumped015` and `gm16`. Outputs:

- `results/phase1/pressure_training_sweep/staged_seeds_0-1-2_aggregate.csv`
- `results/phase1/pressure_training_sweep/clumped015_seeds_0-1-2_aggregate.csv`
- `results/phase1/pressure_training_sweep/gm16_seeds_0-1-2_aggregate.csv`
- `results/phase1/pressure_training_sweep/cross_geometry_shortlist_summary.csv`

Euler-5 means:

| geometry | variant | W | hit | integration error |
| --- | --- | ---: | ---: | ---: |
| staged | standard | 1.251 | 0.120 | 0.652 |
| staged | upper w2 | 1.192 | 0.128 | 0.559 |
| staged | band w1 eta 0.5 | 1.115 | 0.123 | 0.791 |
| clumped015 | standard | 0.984 | 0.345 | 0.542 |
| clumped015 | exact w0.1 | 0.940 | 0.382 | 0.542 |
| clumped015 | band w1 eta 0.5 | 0.941 | 0.385 | 0.517 |
| clumped015 | upper w2 | 0.997 | 0.366 | 0.515 |
| gm16 | standard | 2.059 | 0.331 | 0.811 |
| gm16 | exact w0.1 | 2.087 | 0.307 | 0.795 |
| gm16 | band w1 eta 0.5 | 2.697 | 0.041 | 0.384 |
| gm16 | upper w2 | 2.987 | 0.014 | 0.204 |

Conclusion: direct oracle pressure-budget training is not robust. It can improve some 2D endpoint
metrics, but in 16D the pressure losses reduce integration error by producing much colder fields
with far worse endpoint W and hit rate. This is an over-cooling/under-transport failure. The
current direct pressure objectives should remain oracle probes, not candidate paper methods.

The next useful pressure-training work is diagnostic or mesh-aware: inspect the 16D collapse,
filter the pressure target to the intended solver resolution, or pair pressure training with
pressure-aware sampling rather than evaluating only uniform Euler.

## X2g: capacity and anisotropy audit

We then tested whether the 16D failure was simply under-capacity/under-training, and whether the
anisotropic staged geometry gives a more reliable mechanism test. The audit used `hidden=256`,
`depth=4`, and 4000 epochs.

Outputs:

- `results/phase1/pressure_training_capacity_audit/staged_seeds_0-1-2_aggregate.csv`
- `results/phase1/pressure_training_capacity_audit/gm16_seed0_summary.csv`
- `results/phase1/pressure_training_capacity_audit/capacity_audit_summary.csv`

Euler-5 means:

| audit | variant | W | hit | integration error |
| --- | --- | ---: | ---: | ---: |
| staged strong, 3 seeds | standard | 1.558 | 0.388 | 1.567 |
| staged strong, 3 seeds | upper w2 | 1.375 | 0.408 | 1.295 |
| staged strong, 3 seeds | band w1 eta 0.5 | 1.456 | 0.434 | 1.455 |
| staged strong, 3 seeds | exact w0.1 | 1.783 | 0.447 | 1.852 |
| gm16 strong, seed 0 | standard | 1.911 | 0.539 | 1.381 |
| gm16 strong, seed 0 | upper w2 | 2.535 | 0.176 | 0.392 |
| gm16 strong, seed 0 | band w1 eta 0.5 | 2.682 | 0.260 | 1.917 |

Interpretation: staged is a useful anisotropic mechanism benchmark. With more capacity/training,
`upper_w2` improves Euler-5 W and integration error over strong standard. The same direct
pressure idea still does not survive strong `gm16`: `upper_w2` greatly lowers integration error
but damages endpoint W/hit. So the 16D result is not just a weak-model artifact. The method likely
needs mesh-filtering, projection, or sampler coupling before it can scale beyond coherent
low-dimensional anisotropic geometry.

## X0b: harder oracle-compatible geometries

Two additional Gaussian-mixture benchmarks were added to avoid overfitting decisions to the
symmetric clumped ring:

- `staged_modes`: 2D modes at different distances and angles from a clumped source, intended to
  create less synchronized mode commitment.
- `gaussian_mixture_nd`: high-dimensional Gaussian mixture; the default benchmark uses eight
  simplex-like modes in 16 dimensions.

Exact-field oracle sampler audit (`results/phase1/oracle_sampler_new_geometries.csv`, 512 probes):

| geometry | kappa_E0 | uniform Euler-5 W | E0-warp Euler-5 W | uniform Euler-10 W |
| --- | ---: | ---: | ---: | ---: |
| staged | 2.20 | 1.855 | 0.541 | 1.070 |
| 16D mixture | 1.90 | 1.361 | 0.999 | 1.063 |

Both are useful follow-up problems. The staged geometry is a stronger 2D test than the symmetric
ring because pressure warping clearly beats uniform refinement. The 16D mixture tests whether the
pressure/kappa story survives beyond planar visualization; it remains stiff, but less extremely
than the clumped 2D ring.

First standard-CFM learned baselines (`results/phase1/new_geometry_standard_sampler_eval.csv`):

| run | kappa_E0 | kappa_E1 | uniform Euler-5 W | E1-warp Euler-5 W | E0-warp Euler-5 W | uniform Euler-10 W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| staged standard seed0 | 2.19 | 1.45 | 1.156 | 0.954 | 0.934 | 0.969 |
| 16D mixture standard seed0 | 1.89 | 1.41 | 1.418 | 1.246 | 1.299 | 1.244 |

Both learned fields smooth the oracle pressure concentration, but the problems still separate
uniform and warped integration. The staged benchmark is especially hard for the base 2k MLP
(uniform Euler-5 covers 0 modes, warped/10-step only 1 mode), so it is a good candidate for
capacity, pressure-budget, or coupling experiments rather than a solved toy.

## Predictions scorecard

- **P1 (gain law): confirmed**, with the precise statement "realized gain tracks the *model's*
  stiffness concentration kappa_E1."
- **P2 (estimators): confirmed with a twist.** E2 (residual_log) reproduces oracle kappa within
  3% and powers the audit; but E2 is not interchangeable with E1 for scheduling — the roles are
  complementary, not redundant.
- **P3 (cooling): confirmed** in absolute integration error and in the Burgers paired-coupling
  null (kappa 1.001).
- **P4 (per-sample): unresolved.** The E2-driven controller loses; needs a model-side
  per-sample density.
- **P5 (learnability of pressure layers): partially confirmed.** Capacity and longer training
  recover more of the oracle pressure layer on the clumped benchmark, but the resulting field
  becomes less usable under uniform low-NFE Euler.

## Open items for the paper

- Calibrate E2 -> E1: the audit gap (data-side minus model-side profile) is itself the
  smoothing the model applied; modeling that attenuation would let the data-side head schedule
  too, enabling pre-training schedule selection end to end.
- Decide whether the main method should remain inference-time only, or become a joint
  pressure-aware training + pressure-aware solver method. The capacity audit makes the latter
  plausible, but only if the training target is mesh-aware rather than exact-pressure matching
  under a uniform solver.
- Per-sample density without extra NFE (distill E1 pointwise into a head).
- kappa-driven solver selection (Euler-warped vs Heun) as an automatic policy; both are
  predictable from the survey.
- Image-scale FM and the Gronwall correction to the first-order gain model remain untouched.
