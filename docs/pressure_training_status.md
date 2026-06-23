# Pressure Training Status

This note records the current status of pressure-aware training. It is intentionally short: the
goal is to prevent us from re-testing rejected variants or overclaiming a weak signal.

## Current Question

Can the pressure law provide a training signal that improves low-NFE CFM sampling, rather than
only explaining why pressure-aware time grids help at inference?

The working acceleration is

$$
a_p(x,t) = -\nabla\cdot\Sigma(x,t) - \Sigma(x,t)\nabla\log p_t(x),
$$

where $\Sigma$ is the conditional covariance of the flow-matching target. In the oracle mixture
experiments, this quantity is exact.

## What Was Tested

The current probe is `scripts/train_pressure_viability.py`. It is deliberately oracle-assisted
and should not be treated as a deployable method yet.

Variants:

- `standard`: ordinary CFM.
- `upper_budget`: penalizes material acceleration that is pressure-orthogonal, pressure-opposite,
  or above a pressure-scaled upper budget.
- `pressure_band`: upper budget plus a lower pressure-aligned floor.
- `alignment`: pressure direction only.
- `exact_match`: direct matching to the exact pressure acceleration.

The material acceleration is computed with a differentiable JVP through the velocity model.

## Normalization Decision

Use aggregate pressure-energy normalization:

$$
\mathcal L \sim
\frac{\sum_i \mathrm{violation}_i}{\sum_i \|a_{p,i}\|^2}.
$$

Do not use per-sample pressure normalization. It overweights near-zero-pressure points and turns
the loss into a low-pressure artifact penalty. The rejected result directories are:

- `results/phase1/pressure_training_viability`
- `results/phase1/pressure_training_viability_lowweights`
- `results/phase1/pressure_training_viability_smoke`

The valid result family from the first corrected probe is:

- `results/phase1/pressure_training_viability_globalnorm`

## First Corrected Result

Staged modes, three seeds, 1000 epochs, independent pairing, uniform Euler evaluation:

| variant | Euler-5 W | Euler-5 hit | Euler-5 integration error | Heun-ref W |
| --- | ---: | ---: | ---: | ---: |
| standard | 1.251 | 0.120 | 0.652 | 1.166 |
| upper w1 | 1.197 | 0.126 | 0.598 | 1.106 |
| band w1 | 1.147 | 0.124 | 0.679 | 1.121 |
| exact w0.1 | 1.199 | 0.131 | 0.716 | 1.182 |

Interpretation:

- Pressure-aware training is not dead. Correctly normalized upper/band losses improve endpoint W
  on this small benchmark.
- `upper_w1` is the cleanest hint because it improves both W and integration error.
- `band_w1` has the best W but worse integration error; it may be changing the learned field
  rather than making the field easier to integrate.
- Exact pressure matching is not best, which supports the claim that pressure fidelity alone is
  not the same as low-NFE usefulness.

## Stronger Sweep

The follow-up sweep used a compact staged grid, then carried the most informative candidates to
`clumped015` and `gm16`.

Outputs:

- `results/phase1/pressure_training_sweep/staged_seeds_0-1-2_aggregate.csv`
- `results/phase1/pressure_training_sweep/clumped015_seeds_0-1-2_aggregate.csv`
- `results/phase1/pressure_training_sweep/gm16_seeds_0-1-2_aggregate.csv`
- `results/phase1/pressure_training_sweep/cross_geometry_shortlist_summary.csv`

For `gm16`, the sweep used batch size 64 and diagnostic batch size 128 to keep oracle-Jacobian
cost manageable.

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

Interpretation:

- The direct oracle pressure-budget idea is **not robust** in its current form.
- On 2D staged modes, pressure terms can improve endpoint W; `upper_w2` is the cleanest
  integrability candidate, while stronger band floors trade endpoint W against integration error.
- On clumped modes, the result is mixed. Exact matching and the strong band improve W/hit modestly,
  while `upper_w2` improves integration error but does not improve W.
- On 16D mixtures, the direct pressure losses fail hard: they reduce integration error but destroy
  endpoint W and hit rate. This is an over-cooling/under-transport failure, not a success.

## Do Not Claim Yet

Do not claim a pressure-training method contribution from these results. The stronger sweep shows
that direct oracle pressure-budget regularization can help some 2D cases but fails in 16D.

A pressure-training objective should not enter `lcfm/losses.py` or the method registry until it
survives cross-geometry tests without collapsing high-dimensional endpoint quality.

## Next Test

The next pressure-training direction should not be another direct-weight sweep. Better options:

- diagnose the 16D failure by inspecting transport displacement and mode assignment under
  pressure losses;
- replace direct pressure matching/budgeting with a mesh-filtered pressure target;
- couple pressure training to pressure-aware sampling rather than judging it only under uniform
  Euler;
- consider whether the pressure target needs dimension-aware scaling or projection onto a
  low-rank/local subspace.

The method only graduates if it improves endpoint quality and the relevant low-NFE/integration
tradeoff across geometries.

## Metric Convention

Endpoint Wasserstein metrics use exact finite-sample optimal matching between equally weighted
sample clouds. `wasserstein` is empirical W1 with Euclidean ground cost; `wasserstein2` is
empirical W2, minimizing squared Euclidean cost and reporting the square-rooted mean squared
match distance. These are exact for the sampled clouds; the remaining uncertainty is sample
noise, so final comparisons should use fixed evaluation seeds, enough samples, and multi-seed
uncertainty.

Integration error is a different metric: mean endpoint distance from the low-NFE rollout to a
high-NFE Heun rollout from the same initial samples. For paper-facing results, the reference
should use small time steps first, high order second. The evaluators now default to 1000 Heun
intervals and can report `reference_self_error` by comparing against a finer Heun run.

Use `scripts/metric_convergence_audit.py` before paper-facing comparisons. It runs saved
checkpoints at multiple `n_eval` values and eval seeds to estimate Wasserstein sample noise,
then varies Heun reference intervals to check whether `reference_self_error` is negligible
relative to the method gaps being claimed.

Short audit result: `results/phase1/metric_convergence_audit/staged_capacity_seed0_light/`
compared strong staged `standard` vs `upper_w2`. The old Heun-100 reference was already stable
for this seed: increasing to Heun-500 moved integration error by only about 0.003, while the
method gap was about 0.11. Endpoint W was noisier but still separated at `n_eval=1024`
(`standard` W1 1.424, `upper_w2` W1 1.357). Recommendation: move on, but use larger
`n_eval`/more seeds for paper-facing endpoint-W claims.

Decision reeval: `results/phase1/decision_benchmark_eval/sweep_staged_gm16/` recomputed the
three-seed `pressure_training_sweep` with corrected W1/W2 and Heun-500 reference. On staged,
`upper_w2` improves integration error (0.567 vs 0.660) and modestly improves W1 (1.112 vs
1.154); `band_w1_eta05` has best W1 (1.034) but worse integration error (0.794). On gm16,
pressure variants strongly improve integration error (`upper_w2` 0.204, `band_w1_eta05` 0.378,
standard 0.815) but badly damage endpoint quality and hit rate. This points away from a
universal pressure-training claim and toward a regime/diagnostic story unless we find a
pressure use that preserves endpoint quality in higher dimension.

Next pressure-use candidate: `pressure_aware_minibatch_ot` moves the pressure idea into the
coupling. It augments minibatch OT with a scalar local conditional-velocity-variance proxy, then
trains ordinary CFM on the chosen pairs. This keeps endpoints real and avoids using pressure as a
vector supervision target. Test it first as coupling-only against `independent` and `minibatch_ot`;
only add scalar budget regularization if endpoint quality is preserved.

Coupling/cooling result: `results/phase1/pressure_aware_coupling_benchmark/eval_with_iso_fd/`
trained standard CFM with `independent`, `minibatch_ot`, and `pressure_aware_ot` pairings, plus
an `iso_fd_w05` cooling baseline. On staged, pressure-aware OT modestly improved W1 over
minibatch OT (1.083 vs 1.115) with similar integration error (0.540 vs 0.532), while Iso-FD was
worse (W1 1.842, integration error 1.463). On gm16, all cooling/OT-style interventions hurt
endpoint quality: independent W1/hit was 2.061/0.307, minibatch OT 2.557/0.157, pressure-aware
OT 2.591/0.150, and Iso-FD 5.158/0.000. Iso-FD did reduce gm16 integration error, so this is
the same pattern in sharper form: smoother model ODE, much worse endpoint distribution. The
issue is not just that OT was suspicious; this benchmark is exposing an over-cooling failure
mode in high dimension.

gm16 validity audit: `results/phase1/gm16_validity_audit_allseeds/` checks the 16D benchmark
geometry, target/source metric floors, oracle sampler, and saved model endpoints. The benchmark
is an 8-mode simplex embedded in 16D; the mode-center subspace has rank 7, centers are 6.047
apart, and hit balls of radius 1.6 do not overlap. Target-vs-target W1 is about 1.035 with
100% hit; source-vs-target W1 is about 3.821 with 0% hit. The exact oracle Heun-500 sampler
lands at W1 1.096 with 100% hit, close to the finite-sample target floor, so the distribution
and oracle are coherent. The failure is in learned fields/cooling: independent Heun-500 gets
W1/hit 1.614/0.664, minibatch OT 1.990/0.428, pressure-aware OT 2.011/0.415, and Iso-FD
5.211/0.000. Thus gm16 looks like a valid stress test for endpoint preservation, not an
obvious code/metric artifact.

## Capacity / Anisotropy Audit

After the 16D failure, we checked whether the result was just under-capacity or under-training.
The audit used a larger MLP (`hidden=256`, `depth=4`) and 4000 epochs.

Outputs:

- `results/phase1/pressure_training_capacity_audit/staged_seeds_0-1-2_aggregate.csv`
- `results/phase1/pressure_training_capacity_audit/gm16_seed0_summary.csv`
- `results/phase1/pressure_training_capacity_audit/capacity_audit_summary.csv`

The second strong `gm16` seed was intentionally interrupted before completion; only `gm16_seed0`
should be treated as the strong-16D audit result.

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

Interpretation:

- Staged is a useful anisotropic benchmark. With more capacity/training, `upper_w2` improves
  coarse Euler W and integration error over strong standard; `band_w1_eta05` improves hit but less
  cleanly.
- Exact pressure matching improves high-accuracy Heun quality on staged but is worse under
  Euler-5, so exact fidelity still does not equal low-NFE usefulness.
- Strong `gm16` does not rescue `upper_w2`: it still improves integration error while badly
  degrading endpoint W and hit. Strong `band_w1_eta05` improves the Heun-reference W for seed 0,
  but is worse under Euler-5.
- The likely story is dimension/geometry dependent: pressure regularization can exploit coherent
  anisotropic structure in staged 2D, but direct full-vector pressure penalties over-cool or
  misdirect high-dimensional transport.

So staged is good for developing and visualizing the mechanism, while 16D remains a veto/stress
test. Neither should be the sole design center.

## gm16 Radial Diagnostic

`results/phase1/pressure_aware_coupling_benchmark/diagnostics/gm16_radial_diagnostics.png`
separates the gm16 failure into radial progress toward the assigned mode, nearest-center
distance, orthogonal scatter, and endpoint norm. The important result is that independent,
minibatch OT, and pressure-aware OT are not mainly failing by choosing wildly imbalanced modes.
They are failing by under-traveling toward the modes. Independent reaches farther outward than
minibatch OT or pressure-aware OT: under Heun-500, radial progress is about 0.759 for independent,
0.570 for minibatch OT, and 0.557 for pressure-aware OT, while the target is about 1.0. Orthogonal
distance is already close to target scale for OT-style pairings, so the missing piece is radial
commitment, not transverse noise. Iso-FD is worse and unstable, with zero hit rate. A weaker
Iso-FD setting, w0.1, is less destructive than w0.5 by W1 on gm16 (5.042 vs 5.158), but still
has zero hit rate and remains far worse than independent.

This supports the current caution: cooling/smoothing can improve integration diagnostics while
damaging endpoint distribution quality. A viable pressure method should probably gate or shape
regularization without shortening the learned displacement toward modes.

## Staged Commitment Diagnostic

`scripts/plot_mode_commitment_diagnostics.py` generalizes the gm16 commitment plots to any
mode-mixture run by measuring progress from the source mean toward each assigned mode center.
For staged, the outputs are in
`results/phase1/pressure_aware_coupling_benchmark/diagnostics/staged_radial_diagnostics.png`
and `staged_mode_histograms.png`.

The staged failure is not the same as gm16. Radial progress is already near one for independent,
minibatch OT, pressure-aware OT, and Iso-FD w0.5. The relevant differences are landing accuracy,
integration behavior, and mode allocation, not simple under-travel. In the current Euler-5
comparison, pressure-aware OT has the best W1 among the coupling/cooling methods
(1.083 vs 1.115 minibatch OT and 1.157 independent), with integration error close to minibatch OT
(0.540 vs 0.532).

We now include a weaker Iso-FD w0.1 check. On staged, w0.1 is much less damaging than w0.5 by
endpoint W1 (1.157 vs 1.842), but it does not improve over independent and has worse integration
error than pressure-aware OT or minibatch OT (0.878 vs 0.540/0.532). On gm16, w0.1 remains
zero-hit. Thus staged supports a possible anisotropic/coupling benefit, while gm16 remains the
endpoint-preservation veto test.

## Pressure-Beta Robustness

`results/phase1/pressure_beta_sweep/eval/decision_benchmark_aggregate.csv` sweeps only
`pressure_beta` for pressure-aware minibatch OT, holding `pressure_t="random"`, median bandwidth,
and `reference_pairing="minibatch_ot"` fixed. The sweep used betas 0.05, 0.1, 0.2, 0.5, and 1.0
with seeds 0, 1, and 2 on staged and gm16.

The result is stable but modest. On staged, W1 ranges only from 1.085 to 1.078 as beta increases,
with integration error about 0.538-0.540. This is consistently near, and slightly better in W1
than, the minibatch OT baseline. On gm16, W1 stays about 2.591-2.592 and hit rate about 0.15
across the whole beta range, slightly worse than minibatch OT. Thus beta is not a fragile tuning
knob here, but it also is not a strong rescue lever. The advantage, if any, is robustness and
simplicity rather than a large beta-tuned win.

## Strong GM16 Sinkhorn Follow-Up

After adding Sinkhorn-projected pairing, local strong GM16 probes used the capacity-audit recipe
(`hidden=256`, `depth=4`, `4000` epochs, batch size 64, Euler-5 evaluation). The point was to
avoid judging coupling variants in the earlier low-hit, undertrained regime.

Outputs:

- `results/gm16_strong_sinkhorn_10seed/`
- `results/gm16_pressure_sinkhorn_sweep/`

Ten-seed result:

| variant | W1 mean | W1 std | W2 mean | hit |
| --- | ---: | ---: | ---: | ---: |
| Iso-FD w0.01 | 1.292 | 0.064 | 1.739 | 0.888 |
| minibatch OT | 1.298 | 0.099 | 1.679 | 0.847 |
| pressure-aware Sinkhorn OT | 1.298 | 0.044 | 1.687 | 0.838 |
| pressure-aware OT | 1.314 | 0.088 | 1.711 | 0.849 |
| Sinkhorn OT | 1.342 | 0.115 | 1.750 | 0.839 |
| independent | 1.758 | 0.128 | 2.026 | 0.546 |
| Iso-FD w0.1 | 4.364 | 0.269 | 4.419 | 0.072 |

Interpretation:

- Strong GM16 confirms that OT-style coupling and very light Iso-FD beat independent.
- Iso-FD is highly weight-sensitive: w0.01 is competitive, while w0.1 collapses endpoint
  quality. It is a serious tuned baseline, not something we can ignore.
- Pressure-aware Sinkhorn is stable and competitive with exact OT, but it does not clearly
  dominate tuned Iso-FD or exact minibatch OT.
- The pressure-aware Sinkhorn sweep found a broad useful region around Sinkhorn epsilon scale
  0.2. Varying `pressure_beta` mattered less than epsilon. Thus GM16 currently supports
  Sinkhorn-projected coupling as a robust coupling mechanism more strongly than it supports the
  pressure term as the main driver.
