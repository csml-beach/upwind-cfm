# The Pressure Term: An Exact Momentum Law for Flow Matching

## The Law

For the linear conditional flow-matching interpolant

$$
x_t = (1-t)\,x_0 + t\,x_1, \qquad u = x_1 - x_0,
$$

with any coupling of $(x_0, x_1)$ (in particular the independent one), the marginal velocity
field that CFM regression converges to,

$$
v(x,t) = E[\,u \mid x_t = x\,],
$$

satisfies **exactly**

$$
\boxed{\;
\frac{Dv}{Dt}
\;=\;
\partial_t v + (v\cdot\nabla)v
\;=\;
-\frac{1}{p}\,\nabla\cdot\big(p\,\Sigma\big)
\;=\;
-\nabla\cdot\Sigma \;-\; \Sigma\,\nabla\log p,
\;}
$$

where $p(x,t)$ is the interpolant marginal density and

$$
\Sigma(x,t) = \mathrm{Cov}[\,u \mid x_t = x\,]
$$

is the **conditional covariance of the flow-matching target** — the pressure term.

Derivation (four lines). The pair process $(X_t, U)$ satisfies $\dot X_t = U$, $\dot U = 0$:
a free-streaming, collisionless gas in phase space. Differentiate the momentum density
$p v_i = E[U_i\,\delta(x - X_t)]$ in time, use $\dot U = 0$:

$$
\partial_t(p v_i) + \partial_j\big(p\,E[U_iU_j \mid x]\big) = 0 .
$$

Split the second moment $E[UU^\top\mid x] = vv^\top + \Sigma$ and subtract $v_i\times$ the
continuity equation $\partial_t p + \nabla\cdot(pv) = 0$. The law above remains.

This is the **Jeans equation** of stellar dynamics / the pressureless-Euler system with an
anisotropic pressure tensor $p\Sigma$. It involves no approximation and no closure problem:
unlike physical moment hierarchies, every moment here is directly regressable from data
($v$ from target $u$, the second moment from target $uu^\top$).

## What It Means

- **Trajectory curvature is not noise; it is pressure-gradient forcing, exactly.** The marginal
  flow bends precisely where the dispersion flux $p\Sigma$ has divergence — i.e. where
  conditional ambiguity about the destination is being resolved. "Uncertainty gating" of an
  acceleration penalty is a scalar caricature of this vector identity (and keys on the wrong
  functional: variance *magnitude* rather than dispersion-flux *divergence* — at a symmetric
  high-variance point the required force is zero).
- **Zero-target acceleration penalties (Iso-FM, JVP-to-zero, LC-FD) impose the cold-gas closure
  $\Sigma \equiv 0$.** On multimodal independent-pairing transport the gas is warm; uniform
  straightening fights the data term, which is the observed knife-edge weight sensitivity and
  mode collapse.
- **Deterministic couplings (global OT, Reflow) cool the gas**: $\Sigma \to 0$, the law
  degenerates to $Dv/Dt = 0$ (Benamou–Brenier geodesics), and straightening becomes the
  *correct* closure. The framework places OT-CFM and independent CFM on one axis (gas
  temperature). Minibatch OT cools only partially.
- The leading local truncation error of Euler is $\tfrac{\Delta t^2}{2}\|Dv/Dt\|$, so the law
  yields a **computable, certified local-error field**: the pressure term tells you, pointwise,
  where coarse integration will fail.

## Exactness and How to Re-Verify

For Gaussian source $x_0 \sim N(\mu_0, \sigma_0^2 I)$ and a Gaussian-mixture target (the
`five_modes` / `fan_modes` benchmarks), $v$, $\Sigma$, $p$ are closed-form mixture expressions:
with $\gamma_t^2 = (1-t)^2\sigma_0^2 + t^2\sigma_m^2$, $c_t = t\sigma_m^2 - (1-t)\sigma_0^2$,
component means $m_k = t\mu_k + (1-t)\mu_0$, posterior weights
$w_k \propto \pi_k\,N(x; m_k, \gamma_t^2 I)$:

$$
E[u\mid x,k] = (\mu_k - \mu_0) + \frac{c_t}{\gamma_t^2}(x - m_k), \qquad
\mathrm{Cov}[u \mid x, k] = \Big(\sigma_0^2 + \sigma_m^2 - \frac{c_t^2}{\gamma_t^2}\Big) I,
$$

and $v$, $\Sigma$ follow by mixing over $w_k$. Checking the law is ~50 lines of float64
autograd (jacobians of $v$, $\Sigma$, $\log p$ at random $(x,t)$): it held to a relative
residual of ~1e-13 on the clumped five-mode geometry, on and off the data support, including
$t = 0.99$ (no late-time singularity for $\sigma_m > 0$).

Useful companion identities (Gaussian source): $E[x_0 \mid x_t] = x - t\,v(x,t)$, hence the
score needs no extra network:

$$
\nabla \log p_t(x) = -\frac{x - t\,v(x,t) - \mu_0}{\sigma_0^2\,(1-t)} .
$$

## What Oracle Experiments Established (Stage 0, since discarded)

The closed-form oracle was used to test the law's two possible uses on the clumped five-mode
benchmark (`source_std = 0.15`, Euler 5 steps, the protocol of
`results/five_modes_clumped015_multiseed`). Code and artifacts were deliberately not kept; the
numbers below are the distilled knowledge.

**As a training target: falsified.** Regressing the model's JVP material derivative onto the
exact source $-(1/p)\nabla\cdot(p\Sigma)$ (single-difference control against the zero-target
JVP penalty) made 5-step sampling *worse* at every weight tried (W ≈ 1.6–2.5 vs standard CFM
0.80; coverage 0/5). The mechanism is decisive and general: **the exact field itself is not
uniformly integrable at NFE 5 on this geometry.** Integrating the closed-form $v$ with uniform
5-step Euler gives W = 1.11, coverage 0/5 — *worse than trained standard CFM* (0.80, 4.9/5).
All curvature is compressed into a stiff initial layer — mean $\|Dv/Dt\|$ along interpolant
samples: ~67 at $t=0$, ~16 at $t=0.1$, ~0.6 from $t=0.2$ on (the rarefaction fan where mass
commits to modes) — and uniform steps jump over it. Hence *any* regularizer that only improves
fidelity to the true dynamics inherits this failure; standard CFM's coarse-step usability
partly rests on its own smoothing error. Zero-target straightening fails in the dual way
(integrable but wrong dynamics: path ratio → 1.00, coverage 0/5). Both closures lose to leaving
the field alone.

**As a step-placement oracle: confirmed.** Equalize predicted per-step error
($\Delta t_i \propto \|Dv/Dt\|^{-1/2}$; knots at quantiles of the cumulative $\sqrt{\|a\|}$
profile):

- Exact field, same 5 NFE: W 1.11 → **0.35**, hit 1.7% → **88%**, 5/5 (grid ≈
  $\{0, 0.03, 0.08, 0.18, 0.51, 1\}$). Warped 3 steps beat uniform 10.
- Trained standard CFM (10 seeds, paired): uniform-5 W 0.821 → warped-5 **0.548** (hit 0.46 →
  0.73); **warped-5 beats uniform-10** (0.602).
- The **self-warp** — error profile measured from the trained model's own velocity differences
  along a one-time ~50-step probe rollout, no oracle anywhere — matched the oracle warp
  (0.548 vs 0.555). Deployable on any trained field.
- Composes with training-side methods: directional regularization w2 + warp gave the best
  result recorded on the benchmark (W 0.536 → 0.445, hit 0.76 → 0.87–0.89).
- Geometry prediction validated: with a broad source (ring, `source_std = 1.0`) the profile is
  flat (~10–14 over $t \in [0, 0.4]$, no layer), the exact field is already integrable at
  uniform-5, and warping is a wash — predicted by the oracle before evaluating any model.
- Iso-FD models are a *wrong-field* failure, not a wrong-steps failure: no schedule rescues
  them (straightening erased the initial-layer physics).
- OT-minibatch-paired models retain a layer (partial cooling) and still benefit from warping.

## The Distilled Claim

> The CFM marginal field is the bulk velocity of a free-streaming gas whose pressure tensor is
> the conditional covariance of the training target. Its curvature is exactly the pressure
> gradient. Suppressing that curvature in training (straightening) erases mode-commitment
> physics; enforcing it in training reproduces dynamics too stiff for uniform coarse stepping.
> The law's operational value at low NFE is **step placement**: the pressure term is a
> certified local-truncation-error field, and spending a fixed NFE budget by equalizing it
> (oracle-derived or self-probed) beats doubling the budget on a uniform grid.

## Pitfalls Recorded

- The naive "conditional acceleration matching" target is zero (conditional paths are
  straight); the marginal acceleration is *not* $E[\ddot x_t \mid x_t]$. The gap is exactly the
  pressure term — this is why zero-target penalties are subtly wrong, not approximately right.
- A trajectory-acceleration metric computed as second differences is meaningless across
  non-uniform grids; compare only within a schedule.
- Evaluating recorded checkpoints under a different torch version drifted W by up to ~0.1
  (5-step dynamics amplify forward-pass differences); keep schedule comparisons paired within
  one environment.
- The score identity is for Gaussian sources; non-Gaussian sources need their own score
  estimate.

---

# Phase 1 Strategy: A Priori Error-Controlled Sampling for Flow Matching

## Reframed Paper Claim

> Flow matching comes with a free, exact, **a priori** local-truncation-error estimator: the
> pressure term of its own momentum law. Low-NFE sampling is the problem of equidistributing
> it.

This is error-controlled numerical integration with three inversions of the classical setup,
and the inversions are the contribution:

1. **A priori, not a posteriori.** Classical controllers (embedded pairs, step doubling)
   estimate local error online by spending extra function evaluations — the very resource
   being rationed. Here the error density $\|Dv/Dt\| = \|\tfrac{1}{p}\nabla\cdot(p\Sigma)\|$ is
   a *statistical property of the data and coupling*, computable before sampling starts, at
   zero sampling-time NFE.
2. **Budgeted, not tolerance-driven.** Generative sampling fixes the work (NFE) and asks for
   minimal error — the dual of classical adaptivity. The solution is mesh grading
   (equidistribution), not a feedback controller.
3. **Population-level, hence batched.** One precomputed schedule (or per-sample grids carried
   as data) preserves GPU batching; classical per-trajectory feedback does not.

Phase 0's negative results become the motivation section: straightening = cold-gas closure
(erases mode-commitment physics), fidelity enforcement = faithful to a stiff truth (uniformly
unintegrable). Both training-side closures lose to leaving the field alone and stepping
correctly.

## Theory Deliverables

1. **The law** (four-line proof, already done) and the cold-coupling limit
   ($\Sigma \to 0 \Rightarrow$ straight characteristics; OT/Reflow as coolers).
2. **The gain proposition** — the centerpiece, turning the paper from empirical to predictive.
   With population error density $\bar e(t) = E\,\|Dv/Dt(x_t, t)\|$ along the flow and a
   first-order (no-amplification) error model, $k$-step Euler obeys
   $$
   E_{\text{unif}} \approx \frac{1}{2k}\int_0^1 \bar e\,dt,
   \qquad
   E_{\text{warp}} \approx \frac{1}{2k}\Big(\int_0^1 \sqrt{\bar e}\,dt\Big)^{2},
   $$
   the optimum attained by step density $\propto \sqrt{\bar e}$ (Cauchy–Schwarz). Define the
   **stiffness concentration**
   $$
   \kappa \;=\; \frac{\int_0^1 \bar e\,dt}{\big(\int_0^1 \sqrt{\bar e}\,dt\big)^2} \;\ge\; 1,
   \qquad \kappa = 1 \iff \bar e \text{ constant}.
   $$
   $\kappa$ is the predicted step-efficiency gain of warping over uniform. Sanity check
   against Phase 0: the clumped profile ($\bar e \approx 67 \to 0.6$ in a $\delta \approx 0.1$
   layer) gives $\kappa \approx 3$, matching warped-5 $\gtrsim$ uniform-10–15; the ring profile
   is flat, $\kappa \approx 1$, and warping was indeed a wash. State the Gronwall-factor caveat
   (error propagation ignored) honestly.
3. **Moment hierarchy remark**: differentiating the law along the flow expresses
   $D^2v/Dt^2$ through third conditional moments — i.e. every higher-order integrator has its
   own a priori error density, all regressable from data. (One proposition; Heun's density used
   empirically.)

## The Estimator Ladder

How to get $\bar e(t)$ (and pointwise $\|\hat a(x,t)\|$) without a closed form:

- **E0 — oracle** (Gaussian mixtures only): ground truth for validation.
- **E1 — self-probe** (validated in Phase 0): one fine probe rollout of the trained model,
  $\|\Delta v / \Delta t\|$ along its own flow. Model-only, zero training change. Known
  failure: it reports the *model's* curvature, so an over-straightened field self-reports
  smooth — it cannot diagnose wrong-field failures (Iso-FD looked integrable and wasn't).
- **E2 — dispersion head**: a small second-moment output $M_\phi(x,t) \approx E[uu^\top|x_t=x]$
  (scalar/diagonal closure $m_\phi \approx \tfrac1d E[\|u\|^2|x]$ in high-$d$), trained by
  plain regression next to $v_\theta$; then $\Sigma = M_\phi - vv^\top$, score from
  $\nabla\log p_t = -(x - tv - \mu_0)/(\sigma_0^2(1-t))$, error field
  $\|\nabla\cdot\Sigma + \Sigma\nabla\log p\|$. Measures the *data-side* truth: detects broken
  fields E1 cannot, evaluates pointwise without rollouts (enabling per-sample grids), and
  measures gas temperature independently of the velocity fit.

Phase-1 question for the ladder: E2 ≈ E1 ≈ E0 on healthy fields (toys, vs closed form); E2
disagrees with E1 exactly on straightened fields. That disagreement is itself a diagnostic
deliverable: a **model audit** — "your field is smoother than your data's pressure profile
permits; expect missing modes."

## Sampler Algorithms

- **S1 — global warp** (validated): knots at quantiles of $\sqrt{\bar e}$. Zero overhead,
  drop-in.
- **S2 — per-sample budgeted controller**: each sample carries its own grid from pointwise
  $\|\hat a\|$ (E2), same step *count* for batching; remaining-budget normalization guarantees
  landing at $t=1$ in exactly $k$ steps. Hypothesis: per-sample $\kappa$ exceeds population
  $\kappa$ whenever samples commit at different times (the population mean smears the layer);
  gains over S1 quantify that.
- **S3 — warped higher order**: Heun/midpoint on graded grids; at fixed NFE budget, answer
  "order vs steps" (warped Euler-$k$ vs warped Heun-$k/2$) with the moment-hierarchy density.

## Benchmark Ladder: Central Questions

Before adding another pressure-aware method, the next benchmarks should answer four questions in
order:

1. **Does exact pressure curvature matter numerically?** On oracle-available problems, integrate
   the exact marginal field directly. If pressure-aware schedules or solvers do not help the
   exact field, they are unlikely to help a learned approximation.
2. **Can neural CFM learn the pressure layer?** Compare E0/E2 data-side curvature against E1
   model-side curvature under controlled capacity, training time, and regularization. If the
   learned field smooths away the layer, sampler-side pressure estimates will be misaligned with
   the field actually being integrated.
3. **Does coupling cool the pressure field?** Vary the coupling from independent pairing through
   partial OT, minibatch OT, Sinkhorn, and Reflow-like iterations, then measure dispersion,
   pressure kappa, model kappa, and low-NFE behavior.
4. **Does controlling pressure improve low-NFE sampling?** Only after the first three questions
   are answered should we choose between pressure-aware coupling, pressure-aware time meshes,
   pressure-budget regularization, or abandoning the line.

This is deliberately a benchmark ladder rather than a benchmark zoo: each problem should test
whether the theory survives the transition from exact marginal fields to learned neural fields.

## Training-Side Implications From The Capacity Audit

The capacity audit on the clumped five-mode oracle benchmark changed the status of training-side
pressure methods:

- The pressure layer is learnable: larger or longer-trained standard CFM models move model-side
  kappa toward oracle kappa.
- More faithful pressure-layer learning makes uniform low-NFE Euler worse, not better.
- Warped grids rescue the faithful models, often beating uniform grids with twice the step count.

So the pressure equation should not be used as a naive exact-acceleration target if the evaluation
solver is fixed to uniform Euler. That would train a more truthful but stiffer field — precisely
the failure seen in the earlier exact-pressure matching attempt. The viable training-side idea is
more specific:

> Use the pressure equation to distinguish necessary pressure-driven curvature from spurious
> curvature, while training a field whose pressure layer is matched to the intended numerical
> resolution.

Candidate training objectives:

1. **Pressure-consistent CFM.** Train a velocity model and pressure/dispersion head, then add
   $$
   \left\|
   \frac{Dv_\theta}{Dt}
   +
   \nabla\cdot\Sigma_\phi
   +
   \Sigma_\phi\nabla\log p_t
   \right\|^2 .
   $$
   This is the cleanest PDE-consistency loss, but by itself it likely increases stiffness. It is
   appropriate only if paired with pressure-aware sampling or used as a diagnostic/fidelity term.
2. **Mesh-aware pressure matching.** Replace the exact pressure acceleration $a_p$ by a
   numerically resolvable target $a_{p,h}$:
   $$
   \mathcal L_{\text{mesh-pressure}}
   =
   \left\|
   \frac{Dv_\theta}{Dt}
   -
   a_{p,h}
   \right\|^2 .
   $$
   Here $a_{p,h}$ could be a time-smoothed, clipped, or grid-averaged pressure force determined
   by the intended NFE budget. This is the most plausible training method if the paper wants a
   training contribution rather than an inference-only contribution.
3. **Pressure-budget regularization.** Do not force $Dv_\theta/Dt$ to zero; penalize only the
   part that is not explained by pressure, or the curvature exceeding a pressure-scaled budget:
   $$
   \|P_{\perp a_p}(Dv_\theta/Dt)\|^2
   \quad\text{or}\quad
   [\|Dv_\theta/Dt\| - c\|a_p\|]_+^2 .
   $$
   This preserves pressure-aligned mode-commitment curvature while suppressing roughness not
   justified by the data/coupling.

The most attractive paper direction, if it works, is therefore not "pressure-aware inference"
alone. It is a coupled claim:

> The pressure law exposes a fidelity/integrability tradeoff in CFM. We train pressure-consistent
> or pressure-budgeted fields and sample them on pressure-matched meshes, preserving necessary
> multimodal curvature without paying unnecessary NFE.

Immediate test before committing to this direction: on the clumped oracle benchmark, compare
exact-pressure matching, mesh-filtered pressure matching, and pressure-budget regularization under
both uniform and warped samplers. A method only counts as a win if it improves the Pareto curve of
endpoint quality versus low-NFE integration error, not merely one axis.

Status update from the first viability probe: oracle pressure losses must be normalized by
aggregate pressure energy, not per-sample pressure energy. With corrected normalization on staged
modes, upper-budget and pressure-band losses show a small three-seed improvement over standard
CFM under uniform Euler-5, while exact pressure matching is not best. This keeps pressure-aware
training alive, but only as a fragile hypothesis. The next decision should come from a tuned
upper/band grid repeated across staged, clumped, and 16D mixtures; no pressure-training objective
should enter the main method registry before that. See
`docs/archive/pressure_training_status.md` for the current short status.

Follow-up update: the stronger sweep weakens the direct-training story. Upper/band pressure
losses help some 2D endpoint metrics, but on the 16D mixture they greatly reduce integration
error while destroying endpoint W and hit rate. Direct oracle pressure-budget training should
therefore be treated as a failed or incomplete method, not a paper contribution. If pressure enters
training again, it should be through a mesh-filtered target, a pressure-aware sampler/training
pair, or a diagnostic head rather than naive direct regularization.

## Experiment Ladder

- **X0 — $\kappa$ survey first (the go/no-go).** Before building anything else, measure
  $\kappa$ via E1/E2 on every candidate domain: clumped, ring, fan, spiral, staged modes,
  16D Gaussian mixture, `burgers_autoregressive`, a standard 2D hard pair
  (8-Gaussians → checkerboard), and one image FM model (CIFAR-10 or ImageNet-32 class).
  $\kappa \approx 1$ everywhere realistic
  ⇒ the method is a theory note about synthetic stiff layers — find out in week one, not month
  three.
- **X1 — predicted-vs-realized gain** (signature figure): x-axis $\kappa$ measured by E2,
  y-axis realized warp gain on trained models, one point per geometry × coupling
  (independent / minibatch-OT / global-OT or Reflow). The theory predicts the diagonal; the
  cold-coupling points should slide toward $(1,1)$.
- **X2 — estimator equivalence and the audit**: E0/E1/E2 profiles overlaid on toys;
  the Iso-FD broken-field case where E1 lies and E2 does not.
- **X3 — scale**: Burgers autoregressive (does grading per rollout-step help the frame-to-frame
  flows?) and the image domain at NFE ∈ {3, 5, 10} against the schedule heuristics below.
- **X4 — cooling diagnostics**: $\int \mathrm{tr}\,\Sigma_\phi$ and $\kappa$ across Reflow
  iterations / OT batch sizes — the head *measures* the temperature drop that rectification
  claims. Standalone interpretability result; no sampling needed.

## Baselines

Uniform Euler / midpoint / Heun at matched NFE; uniform at 2× NFE; the tuned-heuristic warps
from diffusion practice adapted to FM time (cosine, Karras/EDM $\rho$-grid, logSNR); a
validation-searched monotone warp (empirical upper bound for *any* global schedule — S1 should
approach it without search); Bespoke-solver-style learned schedules cited and compared
conceptually (they retrain per model; we are training-free given the head). Keep the
training-side trio (standard / Iso-FD / directional) as context and carry the Phase-0
composition result (directional + warp) forward. Distillation methods are out of scope (they
change the learned object); say so explicitly.

Metrics: W2 + hit/coverage (toys), sliced-W2 beyond 2D, RMSE/temporal-TV (Burgers), FID
(images). Never compare finite-difference trajectory statistics across different grids.

## Pre-Registered Predictions

- **P1 (gain law)**: realized warp gain tracks $\kappa$ across geometries and couplings.
- **P2 (estimators)**: E2 ≈ E1 ≈ E0 on healthy fields; E1 ≠ E2 exactly on over-regularized
  fields, and E2 is the one that matches outcome.
- **P3 (cooling)**: $\kappa \to 1$ and warp gain $\to$ none as the coupling becomes
  deterministic; minibatch-OT sits in between (Phase 0 already saw the intermediate point).
- **P4 (per-sample)**: S2 > S1 in proportion to the spread of per-sample commitment times;
  zero on the clumped toy (all commit together), positive on fan/asymmetric geometries.

## Kill Criteria

- X0 finds $\kappa \approx 1$ on all non-synthetic domains → stop; write the theory +
  diagnostics (X4) as a short paper.
- E2 cannot reproduce E0 profiles on toys at reasonable capacity → fall back to E1-only;
  lose per-sample grids and the audit; scope shrinks to S1 + theory.
- S1 fails to beat tuned heuristic warps anywhere despite $\kappa > 1$ → the first-order error
  model is too naive (amplification dominates); investigate Gronwall correction before
  abandoning.

## Scope Exclusions (parked, not forgotten)

SDE/stochastic samplers, consistency distillation, and non-Gaussian-source score estimation remain
parked. Training-side regularization is no longer excluded categorically: the capacity audit makes
pressure-aware training scientifically plausible, but only as a mesh-aware or pressure-budgeted
method. Naive exact-pressure matching remains a known failure mode for uniform low-NFE sampling.

## Milestones

- **M0**: rebuild oracle + probe tooling (one day; everything is specified in this document);
  run X0 $\kappa$ survey → go/no-go per domain.
- **M1**: theory writeup (law, gain proposition with caveat, cold limit, moment hierarchy).
- **M2**: dispersion head E2 + validation X2.
- **M3**: samplers S1–S3.
- **M4**: X1 signature figure; X3 scale experiments.
- **M5**: X4 cooling diagnostics; writing.
