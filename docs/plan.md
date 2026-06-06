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

## Full Derivation Narrative

### 1. Start From The Learned Sampling ODE

Flow matching learns a time-dependent velocity field

$$
v_\theta(x,t),
$$

and samples by solving the ODE

$$
\frac{dx(t)}{dt}
=
v_\theta(x(t),t).
$$

This is the Lagrangian viewpoint: instead of looking at the field at a fixed spatial coordinate, we follow a generated particle as it moves through the learned velocity field.

The central quantity is not only the instantaneous velocity, but how that velocity changes along the trajectory:

$$
\frac{d}{dt}v_\theta(x(t),t).
$$

### 2. Apply The Chain Rule

By the chain rule,

$$
\frac{d}{dt}v_\theta(x(t),t)
=
\partial_t v_\theta(x,t)
+
\left(\frac{dx(t)}{dt}\cdot\nabla_x\right)v_\theta(x,t).
$$

Since the particle itself follows

$$
\frac{dx(t)}{dt}=v_\theta(x(t),t),
$$

we obtain the material derivative:

$$
\frac{Dv_\theta}{Dt}(x,t)
=
\partial_t v_\theta(x,t)
+
(v_\theta(x,t)\cdot\nabla_x)v_\theta(x,t).
$$

We will call this the material residual:

$$
R_\theta(x,t)
=
\partial_t v_\theta(x,t)
+
(v_\theta(x,t)\cdot\nabla_x)v_\theta(x,t).
$$

Interpretation:

- $R_\theta$ is the pathwise acceleration of the learned generative flow.
- $R_\theta \approx 0$ means a particle experiences nearly constant velocity along the learned trajectory.
- Large $R_\theta$ means the local trajectory bends or changes speed quickly.

### 3. Connect The Residual To Low-NFE Solver Error

For a short time step $\Delta t$, Taylor expansion of the true trajectory gives

$$
x(t+\Delta t)
=
x(t)
+
\Delta t\,v_\theta(x(t),t)
+
\frac{\Delta t^2}{2}R_\theta(x(t),t)
+
O(\Delta t^3).
$$

Explicit Euler keeps only the first two terms:

$$
x_{n+1}
=
x_n
+
\Delta t\,v_\theta(x_n,t_n).
$$

Therefore the leading local truncation error contains

$$
\frac{\Delta t^2}{2}R_\theta(x,t).
$$

This gives the numerical motivation for material-residual regularization: if we want reliable low-NFE sampling, we should reduce the residual that dominates coarse-step integration error.

The naive acceleration-regularized objective would be

$$
\mathcal{L}_{acc}
=
\mathbb{E}_{x,t}
\left[
\|R_\theta(x,t)\|^2
\right].
$$

This is the point where the old LC finite-difference loss and Iso-FM live: they approximately penalize this material residual using a finite-difference consistency term along the model trajectory.

### 4. Why Uniform Residual Suppression Is Too Strong

Uniformly forcing $R_\theta \approx 0$ is not obviously correct for generative flow matching.

In multimodal transport, a single intermediate state $x_t$ can be compatible with several different target modes. The conditional target velocity

$$
u = x_1 - x_0
$$

may have high conditional variance near that state:

$$
\mathrm{Var}[u\mid x_t,t].
$$

In that regime, acceleration may be statistically necessary: the marginal flow may need to bend or accelerate as uncertainty resolves and samples commit to modes. Penalizing all material acceleration equally can over-straighten the transport, damage mode coverage, or delay mode commitment.

So the improved question is not:

> Can we make $R_\theta$ small everywhere?

but:

> Can we reduce the part of $R_\theta$ that is numerically harmful for coarse integration, while avoiding strong penalties where acceleration is required by conditional ambiguity?

## Directional-Regularization CFM

The current working method is Directional-Regularization CFM: a flow-matching objective with a material-residual penalty weighted by how quickly the learned velocity changes along its own direction.

$$
\mathcal{L}_{DR\text{-}CFM}
=
\mathcal{L}_{FM}
+
\lambda
\,
\mathbb{E}_{x,t}
\left[
w_{\mathrm{dir}}(x,t)
\,
\|R_\theta(x,t)\|^2
\right].
$$

Here:

- $w_{\mathrm{dir}}$ emphasizes regions where the velocity field changes sharply along the generated trajectory direction.
- $\lambda$ is the global regularization strength.

This replaces the earlier shorthand

$$
\tau(x,t)\|R_\theta(x,t)\|^2
$$

with a directly interpretable directional weighting:

$$
\tau(x,t)
=
w_{\mathrm{dir}}(x,t).
$$

## Directional Weighting

### The Caveat

The earlier plan wrote

$$
\tau_{\mathrm{CFL}}(x,t)
\approx
\frac{\Delta t_{\mathrm{infer}}}
{1+\Delta t_{\mathrm{infer}}L_\theta(x,t)}.
$$

This expression should not be described as a weight that simply "grows when the field is risky." It is closer to a bounded stabilization time scale. It grows with the intended inference step size $\Delta t_{\mathrm{infer}}$, but it shrinks when the local stiffness proxy $L_\theta$ is large.

That behavior can be reasonable for a SUPG-style stabilization parameter, where $\tau$ often behaves like a local time scale. But it is not the same thing as a risk-amplifying loss weight.

We separate two concepts:

1. A **SUPG-style time scale** that normalizes residual stabilization.
2. A **directional weight** that increases when the field changes rapidly along the learned flow direction.

### Local CFL Proxy

Classical CFL reasoning compares a time step to the local spatial scale over which the solution changes. In a learned vector field, we do not have a physical grid spacing $h$, but we can estimate a local inverse length scale using the Jacobian.

Let

$$
L_\theta(x,t)
\approx
\|\nabla_x v_\theta(x,t)\|
$$

or, more cheaply, a directional estimate along the flow:

$$
L_{\mathrm{dir}}(x,t)
=
\frac{
\|(\nabla_x v_\theta(x,t))v_\theta(x,t)\|
}{
\|v_\theta(x,t)\|+\epsilon
}.
$$

Then a dimensionless directional-change proxy is

$$
C_\theta(x,t)
=
\Delta t_{\mathrm{infer}} L_{\mathrm{dir}}(x,t).
$$

This is analogous to a local CFL number: large $C_\theta$ means the inference step is large relative to the local scale on which the velocity field changes.

### Option A: SUPG-Style Stabilization Time Scale

If we want a SUPG-like parameter, a reasonable form is

$$
\tau_{\mathrm{SUPG}}(x,t)
=
\frac{\Delta t_{\mathrm{infer}}}
{1+\Delta t_{\mathrm{infer}}L_{\mathrm{dir}}(x,t)}.
$$

This is a bounded time scale. It should be interpreted as a stabilization scale, not as a direct "penalize stiff regions more" weight.

It may be useful if the regularizer is written in a normalized form such as

$$
\|\tau_{\mathrm{SUPG}} R_\theta\|^2,
$$

or if we derive a residual-based stabilization term that needs a local time-scale coefficient.

### Option B: Directional Weight

If our goal is to penalize residuals more where the learned trajectory direction crosses rapidly changing velocity regions, then the weight should increase with $C_\theta$.

A bounded monotone choice is

$$
w_{\mathrm{dir}}(x,t)
=
\Delta t_{\mathrm{infer}}^2
\,
\frac{C_\theta(x,t)^2}
{1+C_\theta(x,t)^2}.
$$

This says:

- the penalty matters more when we intend to sample with a larger inference step,
- the penalty grows as local stiffness becomes large relative to that step,
- the weight saturates so very stiff outliers do not dominate training.

This is currently the cleaner default for the paper narrative because it directly matches the low-NFE error story.

We should compare both options experimentally:

- **SUPG-scale residual:** $\|\tau_{\mathrm{SUPG}} R_\theta\|^2$
- **Directional residual:** $w_{\mathrm{dir}}\|R_\theta\|^2$

If only one survives, the paper should use the surviving version and describe the other as an ablation.

## Shelved: Uncertainty-Aware Gate

The uncertainty gate was considered as a way to reduce regularization where the conditional transport is ambiguous:

$$
g_{\mathrm{unc}}(x,t)
=
\frac{1}
{1+\kappa \widehat{\mathrm{Var}}[u\mid x_t,t]}.
$$

Here $u=x_1-x_0$ is the flow-matching target velocity, and $\widehat{\mathrm{Var}}[u\mid x_t,t]$ is an estimate of local conditional velocity variance.

Interpretation:

- high variance means many plausible target directions,
- high variance makes $g_{\mathrm{unc}}$ small,
- the residual penalty becomes weaker,
- the model is allowed to accelerate while resolving multimodal ambiguity.

Low variance means the local transport direction is already resolved:

- $g_{\mathrm{unc}}$ is near one,
- residual regularization is strong,
- acceleration is treated as likely numerical roughness rather than necessary mode commitment.

### Practical Data-Dependent Gate

Possible estimators included:

- nearest-neighbor velocity variance around $x_t$,
- minibatch kernel-weighted velocity variance,
- repeated candidate couplings,
- ensemble or dropout disagreement,
- learned auxiliary uncertainty head.

The implemented prototypes used minibatch k-nearest-neighbor and kernel estimates in joint $(x_t,t)$ space. On the current five-mode benchmark they mostly weakened useful regularization and did not improve mode capture, so this branch is shelved until a benchmark clearly needs ambiguity-aware regularization.

## Final Candidate Losses

The core proposed family is:

$$
\mathcal{L}
=
\mathcal{L}_{FM}
+
\lambda
\,
\mathbb{E}_{x,t}
\left[
w_{\mathrm{dir}}(x,t)
\,
\frac{\|R_\theta(x,t)\|^2}{\|v_\theta(x,t)\|^2+\zeta}
\right].
$$

The velocity normalization is optional but likely helpful. It prevents high-speed regions from dominating only because velocities are large. We should test both normalized and unnormalized variants.

Minimum variants:

1. **Standard CFM:** no residual regularization.
2. **Uniform residual:** $\|R_\theta\|^2$.
3. **LC finite difference / Iso-FM-style:** semi-Lagrangian finite-difference residual proxy.
4. **Directional-regularized residual:** $w_{\mathrm{dir}}\|R_\theta\|^2$.

## Paper Narrative

The writeup should proceed in this order:

1. Start from the learned flow-matching ODE.
2. Follow a generated sample along its trajectory.
3. Derive the material residual by the chain rule.
4. Connect the residual to the leading low-NFE Euler truncation error.
5. Explain why uniform residual suppression is too strong for multimodal generative transport.
6. Introduce directional weighting using local velocity change along the learned flow direction.
7. Treat LC finite difference and Iso-FM as close acceleration-regularization baselines.

The intended claim is:

> Acceleration regularization in flow matching should be selective: strong where residuals cause numerical integration error, weak where acceleration reflects unresolved generative ambiguity.

## Immediate Benchmark Needs

The first benchmark should expose both sides of the idea:

- coarse-step fragility under low-NFE inference,
- early-time or multimodal ambiguity where uniform acceleration suppression may over-constrain.

Useful metrics:

- low-NFE Wasserstein or task error,
- fixed low-NFE comparison at the first pass,
- trajectory acceleration / material residual,
- path length ratio,
- mode coverage or mode assignment accuracy on multimodal toy data,
- sensitivity to regularization strength and directional approximation.

## Baselines

Required initial baselines:

- Standard CFM
- LC finite difference as our semi-Lagrangian/Iso-FM-style variant
- Iso-FM-faithful finite-difference loss if its weighting/normalization differs materially
- JVP material-derivative penalty
- SUPG-scale residual
- Directional-Regularization CFM

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
- a clear directional-regularization explanation that predicts when the regularizer should help,
- a meaningful ablation showing whether directional weighting contributes beyond Iso-FD.

## Paper Figure: Regularization Components on Five-Mode Benchmark

A planned figure for the paper showing the effect of each regularization component as a side-by-side scatter plot grid. The goal is to isolate what each piece contributes.

### Proposed panels

**Panel 1 — Standard CFM (no regularization):** baseline, shows mode coverage but high trajectory acceleration and coarse-step error.

**Panel 2 — Uniform material residual (λ‖R‖²):** shows that naive uniform suppression over-straightens trajectories and damages hit rate at the same λ. The over-regularized anchor (λ=10, 0/5 coverage, accel≈0) is already in hand.

**Panel 3 — Directional-Regularization CFM:** tests whether directional weighting can improve coarse-step sampling beyond uniform Iso-FD-style regularization.


### Design notes

- All panels use the same axis limits, color scheme, and metric overlay (W | hit% | acc, coverage N/5).
- Panels should be small (~3×3 inches) to fit as a grid in a paper column.
- Fixed eval seed across all panels for directly comparable sample clouds.
- The λ=10 uniform run can serve as the "over-regularized" anchor.
