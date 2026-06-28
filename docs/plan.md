# Active Plan: Self-Curvature Time Warping for Flow Matching

## Current Thesis

The strongest current paper direction is no longer direct pressure-aware training or
pressure-aware coupling. Those ideas produced useful diagnostics and some positive cases, but
they are not robust enough yet to carry the paper.

The active method candidate is:

> **Self-Curvature Time Warping (SCTW):** a training-free, oracle-free timestep mesh for low-NFE
> flow-matching sampling. It probes a trained velocity field along its own rollout, estimates
> where the model has high material-derivative/self-curvature, and spends Euler steps there.

Pressure theory remains important, but mainly as the derivational and diagnostic lens:

- it explains why marginal CFM trajectories can curve in multimodal transport;
- it explains why zero-acceleration regularizers can erase mode-commitment physics;
- it motivates measuring pathwise acceleration/material derivative;
- it does **not** currently give a robust training objective by itself.

## Method In One Page

Flow matching samples from the learned ODE

$$
\frac{dx}{dt}=v_\theta(x,t).
$$

Along a generated trajectory, the velocity changes according to the material derivative

$$
\frac{D v_\theta}{Dt}
=
\partial_t v_\theta+(v_\theta\cdot\nabla_x)v_\theta.
$$

For a small Euler step,

$$
x(t+\Delta t)
=
x(t)+\Delta t\,v_\theta(x(t),t)
+\frac{\Delta t^2}{2}\frac{D v_\theta}{Dt}(x(t),t)
+O(\Delta t^3).
$$

Thus coarse Euler error is driven, to first order, by the magnitude of the material derivative.
SCTW estimates this quantity without JVPs or an oracle by a fine probe rollout:

$$
e(t_i)\approx
\frac{\|v_\theta(x_{i+1},t_{i+1})-v_\theta(x_i,t_i)\|}{\Delta t}.
$$

It then builds a nonuniform time mesh using a tempered density

$$
\rho(t)=(e(t)+\epsilon)^p.
$$

Current image-scale defaults:

- `profile_samples = 512`
- `profile_fine_steps = 50`
- `warp_power = 0.25`
- `warp_floor = 1e-3`

The raw Euler equal-error exponent would be near `p = 0.5`; `p = 0.25` is a tempered version
that is less brittle at very low NFE.

## What Is Actually Different

SCTW is not novel because it uses nonuniform timesteps. That idea is common. The possible
contribution is narrower:

> A cheap, training-free, native-CFM schedule built from the model's own pathwise
> material-derivative/self-curvature profile, with no learned solver, no dynamic programming,
> no oracle pressure field, and no retraining.

Compared with Iso-FM-style finite-difference regularization, SCTW does not change the learned
field. Iso-FM tries to make the field easier to integrate by reducing material acceleration
during training; SCTW accepts the learned field and changes where the inference budget is spent.

Compared with bespoke or learned solvers, SCTW keeps the update rule fixed as Euler and only
changes the mesh. This is weaker but simpler, cheaper, and easier to isolate scientifically.

Compared with hand schedules such as early/late/symmetric power grids, SCTW is data/model-adaptive.
This distinction matters because staged-shapes prefers an aggressive early schedule, while CIFAR-10
does not. A fixed hand schedule can win on one benchmark for the wrong reason; SCTW should win or
remain competitive without choosing the schedule family by hand.

## Current Evidence

### CIFAR-10 Unconditional Sinkhorn Model

EMA model, 5000 samples, FID lower is better:

| NFE | Uniform | SCTW p=0.25 | SCTW p=0.5 | Best hand power |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 41.4147 | 41.7388 | 44.9989 | 47.8957 |
| 10 | 28.9758 | 28.3254 | 28.7508 | 30.0170 |
| 20 | 23.8106 | 22.9072 | 22.7874 | 23.3659 |
| 50 | 20.9632 | 20.3763 | 20.2047 | 20.3376 |

Interpretation: SCTW beats uniform at NFE 10/20/50 and beats the tested hand power schedules.
The NFE 5 case is not solved: p=0.25 is close to uniform, and p=0.5 is too aggressive.

### Staged-Shapes Easy

Feature W1 lower is better:

| NFE | Uniform | SCTW p=0.25 | SCTW p=0.5 | Best hand power |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 1.5238 | 1.3754 | 1.0118 | 0.9187 |
| 10 | 1.3524 | 1.0482 | 0.8757 | 0.8116 |
| 20 | 1.0977 | 0.8952 | 0.7861 | 0.7433 |
| 50 | 0.8892 | 0.7823 | 0.7286 | 0.7149 |

Interpretation: SCTW beats uniform, but a simple early-concentrated hand schedule beats it.
This is a warning. We should not claim schedule dominance. The better claim is adaptivity across
problems and a principled diagnostic for when nonuniform stepping should help.

## Method Depth Needed Before A Paper Claim

The current method is promising but still shallow. To make it paper-worthy, we need at least one
of the following:

1. **Self-tuned tempering.** Choose `warp_power` from the measured curvature profile instead of
   fixing it. This directly addresses the CIFAR-vs-staged tradeoff.
2. **Equidistribution derivation.** State the numerical-analysis principle clearly: the mesh is
   approximately equalizing estimated local Euler error, possibly with a tempering factor to
   control probe noise and error amplification.
3. **Schedule diagnostics.** Show that the measured profile predicts when uniform Euler fails and
   when hand early/late schedules are appropriate.
4. **Harder baselines.** Compare against uniform Euler, tuned hand schedules, higher-order Heun,
   and at least discuss learned/bespoke solvers and diffusion schedule optimization.

Without one of these, SCTW risks being perceived as a reasonable heuristic rather than a paper
method.

## Current Non-Claims

Do not claim:

- that pressure-aware training is solved;
- that pressure-aware coupling is the main image-scale contribution;
- that SCTW dominates all hand schedules;
- that adaptive timesteps for generative ODEs are new;
- that the method is pressure-aware in implementation.

Do claim, if the next evidence supports it:

- SCTW is a simple self-curvature schedule for native flow matching;
- it improves over uniform Euler on several low-NFE settings;
- it can be more robust than fixed hand schedules across problem geometries;
- the material-derivative profile is an interpretable diagnostic for low-NFE difficulty.

## Immediate Next Step

Build a **schedule-shape diagnostic**:

- plot SCTW profile density and time knots for CIFAR and staged-shapes;
- compare against best hand power grids;
- report early/mid/late step allocation;
- quantify profile concentration and use it to propose an automatic `warp_power`.

This is the shortest path to making the method less ad hoc.
