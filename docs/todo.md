# Active Todo

## Priority 1: Make SCTW Less Ad Hoc

- Add schedule-shape diagnostics for CIFAR and staged-shapes:
  - plot self-curvature profile `e(t)`;
  - plot SCTW knots for `warp_power` values;
  - plot best hand power knots;
  - report early/mid/late step allocation.
- Add scalar profile diagnostics:
  - concentration/kappa;
  - fraction of curvature mass before `t <= 0.2`;
  - entropy or effective support of the profile;
  - max/mean profile ratio.
- Propose and test an automatic `warp_power` rule from the profile diagnostics.

## Priority 2: Strengthen Baselines

- Keep uniform Euler at NFE `{5, 10, 20, 50}`.
- Keep hand power schedules:
  - early;
  - late;
  - symmetric;
  - several `rho` values.
- Add at least one diffusion-style hand schedule if it fits the FM time parameterization:
  - cosine-like;
  - EDM/Karras-like;
  - log-SNR-like only if the parameterization is scientifically honest.
- Keep Heun at matched NFE budget where meaningful.
- Add a validation-searched monotone schedule as an empirical upper bound if it is cheap enough.

## Priority 3: Paper-Facing Experiments

- Re-run SCTW/hand/uniform comparisons with cleaned scripts and fixed evaluation seeds.
- Produce side-by-side plots for:
  - CIFAR samples at low NFE;
  - staged-shapes samples/trajectories;
  - profile/time-grid diagnostics.
- Decide whether the main quantitative image claim uses:
  - unconditional CIFAR only;
  - staged-shapes plus CIFAR;
  - or a third benchmark to avoid overfitting to two cases.

## Priority 4: Theory And Writing

- Write the derivation from the learned FM ODE to material derivative to Euler local error.
- State the equidistribution principle:
  $$
  \Delta t_i^2 e(t_i) \approx \text{constant}
  $$
  or the tempered version actually used.
- Explain why tempering may be needed:
  - probe noise;
  - amplification/Gronwall effects ignored by the local model;
  - very low NFE brittleness;
  - mismatch between population profile and individual sample difficulty.
- Include a "what this is not" paragraph:
  - not pressure-aware implementation;
  - not learned solver;
  - not training regularization;
  - not a claim that adaptive timesteps are new.

## Parked

- Direct pressure-aware training. It is not robust enough yet.
- Pressure-aware coupling as the main claim. CIFAR evidence is mixed/modest.
- Per-sample SCTW. Interesting, but it may break batching or need an extra learned head.
- Higher-order pressure/moment methods. Too much depth before the simpler sampler claim is stable.

## Decision Gate

Continue with SCTW as the main method only if the next diagnostic round shows:

- SCTW beats uniform reliably when the self-curvature profile is concentrated;
- SCTW is competitive with, or more robust than, hand schedules across multiple geometries;
- the profile diagnostics explain when it wins and when it does not.

If tuned hand schedules dominate everywhere, reframe SCTW as a diagnostic for schedule selection
rather than as a standalone sampler.
