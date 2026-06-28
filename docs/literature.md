# Nearby Literature And Current Differentiation

## Short Verdict

The novelty risk is real. Adaptive timestep schedules, learned solvers, and nonuniform samplers
for generative ODEs already exist. The current project should not claim that nonuniform sampling
is new.

The defensible distinction is narrower:

> SCTW is a cheap, training-free, oracle-free, native flow-matching timestep mesh derived from
> the trained model's own pathwise material-derivative/self-curvature profile.

That is different from training-time acceleration regularization, learned solver design, and
validation-optimized schedule search. It is not yet deep enough unless we add self-tuning,
equidistribution analysis, or stronger schedule diagnostics.

## Closest Method Families

### Iso-FM / Acceleration-Regularized Flow Matching

Iso-FM-style methods regularize the velocity field during training, often with a finite-difference
or Jacobian-free approximation to pathwise material acceleration. This is close to our old
LC-finite-difference and JVP residual ideas.

Difference:

- Iso-FM changes the learned vector field.
- SCTW leaves the trained vector field unchanged and changes only the inference time mesh.
- Iso-FM tries to make trajectories straighter or easier; SCTW tries to integrate the trajectory
  the model already learned.

Scientific caveat: this is a clean distinction, but not automatically a stronger method. Iso-FM is
a necessary baseline whenever the paper discusses material derivatives.

### Bespoke / Learned Solvers For Generative Flows

Bespoke solver work learns or optimizes solver coefficients/schedules for a given generative ODE.
This can be stronger than a fixed Euler mesh, but it is a different intervention: the solver itself
is designed or trained.

Difference:

- learned/bespoke solvers optimize the update rule or solver coefficients;
- SCTW keeps Euler fixed and only chooses the mesh;
- SCTW is simpler and cheaper, but likely less expressive.

Paper posture: compare conceptually and, if feasible, include a representative learned-schedule or
validation-searched schedule as an upper-bound baseline.

### GITS / Trajectory-Regularity Timesteps

Trajectory-regularity timestep methods are the closest conceptual neighbor. They also use a notion
of where the generated trajectory is hard to integrate and allocate timesteps nonuniformly.

Difference we need to defend:

- SCTW is stated directly for native flow-matching velocity fields;
- the diagnostic is the learned field's material-derivative/self-curvature profile along FM
  rollouts;
- the proposed mesh is a cheap probe-and-quantile construction rather than a dynamic-programming
  or expensive search procedure;
- the pressure/material-derivative derivation provides a flow-matching-specific interpretation.

This is the dangerous comparison. If we cannot show a crisp distinction or a simpler practical
advantage, reviewers may see SCTW as a small adaptation of existing adaptive schedule ideas.

### Diffusion Adaptive Timestep Schedules

Diffusion samplers often use noise/time schedules, hand-designed grids, adaptive steps, or
validation-tuned schedules. Those methods are usually phrased in diffusion/noise parameterization,
not native CFM velocity fields.

Difference:

- SCTW does not assume a diffusion noise schedule;
- it estimates curvature from the learned FM velocity itself;
- it can be applied to ordinary CFM checkpoints without converting the model into a diffusion
  parameterization.

This distinction is useful but modest. The paper must avoid sounding as if adaptive scheduling is
new.

### Hand Power / Cosine / EDM-Style Schedules

Hand schedules allocate more steps early, late, symmetrically, or according to a fixed power law.
They are simple and strong.

Difference:

- hand schedules require choosing a shape class before seeing the model;
- SCTW measures the model's actual self-curvature profile;
- staged-shapes currently favors an early hand schedule, while CIFAR does not, which supports the
  need for adaptivity.

Scientific caveat: if a tuned hand schedule beats SCTW across all benchmarks, SCTW becomes a
diagnostic rather than a method.

## References To Track

- Bespoke Solvers for Generative Flow Models: https://openreview.net/forum?id=1PXEY7ofFX
- Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models:
  https://arxiv.org/html/2403.01329v1
- GITS / trajectory-regularity timestep scheduling:
  https://proceedings.mlr.press/v235/chen24bm.html
- Adaptive Time-Stepping Schedules for Diffusion Models:
  https://proceedings.mlr.press/v244/chen24c.html
- A-FloPS adaptive flow-path sampling:
  https://arxiv.org/abs/2509.00036

Before writing the paper, re-read these papers carefully and add a table with:

- whether the method changes training;
- whether it learns/optimizes a solver;
- whether it needs a validation objective/reference solver;
- whether it is diffusion-specific or native FM;
- whether it uses per-sample or global schedules;
- whether it preserves GPU batching.

## Novelty Bar

SCTW is plausibly novel enough only if we develop at least one of:

1. **Self-tuned tempering:** choose the warp exponent from the measured profile.
2. **Equidistribution theory:** formalize the mesh as minimizing a first-order Euler error proxy.
3. **Predictive diagnostics:** show profile concentration predicts realized low-NFE gain.
4. **Robust cross-domain evidence:** show it is competitive with tuned hand schedules on multiple
   geometries, not just a lucky shape match.

Without this added depth, the method is likely incremental.
