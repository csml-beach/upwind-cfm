# Experiment Spine

This repo separates problems, training methods, pairings, solvers, sampler schedules, metrics, and
run configuration so we can compare low-NFE flow-matching ideas without duplicating whole scripts.

## Design Goals

- Keep the code small and readable.
- Register methods, pairings, solvers, and schedules by name.
- Keep training regularization separate from inference-time solvers.
- Write comparable run artifacts for every experiment.
- Use Burgers only as an autoregressive dynamics task, not flattened surface generation.

## Layout

```text
lcfm/
  datasets.py       # spiral, mode mixtures, fan modes, burgers_autoregressive
  models.py         # MLP and 1D U-Net
  losses.py         # standard CFM, LC, Iso-FD, directional regularization
  pairing.py        # optional training-time source-target batch pairing
  solvers.py        # Euler, velocity-smoothed Euler, Heun
  schedules.py      # uniform/nonuniform time grids and self-curvature profiles
  metrics.py        # exact empirical W1/W2 matching, path length, acceleration, RMSE, temporal TV
  experiment.py     # shared train/eval loop
scripts/
  run_experiment.py # config-driven runner
configs/
  *.json            # reproducible experiment settings
```

## Registered Datasets

- `spiral`: 2D noise-to-data CFM benchmark.
- `five_modes`: centered ring of five Gaussian target modes, with configurable source scale.
- `fan_modes`: asymmetric source-to-multimodal-target benchmark.
- `staged_modes`: 2D Gaussian mixture with modes at different distances and angles from a
  clumped source, intended to create staged commitment times.
- `gaussian_mixture_nd`: oracle-compatible high-dimensional Gaussian mixture benchmark; the
  default config uses eight simplex-like modes in 16 dimensions.
- `cifar10`: flattened 32x32 RGB image benchmark with optional fake-data smoke mode.
- `staged_shapes_easy`: flattened 32x32 RGB procedural image benchmark with five
  controlled shape modes at unequal positions/scales, intended as an image-like
  analogue of `staged_modes`.
- `burgers_autoregressive`: learns frame-to-next-frame flow and rolls out autoregressively.

The older flattened Burgers surface-generation task should not be used for new comparisons.

## Registered Methods

- `standard_cfm`: baseline CFM regression loss.
- `lc_finite_difference`: backward-characteristic finite-difference Lagrangian consistency.
- `lc_jvp_material_derivative`: analytic material-derivative penalty using a Jacobian-vector product.
- `iso_fm_finite_difference`: forward Iso-FM-style Jacobian-free material-residual baseline.
- `directional_regularization_cfm`: Iso-FD-style residual weighted by a local directional solver-risk proxy. The first implemented variant uses finite differences for both residual and directional weight.

## Pairing Modes

- `independent`: default independent source-target pairing.
- `minibatch_ot`: Hungarian assignment on squared Euclidean source-target minibatch costs before applying the method loss.
- `sinkhorn_ot`: balanced entropic OT plan on squared Euclidean source-target
  costs, projected back to target indices before applying the method loss. The
  default projection is greedy one-to-one matching from Sinkhorn plan mass.
- `pressure_aware_minibatch_ot`: Hungarian assignment with an added scalar local
  conditional-velocity-variance cost. This is a pressure-aware coupling design:
  it changes the minibatch pairing while leaving the CFM velocity target unchanged.
- `pressure_aware_sinkhorn_ot`: the same pressure-aware cost solved with
  Sinkhorn plus projection instead of exact Hungarian assignment.

Pairing is a training-time batch transform. Methods still receive only `(x0, x1)` and do not need method-specific OT logic.

## Registered Solvers

- `euler`: standard explicit Euler.
- `velocity_smoothed_euler`: the previous projection-based velocity smoothing solver, renamed to avoid calling it upwind.
- `heun`: second-order predictor-corrector baseline.

## Registered / Tested Sampler Schedules

Schedules are inference-time choices. They should not be confused with training methods or
pairing rules.

- `uniform`: equally spaced time grid.
- `e1_warped`: Self-Curvature Time Warping. It probes the trained model on a fine Euler rollout,
  estimates the model-side material-derivative/self-curvature profile, and places Euler knots at
  equal mass of a tempered profile.
- `power`: hand-designed nonuniform grids such as early, late, or symmetric power schedules.

The active paper direction lives here, not in `losses.py`: the current method candidate changes
the sampling mesh after training.

## Image Benchmarks

The CIFAR-10 low-NFE benchmark uses `unet2d`, image sample grids, optional FID/KID, step-based
training, checkpoints, resume, and fixed-seed NFE comparisons. See `docs/cifar10_benchmark.md`.

`staged_shapes_easy` uses the same image training/evaluation path but avoids natural-image
ambiguity: a central source blob transports to five simple rendered target modes. Its current
purpose is to test whether sampler schedules behave differently on a deliberately staged,
early-layer image-shaped transport problem.

## Run Artifacts

Each run writes:

```text
runs/{run_name}/
  config.json
  environment.json
  history.json
  metrics.json
  model.pt
```

This is the minimum needed for comparable experiments: config, environment, training trace, metrics, and checkpoint.

## Example

```bash
python3 scripts/run_experiment.py configs/spiral_standard.json
python3 scripts/run_experiment.py configs/spiral_lc_fd.json
python3 scripts/run_experiment.py configs/spiral_lc_jvp.json
```

Install dependencies first:

```bash
python3 -m pip install -r requirements.txt
```
