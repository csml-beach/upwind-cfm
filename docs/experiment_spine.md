# Experiment Spine

This repo now separates problems, methods, solvers, metrics, and run configuration so we can compare Lagrangian Consistency variants against baselines without duplicating whole scripts.

## Design Goals

- Keep the code small and readable.
- Register methods and solvers by name.
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
  metrics.py        # W2, path length, acceleration, RMSE, temporal TV
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

Pairing is a training-time batch transform. Methods still receive only `(x0, x1)` and do not need method-specific OT logic.

## Registered Solvers

- `euler`: standard explicit Euler.
- `velocity_smoothed_euler`: the previous projection-based velocity smoothing solver, renamed to avoid calling it upwind.
- `heun`: second-order predictor-corrector baseline.

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
