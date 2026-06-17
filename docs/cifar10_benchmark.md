# CIFAR-10 Low-NFE Benchmark

This benchmark is the first image-scale test for pressure-aware coupling. The
scientific question is whether coupling design improves low-NFE image quality
against standard CFM, minibatch OT, and a light Iso-FD cooling baseline.

## Scope

- Dataset: CIFAR-10, flattened RGB images in `[-1, 1]`, shape `(3, 32, 32)`.
- Model: repo-native `unet2d` velocity field.
- Sampling: fixed Gaussian prior seeds, Euler NFEs `{5, 10, 20, 50}`.
- Metrics: sample grids always; FID/KID against CIFAR-10 train images when
  `eval.compute_fid_kid=true`.
- Outputs: `config.json`, `environment.json`, `history.json`, `metrics.json`,
  `model.pt`, `checkpoint_latest.pt`, and `samples/*.png`.

## First Suite

The intended first GPU benchmark compares:

- `configs/cifar10_standard.json`
- `configs/cifar10_minibatch_ot.json`
- `configs/cifar10_pressure_aware_ot.json`
- `configs/cifar10_iso_fd_w01.json`

Use batch size 64 for all methods. This is deliberately small enough that exact
Hungarian assignment can run every step for the OT variants.

## Pairing Caveat

Image-space OT is not the same as semantic OT. For CIFAR, minibatch OT and
pressure-aware OT compute assignment costs in fixed downsampled 8x8 RGB feature
space, then apply the chosen permutation to the full 32x32 RGB image vectors.
This keeps the coupling deterministic and cheap, but we should not overclaim it
as a perceptual or class-aware transport geometry.

Pressure-aware OT uses:

- `pressure_beta=0.2`
- `pressure_t="random"`
- `reference_pairing="minibatch_ot"`
- `cost_feature="downsampled_pixels"`

## Smoke Protocol

Local smoke tests use fake CIFAR-like data and tiny UNets:

```bash
python scripts/check_cifar10_benchmark.py
python scripts/run_experiment.py configs/smoke_cifar10_standard.json
python scripts/run_experiment.py configs/smoke_cifar10_pressure_aware_ot.json
```

Smoke success means both runs write a checkpoint, metrics JSON, environment JSON,
history JSON, model weights, and sample grids under `runs/`.

## GPU Handoff

Render one Kubernetes Job per method and seed:

```bash
python scripts/render_cifar10_nrp_job.py \
  --config configs/cifar10_standard.json \
  --seed 0 \
  --run-group first_cifar10_low_nfe \
  --output /tmp/cifar10-standard-seed0.yaml
```

The rendered job clones `https://github.com/csml-beach/upwind-cfm.git`, checks
out an explicit commit SHA, installs `requirements.txt`, and writes results to:

```text
/mnt/data/upwind-cfm/cifar10/{run_group}/runs/{run_name}/
```

After runs finish, sync the result group locally and DVC-track only the selected
paper-facing outputs: final checkpoints, configs, histories, metrics, aggregate
CSVs, and sample grids.
