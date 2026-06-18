# CIFAR-10 Low-NFE Benchmark

This benchmark is the first image-scale test for pressure-aware coupling. The
scientific question is whether coupling design improves low-NFE class-conditional
image quality against standard CFM, minibatch OT, and a light Iso-FD cooling
baseline.

## Scope

- Dataset: CIFAR-10, flattened RGB images in `[-1, 1]`, shape `(3, 32, 32)`,
  with integer class labels.
- Model: repo-native class-conditional `unet2d` velocity field. Labels are added
  through the same embedding pathway as time.
- Sampling: fixed Gaussian prior seeds, Euler NFEs `{5, 10, 20, 50}`.
- Training outputs sample grids; paper-facing FID/KID and classifier accuracy are
  computed afterward with `scripts/evaluate_cifar10_checkpoint.py`.
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

The velocity model is class-conditional, but the current OT cost is still purely
downsampled-pixel geometry. Pairing preserves the target label when it permutes
the target image, so the model sees the correct class for each paired endpoint.
If this benchmark becomes paper-facing, a class-restricted OT variant is a
natural ablation.

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

## Metric Evaluation

Run image metrics after training, not inside the trainer:

```bash
python scripts/evaluate_cifar10_checkpoint.py \
  --run-dir results/cifar10_low_nfe/local/runs/cifar10_standard_seed0 \
  --data-root data \
  --classifier-checkpoint data/classifiers/cifar10_resnet18.pt \
  --n-samples 5000 \
  --nfe-values 5,10,20,50
```

The evaluator uses fixed Gaussian initial samples and balanced labels cycling
through CIFAR classes. It reports global FID/KID from pretrained Inception
features and conditional classifier accuracy from a CIFAR-10 classifier.

Metric assets are DVC-tracked under `artifacts/cifar10_metrics/`:

- `classifier/cifar10_resnet18.pt`: CIFAR-10 ResNet-18 classifier, best test
  accuracy `92.62%` at epoch 29.
- `inception/weights-inception-2015-12-05-6726825d.pth`: pretrained Inception
  feature extractor weights used by TorchMetrics/Torch-Fidelity for FID/KID.
- `eval/standard_25k_eval_5000/`: first 5000-sample evaluation of the standard
  conditional UNet checkpoint.

The classifier can be reproduced with:

```bash
python scripts/train_cifar10_classifier.py \
  --data-root data \
  --out data/classifiers/cifar10_resnet18.pt \
  --device cuda \
  --epochs 30 \
  --batch-size 256
```

## GPU Handoff

For first bring-up, prefer the `gpu-large` VPS documented in
`docs/remote_gpu.md`. It avoids repeated dataset/cache setup while the image
benchmark is still changing.

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
