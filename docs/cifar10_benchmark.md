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
  `model.pt`, optional `model_ema.pt`, `checkpoint_latest.pt`, and
  `samples/*.png`.

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

## Strong Standard Baseline

The first compact conditional UNet baseline was useful for pipeline bring-up, but
it was not strong enough for final method comparison. A larger standard CFM was
trained on the `gpu-large` VPS with:

- Config: `configs/cifar10_standard_large.json`
- Model size: `101.4M` parameters, about `2.07x` the compact `48.9M` UNet.
- Training: `100000` optimizer steps, batch size `64`, AMP, gradient clipping
  `0.5`, learning rate `1e-4`.
- EMA: enabled for the continuation from the step-20000 checkpoint with
  `ema_decay=0.999`.
- Artifact bundle: `artifacts/cifar10_benchmark/standard_large_100k/`

The full optimizer checkpoint is intentionally not DVC-tracked because it is
about 1.9 GB and mostly needed for resuming, not for evaluation. The tracked
bundle keeps the final raw and EMA model weights, run metadata, history, logs,
metric JSON/CSV files, and sample grids.

Final 5000-sample evaluation against CIFAR-10 test references:

| Weights | NFE | FID | KID | Class accuracy |
| --- | ---: | ---: | ---: | ---: |
| Raw | 5 | 40.69 | 0.02569 | 71.70% |
| Raw | 10 | 28.10 | 0.01568 | 82.86% |
| Raw | 20 | 24.88 | 0.01298 | 84.26% |
| Raw | 50 | 23.68 | 0.01067 | 84.18% |
| EMA | 5 | 35.56 | 0.02241 | 77.96% |
| EMA | 10 | 24.75 | 0.01406 | 86.30% |
| EMA | 20 | 21.01 | 0.01105 | 87.84% |
| EMA | 50 | 18.85 | 0.00816 | 87.32% |

Takeaway: capacity alone was not enough at 19000 steps, but longer training plus
EMA substantially improved the baseline. Future CIFAR method comparisons should
use EMA evaluation, otherwise we risk comparing against an artificially weak
image-modeling baseline.

## Strong-Backbone Coupling Comparison

The first method comparison after the strong baseline should use one seed for
each coupling variant and the same large UNet/100k/EMA training recipe:

1. `cifar10_standard_large.json`: independent pairing baseline, already run.
2. `cifar10_minibatch_ot_large_8x8.json`: exact minibatch OT with 8x8 RGB cost.
3. `cifar10_minibatch_ot_large_16x16.json`: exact minibatch OT with 16x16 RGB
   cost.
4. `cifar10_pressure_aware_ot_large_8x8.json`: pressure-aware minibatch OT with
   8x8 RGB cost.
5. `cifar10_pressure_aware_ot_large_16x16.json`: pressure-aware minibatch OT
   with 16x16 RGB cost.

Evaluate the EMA weights for all runs with the fixed 5000-sample protocol.
Primary comparisons are FID/KID and class accuracy at NFE 5, 10, 20, and 50.
NFE 5 and 10 are the most important for the low-NFE claim.

## Deferred Avenues

Keep these as explicit follow-up branches while the first strong-backbone
comparison focuses on independent pairing, minibatch OT, and pressure-aware OT.

1. **Sinkhorn OT instead of exact Hungarian OT.** Exact Hungarian assignment is
   simple and deterministic at batch size 64, but the current implementation
   computes a dense cost matrix and solves the assignment on CPU. A GPU Sinkhorn
   coupling would be approximate, but it could make richer image costs practical,
   including full 32x32 RGB pixel costs or larger batches. This is especially
   relevant if 8x8 or 16x16 downsampled costs look too crude.

2. **Unconditional CIFAR-10 CFM.** The current benchmark is class-conditional,
   so labels already remove much of the multimodal ambiguity that OT might
   otherwise help resolve. An unconditional CIFAR run would test whether OT or
   pressure-aware coupling matters more when class labels no longer detangle the
   target distribution. Primary metrics would be FID/KID; classifier class
   histogram can be diagnostic but not a conditional-accuracy metric.

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
