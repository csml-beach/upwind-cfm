# CIFAR-10 Low-NFE Benchmark

This benchmark is the image-scale test bed for low-NFE flow-matching sampling.
The current active question is whether **Self-Curvature Time Warping (SCTW)**
improves low-NFE sampling from a trained CFM checkpoint against uniform Euler
and strong hand schedules.

Earlier CIFAR experiments also tested pressure-aware coupling. Those results are
kept below because they are informative, but they are no longer the main paper
claim.

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

## Historical First Suite

The first GPU benchmark compared:

- `configs/cifar10_standard.json`
- `configs/cifar10_minibatch_ot.json`
- `configs/cifar10_pressure_aware_ot.json`
- `configs/cifar10_iso_fd_w01.json`

Use batch size 64 for all methods. This was deliberately small enough that exact
Hungarian assignment could run every step for the OT variants.

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

Result location:

- `results/cifar10_coupling_large_100k/`
- `results/cifar10_coupling_large_100k/ema_5000_eval_summary.csv`

All four coupling runs completed on the `gpu-large` VPS and were evaluated from
EMA weights with 5000 generated samples, balanced class labels, CIFAR-10 test
references, and NFEs `{5, 10, 20, 50}`.

| Method | Cost geometry | NFE 5 FID | NFE 10 FID | NFE 20 FID | NFE 50 FID | NFE 50 acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| minibatch OT | 16x16 RGB | **33.85** | **26.15** | **22.72** | **20.61** | **86.32%** |
| pressure-aware OT | 16x16 RGB | 34.66 | 26.49 | 22.99 | 20.77 | 85.72% |
| pressure-aware OT | 8x8 RGB | 35.77 | 26.95 | 23.11 | 20.75 | 84.62% |
| minibatch OT | 8x8 RGB | 36.10 | 27.30 | 23.56 | 21.27 | 85.54% |

Interpretation:

- The proper image metrics do **not** confirm the earlier trainer-side pixel-MSE
  hint that pressure-aware 8x8 was best.
- In this one-seed comparison, exact minibatch OT with 16x16 RGB costs is the
  strongest method across FID, KID, and classifier accuracy.
- Pressure-aware OT is competitive but does not beat the strongest OT baseline.
  This is a useful negative/tempering result: the pressure term should not be
  claimed as an image-scale improvement unless future seeds or variants change
  the picture.
- The image pipeline itself is now usable for method comparison: EMA checkpoints,
  fixed-label sample generation, FID/KID, and conditional classifier accuracy all
  work end to end.

## Unconditional Coupling Probe

The class-conditional CIFAR benchmark may make the coupling problem too easy:
labels already tell the velocity field which semantic mode to target. To make
the test closer in spirit to the staged/multimodal toy problems, we also trained
the same large UNet without class labels as model input.

Result location:

- `results/cifar10_uncond_coupling_large_100k/`
- `results/cifar10_uncond_coupling_large_100k/ema_5000_eval_allseeds_aggregate.csv`
- `results/cifar10_uncond_coupling_large_100k/ema_5000_eval_allseeds_raw.csv`
- `results/cifar10_uncond_coupling_large_100k/figures/metrics_fid_kl.png`
- `results/cifar10_uncond_coupling_large_100k/figures/samples_nfe10_seed0.png`
- `results/cifar10_uncond_coupling_large_100k/pairing_diagnostic_16x16_beta_summary.csv`
- `results/cifar10_uncond_coupling_large_100k/figures/pairing_beta_sensitivity.png`

This probe used the same 100k-step large-UNet/EMA recipe as the conditional
runs, but with `class_conditional=false` and no UNet label embedding.
Evaluation used 5000 generated samples, CIFAR-10 test references, and
classifier-predicted class histogram diagnostics instead of conditional
accuracy.

Three-seed EMA evaluation:

| Method | Cost geometry | NFE 5 FID | NFE 10 FID | NFE 20 FID | NFE 50 FID |
| --- | --- | ---: | ---: | ---: | ---: |
| pressure-aware OT | 16x16 RGB | **41.25 ± 0.25** | **28.48 ± 0.19** | **23.66 ± 0.24** | **20.95 ± 0.21** |
| minibatch OT | 16x16 RGB | 41.34 ± 0.23 | 28.89 ± 0.16 | 23.95 ± 0.21 | 21.07 ± 0.23 |
| independent | none | 52.78 ± 0.20 | 31.31 ± 0.05 | 24.82 ± 0.29 | 21.65 ± 0.27 |

Class-histogram KL-to-uniform:

| Method | NFE 5 KL | NFE 10 KL | NFE 20 KL | NFE 50 KL |
| --- | ---: | ---: | ---: | ---: |
| pressure-aware OT | 0.1225 | 0.0246 | **0.0058** | 0.0022 |
| minibatch OT | 0.1250 | 0.0259 | 0.0064 | **0.0018** |
| independent | 0.2192 | 0.0494 | 0.0094 | 0.0028 |

Interpretation:

- Removing class conditioning makes the low-NFE problem more discriminating:
  independent pairing is much worse at NFE 5/10, while OT-style couplings help.
- Pressure-aware OT modestly beats minibatch OT at every NFE in the three-seed
  aggregate. The gap is small, but it replicated without retuning.
- The main image-scale effect is still OT-style coupling versus independent
  pairing. The pressure-aware term is a smaller improvement on top of that.
- The classifier histogram diagnostics do not suggest class collapse. Entropy is
  near `log(10)` by NFE 20/50 and KL-to-uniform is small.
- This is currently the strongest image-scale positive signal for pressure-aware
  coupling. It should be framed as a modest but consistent low-NFE improvement
  in the harder, label-free CIFAR setting.
- The sample-panel differences are subtle by eye; the paper-facing claim should
  lean on FID/KID and class-balance diagnostics rather than visual inspection
  alone.

Coupling-mechanics diagnostic:

After the three-seed result, we checked whether pressure-aware OT actually
changes minibatch assignments relative to ordinary minibatch OT. For held-out
CIFAR training batches, we recomputed 16x16 RGB minibatch OT and pressure-aware
OT assignments with the same pressure cost used in training.

| pressure beta | assignment rows changed |
| ---: | ---: |
| 0.2 | 0.024% |
| 1.0 | 0.537% |
| 5.0 | 2.100% |
| 10.0 | 3.442% |

This is a serious caveat. At the actual trained setting, `pressure_beta=0.2`,
pressure-aware OT is nearly identical to minibatch OT as an assignment rule. The
small FID gain may therefore be caused by very rare assignment changes,
stochastic training differences, or another indirect effect, not by a strong
pressure-driven recoupling of CIFAR minibatches. Before making a mechanism claim,
we need a stricter control, such as retraining minibatch OT with matched random
number consumption or training a higher-beta pressure-aware variant that actually
changes assignments.

## SCTW Sampler Evaluation

The current sampler-side evaluation uses the unconditional Sinkhorn OT large
UNet/EMA checkpoint and changes only the inference time grid. Training is
unchanged.

Result locations:

- `results/cifar10_uncond_coupling_large_100k/runs/cifar10_uncond_sinkhorn_ot_large_16x16_seed0/eval_e1_warped_5000/`
- `results/cifar10_uncond_coupling_large_100k/runs/cifar10_uncond_sinkhorn_ot_large_16x16_seed0/eval_e1_warped_p025_5000/`
- `results/cifar10_uncond_coupling_large_100k/runs/cifar10_uncond_sinkhorn_ot_large_16x16_seed0/eval_power_baselines_5000/`
- `results/cifar10_uncond_coupling_large_100k/runs/cifar10_uncond_sinkhorn_ot_large_16x16_seed0/time_mesh_sweep_512/`

FID lower is better:

| NFE | Uniform | SCTW p=0.25 | SCTW p=0.5 | Best hand power |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 41.4147 | 41.7388 | 44.9989 | 47.8957 |
| 10 | 28.9758 | 28.3254 | 28.7508 | 30.0170 |
| 20 | 23.8106 | 22.9072 | 22.7874 | 23.3659 |
| 50 | 20.9632 | 20.3763 | 20.2047 | 20.3376 |

Interpretation:

- SCTW improves over uniform at NFE 10/20/50.
- The tempered `p=0.25` setting avoids the severe NFE-5 degradation seen with
  `p=0.5`, but it does not beat uniform at NFE 5.
- The tested hand power schedules are weaker than SCTW on CIFAR, which supports
  the need for model-adaptive scheduling.
- This result should be paired with staged-shapes, where a hand early schedule
  is stronger, to make the honest claim: SCTW is adaptive and interpretable, not
  universally dominant.

## Deferred Avenues

Keep these as explicit follow-up branches while the first strong-backbone
comparison focuses on independent pairing, minibatch OT, and pressure-aware OT.

1. **Sinkhorn OT instead of exact Hungarian OT.** Exact Hungarian assignment is
   simple and deterministic at batch size 64, but the current implementation
   computes a dense cost matrix and solves the assignment on CPU. The implemented
   `sinkhorn_ot` and `pressure_aware_sinkhorn_ot` pairings compute a balanced
   entropic OT plan on torch tensors, then project the plan to target indices
   before applying the usual CFM loss. This is still a hard-pair training
   interface, not a full soft-plan CFM objective. The default greedy projection
   preserves one-to-one target use and keeps labels aligned with full-resolution
   target images. It is useful for testing richer costs, including full 32x32
   RGB pixel costs or larger batches, but it should be reported as approximate
   Sinkhorn-projected coupling rather than exact minibatch OT.

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
