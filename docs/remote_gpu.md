# Remote GPU Notes

This note records non-secret operational details for remote GPU experiments.
Do not store private keys, tokens, or passwords here.

## GPU VPS

- Name: `gpu-large`
- SSH target: `exouser@149.165.154.54`
- Expected use: first CIFAR-10 bring-up and medium/long single-GPU runs before
  moving stable sweeps to Kubernetes pods.

Connect with:

```bash
ssh exouser@149.165.154.54
```

Recommended first check after login:

```bash
nvidia-smi
python3 --version
```

Recommended working layout:

```text
~/upwind-cfm/                 # cloned repo
~/upwind-cfm-data/            # CIFAR-10 and metric caches
~/upwind-cfm-results/         # training outputs for rsync/DVC handoff
```

For CIFAR jobs, keep `dataset_kwargs.data_root` on the VPS-local disk so CIFAR-10
and FID/KID caches are downloaded once and reused across runs.

## Current Setup

As of the first CIFAR setup pass:

- Repo: `~/upwind-cfm`
- Virtualenv: `~/upwind-cfm-venv`
- Data root: `~/upwind-cfm-data`
- Result root: `~/upwind-cfm-results`
- GPU check: `GRID A100X-20C`
- Working PyTorch build: `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`

The VPS driver is too old for the newest CUDA 13 PyTorch wheels. Do not install
plain latest `torch` from PyPI on this machine. Use:

```bash
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1
pip install --no-cache-dir torchmetrics torch-fidelity numpy scipy matplotlib tqdm pandas
```

Old `~/runs` was cleared during setup to free disk. Root disk had about 42 GB
free after cleanup.

## CIFAR Checkpoint Hygiene

For large CIFAR UNet runs, avoid keeping every optimizer checkpoint on the VPS.
The 101M-parameter model's resumable `checkpoint_latest.pt` is about 1.9 GB
because it includes raw weights, optimizer state, scaler state, and EMA weights.
Use sparse checkpointing for long runs:

```json
"checkpoint_every": 10000,
"keep_checkpoints": false,
"stop_on_nonfinite": true,
"save_failed_checkpoint": false
```

For DVC/paper artifacts, prefer standalone evaluable weights:

```text
model.pt
model_ema.pt
config.json
history.json
environment.json
eval_*/*.csv
eval_*/*.json
eval_*/samples/*.png
```

Only sync and DVC-track `checkpoint_latest.pt` when we specifically need to
resume the exact optimizer state later.

## SCTW / Adjoint-SCTW Evaluation Handoff

The remaining CIFAR estimator-equivalence jobs can be run sequentially with
`scripts/run_cifar_equivalence_queue.sh`. Set `WAIT_PID` when an existing GPU evaluation must
finish first. The runner uses the declared pilot/evaluation seeds, writes one probe seed per
output directory, and creates `equivalence_image_queue.DONE` or
`equivalence_image_queue.FAILED` under the suite result root.

The frozen-checkpoint image suite writes to `~/upwind-cfm-results/sctw_adjoint_suite/` and uses
profile caches inside each output directory. Run from `~/upwind-cfm` with the virtualenv active.
No training or optimizer state is involved.

Staged-shapes:

```bash
python scripts/evaluate_staged_shapes_checkpoint.py \
  --run-dir ~/upwind-cfm-results/staged_shapes_easy/runs/staged_shapes_easy_sinkhorn_ot_seed0 \
  --checkpoint model_ema.pt --device cuda --n-samples 1000 --batch-size 64 \
  --profile-samples 100 --profile-batch-size 8 --profile-fine-steps 50 \
  --profile-anchor-count 16 --pilot-seed 1311 --adjoint-probes 4 --adjoint-seed 2718 \
  --eval-seed 1234 --nfe-values 5,10,20,50 \
  --power-rhos 2,3 --reference-intervals 1000 --reference-check-intervals 2000 \
  --methods uniform,sctw,adjoint_sctw,power,vsfm,heun \
  --out-dir ~/upwind-cfm-results/sctw_adjoint_suite/staged_shapes
```

Conditional CIFAR-10:

```bash
python scripts/evaluate_cifar10_checkpoint.py \
  --run-dir ~/upwind-cfm-results/cifar10_coupling_large_100k/runs/cifar10_minibatch_ot_large_16x16_seed0 \
  --checkpoint model_ema.pt --device cuda --metric-device cuda \
  --data-root ~/upwind-cfm-data --n-samples 5000 --batch-size 128 --metric-batch-size 128 \
  --profile-samples 100 --profile-batch-size 4 --profile-fine-steps 50 \
  --profile-anchor-count 16 --pilot-seed 1311 --adjoint-probes 4 --adjoint-seed 2718 \
  --eval-seed 1234 --nfe-values 5,10,20,50 \
  --power-rhos 2,3 --reference-split train \
  --reference-intervals 1000 --reference-check-intervals 2000 \
  --methods uniform,sctw,adjoint_sctw,power,vsfm,heun \
  --classifier-checkpoint artifacts/cifar10_metrics/classifier/cifar10_resnet18.pt \
  --inception-weights artifacts/cifar10_metrics/inception/weights-inception-2015-12-05-6726825d.pth \
  --out-dir ~/upwind-cfm-results/sctw_adjoint_suite/cifar10_conditional
```

Unconditional CIFAR-10:

```bash
python scripts/evaluate_cifar10_checkpoint.py \
  --run-dir ~/upwind-cfm-results/cifar10_uncond_coupling_large_100k/runs/cifar10_uncond_sinkhorn_ot_large_16x16_seed0 \
  --checkpoint model_ema.pt --device cuda --metric-device cuda \
  --data-root ~/upwind-cfm-data --n-samples 5000 --batch-size 128 --metric-batch-size 128 \
  --profile-samples 100 --profile-batch-size 4 --profile-fine-steps 50 \
  --profile-anchor-count 16 --pilot-seed 1311 --adjoint-probes 4 --adjoint-seed 2718 \
  --eval-seed 1234 --nfe-values 5,10,20,50 \
  --power-rhos 2,3 --reference-split train \
  --reference-intervals 1000 --reference-check-intervals 2000 \
  --methods uniform,sctw,adjoint_sctw,power,vsfm,heun \
  --classifier-checkpoint artifacts/cifar10_metrics/classifier/cifar10_resnet18.pt \
  --inception-weights artifacts/cifar10_metrics/inception/weights-inception-2015-12-05-6726825d.pth \
  --out-dir ~/upwind-cfm-results/sctw_adjoint_suite/cifar10_unconditional
```

Before launching, verify `nvidia-smi`, the three checkpoint paths, the classifier artifact, and
the CIFAR cache. The profile-cache JSON records any deterministic chunk fallback from 4 to 2 or 1.
The image training runs retain batch size 64 for coupling comparability. Evaluation uses batch 128
only as a throughput choice; every sampler in a comparison uses the same batching and fixed seed.
