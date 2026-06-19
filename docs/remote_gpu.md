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
