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
