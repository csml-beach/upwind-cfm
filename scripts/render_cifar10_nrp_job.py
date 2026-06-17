#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path


def current_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def tree_is_dirty():
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def slugify(text):
    text = re.sub(r"[^a-z0-9-]+", "-", text.lower())
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:55].strip("-")


def render(args):
    commit = args.commit or current_commit()
    if args.commit is None and tree_is_dirty():
        print("warning: rendering current HEAD while the worktree is dirty; commit first for remote runs", file=sys.stderr)
    config_path = args.config
    variant = Path(config_path).stem.replace("cifar10_", "")
    job_name = args.job_name or slugify(f"cifar10-{variant}-s{args.seed}")
    command = f"""set -eux
apt-get update
apt-get install -y git
python -m pip install --upgrade pip
git clone https://github.com/csml-beach/upwind-cfm.git /tmp/upwind-cfm
cd /tmp/upwind-cfm
git checkout {commit}
pip install -r requirements.txt
python scripts/run_cifar10_job.py \\
  --config {config_path} \\
  --seed {args.seed} \\
  --run-group {args.run_group} \\
  --device cuda \\
  --out-root /mnt/data/upwind-cfm/cifar10 \\
  --data-root /mnt/data/upwind-cfm/datasets
"""
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {args.namespace}
spec:
  ttlSecondsAfterFinished: {args.ttl_seconds}
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: cifar10-train
        image: {args.image}
        securityContext:
          runAsUser: 0
        command: ["sh", "-c"]
        args:
          - |
{chr(10).join("            " + line for line in command.rstrip().splitlines())}
        volumeMounts:
        - name: csml-beach-storage
          mountPath: /mnt/data
        resources:
          limits:
            cpu: "{args.cpu}"
            memory: "{args.memory}"
            nvidia.com/gpu: "{args.gpus}"
          requests:
            cpu: "{args.cpu}"
            memory: "{args.memory}"
            nvidia.com/gpu: "{args.gpus}"
      volumes:
      - name: csml-beach-storage
        persistentVolumeClaim:
          claimName: {args.pvc}
  backoffLimit: 1
"""


def main():
    parser = argparse.ArgumentParser(description="Render a single-GPU NRP CIFAR-10 training Job.")
    parser.add_argument("--config", required=True, help="Config path inside the upwind-cfm repo.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--commit", help="Explicit upwind-cfm commit SHA. Defaults to current HEAD.")
    parser.add_argument("--output", help="Write YAML to this path instead of stdout.")
    parser.add_argument("--job-name")
    parser.add_argument("--namespace", default="csml-beach")
    parser.add_argument("--pvc", default="csml-beach-pvc")
    parser.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    parser.add_argument("--cpu", default="4")
    parser.add_argument("--memory", default="24Gi")
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--ttl-seconds", type=int, default=300)
    args = parser.parse_args()

    yaml = render(args)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml)
        print(path)
    else:
        print(yaml)


if __name__ == "__main__":
    main()
