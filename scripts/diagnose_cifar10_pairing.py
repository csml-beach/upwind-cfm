#!/usr/bin/env python3
"""Diagnose how pressure-aware OT changes CIFAR-10 minibatch pairings."""
import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_cache_dir = Path(tempfile.gettempdir()) / "lcfm_cifar10_pairing_diag_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir / "xdg"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment

from lcfm import datasets  # noqa: F401
from lcfm.pairing import _pairing_cost, _positive_median, pairing_features, pressure_aware_cost
from lcfm.registry import DATASETS, get
from lcfm.utils import set_seed, write_json


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assignment(cost):
    _, col = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(col, device=cost.device, dtype=torch.long)


def assigned_values(cost, order):
    row = torch.arange(cost.shape[0], device=cost.device)
    return cost[row, order]


def entropy_from_counts(counts):
    probs = counts.float() / counts.sum().clamp_min(1)
    nonzero = probs > 0
    return float(-(probs[nonzero] * probs[nonzero].log()).sum().item())


def kl_to_uniform_from_counts(counts):
    probs = counts.float() / counts.sum().clamp_min(1)
    uniform = torch.full_like(probs, 1.0 / probs.numel())
    nonzero = probs > 0
    return float((probs[nonzero] * (probs[nonzero] / uniform[nonzero]).log()).sum().item())


def sample_train_with_labels(problem, batch_size, device):
    idx = torch.randint(problem.train.shape[0], (batch_size,))
    x1 = problem.train[idx].to(device)
    labels = problem.train_labels[idx].to(device)
    x0 = torch.randn(batch_size, problem.dim, device=device)
    return x0, x1, labels


def summarize(rows):
    keys = [
        "disagreement_rate",
        "base_delta_mean",
        "pressure_delta_mean",
        "total_delta_mean",
        "changed_base_delta_mean",
        "changed_pressure_delta_mean",
    ]
    summary = {}
    for key in keys:
        values = [row[key] for row in rows]
        tensor = torch.tensor(values, dtype=torch.float64)
        summary[key] = {
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
        }
    return summary


def plot_histograms(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), dpi=220)
    fig.patch.set_facecolor("white")

    plots = [
        ("disagreement_rate", "assignment disagreement", "fraction of rows"),
        ("base_delta_mean", "pixel-cost change", "PA - OT"),
        ("pressure_delta_mean", "pressure-cost change", "PA - OT"),
    ]
    for ax, (key, title, xlabel) in zip(axes, plots):
        values = [row[key] for row in rows]
        ax.hist(values, bins=24, color="#2563eb", alpha=0.82, edgecolor="white", linewidth=0.5)
        ax.axvline(sum(values) / len(values), color="#dc2626", linewidth=1.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("batches", fontsize=8)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7, colors="#374151")

    fig.suptitle("CIFAR-10 16x16 coupling diagnostic", fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def collect(args):
    device = torch.device(args.device)
    set_seed(args.seed)
    dataset_cls = get(DATASETS, "cifar10")
    problem = dataset_cls(
        {
            "data_root": args.data_root,
            "download": args.download,
            "class_conditional": False,
        }
    )
    config = {
        "pairing_kwargs": {
            "pressure_beta": args.pressure_beta,
            "pressure_t": args.pressure_t,
            "reference_pairing": "minibatch_ot",
            "cost_feature": "downsampled_pixels",
            "image_shape": [3, 32, 32],
            "downsample_size": args.downsample_size,
        }
    }
    beta = float(args.pressure_beta)
    rows = []
    examples = []
    for batch_idx in range(args.n_batches):
        x0, x1, labels = sample_train_with_labels(problem, args.batch_size, device)
        x0_feat = pairing_features(x0, config)
        x1_feat = pairing_features(x1, config)
        base_cost = _pairing_cost(x0_feat, x1_feat)
        total_cost = pressure_aware_cost(x0, x1, config)
        base_scale = _positive_median(base_cost, 1e-8)
        pressure_cost = (total_cost - base_cost) / (beta * base_scale) if beta > 0 else torch.zeros_like(base_cost)

        ot_order = assignment(base_cost)
        pa_order = assignment(total_cost)
        changed = ot_order != pa_order

        base_ot = assigned_values(base_cost, ot_order)
        base_pa = assigned_values(base_cost, pa_order)
        pressure_ot = assigned_values(pressure_cost, ot_order)
        pressure_pa = assigned_values(pressure_cost, pa_order)
        total_ot = assigned_values(total_cost, ot_order)
        total_pa = assigned_values(total_cost, pa_order)

        ot_labels = labels[ot_order]
        pa_labels = labels[pa_order]
        row = {
            "batch": batch_idx,
            "disagreement_rate": float(changed.float().mean().item()),
            "base_ot_mean": float(base_ot.mean().item()),
            "base_pa_mean": float(base_pa.mean().item()),
            "base_delta_mean": float((base_pa - base_ot).mean().item()),
            "pressure_ot_mean": float(pressure_ot.mean().item()),
            "pressure_pa_mean": float(pressure_pa.mean().item()),
            "pressure_delta_mean": float((pressure_pa - pressure_ot).mean().item()),
            "total_ot_mean": float(total_ot.mean().item()),
            "total_pa_mean": float(total_pa.mean().item()),
            "total_delta_mean": float((total_pa - total_ot).mean().item()),
            "ot_label_entropy": entropy_from_counts(torch.bincount(ot_labels.detach().cpu(), minlength=10)),
            "pa_label_entropy": entropy_from_counts(torch.bincount(pa_labels.detach().cpu(), minlength=10)),
            "ot_label_kl": kl_to_uniform_from_counts(torch.bincount(ot_labels.detach().cpu(), minlength=10)),
            "pa_label_kl": kl_to_uniform_from_counts(torch.bincount(pa_labels.detach().cpu(), minlength=10)),
            "changed_count": int(changed.sum().item()),
        }
        if changed.any():
            row.update(
                {
                    "changed_base_delta_mean": float((base_pa[changed] - base_ot[changed]).mean().item()),
                    "changed_pressure_delta_mean": float((pressure_pa[changed] - pressure_ot[changed]).mean().item()),
                    "changed_total_delta_mean": float((total_pa[changed] - total_ot[changed]).mean().item()),
                }
            )
        else:
            row.update(
                {
                    "changed_base_delta_mean": 0.0,
                    "changed_pressure_delta_mean": 0.0,
                    "changed_total_delta_mean": 0.0,
                }
            )
        rows.append(row)

        if len(examples) < args.n_examples and changed.any():
            changed_idx = torch.nonzero(changed, as_tuple=False).flatten()[: args.n_examples - len(examples)]
            for source_idx in changed_idx.detach().cpu().tolist():
                examples.append(
                    {
                        "batch": batch_idx,
                        "source_row": source_idx,
                        "ot_target_col": int(ot_order[source_idx].item()),
                        "pa_target_col": int(pa_order[source_idx].item()),
                        "ot_target_label": int(labels[ot_order[source_idx]].item()),
                        "pa_target_label": int(labels[pa_order[source_idx]].item()),
                        "ot_base_cost": float(base_cost[source_idx, ot_order[source_idx]].item()),
                        "pa_base_cost": float(base_cost[source_idx, pa_order[source_idx]].item()),
                        "ot_pressure_cost": float(pressure_cost[source_idx, ot_order[source_idx]].item()),
                        "pa_pressure_cost": float(pressure_cost[source_idx, pa_order[source_idx]].item()),
                    }
                )
    return rows, examples, summarize(rows)


def main():
    parser = argparse.ArgumentParser(description="Diagnose CIFAR-10 pressure-aware pairing mechanics.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out-dir", default="results/cifar10_pairing_diagnostic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-batches", type=int, default=128)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--pressure-beta", type=float, default=0.2)
    parser.add_argument("--pressure-t", default="random")
    parser.add_argument("--n-examples", type=int, default=32)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, examples, summary = collect(args)
    write_csv(out_dir / "pairing_batches.csv", rows)
    if examples:
        write_csv(out_dir / "changed_examples.csv", examples)
    write_json(out_dir / "summary.json", {"args": vars(args), "summary": summary})
    plot_histograms(rows, out_dir / "pairing_histograms.png")
    print(out_dir / "summary.json")
    print(out_dir / "pairing_batches.csv")
    print(out_dir / "pairing_histograms.png")


if __name__ == "__main__":
    main()
