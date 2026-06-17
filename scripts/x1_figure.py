#!/usr/bin/env python3
"""X1 signature figure: predicted step-efficiency gain (kappa) vs realized
integration-error gain of the global warp at matched NFE.

Realized gain = integration_error(uniform Euler k) / integration_error(warped
Euler k), one point per geometry x coupling (mean +/- std over seeds). The
first-order theory predicts gain = kappa (diagonal).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

MARKERS = {"clumped015": "o", "ring": "s", "fan": "^", "spiral": "D"}
COLORS = {"independent": "#2563eb", "minibatch_ot": "#dc2626"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--kappa-key", default="kappa_e1", choices=["kappa_e0", "kappa_e1", "kappa_e2"])
    parser.add_argument("--warp-schedule", default=None, help="Defaults to euler{steps}_warp_e1.")
    args = parser.parse_args()
    warp_schedule = args.warp_schedule or f"euler{args.steps}_warp_e1"

    rows = list(csv.DictReader(open(args.csv_path)))
    by_run = defaultdict(dict)
    for row in rows:
        by_run[(row["group"], row["seed"])][row["schedule"]] = row

    points = defaultdict(lambda: {"kappa": [], "gain": []})
    for (group, _), schedules in by_run.items():
        uni = schedules.get(f"euler{args.steps}_uniform")
        warp = schedules.get(warp_schedule)
        if not uni or not warp or not uni.get(args.kappa_key):
            continue
        points[group]["kappa"].append(float(uni[args.kappa_key]))
        points[group]["gain"].append(float(uni["integration_error"]) / float(warp["integration_error"]))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    lim = 1.0
    print(f"{'group':>24} {'kappa':>14} {'realized gain':>16}")
    for group, data in sorted(points.items()):
        kap = torch.tensor(data["kappa"])
        gain = torch.tensor(data["gain"])
        geometry, coupling = group.rsplit("_", 1) if not group.endswith("minibatch_ot") else (group[: -len("_minibatch_ot")], "minibatch_ot")
        if coupling != "minibatch_ot":
            geometry, coupling = group.rsplit("_independent")[0], "independent"
        ax.errorbar(
            kap.mean(),
            gain.mean(),
            xerr=kap.std(unbiased=False),
            yerr=gain.std(unbiased=False),
            marker=MARKERS.get(geometry, "x"),
            color=COLORS.get(coupling, "#111827"),
            markersize=9,
            capsize=3,
            label=f"{geometry} ({coupling})",
        )
        lim = max(lim, float(kap.max()), float(gain.max()))
        print(f"{group:>24} {kap.mean():7.2f}+/-{kap.std(unbiased=False):5.2f} {gain.mean():9.2f}+/-{gain.std(unbiased=False):5.2f}")
    lim *= 1.15
    ax.plot([1, lim], [1, lim], color="#9ca3af", linestyle=":", label="predicted (gain = kappa)")
    ax.axhline(1.0, color="#9ca3af", linewidth=0.6)
    ax.set_xlabel(f"predicted gain  ({args.kappa_key})")
    ax.set_ylabel(f"realized gain  (int. err uniform / warped, Euler {args.steps})")
    ax.set_title("X1: stiffness concentration predicts warp gain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
