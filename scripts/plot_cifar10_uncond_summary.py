#!/usr/bin/env python3
"""Plot unconditional CIFAR-10 coupling summary figures."""
import argparse
import csv
import os
import tempfile
from pathlib import Path

_cache_dir = Path(tempfile.gettempdir()) / "lcfm_cifar10_plot_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir / "xdg"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


METHODS = [
    ("independent", "Independent", "#4b5563"),
    ("minibatch_ot_16x16", "Minibatch OT", "#2563eb"),
    ("pressure_aware_ot_16x16", "Pressure-aware OT", "#dc2626"),
]

RUN_BASES = {
    "independent": "cifar10_uncond_standard_large",
    "minibatch_ot_16x16": "cifar10_uncond_minibatch_ot_large_16x16",
    "pressure_aware_ot_16x16": "cifar10_uncond_pressure_aware_ot_large_16x16",
}


def read_aggregate(path):
    rows = []
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            parsed["nfe"] = int(row["nfe"])
            for key in (
                "fid_mean",
                "fid_std",
                "kid_mean_mean",
                "kid_mean_std",
                "classifier_prediction_kl_to_uniform_mean",
                "classifier_prediction_kl_to_uniform_std",
            ):
                parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def by_method(rows):
    grouped = {}
    for method, _, _ in METHODS:
        grouped[method] = sorted([row for row in rows if row["method"] == method], key=lambda row: row["nfe"])
    return grouped


def plot_metrics(rows, output):
    grouped = by_method(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), dpi=220)
    fig.patch.set_facecolor("white")

    for method, label, color in METHODS:
        method_rows = grouped[method]
        x = [row["nfe"] for row in method_rows]
        axes[0].errorbar(
            x,
            [row["fid_mean"] for row in method_rows],
            yerr=[row["fid_std"] for row in method_rows],
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            capsize=3,
            label=label,
        )
        axes[1].errorbar(
            x,
            [row["classifier_prediction_kl_to_uniform_mean"] for row in method_rows],
            yerr=[row["classifier_prediction_kl_to_uniform_std"] for row in method_rows],
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            capsize=3,
            label=label,
        )

    axes[0].set_ylabel("FID (lower is better)")
    axes[1].set_ylabel("Class histogram KL to uniform")
    for ax in axes:
        ax.set_xlabel("Euler NFE")
        ax.set_xticks([5, 10, 20, 50])
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8, colors="#374151")
        ax.yaxis.label.set_size(9)
        ax.xaxis.label.set_size(9)

    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_title("Image quality", fontsize=10, pad=8)
    axes[1].set_title("Class balance", fontsize=10, pad=8)
    fig.suptitle("Unconditional CIFAR-10, 3 seeds, EMA evaluation", fontsize=11, y=1.02)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def load_font(size):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def sample_grid_path(root, method, seed, nfe):
    run = f"{RUN_BASES[method]}_seed{seed}"
    return Path(root) / "runs" / run / "eval_ema_5000" / "samples" / f"nfe_{nfe}.png"


def resize_grid(image, target_width=384):
    if image.width >= target_width:
        return image
    scale = target_width / image.width
    target_height = int(round(image.height * scale))
    return image.resize((target_width, target_height), Image.Resampling.BICUBIC)


def plot_sample_panel(root, output, seed=0, nfe=10):
    panels = []
    for method, label, _ in METHODS:
        path = sample_grid_path(root, method, seed, nfe)
        if not path.exists():
            raise FileNotFoundError(f"Missing sample grid: {path}")
        panels.append((label, resize_grid(Image.open(path).convert("RGB"))))

    label_width = 175
    row_gap = 24
    top_pad = 64
    right_pad = 24
    bottom_pad = 18
    grid_width = max(image.width for _, image in panels)
    grid_height = max(image.height for _, image in panels)
    width = max(label_width + grid_width + right_pad, 760)
    canvas = Image.new(
        "RGB",
        (width, top_pad + len(panels) * grid_height + (len(panels) - 1) * row_gap + bottom_pad),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20)
    label_font = load_font(16)
    small_font = load_font(12)
    draw.text((label_width, 14), f"Unconditional CIFAR-10 samples, seed {seed}, NFE {nfe}", fill="#111827", font=title_font)
    draw.text((label_width, 40), "EMA checkpoints; same evaluation protocol across methods", fill="#6b7280", font=small_font)

    y = top_pad
    for label, image in panels:
        draw.text((18, y + 8), label, fill="#111827", font=label_font)
        canvas.paste(image, (label_width, y))
        y += grid_height + row_gap

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description="Plot unconditional CIFAR-10 coupling summary figures.")
    parser.add_argument("--root", default="results/cifar10_uncond_coupling_large_100k")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-nfe", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    figures = root / "figures"
    rows = read_aggregate(root / "ema_5000_eval_allseeds_aggregate.csv")
    plot_metrics(rows, figures / "metrics_fid_kl.png")
    plot_sample_panel(root, figures / f"samples_nfe{args.sample_nfe}_seed{args.sample_seed}.png", args.sample_seed, args.sample_nfe)
    print(figures / "metrics_fid_kl.png")
    print(figures / f"samples_nfe{args.sample_nfe}_seed{args.sample_seed}.png")


if __name__ == "__main__":
    main()
