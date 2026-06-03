#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def get_nested(data, path, default=None):
    cur = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def flat_tags(config):
    tags = config.get("tags", {})
    if not isinstance(tags, dict):
        return {}
    return {f"tag_{key}": value for key, value in tags.items()}


def row_from_run(root, metrics_path):
    run_dir = metrics_path.parent
    config_path = run_dir / "config.json"
    env_path = run_dir / "environment.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json next to {metrics_path}")

    config = read_json(config_path)
    metrics = read_json(metrics_path)
    env = read_json(env_path) if env_path.exists() else {}
    relative = run_dir.relative_to(root)
    parts = relative.parts

    row = {
        "relative_run_dir": str(relative),
        "group": parts[0] if len(parts) > 1 else "",
        "run_name": run_dir.name,
        "dataset": config.get("dataset"),
        "method": config.get("method"),
        "model": config.get("model"),
        "seed": config.get("seed"),
        "device": config.get("device", "auto"),
        "solver": config.get("solver"),
        "steps": get_nested(config, "solver_kwargs.steps"),
        "noise": get_nested(config, "solver_kwargs.noise", 0.0),
        "epochs": get_nested(config, "train.epochs"),
        "batch_size": get_nested(config, "train.batch_size"),
        "lr": get_nested(config, "train.lr"),
        "weight": get_nested(config, "method_kwargs.weight"),
        "dt": get_nested(config, "method_kwargs.dt"),
        "git_commit": env.get("git_commit"),
        "python": env.get("python"),
        "torch": env.get("torch"),
        "cuda_available": env.get("cuda_available"),
    }
    row.update(flat_tags(config))
    row.update(metrics)
    return row


def collect_rows(root):
    root = Path(root)
    rows = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        rows.append(row_from_run(root, metrics_path))
    return rows


def columns_for(rows):
    preferred = [
        "relative_run_dir",
        "group",
        "run_name",
        "dataset",
        "method",
        "model",
        "seed",
        "device",
        "solver",
        "steps",
        "noise",
        "epochs",
        "batch_size",
        "lr",
        "weight",
        "dt",
    ]
    metric_cols = sorted(
        key
        for row in rows
        for key in row
        if key not in preferred
        and key not in {"git_commit", "python", "torch", "cuda_available"}
        and not key.startswith("tag_")
    )
    tag_cols = sorted(key for row in rows for key in row if key.startswith("tag_"))
    env_cols = ["git_commit", "python", "torch", "cuda_available"]
    seen = set()
    cols = []
    for col in preferred + tag_cols + metric_cols + env_cols:
        if col not in seen:
            seen.add(col)
            cols.append(col)
    return cols


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, columns):
    def fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(rows, columns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Run artifact root to scan recursively.")
    parser.add_argument("--output", default=None, help="CSV output path.")
    parser.add_argument("--markdown", default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect_rows(root)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {root}")
    columns = columns_for(rows)
    output = args.output or str(root / "summary.csv")
    write_csv(output, rows, columns)
    if args.markdown:
        write_markdown(args.markdown, rows, columns)
    print(f"wrote {len(rows)} rows to {output}")
    if args.markdown:
        print(f"wrote markdown to {args.markdown}")


if __name__ == "__main__":
    sys.exit(main())
