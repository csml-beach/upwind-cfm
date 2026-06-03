#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm.plotting import plot_spiral_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--n-traj", type=int, default=24)
    parser.add_argument("--n-final", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--noise", type=float, default=None)
    args = parser.parse_args()
    output = plot_spiral_run(
        args.run_dir,
        args.output,
        args.eval_seed,
        args.n_traj,
        args.n_final,
        args.steps,
        args.noise,
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
