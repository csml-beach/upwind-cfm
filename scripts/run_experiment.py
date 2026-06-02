#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm.experiment import run
from lcfm.utils import read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a JSON experiment config.")
    args = parser.parse_args()
    run(read_json(args.config))


if __name__ == "__main__":
    main()
