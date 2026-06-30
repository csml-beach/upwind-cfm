#!/usr/bin/env python3
"""Generate cached Burgers solution-map data with Dedalus.

This script is intended for the Linux VPS where Dedalus is installed. It writes
an `.npz` cache containing normalized trajectories for `burgers_solution_map`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def _periodic_delta(x, center):
    return np.angle(np.exp(1j * (x - center)))


def sample_fourier_ic(x, rng, modes, spectral_decay, scale, n_bumps=0, bump_width=0.25, bump_scale=0.0):
    u0 = np.zeros_like(x)
    for k in range(1, modes + 1):
        mode_scale = 1.0 / (k**spectral_decay)
        u0 += rng.normal(0.0, mode_scale) * np.sin(k * x)
        u0 += rng.normal(0.0, mode_scale) * np.cos(k * x)
    for _ in range(n_bumps):
        center = rng.uniform(0.0, 2.0 * np.pi)
        width = bump_width * rng.uniform(0.6, 1.4)
        amplitude = rng.normal(0.0, bump_scale)
        u0 += amplitude * np.exp(-0.5 * (_periodic_delta(x, center) / width) ** 2)
    u0 = u0 - u0.mean()
    u0 = scale * u0 / (u0.std() + 1e-8)
    return u0


def solve_one(u0, args):
    import dedalus.public as d3

    coords = d3.CartesianCoordinates("x")
    dist = d3.Distributor(coords, dtype=np.float64)
    basis = d3.RealFourier(coords["x"], size=args.nx, bounds=(0, 2 * np.pi), dealias=args.dealias)
    u = dist.Field(name="u", bases=basis)
    dx = lambda field: d3.Differentiate(field, coords["x"])
    problem = d3.IVP([u], namespace={"u": u, "nu": args.nu, "dx": dx})
    if args.form == "advective":
        problem.add_equation("dt(u) - nu*dx(dx(u)) = - u*dx(u)")
    elif args.form == "conservative":
        problem.add_equation("dt(u) - nu*dx(dx(u)) = - dx(0.5*u*u)")
    else:
        raise ValueError("form must be advective or conservative.")
    timestepper = getattr(d3, args.timestepper)
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = args.final_time
    u["g"] = u0

    save_times = np.linspace(0.0, args.final_time, args.nt)
    frames = []
    next_idx = 0
    while solver.proceed:
        while next_idx < len(save_times) and solver.sim_time >= save_times[next_idx] - 1e-12:
            u.change_scales(1)
            frames.append(np.array(u["g"], copy=True))
            next_idx += 1
        solver.step(args.dt)
    while next_idx < len(save_times):
        u.change_scales(1)
        frames.append(np.array(u["g"], copy=True))
        next_idx += 1
    arr = np.stack(frames[: args.nt])
    if not np.isfinite(arr).all():
        raise FloatingPointError("Dedalus produced non-finite Burgers trajectory.")
    return arr


def make_split(n_samples, seed, args):
    import dedalus.public as d3

    coords = d3.CartesianCoordinates("x")
    dist = d3.Distributor(coords, dtype=np.float64)
    basis = d3.RealFourier(coords["x"], size=args.nx, bounds=(0, 2 * np.pi), dealias=args.dealias)
    x = np.asarray(dist.local_grid(basis))
    rng = np.random.default_rng(seed)
    videos = []
    for index in range(n_samples):
        modes = args.n_fourier_modes
        spectral_decay = args.spectral_decay
        ic_scale = args.ic_scale
        n_bumps = args.n_bumps
        if args.randomize_ic:
            modes = int(rng.integers(args.min_fourier_modes, args.n_fourier_modes + 1))
            spectral_decay = float(rng.uniform(args.min_spectral_decay, args.spectral_decay))
            ic_scale = float(rng.uniform(args.min_ic_scale, args.ic_scale))
            if args.n_bumps > 0:
                n_bumps = int(rng.integers(0, args.n_bumps + 1))
        u0 = sample_fourier_ic(
            x,
            rng,
            modes,
            spectral_decay,
            ic_scale,
            n_bumps=n_bumps,
            bump_width=args.bump_width,
            bump_scale=args.bump_scale,
        )
        video = solve_one(u0, args)
        videos.append(video)
        if (index + 1) % args.log_every == 0 or index + 1 == n_samples:
            print(f"generated {index + 1}/{n_samples}", flush=True)
    return np.stack(videos)


def normalize_per_video(videos):
    mean = videos.mean(axis=(1, 2), keepdims=True)
    std = videos.std(axis=(1, 2), keepdims=True)
    return (videos - mean) / (std + 1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=64)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nt", type=int, default=64)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--final-time", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=2e-4)
    parser.add_argument("--n-fourier-modes", type=int, default=5)
    parser.add_argument("--min-fourier-modes", type=int, default=2)
    parser.add_argument("--spectral-decay", type=float, default=1.2)
    parser.add_argument("--min-spectral-decay", type=float, default=0.8)
    parser.add_argument("--ic-scale", type=float, default=0.8)
    parser.add_argument("--min-ic-scale", type=float, default=0.45)
    parser.add_argument("--randomize-ic", action="store_true")
    parser.add_argument("--n-bumps", type=int, default=0)
    parser.add_argument("--bump-width", type=float, default=0.25)
    parser.add_argument("--bump-scale", type=float, default=0.35)
    parser.add_argument("--dealias", type=float, default=1.5)
    parser.add_argument("--form", choices=["advective", "conservative"], default="advective")
    parser.add_argument("--timestepper", default="RK222")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-seed-offset", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating Dedalus Burgers cache: {out}", flush=True)
    train = make_split(args.n_train, args.seed, args)
    test = make_split(args.n_test, args.seed + args.test_seed_offset, args)
    train_norm = normalize_per_video(train).astype(np.float32)
    test_norm = normalize_per_video(test).astype(np.float32)
    metadata = vars(args).copy()
    np.savez_compressed(
        out,
        train_videos=train_norm,
        test_videos=test_norm,
        raw_train_min=np.array(train.min()),
        raw_train_max=np.array(train.max()),
        raw_test_min=np.array(test.min()),
        raw_test_max=np.array(test.max()),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Saved {out}", flush=True)
    print(f"train {train_norm.shape} test {test_norm.shape}", flush=True)
    print(f"raw min/max train=({train.min():.4f}, {train.max():.4f}) test=({test.min():.4f}, {test.max():.4f})", flush=True)


if __name__ == "__main__":
    main()
