#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from lcfm.losses import material_derivative_jvp


class LinearVelocity(nn.Module):
    def __init__(self, matrix, time_vector):
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.register_buffer("time_vector", time_vector)

    def forward(self, x, t):
        return x @ self.matrix.T + t * self.time_vector


def main():
    torch.manual_seed(123)
    dtype = torch.float64
    matrix = torch.tensor(
        [[0.7, -0.2, 0.1], [0.3, 0.4, -0.5], [-0.1, 0.2, 0.6]],
        dtype=dtype,
    )
    time_vector = torch.tensor([0.5, -0.3, 0.2], dtype=dtype)
    model = LinearVelocity(matrix, time_vector)

    x = torch.randn(5, 3, dtype=dtype, requires_grad=True)
    t = torch.rand(5, 1, dtype=dtype, requires_grad=True)
    velocity = model(x, t)

    computed = material_derivative_jvp(model, x, t, velocity)
    expected = velocity @ matrix.T + time_vector
    max_error = torch.max(torch.abs(computed - expected)).item()

    print(f"computed:\n{computed}")
    print(f"expected:\n{expected}")
    print(f"max_error: {max_error:.3e}")

    tolerance = 1e-10
    if max_error > tolerance:
        raise SystemExit(f"material derivative JVP check failed: {max_error} > {tolerance}")
    print("material derivative JVP check passed")


if __name__ == "__main__":
    main()
