from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triton_backend.stencil_t0 import launch_t0


def initialize(n: int):
    x = torch.linspace(
        0.0,
        1.0,
        n,
        device="cuda",
        dtype=torch.float64,
    )

    y = torch.linspace(
        0.0,
        1.0,
        n,
        device="cuda",
        dtype=torch.float64,
    )

    yy, xx = torch.meshgrid(
        y,
        x,
        indexing="ij",
    )

    phi = (
        torch.exp(
            -40.0
            * (
                (xx - 0.30) ** 2
                + (yy - 0.60) ** 2
            )
        )
    ).contiguous()

    u = (
        0.5
        + 0.1
        * torch.sin(
            2.0 * math.pi * yy
        )
    ).contiguous()

    v = (
        -0.2
        + 0.1
        * torch.cos(
            2.0 * math.pi * xx
        )
    ).contiguous()

    return phi, u, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=512,
    )
    args = parser.parse_args()

    n = args.n

    phi_a, u, v = initialize(n)
    phi_b = torch.empty_like(phi_a)

    kwargs = dict(
        nx=n,
        ny=n,
        dx=1.0 / (n - 1),
        dy=1.0 / (n - 1),
        diffusivity=1.0e-3,
        dt=1.0e-8,
        block_size=256,
        num_warps=8,
    )

    # Compilation + warmup.
    for _ in range(50):
        launch_t0(
            phi_a,
            u,
            v,
            phi_b,
            **kwargs,
        )

        phi_a, phi_b = (
            phi_b,
            phi_a,
        )

    torch.cuda.synchronize()

    # Small deterministic region for profiling.
    for _ in range(20):
        launch_t0(
            phi_a,
            u,
            v,
            phi_b,
            **kwargs,
        )

        phi_a, phi_b = (
            phi_b,
            phi_a,
        )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
