from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triton_backend.stencil_t0 import advance_t0


SIZES = [
    512,
    1024,
    2048,
    4096,
]

WARMUP_STEPS = 50
TIMED_STEPS = 500
REPEATS = 7

DIFFUSIVITY = 1.0e-3
DT = 1.0e-8

BLOCK_SIZE = 256
NUM_WARPS = 8


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
        + 0.25
        * torch.exp(
            -60.0
            * (
                (xx - 0.75) ** 2
                + (yy - 0.25) ** 2
            )
        )
    ).contiguous()

    u = (
        0.50
        + 0.10
        * torch.sin(
            2.0 * math.pi * yy
        )
    ).contiguous()

    v = (
        -0.20
        + 0.10
        * torch.cos(
            2.0 * math.pi * xx
        )
    ).contiguous()

    return phi, u, v


def benchmark(n: int):
    phi_initial, u, v = initialize(n)

    phi_a = phi_initial.clone()
    phi_b = torch.empty_like(phi_a)

    dx = 1.0 / (n - 1)
    dy = 1.0 / (n - 1)

    current_is_a = True

    # Compilation + warmup excluded from timings.
    current_is_a = advance_t0(
        phi_a,
        phi_b,
        u,
        v,
        nx=n,
        ny=n,
        dx=dx,
        dy=dy,
        diffusivity=DIFFUSIVITY,
        dt=DT,
        steps=WARMUP_STEPS,
        current_is_a=current_is_a,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    torch.cuda.synchronize()

    samples_ms = []

    for _ in range(REPEATS):
        start = torch.cuda.Event(
            enable_timing=True
        )

        stop = torch.cuda.Event(
            enable_timing=True
        )

        start.record()

        current_is_a = advance_t0(
            phi_a,
            phi_b,
            u,
            v,
            nx=n,
            ny=n,
            dx=dx,
            dy=dy,
            diffusivity=DIFFUSIVITY,
            dt=DT,
            steps=TIMED_STEPS,
            current_is_a=current_is_a,
            block_size=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )

        stop.record()
        stop.synchronize()

        elapsed_ms = (
            start.elapsed_time(stop)
            / TIMED_STEPS
        )

        samples_ms.append(
            elapsed_ms
        )

    median_ms = statistics.median(
        samples_ms
    )

    mean_ms = statistics.mean(
        samples_ms
    )

    std_ms = statistics.pstdev(
        samples_ms
    )

    cells = n * n

    gcell_per_s = (
        cells
        / (
            median_ms
            * 1.0e-3
        )
        / 1.0e9
    )

    return (
        median_ms,
        mean_ms,
        std_ms,
        gcell_per_s,
    )


def main():
    print(
        "nx,ny,"
        "block_size,num_warps,"
        "median_step_us,"
        "mean_step_us,"
        "std_step_us,"
        "gcell_per_s"
    )

    for n in SIZES:
        (
            median_ms,
            mean_ms,
            std_ms,
            gcell_per_s,
        ) = benchmark(n)

        print(
            f"{n},{n},"
            f"{BLOCK_SIZE},{NUM_WARPS},"
            f"{median_ms * 1000.0:.10g},"
            f"{mean_ms * 1000.0:.10g},"
            f"{std_ms * 1000.0:.10g},"
            f"{gcell_per_s:.10g}"
        )


if __name__ == "__main__":
    main()
