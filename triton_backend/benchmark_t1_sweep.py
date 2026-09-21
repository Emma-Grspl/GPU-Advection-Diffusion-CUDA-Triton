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

CONFIGS = [
    (64, 2),
    (128, 4),
    (256, 4),
    (256, 8),
    (512, 4),
    (512, 8),
    (1024, 8),
]

DIFFUSIVITY = 1.0e-3
DT = 1.0e-8

WARMUP_STEPS = 30
TIMED_STEPS = 500
ROUNDS = 7


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
            2.0
            * math.pi
            * yy
        )
    ).contiguous()

    v = (
        -0.20
        + 0.10
        * torch.cos(
            2.0
            * math.pi
            * xx
        )
    ).contiguous()

    return phi, u, v


def gpu_preheat():
    n = 4096

    phi, u, v = initialize(n)

    phi_b = torch.empty_like(phi)

    advance_t0(
        phi,
        phi_b,
        u,
        v,
        nx=n,
        ny=n,
        dx=1.0 / (n - 1),
        dy=1.0 / (n - 1),
        diffusivity=DIFFUSIVITY,
        dt=DT,
        steps=1000,
        current_is_a=True,
        block_size=256,
        num_warps=8,
    )

    torch.cuda.synchronize()

    del phi, phi_b, u, v
    torch.cuda.empty_cache()


def benchmark_size(n: int):
    phi_initial, u, v = initialize(n)

    dx = 1.0 / (n - 1)
    dy = 1.0 / (n - 1)

    samples = {
        config: []
        for config in CONFIGS
    }

    states = {}

    # Compile and warm every configuration before measurement.
    for block_size, num_warps in CONFIGS:
        phi_a = phi_initial.clone()
        phi_b = torch.empty_like(phi_a)

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
            current_is_a=True,
            block_size=block_size,
            num_warps=num_warps,
        )

        torch.cuda.synchronize()

        states[(block_size, num_warps)] = [
            phi_a,
            phi_b,
            current_is_a,
        ]

    # Cyclic order so each config appears in each relative position.
    for round_index in range(ROUNDS):
        for position in range(len(CONFIGS)):
            config_index = (
                position
                + round_index
            ) % len(CONFIGS)

            block_size, num_warps = (
                CONFIGS[config_index]
            )

            phi_a, phi_b, current_is_a = (
                states[
                    (block_size, num_warps)
                ]
            )

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
                block_size=block_size,
                num_warps=num_warps,
            )

            stop.record()
            stop.synchronize()

            elapsed_ms = (
                start.elapsed_time(stop)
                / TIMED_STEPS
            )

            samples[
                (block_size, num_warps)
            ].append(
                elapsed_ms
            )

            states[
                (block_size, num_warps)
            ][2] = current_is_a

    for block_size, num_warps in CONFIGS:
        values = samples[
            (block_size, num_warps)
        ]

        median_ms = statistics.median(
            values
        )

        mean_ms = statistics.mean(
            values
        )

        std_ms = statistics.pstdev(
            values
        )

        cells = n * n

        gcell = (
            cells
            / (
                median_ms
                * 1.0e-3
            )
            / 1.0e9
        )

        print(
            f"{n},{n},"
            f"{block_size},"
            f"{num_warps},"
            f"{median_ms * 1000.0:.10g},"
            f"{mean_ms * 1000.0:.10g},"
            f"{std_ms * 1000.0:.10g},"
            f"{gcell:.10g}"
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

    gpu_preheat()

    for n in SIZES:
        benchmark_size(n)


if __name__ == "__main__":
    main()
