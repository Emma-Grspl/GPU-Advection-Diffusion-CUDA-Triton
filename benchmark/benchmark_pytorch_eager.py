from __future__ import annotations

import math
import statistics
import time

import torch


DTYPE = torch.float64
DEVICE = torch.device("cuda")

SIZES = [512, 1024, 2048, 4096]

WARMUP_STEPS = 50
TIMED_STEPS = 200
REPEATS = 7

DIFFUSIVITY = 1.0e-3
DT = 1.0e-8


@torch.no_grad()
def initialize_fields(nx: int, ny: int):
    x = torch.linspace(
        0.0,
        1.0,
        nx,
        device=DEVICE,
        dtype=DTYPE,
    )

    y = torch.linspace(
        0.0,
        1.0,
        ny,
        device=DEVICE,
        dtype=DTYPE,
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
    )

    u = (
        0.50
        + 0.10
        * torch.sin(
            2.0 * math.pi * yy
        )
    )

    v = (
        -0.20
        + 0.10
        * torch.cos(
            2.0 * math.pi * xx
        )
    )

    return (
        phi.contiguous(),
        u.contiguous(),
        v.contiguous(),
    )


@torch.no_grad()
def step(
    phi_old: torch.Tensor,
    phi_new: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
):
    center = phi_old[1:-1, 1:-1]

    left = phi_old[1:-1, :-2]
    right = phi_old[1:-1, 2:]

    bottom = phi_old[:-2, 1:-1]
    top = phi_old[2:, 1:-1]

    dphi_dx = (
        right - left
    ) / (2.0 * dx)

    dphi_dy = (
        top - bottom
    ) / (2.0 * dy)

    laplacian = (
        (
            right
            - 2.0 * center
            + left
        )
        / (dx * dx)
        +
        (
            top
            - 2.0 * center
            + bottom
        )
        / (dy * dy)
    )

    phi_new[1:-1, 1:-1] = (
        center
        - DT
        * (
            u[1:-1, 1:-1]
            * dphi_dx
            +
            v[1:-1, 1:-1]
            * dphi_dy
        )
        + DIFFUSIVITY
        * DT
        * laplacian
    )

    # Exact boundary preservation.
    phi_new[0, :] = phi_old[0, :]
    phi_new[-1, :] = phi_old[-1, :]
    phi_new[:, 0] = phi_old[:, 0]
    phi_new[:, -1] = phi_old[:, -1]


@torch.no_grad()
def advance(
    phi_a: torch.Tensor,
    phi_b: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
    steps: int,
    current_is_a: bool,
):
    for _ in range(steps):
        if current_is_a:
            step(
                phi_a,
                phi_b,
                u,
                v,
                dx,
                dy,
            )
        else:
            step(
                phi_b,
                phi_a,
                u,
                v,
                dx,
                dy,
            )

        current_is_a = not current_is_a

    return current_is_a


def benchmark_size(nx: int, ny: int):
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    phi_initial, u, v = initialize_fields(
        nx,
        ny,
    )

    phi_a = phi_initial.clone()
    phi_b = torch.empty_like(phi_a)

    current_is_a = True

    current_is_a = advance(
        phi_a,
        phi_b,
        u,
        v,
        dx,
        dy,
        WARMUP_STEPS,
        current_is_a,
    )

    torch.cuda.synchronize()

    event_samples_ms = []
    wall_samples_ms = []

    for _ in range(REPEATS):
        start_event = torch.cuda.Event(
            enable_timing=True
        )

        stop_event = torch.cuda.Event(
            enable_timing=True
        )

        torch.cuda.synchronize()

        wall_start = time.perf_counter()

        start_event.record()

        current_is_a = advance(
            phi_a,
            phi_b,
            u,
            v,
            dx,
            dy,
            TIMED_STEPS,
            current_is_a,
        )

        stop_event.record()
        stop_event.synchronize()

        wall_stop = time.perf_counter()

        event_ms = (
            start_event.elapsed_time(
                stop_event
            )
            / TIMED_STEPS
        )

        wall_ms = (
            (wall_stop - wall_start)
            * 1000.0
            / TIMED_STEPS
        )

        event_samples_ms.append(
            event_ms
        )

        wall_samples_ms.append(
            wall_ms
        )

    event_median_ms = statistics.median(
        event_samples_ms
    )

    wall_median_ms = statistics.median(
        wall_samples_ms
    )

    event_std_ms = statistics.pstdev(
        event_samples_ms
    )

    wall_std_ms = statistics.pstdev(
        wall_samples_ms
    )

    cells = nx * ny

    event_gcell = (
        cells
        / (event_median_ms * 1.0e-3)
        / 1.0e9
    )

    wall_gcell = (
        cells
        / (wall_median_ms * 1.0e-3)
        / 1.0e9
    )

    return (
        event_median_ms,
        wall_median_ms,
        event_std_ms,
        wall_std_ms,
        event_gcell,
        wall_gcell,
    )


def main():
    print(
        "nx,ny,"
        "event_median_us,"
        "wall_median_us,"
        "event_std_us,"
        "wall_std_us,"
        "event_gcell_per_s,"
        "wall_gcell_per_s"
    )

    for n in SIZES:
        (
            event_median_ms,
            wall_median_ms,
            event_std_ms,
            wall_std_ms,
            event_gcell,
            wall_gcell,
        ) = benchmark_size(
            n,
            n,
        )

        print(
            f"{n},"
            f"{n},"
            f"{event_median_ms * 1000.0:.10g},"
            f"{wall_median_ms * 1000.0:.10g},"
            f"{event_std_ms * 1000.0:.10g},"
            f"{wall_std_ms * 1000.0:.10g},"
            f"{event_gcell:.10g},"
            f"{wall_gcell:.10g}"
        )


if __name__ == "__main__":
    main()
