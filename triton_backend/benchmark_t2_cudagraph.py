from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triton_backend.stencil_t0 import (
    advance_t0,
    launch_t0,
)


SIZES = [
    512,
    1024,
    2048,
    4096,
]

BLOCK_SIZE = 256
NUM_WARPS = 8

GRAPH_STEPS = 100

WARMUP_STEPS = 50
TIMED_STEPS = 500
REPEATS = 7

DIFFUSIVITY = 1.0e-3
DT = 1.0e-8


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


def capture_graph(
    phi_initial,
    u,
    v,
    *,
    n,
):
    if GRAPH_STEPS % 2 != 0:
        raise RuntimeError(
            "GRAPH_STEPS must be even."
        )

    dx = 1.0 / (n - 1)
    dy = 1.0 / (n - 1)

    static_a = phi_initial.clone()
    static_b = torch.empty_like(
        static_a
    )

    # Triton compilation outside capture.
    warm_a = phi_initial.clone()
    warm_b = torch.empty_like(
        warm_a
    )

    warm_stream = torch.cuda.Stream()

    warm_stream.wait_stream(
        torch.cuda.current_stream()
    )

    with torch.cuda.stream(
        warm_stream
    ):
        current_is_a = True

        for _ in range(10):
            if current_is_a:
                launch_t0(
                    warm_a,
                    u,
                    v,
                    warm_b,
                    nx=n,
                    ny=n,
                    dx=dx,
                    dy=dy,
                    diffusivity=DIFFUSIVITY,
                    dt=DT,
                    block_size=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )
            else:
                launch_t0(
                    warm_b,
                    u,
                    v,
                    warm_a,
                    nx=n,
                    ny=n,
                    dx=dx,
                    dy=dy,
                    diffusivity=DIFFUSIVITY,
                    dt=DT,
                    block_size=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )

            current_is_a = (
                not current_is_a
            )

    torch.cuda.current_stream().wait_stream(
        warm_stream
    )

    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()

    current_is_a = True

    with torch.cuda.graph(graph):
        for _ in range(GRAPH_STEPS):
            if current_is_a:
                launch_t0(
                    static_a,
                    u,
                    v,
                    static_b,
                    nx=n,
                    ny=n,
                    dx=dx,
                    dy=dy,
                    diffusivity=DIFFUSIVITY,
                    dt=DT,
                    block_size=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )
            else:
                launch_t0(
                    static_b,
                    u,
                    v,
                    static_a,
                    nx=n,
                    ny=n,
                    dx=dx,
                    dy=dy,
                    diffusivity=DIFFUSIVITY,
                    dt=DT,
                    block_size=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )

            current_is_a = (
                not current_is_a
            )

    return (
        graph,
        static_a,
        static_b,
    )


def benchmark_eager(
    phi_initial,
    u,
    v,
    n,
):
    phi_a = phi_initial.clone()
    phi_b = torch.empty_like(
        phi_a
    )

    dx = 1.0 / (n - 1)
    dy = 1.0 / (n - 1)

    current_is_a = True

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

    samples = []

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

        samples.append(
            start.elapsed_time(stop)
            / TIMED_STEPS
        )

    return samples


def benchmark_graph(
    phi_initial,
    u,
    v,
    n,
):
    graph, static_a, static_b = (
        capture_graph(
            phi_initial,
            u,
            v,
            n=n,
        )
    )

    # Keep graph-owned buffers alive.
    _ = (
        static_a,
        static_b,
    )

    # Warm graph replay itself.
    for _ in range(5):
        graph.replay()

    torch.cuda.synchronize()

    graph_replays = (
        TIMED_STEPS
        // GRAPH_STEPS
    )

    if (
        graph_replays
        * GRAPH_STEPS
        != TIMED_STEPS
    ):
        raise RuntimeError(
            "TIMED_STEPS must be divisible "
            "by GRAPH_STEPS."
        )

    samples = []

    for _ in range(REPEATS):
        start = torch.cuda.Event(
            enable_timing=True
        )

        stop = torch.cuda.Event(
            enable_timing=True
        )

        start.record()

        for _ in range(
            graph_replays
        ):
            graph.replay()

        stop.record()
        stop.synchronize()

        samples.append(
            start.elapsed_time(stop)
            / TIMED_STEPS
        )

    return samples


def summarize(
    samples,
    n,
):
    median_ms = statistics.median(
        samples
    )

    mean_ms = statistics.mean(
        samples
    )

    std_ms = statistics.pstdev(
        samples
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
        "eager_median_us,"
        "graph_median_us,"
        "graph_speedup,"
        "eager_gcell_per_s,"
        "graph_gcell_per_s,"
        "graph_steps"
    )

    for n in SIZES:
        phi, u, v = initialize(n)

        eager_samples = benchmark_eager(
            phi,
            u,
            v,
            n,
        )

        graph_samples = benchmark_graph(
            phi,
            u,
            v,
            n,
        )

        (
            eager_ms,
            _,
            _,
            eager_gcell,
        ) = summarize(
            eager_samples,
            n,
        )

        (
            graph_ms,
            _,
            _,
            graph_gcell,
        ) = summarize(
            graph_samples,
            n,
        )

        speedup = (
            eager_ms
            / graph_ms
        )

        print(
            f"{n},{n},"
            f"{eager_ms * 1000.0:.10g},"
            f"{graph_ms * 1000.0:.10g},"
            f"{speedup:.10g},"
            f"{eager_gcell:.10g},"
            f"{graph_gcell:.10g},"
            f"{GRAPH_STEPS}"
        )


if __name__ == "__main__":
    main()
