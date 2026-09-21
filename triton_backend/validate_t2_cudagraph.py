from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triton_backend.stencil_t0 import (
    advance_t0,
    launch_t0,
)


BLOCK_SIZE = 256
NUM_WARPS = 8

GRAPH_STEPS = 100
TOTAL_STEPS = 1000


def make_fields(nx: int, ny: int):
    x = torch.linspace(
        0.0,
        1.0,
        nx,
        device="cuda",
        dtype=torch.float64,
    )

    y = torch.linspace(
        0.0,
        1.0,
        ny,
        device="cuda",
        dtype=torch.float64,
    )

    yy, xx = torch.meshgrid(
        y,
        x,
        indexing="ij",
    )

    phi = (
        torch.sin(3.1 * xx)
        * torch.cos(2.7 * yy)
        + 0.2 * xx * yy
    ).contiguous()

    u = (
        0.4
        + 0.1 * yy
    ).contiguous()

    v = (
        -0.2
        + 0.05 * xx
    ).contiguous()

    return phi, u, v


def capture_graph(
    phi_initial,
    u,
    v,
    *,
    nx,
    ny,
    dx,
    dy,
    diffusivity,
    dt,
):
    if GRAPH_STEPS % 2 != 0:
        raise ValueError(
            "GRAPH_STEPS must be even "
            "for ping-pong replay."
        )

    static_a = phi_initial.clone()
    static_b = torch.empty_like(static_a)

    # Compile/warm Triton outside graph capture.
    warm_a = phi_initial.clone()
    warm_b = torch.empty_like(warm_a)

    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(
        torch.cuda.current_stream()
    )

    with torch.cuda.stream(warm_stream):
        current_is_a = True

        for _ in range(10):
            if current_is_a:
                launch_t0(
                    warm_a,
                    u,
                    v,
                    warm_b,
                    nx=nx,
                    ny=ny,
                    dx=dx,
                    dy=dy,
                    diffusivity=diffusivity,
                    dt=dt,
                    block_size=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )
            else:
                launch_t0(
                    warm_b,
                    u,
                    v,
                    warm_a,
                    nx=nx,
                    ny=ny,
                    dx=dx,
                    dy=dy,
                    diffusivity=diffusivity,
                    dt=dt,
                    block_size=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )

            current_is_a = not current_is_a

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
                    nx=nx,
                    ny=ny,
                    dx=dx,
                    dy=dy,
                    diffusivity=diffusivity,
                    dt=dt,
                    block_size=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )
            else:
                launch_t0(
                    static_b,
                    u,
                    v,
                    static_a,
                    nx=nx,
                    ny=ny,
                    dx=dx,
                    dy=dy,
                    diffusivity=diffusivity,
                    dt=dt,
                    block_size=BLOCK_SIZE,
                    num_warps=NUM_WARPS,
                )

            current_is_a = not current_is_a

    if not current_is_a:
        raise RuntimeError(
            "Captured graph does not finish in phi_a."
        )

    return graph, static_a, static_b


def validate_case(nx: int, ny: int):
    dx = 0.7 / (nx - 1)
    dy = 1.3 / (ny - 1)

    diffusivity = 1.0e-3
    dt = 1.0e-5

    phi0, u, v = make_fields(
        nx,
        ny,
    )

    # -----------------------------
    # Eager Triton reference
    # -----------------------------
    eager_a = phi0.clone()
    eager_b = torch.empty_like(eager_a)

    eager_current_is_a = advance_t0(
        eager_a,
        eager_b,
        u,
        v,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        diffusivity=diffusivity,
        dt=dt,
        steps=TOTAL_STEPS,
        current_is_a=True,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    torch.cuda.synchronize()

    eager_result = (
        eager_a
        if eager_current_is_a
        else eager_b
    )

    # -----------------------------
    # CUDA Graph version
    # -----------------------------
    graph, graph_a, graph_b = capture_graph(
        phi0,
        u,
        v,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        diffusivity=diffusivity,
        dt=dt,
    )

    n_replays = (
        TOTAL_STEPS
        // GRAPH_STEPS
    )

    for _ in range(n_replays):
        graph.replay()

    torch.cuda.synchronize()

    graph_result = graph_a

    diff = graph_result - eager_result

    max_abs = (
        diff.abs().max().item()
    )

    rel_l2 = (
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(
            eager_result
        )
    ).item()

    exact = torch.equal(
        graph_result,
        eager_result,
    )

    passed = (
        max_abs <= 1.0e-15
        and rel_l2 <= 1.0e-15
    )

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{nx}x{ny} "
        f"steps={TOTAL_STEPS} "
        f"graph_steps={GRAPH_STEPS} "
        f"exact={exact} "
        f"rel_L2={rel_l2:.6e} "
        f"Linf={max_abs:.6e}"
    )

    if not passed:
        raise SystemExit(1)


def main():
    print(
        "CUDA Graph validation"
    )

    validate_case(
        37,
        51,
    )

    validate_case(
        512,
        512,
    )

    print()
    print(
        "TRITON T2 CUDA GRAPH VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
