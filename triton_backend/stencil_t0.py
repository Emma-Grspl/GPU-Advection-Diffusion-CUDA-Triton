from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def advection_diffusion_t0_kernel(
    phi_old_ptr,
    u_ptr,
    v_ptr,
    phi_new_ptr,
    nx,
    ny,
    dt: tl.float64,
    inv_2dx: tl.float64,
    inv_2dy: tl.float64,
    inv_dx2: tl.float64,
    inv_dy2: tl.float64,
    diffusivity_dt: tl.float64,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = (
        tl.program_id(axis=0)
        * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    valid = offsets < n_elements

    j = offsets // nx
    i = offsets - j * nx

    interior = (
        valid
        & (i > 0)
        & (j > 0)
        & (i + 1 < nx)
        & (j + 1 < ny)
    )

    center = tl.load(
        phi_old_ptr + offsets,
        mask=valid,
        other=0.0,
    )

    left = tl.load(
        phi_old_ptr + offsets - 1,
        mask=interior,
        other=0.0,
    )

    right = tl.load(
        phi_old_ptr + offsets + 1,
        mask=interior,
        other=0.0,
    )

    bottom = tl.load(
        phi_old_ptr + offsets - nx,
        mask=interior,
        other=0.0,
    )

    top = tl.load(
        phi_old_ptr + offsets + nx,
        mask=interior,
        other=0.0,
    )

    u = tl.load(
        u_ptr + offsets,
        mask=interior,
        other=0.0,
    )

    v = tl.load(
        v_ptr + offsets,
        mask=interior,
        other=0.0,
    )

    dphi_dx = (
        right - left
    ) * inv_2dx

    dphi_dy = (
        top - bottom
    ) * inv_2dy

    laplacian = (
        (
            right
            - 2.0 * center
            + left
        )
        * inv_dx2
        +
        (
            top
            - 2.0 * center
            + bottom
        )
        * inv_dy2
    )

    updated = (
        center
        - dt
        * (
            u * dphi_dx
            + v * dphi_dy
        )
        + diffusivity_dt
        * laplacian
    )

    output = tl.where(
        interior,
        updated,
        center,
    )

    tl.store(
        phi_new_ptr + offsets,
        output,
        mask=valid,
    )


def launch_t0(
    phi_old: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    phi_new: torch.Tensor,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    diffusivity: float,
    dt: float,
    block_size: int = 256,
    num_warps: int = 8,
) -> None:
    n_elements = nx * ny

    grid = (
        triton.cdiv(
            n_elements,
            block_size,
        ),
    )

    advection_diffusion_t0_kernel[grid](
        phi_old,
        u,
        v,
        phi_new,
        nx,
        ny,
        dt,
        1.0 / (2.0 * dx),
        1.0 / (2.0 * dy),
        1.0 / (dx * dx),
        1.0 / (dy * dy),
        diffusivity * dt,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def advance_t0(
    phi_a: torch.Tensor,
    phi_b: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    diffusivity: float,
    dt: float,
    steps: int,
    current_is_a: bool = True,
    block_size: int = 256,
    num_warps: int = 8,
) -> bool:
    for _ in range(steps):
        if current_is_a:
            launch_t0(
                phi_a,
                u,
                v,
                phi_b,
                nx=nx,
                ny=ny,
                dx=dx,
                dy=dy,
                diffusivity=diffusivity,
                dt=dt,
                block_size=block_size,
                num_warps=num_warps,
            )
        else:
            launch_t0(
                phi_b,
                u,
                v,
                phi_a,
                nx=nx,
                ny=ny,
                dx=dx,
                dy=dy,
                diffusivity=diffusivity,
                dt=dt,
                block_size=block_size,
                num_warps=num_warps,
            )

        current_is_a = not current_is_a

    return current_is_a
