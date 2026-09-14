"""
PyTorch eager implementation of the 2-D advection-diffusion stencil.

The mathematical operator and flattened memory layout match the
historical NumPy and C++ CPU reference implementations.
"""

from __future__ import annotations

import torch


def _validate_inputs(
    phi_old: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    diffusivity: float,
    dt: float,
) -> None:
    expected_size = nx * ny

    if nx < 3 or ny < 3:
        raise ValueError(
            "nx and ny must both be at least 3."
        )

    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(
            "dx and dy must be strictly positive."
        )

    if diffusivity < 0.0:
        raise ValueError(
            "diffusivity cannot be negative."
        )

    if dt <= 0.0:
        raise ValueError(
            "dt must be strictly positive."
        )

    for name, tensor in (
        ("phi_old", phi_old),
        ("u", u),
        ("v", v),
    ):
        if tensor.ndim != 1:
            raise ValueError(
                f"{name} must be a flattened 1-D tensor."
            )

        if tensor.numel() != expected_size:
            raise ValueError(
                f"{name} has {tensor.numel()} values, "
                f"expected {expected_size}."
            )

        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous in memory."
            )

        if tensor.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise TypeError(
                f"{name} must use float32 or float64."
            )

    if not (
        phi_old.dtype
        == u.dtype
        == v.dtype
    ):
        raise TypeError(
            "phi_old, u and v must use the same dtype."
        )

    if not (
        phi_old.device
        == u.device
        == v.device
    ):
        raise ValueError(
            "phi_old, u and v must be on the same device."
        )


def advance_scalar_torch(
    phi_old: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    diffusivity: float,
    dt: float,
) -> torch.Tensor:
    """
    Advance one explicit advection-diffusion step.

    Flattened layout:
        k = i + j * nx

    Boundary values are preserved.
    """

    _validate_inputs(
        phi_old,
        u,
        v,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        diffusivity=diffusivity,
        dt=dt,
    )

    # reshape() creates a view here because the input
    # flattened tensors are contiguous.
    phi = phi_old.reshape(
        ny,
        nx,
    )

    velocity_u = u.reshape(
        ny,
        nx,
    )

    velocity_v = v.reshape(
        ny,
        nx,
    )

    # clone() creates the independent output buffer and
    # automatically preserves all boundary values.
    phi_new = phi.clone()

    center = phi[
        1:-1,
        1:-1,
    ]

    left = phi[
        1:-1,
        :-2,
    ]

    right = phi[
        1:-1,
        2:,
    ]

    bottom = phi[
        :-2,
        1:-1,
    ]

    top = phi[
        2:,
        1:-1,
    ]

    dphi_dx = (
        right - left
    ) / (
        2.0 * dx
    )

    dphi_dy = (
        top - bottom
    ) / (
        2.0 * dy
    )

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

    advection = (
        velocity_u[
            1:-1,
            1:-1,
        ]
        * dphi_dx
        +
        velocity_v[
            1:-1,
            1:-1,
        ]
        * dphi_dy
    )

    phi_new[
        1:-1,
        1:-1,
    ] = (
        center
        - dt * advection
        + diffusivity
        * dt
        * laplacian
    )

    return phi_new.reshape(-1)
