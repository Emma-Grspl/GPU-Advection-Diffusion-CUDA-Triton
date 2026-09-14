"""
Historical NumPy/Python reference for the 2-D advection-diffusion stencil.

The discretization is extracted from the original Master's counterflow
combustion project. Problem-specific nitrogen boundary conditions and
species clipping are deliberately excluded from this generic kernel.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


def flat_index(
    i: int,
    j: int,
    nx: int,
) -> int:
    """Return the x-contiguous flattened index k = i + j * nx."""

    return i + j * nx


def advance_scalar_legacy(
    phi_old: FloatArray,
    u: FloatArray,
    v: FloatArray,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    diffusivity: float,
    dt: float,
) -> FloatArray:
    """
    Advance one explicit advection-diffusion step.

    This intentionally uses Python loops so that it remains close to the
    historical Master's implementation rather than acting as an optimized
    NumPy implementation.
    """

    expected_size = nx * ny

    for name, array in (
        ("phi_old", phi_old),
        ("u", u),
        ("v", v),
    ):
        if array.ndim != 1:
            raise ValueError(
                f"{name} must be a flattened 1-D array."
            )

        if array.size != expected_size:
            raise ValueError(
                f"{name} has size {array.size}, "
                f"expected {expected_size}."
            )

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

    phi_new = phi_old.copy()

    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            center = flat_index(
                i,
                j,
                nx,
            )

            left = flat_index(
                i - 1,
                j,
                nx,
            )

            right = flat_index(
                i + 1,
                j,
                nx,
            )

            bottom = flat_index(
                i,
                j - 1,
                nx,
            )

            top = flat_index(
                i,
                j + 1,
                nx,
            )

            dphi_dx = (
                phi_old[right]
                - phi_old[left]
            ) * inv_2dx

            dphi_dy = (
                phi_old[top]
                - phi_old[bottom]
            ) * inv_2dy

            laplacian = (
                phi_old[right]
                - 2.0 * phi_old[center]
                + phi_old[left]
            ) * inv_dx2 + (
                phi_old[top]
                - 2.0 * phi_old[center]
                + phi_old[bottom]
            ) * inv_dy2

            advection = (
                u[center] * dphi_dx
                + v[center] * dphi_dy
            )

            phi_new[center] = (
                phi_old[center]
                - dt * advection
                + diffusivity
                * dt
                * laplacian
            )

    return phi_new
