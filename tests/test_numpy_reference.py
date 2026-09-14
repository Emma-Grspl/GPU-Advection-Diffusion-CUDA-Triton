import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from reference.numpy.legacy_stencil import (
    advance_scalar_legacy,
    flat_index,
)


def test_flat_index_is_x_contiguous():
    nx = 11

    assert flat_index(0, 0, nx) == 0
    assert flat_index(1, 0, nx) == 1
    assert flat_index(0, 1, nx) == nx
    assert flat_index(7, 3, nx) == 7 + 3 * nx


def test_constant_field_is_preserved():
    nx = 17
    ny = 13

    phi = np.full(
        nx * ny,
        3.25,
        dtype=np.float64,
    )

    u = np.full(
        nx * ny,
        0.7,
        dtype=np.float64,
    )

    v = np.full(
        nx * ny,
        -0.2,
        dtype=np.float64,
    )

    result = advance_scalar_legacy(
        phi,
        u,
        v,
        nx=nx,
        ny=ny,
        dx=0.01,
        dy=0.015,
        diffusivity=1.0e-4,
        dt=1.0e-4,
    )

    np.testing.assert_array_equal(
        result,
        phi,
    )


def test_boundaries_are_preserved():
    nx = 9
    ny = 7

    rng = np.random.default_rng(42)

    phi = rng.normal(
        size=nx * ny,
    )

    u = rng.normal(
        size=nx * ny,
    )

    v = rng.normal(
        size=nx * ny,
    )

    result = advance_scalar_legacy(
        phi,
        u,
        v,
        nx=nx,
        ny=ny,
        dx=0.1,
        dy=0.2,
        diffusivity=0.01,
        dt=0.001,
    )

    for i in range(nx):
        assert (
            result[flat_index(i, 0, nx)]
            == phi[flat_index(i, 0, nx)]
        )

        assert (
            result[flat_index(i, ny - 1, nx)]
            == phi[flat_index(i, ny - 1, nx)]
        )

    for j in range(ny):
        assert (
            result[flat_index(0, j, nx)]
            == phi[flat_index(0, j, nx)]
        )

        assert (
            result[flat_index(nx - 1, j, nx)]
            == phi[flat_index(nx - 1, j, nx)]
        )
