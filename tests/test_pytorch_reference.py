from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from pytorch.stencil import (
    advance_scalar_torch,
)

from reference.numpy.legacy_stencil import (
    advance_scalar_legacy,
)


def numpy_to_torch(
    array: np.ndarray,
) -> torch.Tensor:
    return torch.from_numpy(
        array.copy()
    )


def test_constant_field_is_preserved():
    nx = 17
    ny = 13

    size = nx * ny

    phi = torch.full(
        (size,),
        3.25,
        dtype=torch.float64,
    )

    u = torch.full(
        (size,),
        0.7,
        dtype=torch.float64,
    )

    v = torch.full(
        (size,),
        -0.2,
        dtype=torch.float64,
    )

    result = advance_scalar_torch(
        phi,
        u,
        v,
        nx=nx,
        ny=ny,
        dx=0.013,
        dy=0.021,
        diffusivity=0.004,
        dt=2.0e-4,
    )

    torch.testing.assert_close(
        result,
        phi,
        rtol=0.0,
        atol=0.0,
    )


def test_boundaries_are_preserved():
    nx = 11
    ny = 9

    generator = torch.Generator()
    generator.manual_seed(42)

    phi = torch.rand(
        nx * ny,
        generator=generator,
        dtype=torch.float64,
    )

    u = torch.rand(
        nx * ny,
        generator=generator,
        dtype=torch.float64,
    )

    v = torch.rand(
        nx * ny,
        generator=generator,
        dtype=torch.float64,
    )

    result = advance_scalar_torch(
        phi,
        u,
        v,
        nx=nx,
        ny=ny,
        dx=0.03,
        dy=0.07,
        diffusivity=0.002,
        dt=1.0e-4,
    )

    phi_2d = phi.reshape(
        ny,
        nx,
    )

    result_2d = result.reshape(
        ny,
        nx,
    )

    torch.testing.assert_close(
        result_2d[0, :],
        phi_2d[0, :],
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        result_2d[-1, :],
        phi_2d[-1, :],
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        result_2d[:, 0],
        phi_2d[:, 0],
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        result_2d[:, -1],
        phi_2d[:, -1],
        rtol=0.0,
        atol=0.0,
    )


def test_matches_numpy_fp64():
    nx = 37
    ny = 51

    dx = 0.017
    dy = 0.011

    diffusivity = 0.002
    dt = 5.0e-5

    rng = np.random.default_rng(
        1234
    )

    phi_np = rng.uniform(
        0.2,
        0.8,
        size=nx * ny,
    ).astype(
        np.float64
    )

    u_np = rng.uniform(
        -0.5,
        0.5,
        size=nx * ny,
    ).astype(
        np.float64
    )

    v_np = rng.uniform(
        -0.5,
        0.5,
        size=nx * ny,
    ).astype(
        np.float64
    )

    numpy_result = (
        advance_scalar_legacy(
            phi_np,
            u_np,
            v_np,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
        )
    )

    torch_result = (
        advance_scalar_torch(
            numpy_to_torch(
                phi_np
            ),
            numpy_to_torch(
                u_np
            ),
            numpy_to_torch(
                v_np
            ),
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
        )
        .numpy()
    )

    np.testing.assert_allclose(
        torch_result,
        numpy_result,
        rtol=5.0e-14,
        atol=5.0e-14,
    )


def test_multistep_matches_numpy_fp64():
    nx = 41
    ny = 27

    dx = 0.025
    dy = 0.04

    diffusivity = 0.0015
    dt = 2.0e-5

    steps = 100

    x = np.arange(
        nx,
        dtype=np.float64,
    ) * dx

    y = np.arange(
        ny,
        dtype=np.float64,
    ) * dy

    X, Y = np.meshgrid(
        x,
        y,
        indexing="xy",
    )

    lx = (
        (nx - 1)
        * dx
    )

    ly = (
        (ny - 1)
        * dy
    )

    phi_np = (
        0.5
        + 0.1
        * np.sin(
            2.0
            * np.pi
            * X
            / lx
        )
        * np.cos(
            np.pi
            * Y
            / ly
        )
    ).ravel(
        order="C"
    )

    u_np = np.full(
        nx * ny,
        0.3,
        dtype=np.float64,
    )

    v_np = np.full(
        nx * ny,
        -0.2,
        dtype=np.float64,
    )

    numpy_current = (
        phi_np.copy()
    )

    torch_current = (
        numpy_to_torch(
            phi_np
        )
    )

    torch_u = numpy_to_torch(
        u_np
    )

    torch_v = numpy_to_torch(
        v_np
    )

    for _ in range(steps):
        numpy_current = (
            advance_scalar_legacy(
                numpy_current,
                u_np,
                v_np,
                nx=nx,
                ny=ny,
                dx=dx,
                dy=dy,
                diffusivity=diffusivity,
                dt=dt,
            )
        )

        torch_current = (
            advance_scalar_torch(
                torch_current,
                torch_u,
                torch_v,
                nx=nx,
                ny=ny,
                dx=dx,
                dy=dy,
                diffusivity=diffusivity,
                dt=dt,
            )
        )

    difference = (
        torch_current.numpy()
        - numpy_current
    )

    relative_l2 = (
        np.linalg.norm(
            difference
        )
        / np.linalg.norm(
            numpy_current
        )
    )

    max_abs = np.max(
        np.abs(
            difference
        )
    )

    assert (
        relative_l2
        < 5.0e-12
    )

    assert (
        max_abs
        < 5.0e-12
    )


def test_float32_is_supported():
    nx = 9
    ny = 7

    phi = torch.ones(
        nx * ny,
        dtype=torch.float32,
    )

    u = torch.zeros(
        nx * ny,
        dtype=torch.float32,
    )

    v = torch.zeros(
        nx * ny,
        dtype=torch.float32,
    )

    result = advance_scalar_torch(
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

    assert (
        result.dtype
        == torch.float32
    )
