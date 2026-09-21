from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triton_backend.stencil_t0 import advance_t0


def numpy_step(
    phi_old,
    u,
    v,
    dx,
    dy,
    diffusivity,
    dt,
):
    phi_new = phi_old.copy()

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
        - dt
        * (
            u[1:-1, 1:-1]
            * dphi_dx
            +
            v[1:-1, 1:-1]
            * dphi_dy
        )
        + diffusivity
        * dt
        * laplacian
    )

    return phi_new


def main():
    nx = 37
    ny = 51

    dx = 0.7 / (nx - 1)
    dy = 1.3 / (ny - 1)

    diffusivity = 1.0e-3
    dt = 1.0e-5

    x = np.linspace(
        0.0,
        1.0,
        nx,
        dtype=np.float64,
    )

    y = np.linspace(
        0.0,
        1.0,
        ny,
        dtype=np.float64,
    )

    yy, xx = np.meshgrid(
        y,
        x,
        indexing="ij",
    )

    phi0 = (
        np.sin(3.1 * xx)
        * np.cos(2.7 * yy)
        + 0.2 * xx * yy
    )

    u_np = (
        0.4
        + 0.1 * yy
    )

    v_np = (
        -0.2
        + 0.05 * xx
    )

    for steps in (1, 10, 1000):
        ref = phi0.copy()

        for _ in range(steps):
            ref = numpy_step(
                ref,
                u_np,
                v_np,
                dx,
                dy,
                diffusivity,
                dt,
            )

        phi_a = torch.tensor(
            phi0,
            device="cuda",
            dtype=torch.float64,
        ).contiguous()

        phi_b = torch.empty_like(
            phi_a
        )

        u = torch.tensor(
            u_np,
            device="cuda",
            dtype=torch.float64,
        ).contiguous()

        v = torch.tensor(
            v_np,
            device="cuda",
            dtype=torch.float64,
        ).contiguous()

        current_is_a = advance_t0(
            phi_a,
            phi_b,
            u,
            v,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
            steps=steps,
            current_is_a=True,
        )

        torch.cuda.synchronize()

        result = (
            phi_a
            if current_is_a
            else phi_b
        ).cpu().numpy()

        diff = result - ref

        rel_l2 = (
            np.linalg.norm(diff)
            / np.linalg.norm(ref)
        )

        linf = np.max(
            np.abs(diff)
        )

        passed = (
            rel_l2 <= 1.0e-12
            and linf <= 1.0e-12
        )

        print(
            f"[{'PASS' if passed else 'FAIL'}] "
            f"steps={steps} "
            f"rel_L2={rel_l2:.6e} "
            f"Linf={linf:.6e}"
        )

        if not passed:
            raise SystemExit(1)

    print()
    print(
        "TRITON T0 VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()
