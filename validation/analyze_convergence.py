from __future__ import annotations

from pathlib import Path
import csv
import subprocess
import sys
import tempfile

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


RESULT_DIR = (
    ROOT
    / "validation"
    / "results"
)

CPP_EXECUTABLE = (
    ROOT
    / "build"
    / "cpu_stencil_cli"
)


# ============================================================
# Generic backend utilities
# ============================================================

def write_cpp_input(
    path: Path,
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
):
    with path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "k",
            "phi",
            "u",
            "v",
        ])

        for k in range(phi.size):
            writer.writerow([
                k,
                format(
                    float(phi[k]),
                    ".17g",
                ),
                format(
                    float(u[k]),
                    ".17g",
                ),
                format(
                    float(v[k]),
                    ".17g",
                ),
            ])


def run_cpp(
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    diffusivity: float,
    dt: float,
    steps: int,
):
    if not CPP_EXECUTABLE.exists():
        raise FileNotFoundError(
            "build/cpu_stencil_cli absent. "
            "Lance d'abord cmake --build build."
        )

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)

        input_path = (
            temp_dir
            / "input.csv"
        )

        output_path = (
            temp_dir
            / "output.csv"
        )

        write_cpp_input(
            input_path,
            phi,
            u,
            v,
        )

        subprocess.run(
            [
                str(CPP_EXECUTABLE),
                str(nx),
                str(ny),
                repr(dx),
                repr(dy),
                repr(diffusivity),
                repr(dt),
                str(input_path),
                str(output_path),
                str(steps),
            ],
            check=True,
        )

        result = np.loadtxt(
            output_path,
            delimiter=",",
            skiprows=1,
            usecols=1,
            dtype=np.float64,
        )

    return np.atleast_1d(
        result
    )


def run_numpy(
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    diffusivity: float,
    dt: float,
    steps: int,
):
    current = phi.copy()

    for _ in range(steps):
        current = advance_scalar_legacy(
            current,
            u,
            v,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
        )

    return current



def run_torch(
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    diffusivity: float,
    dt: float,
    steps: int,
):
    current = torch.from_numpy(
        phi.copy()
    )

    velocity_u = torch.from_numpy(
        u.copy()
    )

    velocity_v = torch.from_numpy(
        v.copy()
    )

    for _ in range(steps):
        current = advance_scalar_torch(
            current,
            velocity_u,
            velocity_v,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
        )

    return (
        current
        .detach()
        .cpu()
        .numpy()
    )

def relative_l2(
    candidate: np.ndarray,
    reference: np.ndarray,
):
    return float(
        np.linalg.norm(
            candidate - reference
        )
        / np.linalg.norm(reference)
    )


def observed_order(
    previous_error: float,
    current_error: float,
):
    return float(
        np.log2(
            previous_error
            / current_error
        )
    )


# ============================================================
# Spatial convergence
# ============================================================

def spatial_convergence():
    print()
    print("Spatial convergence")
    print("===================")

    sizes = [
        17,
        33,
        65,
        129,
        257,
    ]

    lx = 1.3
    ly = 0.9

    velocity_x = 0.70
    velocity_y = -0.35

    diffusivity = 0.02

    # This is an operator probe, not a physical
    # time-integration experiment.
    #
    # Using dt = 1 avoids subtractive amplification when
    # recovering L_h(phi) from:
    #
    #     phi_new = phi_old + dt * L_h(phi)
    #
    dt = 1.0

    kx = (
        2.0
        * np.pi
        / lx
    )

    ky = (
        np.pi
        / ly
    )

    rows = []

    previous_numpy_error = None
    previous_cpp_error = None
    previous_torch_error = None

    for n in sizes:
        nx = n
        ny = n

        dx = (
            lx
            / (nx - 1)
        )

        dy = (
            ly
            / (ny - 1)
        )

        x = np.linspace(
            0.0,
            lx,
            nx,
            dtype=np.float64,
        )

        y = np.linspace(
            0.0,
            ly,
            ny,
            dtype=np.float64,
        )

        X, Y = np.meshgrid(
            x,
            y,
            indexing="xy",
        )

        phi_2d = (
            np.sin(kx * X)
            * np.cos(ky * Y)
        )

        dphi_dx = (
            kx
            * np.cos(kx * X)
            * np.cos(ky * Y)
        )

        dphi_dy = (
            -ky
            * np.sin(kx * X)
            * np.sin(ky * Y)
        )

        laplacian = (
            -(
                kx**2
                + ky**2
            )
            * phi_2d
        )

        exact_operator = (
            -velocity_x
            * dphi_dx
            - velocity_y
            * dphi_dy
            + diffusivity
            * laplacian
        )

        phi = phi_2d.ravel(
            order="C"
        )

        u = np.full(
            nx * ny,
            velocity_x,
            dtype=np.float64,
        )

        v = np.full(
            nx * ny,
            velocity_y,
            dtype=np.float64,
        )

        numpy_result = run_numpy(
            phi,
            u,
            v,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
            steps=1,
        )

        cpp_result = run_cpp(
            phi,
            u,
            v,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
            steps=1,
        )

        torch_result = run_torch(
            phi,
            u,
            v,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
            steps=1,
        )

        numerical_numpy = (
            (
                numpy_result
                - phi
            )
            / dt
        ).reshape(
            ny,
            nx,
        )

        numerical_cpp = (
            (
                cpp_result
                - phi
            )
            / dt
        ).reshape(
            ny,
            nx,
        )

        numerical_torch = (
            (
                torch_result
                - phi
            )
            / dt
        ).reshape(
            ny,
            nx,
        )

        # Boundaries are intentionally excluded because
        # advance_scalar() preserves them rather than applying
        # the differential operator there.
        interior = (
            np.s_[
                1:-1,
                1:-1,
            ]
        )

        exact_interior = (
            exact_operator[
                interior
            ]
        )

        numpy_interior = (
            numerical_numpy[
                interior
            ]
        )

        cpp_interior = (
            numerical_cpp[
                interior
            ]
        )

        torch_interior = (
            numerical_torch[
                interior
            ]
        )

        numpy_error = relative_l2(
            numpy_interior,
            exact_interior,
        )

        cpp_error = relative_l2(
            cpp_interior,
            exact_interior,
        )

        torch_error = relative_l2(
            torch_interior,
            exact_interior,
        )

        backend_error = relative_l2(
            cpp_interior,
            numpy_interior,
        )

        torch_backend_error = relative_l2(
            torch_interior,
            numpy_interior,
        )

        numpy_order = (
            None
            if previous_numpy_error is None
            else observed_order(
                previous_numpy_error,
                numpy_error,
            )
        )

        cpp_order = (
            None
            if previous_cpp_error is None
            else observed_order(
                previous_cpp_error,
                cpp_error,
            )
        )

        torch_order = (
            None
            if previous_torch_error is None
            else observed_order(
                previous_torch_error,
                torch_error,
            )
        )

        rows.append({
            "n": n,
            "dx": dx,
            "dy": dy,
            "numpy_error": numpy_error,
            "cpp_error": cpp_error,
            "torch_error": torch_error,
            "backend_error": backend_error,
            "torch_backend_error": torch_backend_error,
            "numpy_order": numpy_order,
            "cpp_order": cpp_order,
            "torch_order": torch_order,
        })

        print(
            f"N={n:3d} | "
            f"E_np={numpy_error:.6e} | "
            f"E_cpp={cpp_error:.6e} | "
            f"E_pt={torch_error:.6e} | "
            f"p_cpp="
            f"{'-' if cpp_order is None else f'{cpp_order:.6f}'} | "
            f"p_pt="
            f"{'-' if torch_order is None else f'{torch_order:.6f}'}"
        )

        previous_numpy_error = (
            numpy_error
        )

        previous_cpp_error = (
            cpp_error
        )

        previous_torch_error = (
            torch_error
        )

    cpp_orders = [
        row["cpp_order"]
        for row in rows
        if row["cpp_order"]
        is not None
    ]

    torch_orders = [
        row["torch_order"]
        for row in rows
        if row["torch_order"]
        is not None
    ]

    cpp_mean_order = float(
        np.mean(
            cpp_orders[-3:]
        )
    )

    torch_mean_order = float(
        np.mean(
            torch_orders[-3:]
        )
    )

    max_torch_backend_error = max(
        row["torch_backend_error"]
        for row in rows
    )

    passed = (
        1.90
        <= cpp_mean_order
        <= 2.10
        and
        1.90
        <= torch_mean_order
        <= 2.10
        and
        max_torch_backend_error
        <= 5.0e-14
    )

    print()
    print(
        f"C++ asymptotic order   : "
        f"{cpp_mean_order:.6f}"
    )

    print(
        f"PyTorch asymptotic order: "
        f"{torch_mean_order:.6f}"
    )

    print(
        f"Max PyTorch/NumPy error : "
        f"{max_torch_backend_error:.6e}"
    )

    print(
        "Spatial result       : "
        + (
            "PASS"
            if passed
            else "FAIL"
        )
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        RESULT_DIR
        / "spatial_convergence.csv"
    )

    with output.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "n",
                "dx",
                "dy",
                "numpy_error",
                "cpp_error",
                "torch_error",
                "backend_error",
                "torch_backend_error",
                "numpy_order",
                "cpp_order",
                "torch_order",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return passed


# ============================================================
# Temporal convergence
# ============================================================

def temporal_convergence():
    print()
    print("Temporal convergence")
    print("====================")

    nx = 33
    ny = 33

    lx = 1.0
    ly = 1.0

    dx = (
        lx
        / (nx - 1)
    )

    dy = (
        ly
        / (ny - 1)
    )

    diffusivity = 0.01

    final_time = 0.1

    step_counts = [
        5,
        10,
        20,
        40,
        80,
    ]

    x = np.linspace(
        0.0,
        lx,
        nx,
        dtype=np.float64,
    )

    y = np.linspace(
        0.0,
        ly,
        ny,
        dtype=np.float64,
    )

    X, Y = np.meshgrid(
        x,
        y,
        indexing="xy",
    )

    phi0_2d = (
        np.sin(
            np.pi * X / lx
        )
        * np.sin(
            np.pi * Y / ly
        )
    )

    # Enforce exact homogeneous Dirichlet boundaries instead
    # of retaining sin(pi) roundoff values.
    phi0_2d[0, :] = 0.0
    phi0_2d[-1, :] = 0.0
    phi0_2d[:, 0] = 0.0
    phi0_2d[:, -1] = 0.0

    phi0 = phi0_2d.ravel(
        order="C"
    )

    u = np.zeros(
        nx * ny,
        dtype=np.float64,
    )

    v = np.zeros(
        nx * ny,
        dtype=np.float64,
    )

    # Exact eigenvalue of the discrete five-point Laplacian.
    #
    # Using the exact semi-discrete solution isolates temporal
    # error from spatial truncation error.
    lambda_x = (
        -4.0
        / dx**2
        * np.sin(
            np.pi
            / (
                2.0
                * (nx - 1)
            )
        )**2
    )

    lambda_y = (
        -4.0
        / dy**2
        * np.sin(
            np.pi
            / (
                2.0
                * (ny - 1)
            )
        )**2
    )

    lambda_h = (
        lambda_x
        + lambda_y
    )

    exact_2d = (
        phi0_2d
        * np.exp(
            diffusivity
            * lambda_h
            * final_time
        )
    )

    exact = exact_2d[
        1:-1,
        1:-1,
    ]

    rows = []

    previous_numpy_error = None
    previous_cpp_error = None
    previous_torch_error = None

    for steps in step_counts:
        dt = (
            final_time
            / steps
        )

        numpy_result = run_numpy(
            phi0,
            u,
            v,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
            steps=steps,
        )

        cpp_result = run_cpp(
            phi0,
            u,
            v,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
            steps=steps,
        )

        torch_result = run_torch(
            phi0,
            u,
            v,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            diffusivity=diffusivity,
            dt=dt,
            steps=steps,
        )

        numpy_interior = (
            numpy_result.reshape(
                ny,
                nx,
            )[1:-1, 1:-1]
        )

        cpp_interior = (
            cpp_result.reshape(
                ny,
                nx,
            )[1:-1, 1:-1]
        )

        torch_interior = (
            torch_result.reshape(
                ny,
                nx,
            )[1:-1, 1:-1]
        )

        numpy_error = relative_l2(
            numpy_interior,
            exact,
        )

        cpp_error = relative_l2(
            cpp_interior,
            exact,
        )

        torch_error = relative_l2(
            torch_interior,
            exact,
        )

        backend_error = relative_l2(
            cpp_interior,
            numpy_interior,
        )

        torch_backend_error = relative_l2(
            torch_interior,
            numpy_interior,
        )

        numpy_order = (
            None
            if previous_numpy_error is None
            else observed_order(
                previous_numpy_error,
                numpy_error,
            )
        )

        cpp_order = (
            None
            if previous_cpp_error is None
            else observed_order(
                previous_cpp_error,
                cpp_error,
            )
        )

        torch_order = (
            None
            if previous_torch_error is None
            else observed_order(
                previous_torch_error,
                torch_error,
            )
        )

        c_diff = (
            diffusivity
            * dt
            * (
                1.0 / dx**2
                + 1.0 / dy**2
            )
        )

        rows.append({
            "steps": steps,
            "dt": dt,
            "c_diff": c_diff,
            "numpy_error": numpy_error,
            "cpp_error": cpp_error,
            "torch_error": torch_error,
            "backend_error": backend_error,
            "torch_backend_error": torch_backend_error,
            "numpy_order": numpy_order,
            "cpp_order": cpp_order,
            "torch_order": torch_order,
        })

        print(
            f"steps={steps:3d} | "
            f"dt={dt:.6e} | "
            f"C_diff={c_diff:.6f} | "
            f"E_cpp={cpp_error:.6e} | "
            f"E_pt={torch_error:.6e} | "
            f"p_cpp="
            f"{'-' if cpp_order is None else f'{cpp_order:.6f}'} | "
            f"p_pt="
            f"{'-' if torch_order is None else f'{torch_order:.6f}'}"
        )

        previous_numpy_error = (
            numpy_error
        )

        previous_cpp_error = (
            cpp_error
        )

        previous_torch_error = (
            torch_error
        )

    cpp_orders = [
        row["cpp_order"]
        for row in rows
        if row["cpp_order"]
        is not None
    ]

    torch_orders = [
        row["torch_order"]
        for row in rows
        if row["torch_order"]
        is not None
    ]

    cpp_mean_order = float(
        np.mean(
            cpp_orders[-3:]
        )
    )

    torch_mean_order = float(
        np.mean(
            torch_orders[-3:]
        )
    )

    max_torch_backend_error = max(
        row["torch_backend_error"]
        for row in rows
    )

    passed = (
        0.95
        <= cpp_mean_order
        <= 1.10
        and
        0.95
        <= torch_mean_order
        <= 1.10
        and
        max_torch_backend_error
        <= 5.0e-12
    )

    print()
    print(
        f"C++ asymptotic order   : "
        f"{cpp_mean_order:.6f}"
    )

    print(
        f"PyTorch asymptotic order: "
        f"{torch_mean_order:.6f}"
    )

    print(
        f"Max PyTorch/NumPy error : "
        f"{max_torch_backend_error:.6e}"
    )

    print(
        "Temporal result      : "
        + (
            "PASS"
            if passed
            else "FAIL"
        )
    )

    output = (
        RESULT_DIR
        / "temporal_convergence.csv"
    )

    with output.open(
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "steps",
                "dt",
                "c_diff",
                "numpy_error",
                "cpp_error",
                "torch_error",
                "backend_error",
                "torch_backend_error",
                "numpy_order",
                "cpp_order",
                "torch_order",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return passed


# ============================================================
# Main
# ============================================================

def main():
    print(
        "Analytical convergence study"
    )

    print(
        "============================"
    )

    spatial_pass = (
        spatial_convergence()
    )

    temporal_pass = (
        temporal_convergence()
    )

    print()
    print("Summary")
    print("-------")

    print(
        "Spatial  O(h^2)  : "
        + (
            "PASS"
            if spatial_pass
            else "FAIL"
        )
    )

    print(
        "Temporal O(dt)   : "
        + (
            "PASS"
            if temporal_pass
            else "FAIL"
        )
    )

    if not (
        spatial_pass
        and temporal_pass
    ):
        raise SystemExit(1)

    print()
    print(
        "CONVERGENCE RESULT: PASS"
    )


if __name__ == "__main__":
    main()
