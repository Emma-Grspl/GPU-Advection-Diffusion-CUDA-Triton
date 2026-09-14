from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from reference.numpy.legacy_stencil import (
    advance_scalar_legacy,
)


@dataclass(frozen=True)
class Case:
    name: str

    nx: int
    ny: int

    dx: float
    dy: float

    diffusivity: float
    dt: float

    steps: int

    kind: str

    seed: int = 0


CASES = [
    Case(
        name="constant_field",
        nx=17,
        ny=13,
        dx=0.013,
        dy=0.021,
        diffusivity=0.004,
        dt=2.0e-4,
        steps=1,
        kind="constant",
    ),
    Case(
        name="pure_diffusion",
        nx=19,
        ny=17,
        dx=0.05,
        dy=0.04,
        diffusivity=0.01,
        dt=1.0e-3,
        steps=100,
        kind="diffusion",
    ),
    Case(
        name="pure_advection_one_step",
        nx=23,
        ny=15,
        dx=0.04,
        dy=0.07,
        diffusivity=0.0,
        dt=1.0e-4,
        steps=1,
        kind="advection",
    ),
    Case(
        name="combined_anisotropic",
        nx=31,
        ny=18,
        dx=0.02,
        dy=0.035,
        diffusivity=0.003,
        dt=1.0e-4,
        steps=100,
        kind="combined",
    ),
    Case(
        name="irregular_37x51",
        nx=37,
        ny=51,
        dx=0.017,
        dy=0.011,
        diffusivity=0.002,
        dt=5.0e-5,
        steps=10,
        kind="combined",
    ),
    Case(
        name="random_seed_0",
        nx=29,
        ny=21,
        dx=0.03,
        dy=0.025,
        diffusivity=0.001,
        dt=1.0e-5,
        steps=1,
        kind="random",
        seed=0,
    ),
    Case(
        name="random_seed_42",
        nx=63,
        ny=65,
        dx=0.01,
        dy=0.013,
        diffusivity=0.002,
        dt=1.0e-5,
        steps=10,
        kind="random",
        seed=42,
    ),
    Case(
        name="multistep_1000",
        nx=41,
        ny=27,
        dx=0.025,
        dy=0.04,
        diffusivity=0.0015,
        dt=2.0e-5,
        steps=1000,
        kind="combined",
    ),
]


def coordinates(
    case: Case,
):
    x = (
        np.arange(
            case.nx,
            dtype=np.float64,
        )
        * case.dx
    )

    y = (
        np.arange(
            case.ny,
            dtype=np.float64,
        )
        * case.dy
    )

    # Shape = (ny, nx).
    #
    # C-order flattening therefore makes x / i
    # the contiguous index:
    #
    #     k = i + j * nx
    #
    X, Y = np.meshgrid(
        x,
        y,
        indexing="xy",
    )

    return X, Y


def smooth_fields(
    case: Case,
):
    X, Y = coordinates(case)

    lx = max(
        (case.nx - 1) * case.dx,
        case.dx,
    )

    ly = max(
        (case.ny - 1) * case.dy,
        case.dy,
    )

    xhat = X / lx
    yhat = Y / ly

    phi = (
        0.5
        + 0.12
        * np.sin(
            2.0 * np.pi * xhat
        )
        * np.cos(
            np.pi * yhat
        )
        + 0.03 * xhat
        + 0.02 * yhat**2
    )

    u = (
        0.30
        + 0.05
        * np.cos(
            2.0 * np.pi * yhat
        )
    )

    v = (
        -0.20
        + 0.04
        * np.sin(
            2.0 * np.pi * xhat
        )
    )

    return (
        phi.ravel(order="C"),
        u.ravel(order="C"),
        v.ravel(order="C"),
    )


def build_inputs(
    case: Case,
):
    if case.kind == "constant":
        size = (
            case.nx * case.ny
        )

        return (
            np.full(
                size,
                3.25,
                dtype=np.float64,
            ),
            np.full(
                size,
                0.7,
                dtype=np.float64,
            ),
            np.full(
                size,
                -0.2,
                dtype=np.float64,
            ),
        )

    if case.kind == "random":
        rng = np.random.default_rng(
            case.seed
        )

        size = (
            case.nx * case.ny
        )

        phi = rng.uniform(
            0.2,
            0.8,
            size=size,
        ).astype(
            np.float64
        )

        u = rng.uniform(
            -0.5,
            0.5,
            size=size,
        ).astype(
            np.float64
        )

        v = rng.uniform(
            -0.5,
            0.5,
            size=size,
        ).astype(
            np.float64
        )

        return phi, u, v

    phi, u, v = smooth_fields(
        case
    )

    if case.kind == "diffusion":
        u.fill(0.0)
        v.fill(0.0)

    return phi, u, v


def run_numpy(
    case: Case,
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
):
    current = phi.copy()

    for _ in range(
        case.steps
    ):
        current = (
            advance_scalar_legacy(
                current,
                u,
                v,
                nx=case.nx,
                ny=case.ny,
                dx=case.dx,
                dy=case.dy,
                diffusivity=(
                    case.diffusivity
                ),
                dt=case.dt,
            )
        )

    return current


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
        writer = csv.writer(
            file
        )

        writer.writerow([
            "k",
            "phi",
            "u",
            "v",
        ])

        for k in range(
            phi.size
        ):
            writer.writerow([
                k,
                repr(float(phi[k])),
                repr(float(u[k])),
                repr(float(v[k])),
            ])


def read_cpp_output(
    path: Path,
):
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=1,
        dtype=np.float64,
    )

    return np.atleast_1d(
        values
    )


def run_cpp(
    case: Case,
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
):
    executable = (
        ROOT
        / "build"
        / "cpu_stencil_cli"
    )

    if not executable.exists():
        raise FileNotFoundError(
            "build/cpu_stencil_cli absent. "
            "Lance cmake --build build."
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        input_path = (
            tmpdir
            / "input.csv"
        )

        output_path = (
            tmpdir
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
                str(executable),
                str(case.nx),
                str(case.ny),
                repr(case.dx),
                repr(case.dy),
                repr(
                    case.diffusivity
                ),
                repr(case.dt),
                str(input_path),
                str(output_path),
                str(case.steps),
            ],
            check=True,
        )

        return read_cpp_output(
            output_path
        )


def error_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
):
    difference = (
        candidate
        - reference
    )

    reference_norm = (
        np.linalg.norm(
            reference
        )
    )

    absolute_norm = (
        np.linalg.norm(
            difference
        )
    )

    if reference_norm == 0.0:
        relative_l2 = (
            absolute_norm
        )
    else:
        relative_l2 = (
            absolute_norm
            / reference_norm
        )

    max_abs = float(
        np.max(
            np.abs(
                difference
            )
        )
    )

    return (
        float(relative_l2),
        max_abs,
    )


def stability_indicators(
    case: Case,
    u: np.ndarray,
    v: np.ndarray,
):
    advective = (
        case.dt
        * (
            np.max(
                np.abs(u)
            )
            / case.dx
            +
            np.max(
                np.abs(v)
            )
            / case.dy
        )
    )

    diffusive = (
        case.diffusivity
        * case.dt
        * (
            1.0 / case.dx**2
            +
            1.0 / case.dy**2
        )
    )

    return (
        float(advective),
        float(diffusive),
    )


def tolerances(
    steps: int,
):
    if steps == 1:
        return (
            5.0e-14,
            5.0e-14,
        )

    if steps <= 100:
        return (
            5.0e-13,
            5.0e-13,
        )

    return (
        5.0e-12,
        5.0e-12,
    )


def main():
    print(
        "CPU robustness suite"
    )

    print(
        "===================="
    )

    failures = []

    for case in CASES:
        phi, u, v = (
            build_inputs(case)
        )

        numpy_result = run_numpy(
            case,
            phi,
            u,
            v,
        )

        cpp_result = run_cpp(
            case,
            phi,
            u,
            v,
        )

        relative_l2, max_abs = (
            error_metrics(
                numpy_result,
                cpp_result,
            )
        )

        advective, diffusive = (
            stability_indicators(
                case,
                u,
                v,
            )
        )

        rel_tol, abs_tol = (
            tolerances(
                case.steps
            )
        )

        passed = (
            relative_l2 <= rel_tol
            and max_abs <= abs_tol
        )

        state = (
            "PASS"
            if passed
            else "FAIL"
        )

        print()
        print(
            f"[{state}] {case.name}"
        )

        print(
            f"  grid       : "
            f"{case.nx} x {case.ny}"
        )

        print(
            f"  steps      : "
            f"{case.steps}"
        )

        print(
            f"  C_adv      : "
            f"{advective:.6e}"
        )

        print(
            f"  C_diff     : "
            f"{diffusive:.6e}"
        )

        print(
            f"  rel L2     : "
            f"{relative_l2:.6e}"
        )

        print(
            f"  max abs    : "
            f"{max_abs:.6e}"
        )

        if not passed:
            failures.append(
                case.name
            )

    print()
    print(
        "Summary"
    )

    print(
        "-------"
    )

    print(
        f"Passed: "
        f"{len(CASES) - len(failures)}"
        f"/{len(CASES)}"
    )

    if failures:
        print(
            "Failed cases:"
        )

        for name in failures:
            print(
                f"  - {name}"
            )

        raise SystemExit(1)

    print(
        "ROBUSTNESS RESULT: PASS"
    )


if __name__ == "__main__":
    main()
