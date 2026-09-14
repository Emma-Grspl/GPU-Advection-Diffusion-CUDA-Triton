from __future__ import annotations

from pathlib import Path
import csv
import io
import subprocess
import sys

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


NX = 13
NY = 11

DX = 0.013
DY = 0.021

DIFFUSIVITY = 0.004
DT = 0.0002

RELATIVE_TOLERANCE = 5.0e-14
ABSOLUTE_TOLERANCE = 5.0e-14


def build_reference_inputs():
    size = NX * NY

    phi_old = np.zeros(
        size,
        dtype=np.float64,
    )

    u = np.zeros(
        size,
        dtype=np.float64,
    )

    v = np.zeros(
        size,
        dtype=np.float64,
    )

    for j in range(NY):
        for i in range(NX):
            k = flat_index(
                i,
                j,
                NX,
            )

            x_index = float(i)
            y_index = float(j)

            phi_old[k] = (
                0.25
                + 0.01 * x_index
                + 0.02 * y_index
                + 0.001
                * x_index
                * y_index
                + 0.0003
                * x_index
                * x_index
                - 0.0002
                * y_index
                * y_index
            )

            u[k] = (
                0.30
                + 0.02 * x_index
                - 0.01 * y_index
            )

            v[k] = (
                -0.20
                + 0.005 * x_index
                + 0.015 * y_index
            )

    return phi_old, u, v


def run_cpp_reference():
    executable = (
        ROOT
        / "build"
        / "cpu_reference_case"
    )

    if not executable.exists():
        raise FileNotFoundError(
            "Executable C++ absent. "
            "Lance d'abord cmake --build build."
        )

    completed = subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )

    reader = csv.DictReader(
        io.StringIO(
            completed.stdout
        )
    )

    rows = list(reader)

    if len(rows) != NX * NY:
        raise RuntimeError(
            f"Le C++ a produit {len(rows)} valeurs, "
            f"attendu : {NX * NY}."
        )

    values = np.empty(
        NX * NY,
        dtype=np.float64,
    )

    for row in rows:
        k = int(row["k"])

        values[k] = float(
            row["phi_new"]
        )

    return values


def main():
    phi_old, u, v = (
        build_reference_inputs()
    )

    numpy_result = (
        advance_scalar_legacy(
            phi_old,
            u,
            v,
            nx=NX,
            ny=NY,
            dx=DX,
            dy=DY,
            diffusivity=DIFFUSIVITY,
            dt=DT,
        )
    )

    cpp_result = (
        run_cpp_reference()
    )

    difference = (
        cpp_result
        - numpy_result
    )

    absolute_error = np.abs(
        difference
    )

    max_abs = float(
        np.max(
            absolute_error
        )
    )

    denominator = float(
        np.linalg.norm(
            numpy_result
        )
    )

    relative_l2 = float(
        np.linalg.norm(
            difference
        )
        / denominator
    )

    exact_matches = int(
        np.count_nonzero(
            cpp_result
            == numpy_result
        )
    )

    total_values = (
        NX * NY
    )

    worst_index = int(
        np.argmax(
            absolute_error
        )
    )

    worst_i = (
        worst_index % NX
    )

    worst_j = (
        worst_index // NX
    )

    print(
        "NumPy <-> C++ CPU equivalence"
    )

    print(
        "============================"
    )

    print(
        f"Grid              : "
        f"{NX} x {NY}"
    )

    print(
        f"dx / dy           : "
        f"{DX} / {DY}"
    )

    print(
        f"Values compared   : "
        f"{total_values}"
    )

    print(
        f"Exact FP64 matches: "
        f"{exact_matches}/{total_values}"
    )

    print()

    print(
        f"Relative L2 error : "
        f"{relative_l2:.16e}"
    )

    print(
        f"Max absolute error: "
        f"{max_abs:.16e}"
    )

    print(
        f"Worst location    : "
        f"(i={worst_i}, j={worst_j})"
    )

    print(
        f"NumPy value       : "
        f"{numpy_result[worst_index]:.17g}"
    )

    print(
        f"C++ value         : "
        f"{cpp_result[worst_index]:.17g}"
    )

    print()

    relative_pass = (
        relative_l2
        <= RELATIVE_TOLERANCE
    )

    absolute_pass = (
        max_abs
        <= ABSOLUTE_TOLERANCE
    )

    if (
        relative_pass
        and absolute_pass
    ):
        print(
            "RESULT: PASS"
        )

        return

    print(
        "RESULT: FAIL"
    )

    raise SystemExit(1)


if __name__ == "__main__":
    main()
