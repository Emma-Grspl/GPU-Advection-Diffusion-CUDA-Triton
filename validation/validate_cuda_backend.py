from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from validation.run_robustness_suite import (
    CASES,
    build_inputs,
    error_metrics,
    read_cpp_output,
    run_numpy,
    tolerances,
    write_cpp_input,
)


CUDA_EXECUTABLE = (
    ROOT
    / "build-cuda"
    / "cuda_stencil_cli"
)


def run_cuda(
    case,
    phi,
    u,
    v,
):
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
                str(CUDA_EXECUTABLE),
                str(case.nx),
                str(case.ny),
                repr(case.dx),
                repr(case.dy),
                repr(case.diffusivity),
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


def main():
    if not CUDA_EXECUTABLE.exists():
        print(
            "CUDA VALIDATION: SKIPPED"
        )

        print(
            f"Executable absent: {CUDA_EXECUTABLE}"
        )

        print(
            "Build the CUDA backend on an NVIDIA "
            "system before numerical validation."
        )

        return

    print(
        "CUDA backend validation"
    )

    print(
        "======================="
    )

    failures = []

    for case in CASES:
        phi, u, v = (
            build_inputs(case)
        )

        reference = run_numpy(
            case,
            phi,
            u,
            v,
        )

        candidate = run_cuda(
            case,
            phi,
            u,
            v,
        )

        relative_l2, max_abs = (
            error_metrics(
                reference,
                candidate,
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

        print(
            f"[{state}] "
            f"{case.name:28s} "
            f"L2={relative_l2:.6e} "
            f"Linf={max_abs:.6e}"
        )

        if not passed:
            failures.append(
                case.name
            )

    print()

    print(
        f"Passed: "
        f"{len(CASES) - len(failures)}"
        f"/{len(CASES)}"
    )

    if failures:
        print(
            "CUDA VALIDATION RESULT: FAIL"
        )

        raise SystemExit(1)

    print(
        "CUDA VALIDATION RESULT: PASS"
    )


if __name__ == "__main__":
    main()
