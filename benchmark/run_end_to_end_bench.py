from __future__ import annotations

import csv
import math
import statistics
import subprocess
import time
from pathlib import Path

REPO = Path.cwd()

CPU_EXE = REPO / "build" / "cpu_stencil_cli"

if not CPU_EXE.exists():
    CPU_EXE = (
        REPO
        / "build-cuda"
        / "cpu_stencil_cli"
    )

CUDA_EXE = (
    REPO
    / "build-cuda"
    / "cuda_stencil_cli"
)

if not CPU_EXE.exists():
    raise SystemExit(
        f"CPU executable not found: {CPU_EXE}"
    )
if not CUDA_EXE.exists():
    raise SystemExit(f"CUDA executable not found: {CUDA_EXE}")

TMP_ROOT = Path("/tmp/gpu_advection_diffusion_bench")
INPUT_DIR = TMP_ROOT / "inputs"
OUTPUT_DIR = TMP_ROOT / "outputs"

RESULT_DIR = REPO / "benchmark" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = RESULT_DIR / "end_to_end_v100.csv"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tailles raisonnables pour un benchmark provisoire avec I/O CSV
SIZES = [256, 512, 1024]

# Paramètres numériques stables
DIFFUSIVITY = 5.0e-4
DT = 1.0e-4
STEPS = 200

WARMUP = 1
REPEATS = 5


def generate_input_csv(nx: int, ny: int) -> Path:
    path = INPUT_DIR / f"input_{nx}x{ny}.csv"
    if path.exists():
        return path

    print(f"[generate] {path}")
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "phi", "u", "v"])

        for j in range(ny):
            y = 0.0 if ny == 1 else j / (ny - 1)
            for i in range(nx):
                x = 0.0 if nx == 1 else i / (nx - 1)
                k = j * nx + i

                phi = (
                    math.exp(-40.0 * ((x - 0.30) ** 2 + (y - 0.60) ** 2))
                    + 0.25 * math.exp(-60.0 * ((x - 0.75) ** 2 + (y - 0.25) ** 2))
                )
                u = 0.50 + 0.10 * math.sin(2.0 * math.pi * y)
                v = -0.20 + 0.10 * math.cos(2.0 * math.pi * x)

                writer.writerow([k, f"{phi:.17g}", f"{u:.17g}", f"{v:.17g}"])

    return path


def run_once(executable: Path, nx: int, ny: int, input_csv: Path, output_csv: Path) -> float:
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    cmd = [
        str(executable),
        str(nx),
        str(ny),
        f"{dx:.17g}",
        f"{dy:.17g}",
        f"{DIFFUSIVITY:.17g}",
        f"{DT:.17g}",
        str(input_csv),
        str(output_csv),
        str(STEPS),
    ]

    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    rows = []

    configs = [
        ("cpu", CPU_EXE),
        ("cuda", CUDA_EXE),
    ]

    print("End-to-end benchmark V100")
    print("=========================")
    print(f"steps      : {STEPS}")
    print(f"diffusivity: {DIFFUSIVITY}")
    print(f"dt         : {DT}")
    print(f"warmup     : {WARMUP}")
    print(f"repeats    : {REPEATS}")
    print()

    for nx in SIZES:
        ny = nx
        input_csv = generate_input_csv(nx, ny)

        for backend, exe in configs:
            print(f"[{backend}] {nx} x {ny}")

            for iwarm in range(WARMUP):
                output_csv = OUTPUT_DIR / f"{backend}_{nx}x{ny}_warmup_{iwarm+1}.csv"
                elapsed = run_once(exe, nx, ny, input_csv, output_csv)
                print(f"  warmup {iwarm+1}: {elapsed:.6f} s")

            samples = []
            for irun in range(REPEATS):
                output_csv = OUTPUT_DIR / f"{backend}_{nx}x{ny}_run_{irun+1}.csv"
                elapsed = run_once(exe, nx, ny, input_csv, output_csv)
                samples.append(elapsed)
                rows.append({
                    "backend": backend,
                    "nx": nx,
                    "ny": ny,
                    "steps": STEPS,
                    "run": irun + 1,
                    "seconds": elapsed,
                })
                print(f"  run {irun+1}: {elapsed:.6f} s")

            mean_value = statistics.mean(samples)
            median_value = statistics.median(samples)
            std_value = statistics.pstdev(samples) if len(samples) > 1 else 0.0

            print(f"  mean   : {mean_value:.6f} s")
            print(f"  median : {median_value:.6f} s")
            print(f"  std    : {std_value:.6f} s")
            print()

    with RESULT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["backend", "nx", "ny", "steps", "run", "seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved raw timings to:")
    print(RESULT_CSV)

    print()
    print("Median speedups (CPU / CUDA)")
    print("============================")

    by_key = {}
    for row in rows:
        key = (row["backend"], row["nx"], row["ny"])
        by_key.setdefault(key, []).append(row["seconds"])

    for nx in SIZES:
        cpu_times = by_key[("cpu", nx, nx)]
        cuda_times = by_key[("cuda", nx, nx)]
        cpu_med = statistics.median(cpu_times)
        cuda_med = statistics.median(cuda_times)
        speedup = cpu_med / cuda_med
        print(
            f"{nx:4d} x {nx:<4d} : "
            f"CPU median = {cpu_med:.6f} s | "
            f"CUDA median = {cuda_med:.6f} s | "
            f"speedup = {speedup:.3f}x"
        )


if __name__ == "__main__":
    main()
