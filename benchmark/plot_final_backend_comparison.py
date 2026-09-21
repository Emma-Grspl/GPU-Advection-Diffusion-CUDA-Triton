from pathlib import Path

import csv
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "benchmark"
    / "results"
    / "final_backend_comparison_v100_5backends.csv"
)

OUT = (
    ROOT
    / "assets"
    / "performance"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


with CSV_PATH.open(newline="") as f:
    rows = list(
        csv.DictReader(f)
    )


sizes = [
    int(row["nx"])
    for row in rows
]

labels = [
    f"{n}²"
    for n in sizes
]


def values(key):
    return [
        float(row[key])
        for row in rows
    ]


# --------------------------------------------------
# 1. Time per timestep
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    labels,
    values("cpp_cpu_us"),
    marker="o",
    label="C++ CPU",
)

plt.plot(
    labels,
    values("pytorch_eager_us"),
    marker="o",
    label="PyTorch eager",
)

plt.plot(
    labels,
    values("triton_eager_us"),
    marker="o",
    label="Triton eager",
)

plt.plot(
    labels,
    values("triton_graph_us"),
    marker="o",
    label="Triton + CUDA Graph",
)

plt.plot(
    labels,
    values("cuda_c0_us"),
    marker="o",
    label="CUDA C0",
)

plt.yscale("log")

plt.xlabel("Grid size")
plt.ylabel("Time per timestep [µs]")
plt.title(
    "Advection–diffusion timestep latency"
)
plt.grid(
    True,
    alpha=0.25,
)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT
    / "final_backend_time_per_step.png",
    dpi=200,
)

plt.savefig(
    OUT
    / "final_backend_time_per_step.pdf",
)

plt.close()


# --------------------------------------------------
# 2. Throughput
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

for key, label in [
    (
        "cpp_cpu_gcell_s",
        "C++ CPU",
    ),
    (
        "pytorch_eager_gcell_s",
        "PyTorch eager",
    ),
    (
        "triton_eager_gcell_s",
        "Triton eager",
    ),
    (
        "triton_graph_gcell_s",
        "Triton + CUDA Graph",
    ),
    (
        "cuda_c0_gcell_s",
        "CUDA C0",
    ),
]:
    plt.plot(
        labels,
        values(key),
        marker="o",
        label=label,
    )

plt.xlabel("Grid size")
plt.ylabel("Throughput [GCell/s]")
plt.title(
    "Backend throughput"
)
plt.grid(
    True,
    alpha=0.25,
)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT
    / "final_backend_throughput.png",
    dpi=200,
)

plt.savefig(
    OUT
    / "final_backend_throughput.pdf",
)

plt.close()


# --------------------------------------------------
# 3. GPU speedup versus PyTorch eager
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    labels,
    values(
        "triton_eager_vs_cpp_speedup"
    ),
    marker="o",
    label="Triton eager vs CPU",
)

plt.plot(
    labels,
    values(
        "triton_graph_vs_cpp_speedup"
    ),
    marker="o",
    label="Triton Graph vs CPU",
)

plt.plot(
    labels,
    values(
        "cuda_vs_cpp_speedup"
    ),
    marker="o",
    label="CUDA C0 vs CPU",
)

plt.xlabel("Grid size")
plt.ylabel("Speedup over C++ CPU")
plt.title(
    "GPU acceleration over single-threaded C++"
)
plt.grid(
    True,
    alpha=0.25,
)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT
    / "final_backend_speedup_vs_cpu.png",
    dpi=200,
)

plt.savefig(
    OUT
    / "final_backend_speedup_vs_cpu.pdf",
)

plt.close()


# --------------------------------------------------
# 4. Triton eager vs CUDA Graph
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    labels,
    values("triton_eager_us"),
    marker="o",
    label="Triton eager",
)

plt.plot(
    labels,
    values("triton_graph_us"),
    marker="o",
    label="Triton + CUDA Graph",
)

plt.plot(
    labels,
    values("cuda_c0_us"),
    marker="o",
    label="CUDA C0",
)

plt.xlabel("Grid size")
plt.ylabel("Time per timestep [µs]")
plt.title(
    "Triton dispatch overhead and CUDA Graph replay"
)
plt.grid(
    True,
    alpha=0.25,
)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT
    / "triton_graph_vs_eager_cuda.png",
    dpi=200,
)

plt.savefig(
    OUT
    / "triton_graph_vs_eager_cuda.pdf",
)

plt.close()


print(
    "Figures written to:",
    OUT,
)
