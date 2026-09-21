from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results"
FIGURES = ROOT / "assets" / "performance"

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark file: {path}"
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def save_figure(
    fig: plt.Figure,
    stem: str,
) -> None:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"

    fig.tight_layout()

    fig.savefig(
        png,
        dpi=220,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_backend_time() -> None:
    rows = read_csv(
        RESULTS
        / "final_backend_comparison_v100.csv"
    )

    n = [
        int(row["nx"])
        for row in rows
    ]

    cpp = [
        float(row["cpp_cpu_us"])
        for row in rows
    ]

    pytorch = [
        float(row["pytorch_eager_us"])
        for row in rows
    ]

    cuda = [
        float(row["cuda_c0_us"])
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        n,
        cpp,
        marker="o",
        linewidth=2,
        label="C++ CPU",
    )

    ax.plot(
        n,
        pytorch,
        marker="s",
        linewidth=2,
        label="PyTorch eager GPU",
    )

    ax.plot(
        n,
        cuda,
        marker="^",
        linewidth=2,
        label="CUDA C0",
    )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Grid size N for N × N"
    )

    ax.set_ylabel(
        "Median time per timestep [µs]"
    )

    ax.set_title(
        "Advection–diffusion stencil — compute time"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend()

    save_figure(
        fig,
        "backend_time_per_step",
    )


def plot_backend_throughput() -> None:
    rows = read_csv(
        RESULTS
        / "final_backend_comparison_v100.csv"
    )

    n = [
        int(row["nx"])
        for row in rows
    ]

    cpp = [
        float(row["cpp_cpu_gcell_per_s"])
        for row in rows
    ]

    pytorch = [
        float(row["pytorch_gcell_per_s"])
        for row in rows
    ]

    cuda = [
        float(row["cuda_gcell_per_s"])
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        n,
        cpp,
        marker="o",
        linewidth=2,
        label="C++ CPU",
    )

    ax.plot(
        n,
        pytorch,
        marker="s",
        linewidth=2,
        label="PyTorch eager GPU",
    )

    ax.plot(
        n,
        cuda,
        marker="^",
        linewidth=2,
        label="CUDA C0",
    )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xlabel(
        "Grid size N for N × N"
    )

    ax.set_ylabel(
        "Throughput [GCell/s]"
    )

    ax.set_title(
        "Advection–diffusion stencil — throughput"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    save_figure(
        fig,
        "backend_throughput",
    )


def plot_speedups() -> None:
    rows = read_csv(
        RESULTS
        / "final_backend_comparison_v100.csv"
    )

    n = [
        int(row["nx"])
        for row in rows
    ]

    labels = [
        f"{value}²"
        for value in n
    ]

    pytorch_vs_cpp = [
        float(
            row["pytorch_speedup_vs_cpp"]
        )
        for row in rows
    ]

    cuda_vs_pytorch = [
        float(
            row["cuda_speedup_vs_pytorch"]
        )
        for row in rows
    ]

    cuda_vs_cpp = [
        float(
            row["cuda_speedup_vs_cpp"]
        )
        for row in rows
    ]

    x = list(
        range(len(labels))
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(8.0, 4.8)
    )

    ax.bar(
        [
            value - width
            for value in x
        ],
        pytorch_vs_cpp,
        width=width,
        label="PyTorch / C++",
    )

    ax.bar(
        x,
        cuda_vs_pytorch,
        width=width,
        label="CUDA / PyTorch",
    )

    ax.bar(
        [
            value + width
            for value in x
        ],
        cuda_vs_cpp,
        width=width,
        label="CUDA / C++",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_xlabel(
        "Grid size"
    )

    ax.set_ylabel(
        "Speedup [×]"
    )

    ax.set_title(
        "Backend speedups"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    save_figure(
        fig,
        "backend_speedups",
    )


def plot_c0_vs_c2() -> None:
    rows = read_csv(
        RESULTS
        / "cuda_c0_vs_c2_v100.csv"
    )

    n = [
        int(row["nx"])
        for row in rows
    ]

    labels = [
        f"{value}²"
        for value in n
    ]

    c0 = [
        float(row["c0_median_us"])
        for row in rows
    ]

    c2 = [
        float(row["c2_median_us"])
        for row in rows
    ]

    change = [
        float(row["c2_percent_change"])
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    ax.plot(
        n,
        c0,
        marker="o",
        linewidth=2,
        label="C0 global memory",
    )

    ax.plot(
        n,
        c2,
        marker="s",
        linewidth=2,
        label="C2 shared memory",
    )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Grid size N for N × N"
    )

    ax.set_ylabel(
        "Median time per timestep [µs]"
    )

    ax.set_title(
        "CUDA C0 vs C2"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend()

    save_figure(
        fig,
        "cuda_c0_vs_c2_time",
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    bars = ax.bar(
        labels,
        change,
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Grid size"
    )

    ax.set_ylabel(
        "C2 performance change [%]"
    )

    ax.set_title(
        "Shared-memory tiling impact"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        change,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            value,
            f"{value:.2f}%",
            ha="center",
            va=(
                "top"
                if value < 0.0
                else "bottom"
            ),
        )

    save_figure(
        fig,
        "cuda_c2_relative_change",
    )


def plot_block_sweep() -> None:
    rows = read_csv(
        RESULTS
        / "cuda_block_sweep_balanced_v100.csv"
    )

    configurations = []

    for row in rows:
        config = (
            int(row["block_x"]),
            int(row["block_y"]),
        )

        if config not in configurations:
            configurations.append(
                config
            )

    fig, ax = plt.subplots(
        figsize=(7.6, 4.8)
    )

    for block_x, block_y in configurations:
        selected = [
            row
            for row in rows
            if int(row["block_x"])
            == block_x
            and int(row["block_y"])
            == block_y
        ]

        selected.sort(
            key=lambda row: int(
                row["nx"]
            )
        )

        n = [
            int(row["nx"])
            for row in selected
        ]

        throughput = [
            float(
                row["gcell_per_s"]
            )
            for row in selected
        ]

        ax.plot(
            n,
            throughput,
            marker="o",
            linewidth=1.8,
            label=(
                f"{block_x}×{block_y}"
            ),
        )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xlabel(
        "Grid size N for N × N"
    )

    ax.set_ylabel(
        "Throughput [GCell/s]"
    )

    ax.set_title(
        "CUDA block-geometry sweep"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        title="Block",
    )

    save_figure(
        fig,
        "cuda_block_sweep",
    )


def main() -> None:
    plot_backend_time()
    plot_backend_throughput()
    plot_speedups()
    plot_c0_vs_c2()
    plot_block_sweep()

    print()
    print(
        f"All figures written to: {FIGURES}"
    )


if __name__ == "__main__":
    main()
