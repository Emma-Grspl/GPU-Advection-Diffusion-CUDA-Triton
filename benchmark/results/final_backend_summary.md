# Backend performance comparison

## Configuration

Precision:
- FP64

Backends:
- optimized single-threaded C++ baseline
- PyTorch 2.8 eager on CUDA
- custom CUDA C0 fused kernel

CUDA C0 configuration:
- block: 32 x 8
- one output cell per thread
- one fused kernel per timestep
- global-memory implementation

GPU:
- NVIDIA Tesla V100-SXM2

## Compute-only results

| Grid | C++ CPU | PyTorch eager | CUDA C0 | PyTorch / CPU | CUDA / PyTorch | CUDA / CPU |
|---|---:|---:|---:|---:|---:|---:|
| 512² | 581.45 us | 628.91 us | 10.87 us | 0.92x | 57.88x | 53.51x |
| 1024² | 4691.11 us | 633.88 us | 44.17 us | 7.40x | 14.35x | 106.20x |
| 2048² | 18658.59 us | 2182.52 us | 166.30 us | 8.55x | 13.12x | 112.20x |
| 4096² | 73156.28 us | 8475.38 us | 652.17 us | 8.63x | 13.00x | 112.17x |

## Interpretation

For the smallest tested grid, 512², PyTorch eager remains slightly
slower than the optimized single-threaded C++ implementation because
GPU dispatch and multiple eager tensor operations dominate the runtime.

From 1024² upward, PyTorch increasingly amortizes this overhead and
becomes approximately 7-9x faster than the CPU implementation.

The custom CUDA C0 backend performs the complete stencil update in one
fused kernel per timestep. In the large-grid regime, it reaches roughly
25-26 GCell/s and is approximately:

- 13x faster than PyTorch eager
- 112x faster than the optimized single-threaded C++ baseline

The performance difference between PyTorch eager and CUDA C0 is
primarily attributable to kernel fusion and reduced intermediate memory
traffic and launch overhead.

## CUDA optimization study

C1:
- block-size tuning
- no robust material improvement
- 32 x 8 retained

C2:
- explicit shared-memory tiling
- numerically identical to C0
- slower than C0
- rejected as an optimization

Final CUDA implementation:
- C0 global-memory kernel
- block = 32 x 8
- FP64
