# GPU backend comparison — V100-SXM2-16GB

All timings use FP64 and resident GPU data.

## CUDA C0

- custom fused CUDA kernel
- block: 32 x 8
- one kernel launch per timestep
- global-memory implementation
- C1 block tuning showed no material improvement
- C2 shared-memory tiling was slower

## PyTorch eager

- PyTorch 2.8.0
- FP64
- eager execution
- multiple tensor operations and kernel launches per timestep
- CUDA Events used for GPU elapsed time

## Results

| Grid | CUDA C0 | PyTorch eager | CUDA speedup |
|---|---:|---:|---:|
| 512² | 10.866 us | 628.912 us | 57.88x |
| 1024² | 44.172 us | 633.882 us | 14.35x |
| 2048² | 166.302 us | 2182.523 us | 13.12x |
| 4096² | 652.173 us | 8475.385 us | 13.00x |

At 4096²:

- CUDA C0: ~25.73 GCell/s
- PyTorch eager: ~1.98 GCell/s

The custom CUDA kernel is therefore approximately 13x faster than
PyTorch eager in the large-grid regime.

The particularly large gap at 512² is primarily a latency effect:
PyTorch eager executes the stencil as several tensor operations and
therefore incurs multiple GPU kernel launches per timestep, whereas
CUDA C0 evaluates the complete stencil in one fused kernel.
