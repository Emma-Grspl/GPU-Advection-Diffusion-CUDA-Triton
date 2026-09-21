# CUDA optimization summary — V100

## C0 — global-memory baseline

Configuration:

- FP64
- block: 32 x 8 = 256 threads
- one output cell per thread
- centered advection-diffusion stencil
- separate old/new buffers
- x-index mapped to threadIdx.x

Validation:

- NumPy/C++/PyTorch/CUDA robustness suite: PASS
- CUDA cases: 8/8 PASS
- irregular grids: PASS
- multistep 1000: PASS

Performance:

- plateau around 23–26 GCell/s for large grids
- 4096 x 4096 Nsight Systems, V100-SXM2-16GB:
  - median kernel time: 642.755 us
  - mean kernel time: 643.008 us
  - min: 641.731 us
  - max: 647.171 us
  - stddev: 1.440 us
  - registers/thread: 30
  - dynamic shared memory: 0

## C1 — block geometry sweep

Tested:

- 8 x 8
- 16 x 8
- 32 x 8
- 16 x 16
- 32 x 16

Grid sizes:

- 512^2
- 1024^2
- 2048^2
- 4096^2

Result:

No block geometry gives a robust material improvement across grid
sizes. Differences on large grids remain around the sub-percent level.

Decision:

Keep the original 32 x 8 configuration.

## C2 — explicit shared-memory tile

Design:

- phi tile loaded cooperatively into shared memory
- one-cell halo
- u and v remain in global memory
- block geometry unchanged at 32 x 8
- tile size:
  (32 + 2) x (8 + 2) = 340 doubles
  = 2720 bytes/block

Validation:

- C0 vs C2, irregular 37 x 51 grid
- 1 step: exact match
- 10 steps: exact match
- 1000 steps: exact match
- result: PASS

A/B CUDA Events benchmark:

C2 was slower than C0 for every tested grid:

- 512^2:  -5.27%
- 1024^2: -1.66%
- 2048^2: -1.27%
- 4096^2: -1.77%

Same-GPU Nsight Systems comparison, V100-SXM2-16GB:

C0:
- median: 642.755 us
- registers/thread: 30
- dynamic shared memory: 0

C2:
- median: 661.028 us
- registers/thread: 28
- dynamic shared memory: ~2.66 KiB/block

C2 therefore increases kernel latency by approximately 2.8% in the
instrumented same-GPU comparison.

## Conclusion

Explicit shared-memory tiling does not improve this stencil on the V100.

The global-memory C0 implementation already benefits sufficiently from
hardware caching and coalesced accesses. The extra cooperative tile
loading, halo handling, and synchronization introduced by C2 outweigh
the reduction in explicit global-memory neighbor loads.

Final CUDA implementation retained for subsequent comparisons:

    C0 global-memory kernel
    block = 32 x 8
    FP64
