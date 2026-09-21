# Triton optimization study — V100-SXM2-16GB

## Environment

- GPU: NVIDIA Tesla V100-SXM2-16GB
- Compute capability: 7.0
- PyTorch: 2.8.0
- Triton: 3.4.0
- Precision: FP64
- Stencil: 2-D advection-diffusion
- Memory layout: x-index contiguous
- Ping-pong state buffers
- Boundaries preserved exactly

Despite the V100 not belonging to the currently advertised modern
Triton hardware baseline, Triton 3.4.0 successfully compiled and
executed the kernels in the Jean Zay PyTorch environment.

## T0 — baseline Triton kernel

The baseline implementation maps one Triton program to a contiguous
1-D block of flattened grid cells.

The complete advection-diffusion update is fused into one kernel per
timestep.

Runtime scalar coefficients were explicitly typed as `tl.float64`.
Without this explicit typing, Python floating-point scalar arguments
were initially lowered with insufficient precision and produced errors
around 1e-12.

After the FP64 correction:

- 1 step: exact agreement with NumPy
- 10 steps: Linf = 1.39e-17
- 1000 steps: Linf = 1.11e-16

T0 is therefore validated numerically in FP64.

## T1 — launch-configuration tuning

The following `(BLOCK_SIZE, num_warps)` configurations were evaluated:

- 64 / 2
- 128 / 4
- 256 / 4
- 256 / 8
- 512 / 4
- 512 / 8
- 1024 / 8

A cyclic measurement order was used to reduce ordering and GPU-state
bias.

The configuration

    BLOCK_SIZE = 256
    num_warps  = 8

was retained as the best global compromise.

For large grids it reaches approximately 25-26 GCell/s, essentially
matching the hand-written CUDA C0 implementation.

### Same-GPU comparison

| Grid | CUDA C0 [us] | Triton T1 eager [us] |
|---:|---:|---:|
| 512²  | 10.874 | 35.779 |
| 1024² | 44.125 | 44.142 |
| 2048² | 166.328 | 166.336 |
| 4096² | 652.171 | 653.707 |

From 1024² upward, CUDA and Triton differ by less than approximately
0.3%.

The 512² discrepancy was investigated with Nsight Systems rather than
being attributed directly to the generated kernel.

## Nsight Systems diagnosis

For 512², Nsight Systems reports:

- Triton kernel median: 10.208 us
- `cuLaunchKernelEx` median host API duration: 6.076 us
- eager iterative timestep: approximately 35.8 us

For 4096²:

- Triton kernel median: 645.628 us
- `cuLaunchKernelEx` median host API duration: 6.259 us
- eager iterative timestep: approximately 653.7 us

The 512² slowdown is therefore not caused by poor intrinsic GPU kernel
performance.

For short kernels, repeated Python/Triton dispatch leaves significant
inter-launch gaps on the GPU timeline. For large kernels, host-side
submission can be overlapped with the much longer GPU execution and
the overhead becomes effectively hidden.

The intrinsic Triton stencil kernel is therefore comparable in
performance to the hand-written CUDA kernel.

## T2 — CUDA Graph replay

To remove repeated host-side dispatch, 100 complete Triton timesteps
were captured in a CUDA Graph.

An even graph length preserves the ping-pong state convention:
after 100 steps, the current state is again stored in the same static
buffer.

### Numerical validation

CUDA Graph replay was compared directly with eager Triton execution
for 1000 timesteps.

Results:

- 37x51: bitwise exact
- 512x512: bitwise exact

For both cases:

    rel_L2 = 0
    Linf   = 0

CUDA Graph replay therefore preserves the numerical trajectory exactly.

### Performance

| Grid | Triton eager [us/step] | CUDA Graph [us/step] | Speedup |
|---:|---:|---:|---:|
| 512²  | 36.104 | 9.674   | 3.732x |
| 1024² | 44.161 | 43.060  | 1.026x |
| 2048² | 166.168 | 165.097 | 1.006x |
| 4096² | 653.931 | 652.845 | 1.002x |

CUDA Graph timings are amortized per timestep over graph replays
containing 100 timesteps.

At 512², CUDA Graph replay removes most of the dispatch bottleneck and
increases throughput from approximately 7.26 GCell/s to 27.10 GCell/s.

For larger grids the benefit becomes progressively smaller because the
GPU kernel execution already dominates the timestep cost.

## Final conclusion

The optimization study separates two fundamentally different effects:

1. GPU kernel efficiency.
2. Host-side orchestration overhead.

Triton generates an FP64 stencil kernel whose intrinsic GPU performance
is essentially equivalent to the hand-written CUDA implementation on
the tested V100.

For sufficiently large grids, eager Triton already achieves this
performance directly.

For smaller workloads, repeated Python-side kernel dispatch becomes the
dominant bottleneck. Capturing the iterative workload in a CUDA Graph
removes most of this overhead and restores near-kernel-limited
performance.

The final retained Triton configuration is:

    BLOCK_SIZE = 256
    num_warps  = 8

with CUDA Graph replay as an optional execution strategy for repeated
fixed-shape timesteps.
