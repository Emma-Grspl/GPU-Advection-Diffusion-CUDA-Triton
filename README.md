# GPU-Advection-Diffusion-CUDA-Triton
GPU optimization study of a real 2D advection–diffusion stencil extracted from a validated C++ reactive-flow solver, comparing CPU, PyTorch, CUDA C++ and Triton implementations with Nsight profiling.

## Performance results

All GPU timings reported below exclude initialization, host-device
transfers and file I/O. Fields remain resident on the GPU throughout
the timed iterative computation.

### Backend comparison

![Final backend latency](assets/performance/final_backend_time_per_step.png)

The optimized single-threaded C++ implementation provides the CPU
reference baseline. PyTorch eager executes the same stencil using
multiple tensor operations, while CUDA and Triton fuse the complete
update into a single GPU kernel per timestep.

For large grids, both the hand-written CUDA C0 kernel and the Triton
kernel sustain approximately 25–26 GCell/s on the tested
V100-SXM2-16GB.

![Final backend throughput](assets/performance/final_backend_throughput.png)

![GPU speedup over CPU](assets/performance/final_backend_speedup_vs_cpu.png)

### Triton optimization

The final Triton launch configuration is:

- BLOCK_SIZE = 256
- num_warps = 8
- FP64 arithmetic

Nsight Systems showed that the intrinsic Triton kernel performance is
comparable to the hand-written CUDA implementation.

At 512², the Triton kernel itself executes in approximately 10.2 us,
but repeated eager launches from the Python loop increase the measured
iterative timestep to approximately 36 us.

Capturing 100 fixed-shape timesteps in a CUDA Graph removes most of
this host-side dispatch overhead:

- eager Triton: 36.10 us/step
- CUDA Graph replay: 9.67 us/step
- speedup: 3.73x

The CUDA Graph result is an amortized per-timestep cost over graph
replays containing 100 timesteps.

For larger grids, kernel execution dominates and the benefit of graph
replay becomes small.

![Triton CUDA Graph](assets/performance/triton_graph_vs_eager_cuda.png)

CUDA Graph replay was validated against eager Triton for 1000
timesteps and produced bitwise-identical results on both an irregular
37x51 grid and a 512x512 grid.

### CUDA optimization study

CUDA block-size tuning produced only sub-percent differences for large
grids, so the 32x8 launch configuration was retained.

The explicit shared-memory C2 implementation was also evaluated.
For a 32x8 block, the tile occupies 2720 bytes, approximately
2.66 KiB of shared memory per block.

C2 remained slower than the global-memory C0 implementation. The
additional cooperative halo loads and block synchronization outweighed
the benefit of explicit shared-memory reuse for this stencil on V100.

The retained CUDA implementation is therefore:

- FP64
- one fused kernel per timestep
- 32x8 thread blocks
- coalesced x-direction accesses
- global-memory stencil reads
