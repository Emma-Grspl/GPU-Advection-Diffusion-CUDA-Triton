# GPU Advection Diffusion with CUDA and Triton

This project studies how a real scientific stencil behaves on CPU and GPU.

The stencil comes from my C++ counterflow combustion solver (https://github.com/Emma-Grspl/Counterflow-Combustion-CPP) and solves a 2D advection diffusion equation. The goal is simple: implement the same numerical update with several backends, check that they give the same result, then measure where the computation spends time.

The project compares:

- NumPy
- C++ on one CPU thread
- PyTorch on GPU
- CUDA C++
- Triton
- Triton with CUDA Graphs

The main tests were run in FP64 on an NVIDIA Tesla V100 SXM2 16 GB.

---

## Main results

The CUDA and Triton kernels reach about 25 to 26 GCell/s on large grids.

For grids from `1024 x 1024` upward, CUDA and Triton have almost the same performance.

At `512 x 512`, Triton is slower in eager execution even though the GPU kernel itself is already fast. Nsight Systems showed that most of the extra time comes from launching many short kernels from Python.

Using CUDA Graphs reduces the Triton cost at `512 x 512` from about 36.1 microseconds to 9.67 microseconds per timestep, which gives a 3.73x speedup.

A second CUDA kernel using shared memory was also tested. It was slightly slower than the simpler CUDA kernel, so the simpler version was kept.

![Backend time per step](assets/performance/final_backend_time_per_step.png)

![Backend throughput](assets/performance/final_backend_throughput.png)

![GPU speedup over CPU](assets/performance/final_backend_speedup_vs_cpu.png)

---

## Numerical problem

The code solves the 2D advection diffusion equation

$$ \frac{\partial \phi}{\partial t} + u\frac{\partial \phi}{\partial x} + v\frac{\partial \phi}{\partial y} = D\nabla^2\phi.$$

Here:

- $\phi(x,y,t)$ is the transported scalar field
- $u(x,y)$ and $v(x,y)$ are the velocity components
- $D$ is the diffusion coefficient

The same centered finite difference update is used in every implementation.

All backends use:

- FP64 arithmetic for the main study
- row major storage
- contiguous x direction memory layout
- separate old and new state arrays
- ping pong time integration
- fixed boundary values
- independent `Nx`, `Ny`, `dx`, and `dy`

This makes the performance comparison meaningful because every backend solves the same numerical problem.

---

## Implementations

| Backend | Purpose |
| --- | --- |
| NumPy | Numerical reference |
| C++ CPU | CPU performance reference |
| PyTorch | Standard GPU tensor implementation |
| CUDA C0 | Main CUDA kernel |
| CUDA C1 | CUDA block size study |
| CUDA C2 | Shared memory experiment |
| Triton T0 | First Triton kernel |
| Triton T1 | Tuned Triton kernel |
| Triton T2 | Triton with CUDA Graphs |

---

## Numerical validation

Performance was measured only after checking correctness.

The CUDA implementation was compared with the NumPy reference on:

- constant fields
- diffusion only cases
- advection only cases
- combined advection and diffusion
- irregular grid sizes
- anisotropic grids
- random initial conditions
- integrations up to 1000 timesteps

The CUDA results stayed at FP64 numerical precision.

The Triton implementation was also checked against the reference. After using `tl.float64` for runtime scalar coefficients, the maximum error remained around machine precision.

For 1000 timesteps, CUDA Graph execution and eager Triton produced identical results on both `37 x 51` and `512 x 512` grids.

---

## CUDA study

### C0: simple CUDA kernel

The first CUDA version uses one thread per grid cell and one fused kernel per timestep.

The retained block size is:

```text
blockDim = (32, 8)
threads per block = 256
```

For large grids, this kernel reaches about 25 to 26 GCell/s on the V100.

### C1: block size study

Several block sizes were tested:

```text
8 x 8
16 x 8
32 x 8
16 x 16
32 x 16
```

For large grids, the performance difference stays below about one percent.

The original `32 x 8` configuration was kept because it gives stable performance and maps well to contiguous x direction memory access.

![CUDA block sweep](assets/performance/cuda_block_sweep.png)

### C2: shared memory experiment

A second CUDA kernel loads the stencil tile into shared memory before computing the update.

For a `32 x 8` block, the shared tile contains a one cell halo around the block.

The result was correct, but the kernel was slightly slower than C0.

The extra work comes from:

- loading the halo
- extra indexing
- synchronizing the block

Nsight also showed that register use was not the cause of the slowdown.

This experiment was kept in the repository because it is a useful negative result: shared memory does not automatically make a stencil faster.

---

## Triton study

The Triton tests use:

```text
PyTorch 2.8.0
Triton 3.4.0
CUDA 12.8 runtime
NVIDIA Tesla V100 SXM2 16 GB
```

### T0: first Triton kernel

The first Triton version computes the complete stencil in one kernel.

A precision issue appeared during validation: Python scalar coefficients had to be explicitly converted to `tl.float64`.

After this change, the Triton result matched the reference at machine precision.

### T1: launch tuning

The following configurations were tested:

```text
64   / 2 warps
128  / 4 warps
256  / 4 warps
256  / 8 warps
512  / 4 warps
512  / 8 warps
1024 / 8 warps
```

The retained configuration is:

```text
BLOCK_SIZE = 256
num_warps  = 8
```

Representative results are:

| Grid | CUDA C0 | Triton eager |
| ---: | ---: | ---: |
| `512 x 512` | 10.874 microseconds | 35.779 microseconds |
| `1024 x 1024` | 44.125 microseconds | 44.142 microseconds |
| `2048 x 2048` | 166.328 microseconds | 166.336 microseconds |
| `4096 x 4096` | 652.171 microseconds | 653.707 microseconds |

From `1024 x 1024` upward, CUDA and Triton differ by less than about **0.3 percent**.

### Nsight analysis at 512 x 512

The `512 x 512` case looked much slower in eager Triton, so it was profiled with Nsight Systems.

The profile showed:

```text
Triton kernel median      about 10.21 microseconds
CUDA launch call median   about  6.08 microseconds
eager timestep            about 35.8 microseconds
```

The GPU kernel itself is already close to the CUDA kernel. The main problem is the time between short kernel launches.

For larger grids, the kernel runs long enough that this launch cost becomes much less important.

---

## CUDA Graphs

The workload repeats the same kernel many times with fixed tensor shapes and fixed device memory addresses.

This makes it a good candidate for CUDA Graphs.

The final version captures 100 timesteps and replays them as one graph.

| Grid | Triton eager | Triton with CUDA Graphs | Speedup |
| ---: | ---: | ---: | ---: |
| `512 x 512` | 36.104 microseconds | **9.674 microseconds** | **3.73x** |
| `1024 x 1024` | 44.161 microseconds | **43.060 microseconds** | 1.026x |
| `2048 x 2048` | 166.168 microseconds | **165.097 microseconds** | 1.006x |
| `4096 x 4096` | 653.931 microseconds | **652.845 microseconds** | 1.002x |

At `512 x 512`, throughput rises from about **7.26 GCell/s** to about **27.10 GCell/s**.

![Triton eager and CUDA Graph comparison](assets/performance/triton_graph_vs_eager_cuda.png)

This confirms that the small grid slowdown came mainly from repeated launch overhead, not from a slow Triton GPU kernel.

---

## What I learned

This project gave me a practical view of several GPU performance topics:

- memory layout
- CUDA thread blocks
- GPU kernel fusion
- block size tuning
- shared memory
- launch overhead
- Nsight profiling
- Triton kernel tuning
- CUDA Graphs

The main result is that the best optimization depends on the problem size.

For large grids, CUDA and Triton are limited mainly by the GPU work itself and reach almost the same throughput.

For smaller repeated workloads, launch overhead becomes important and CUDA Graphs can make a large difference.

The shared memory experiment also showed that a more complex kernel is not always a faster kernel.

---

## Reproducing the results

### Build the CUDA code

```bash
cmake -S . -B build-cuda
cmake --build build-cuda -j
```

### CUDA validation

```bash
python validation/validate_cuda_backend.py
./build-cuda/validate_cuda_shared
```

### CUDA benchmarks

```bash
./build-cuda/benchmark_cuda_c0
./build-cuda/benchmark_cuda_blocks
./build-cuda/benchmark_cuda_c0_vs_c2
```

### CUDA profiling

```bash
./build-cuda/profile_cuda_c0
./build-cuda/profile_cuda_c2
```

### C++ CPU benchmark

```bash
benchmark/benchmark_cpp_cpu
```

### PyTorch benchmark

```bash
python benchmark/benchmark_pytorch_eager.py
```

### Triton validation

```bash
python triton_backend/validate_t0.py
```

### Triton benchmark and tuning

```bash
python triton_backend/benchmark_t0.py
python triton_backend/benchmark_t1_sweep.py
```

### Triton profiling

```bash
python triton_backend/profile_t1.py --n 512
python triton_backend/profile_t1.py --n 4096
```

### CUDA Graph validation and benchmark

```bash
python triton_backend/validate_t2_cudagraph.py
python triton_backend/benchmark_t2_cudagraph.py
```

---

## Repository structure

```text
GPU-Advection-Diffusion-CUDA-Triton/
|
|-- cpu/
|   |-- include/
|   `-- src/
|
|-- cuda/
|   |-- include/
|   `-- src/
|
|-- triton_backend/
|
|-- benchmark/
|   `-- results/
|
|-- validation/
|   `-- results/
|
|-- assets/
|   `-- performance/
|
|-- CMakeLists.txt
`-- README.md
```

---

## Project origin

The stencil comes from my C++ counterflow combustion solver:

`Counterflow-Combustion-CPP`

That solver is itself a C++ version of my Master's project, originally written in Python.

This GPU project therefore follows the same scientific problem across several stages:

```text
Python and NumPy
        |
        v
C++ CPU
        |
        v
CUDA C++
        |
        v
Triton
```

The goal was not only to make the code faster, but also to understand where the time is spent and which GPU optimizations are useful for this stencil.

---

## Final configurations

### CUDA

```text
Precision       FP64
Kernel          one fused stencil kernel per timestep
Block size      32 x 8
Threads/block   256
Memory          global memory stencil reads
```

### Triton

```text
Precision       FP64
BLOCK_SIZE      256
num_warps       8
Kernel          one fused stencil kernel per timestep
```

CUDA Graphs can be added when many short timesteps are repeated with fixed shapes.

---

## Conclusion

The simple CUDA kernel reaches about 25 to 26 GCell/s on the V100.

Triton reaches almost the same performance on large grids.

At `512 x 512`, the Triton kernel itself is already fast, but repeated Python launches add a large cost. CUDA Graphs remove most of that cost and reduce the timestep from about 36.1 microseconds to 9.67 microseconds.

The shared memory version does not improve performance on this stencil, which is also an important result.

Overall, this project shows how validation, profiling, CUDA, Triton and CUDA Graphs can be used together to understand and improve the performance of a real scientific stencil.
