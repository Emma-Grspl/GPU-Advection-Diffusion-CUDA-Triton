# Kernel Specification

## 1. Objective

This project studies the implementation and GPU optimization of a real two-dimensional advection-diffusion finite-difference stencil extracted from the validated `Counterflow-Combustion-CPP` solver.

The objective is not to redesign the numerical method for each backend.

All implementations must compute the same discrete operator before performance comparisons are considered valid.

The compared backends are:

* historical NumPy CPU reference;
* C++ CPU reference;
* PyTorch CUDA;
* CUDA C++;
* Triton.

FP64 is the primary scientific-computing precision.

FP32 is treated as a secondary precision/performance experiment.

---

## 2. Continuous equation

The transported scalar satisfies

$$
\frac{\partial \phi}{\partial t}
+
u\frac{\partial\phi}{\partial x}
+
v\frac{\partial\phi}{\partial y}
=
D\nabla^2\phi.
$$

Here:

* $\phi$ is the transported scalar;
* $u$ is the horizontal velocity;
* $v$ is the vertical velocity;
* $D$ is the diffusivity.

---

## 3. Spatial discretization

The first derivatives use centered finite differences:

$$
D_x\phi_{i,j}
=
\frac{
\phi_{i+1,j}-\phi_{i-1,j}
}{
2\Delta x
},
$$

$$
D_y\phi_{i,j}
=
\frac{
\phi_{i,j+1}-\phi_{i,j-1}
}{
2\Delta y
}.
$$

The Laplacian is

$$
\Delta_h\phi_{i,j}
=
\frac{
\phi_{i+1,j}
-2\phi_{i,j}
+\phi_{i-1,j}
}{
\Delta x^2
}
+
\frac{
\phi_{i,j+1}
-2\phi_{i,j}
+\phi_{i,j-1}
}{
\Delta y^2
}.
$$

---

## 4. Time discretization

A Forward Euler update is used:

$$
\phi_{i,j}^{n+1}
=
\phi_{i,j}^{n}
-
\Delta t
\left(
u_{i,j}D_x\phi_{i,j}
+
v_{i,j}D_y\phi_{i,j}
\right)
+
D\Delta t\,
\Delta_h\phi_{i,j}.
$$

Only interior points are updated:

$$
1\le i<N_x-1,
$$

$$
1\le j<N_y-1.
$$

---

## 5. Memory layout

All implementations use the flattened index

$$
k(i,j)=i+jN_x.
$$

Therefore the $x$ index is contiguous in memory.

For fixed $j$,

$$
k(i+1,j)=k(i,j)+1.
$$

For fixed $i$,

$$
k(i,j+1)=k(i,j)+N_x.
$$

This convention is inherited from the original NumPy Master's implementation and from the C++ Counterflow solver.

The memory layout is part of the benchmark specification and must not be changed independently by one backend.

---

## 6. Input and output buffers

The kernel receives:

* `phi_old`;
* `u`;
* `v`;
* `Nx`;
* `Ny`;
* `dx`;
* `dy`;
* `diffusivity`;
* `dt`.

The kernel produces:

* `phi_new`.

`phi_old` and `phi_new` must be distinct buffers.

No implementation may update `phi_old` in place.

---

## 7. Boundary convention

The benchmarked operator updates only interior cells.

Boundary values are preserved:

$$
\phi^{n+1}_{i,j}
=
\phi^n_{i,j}
$$

for all boundary points.

Problem-specific Counterflow boundary conditions are deliberately excluded from the kernel benchmark.

Species clipping is also excluded.

These operations are separate from the generic advection-diffusion stencil.

---

## 8. Precision

The primary benchmark uses

$$
\mathrm{FP64}.
$$

All CPU and GPU implementations must support FP64.

A secondary benchmark studies

$$
\mathrm{FP32}
$$

and quantifies the performance/accuracy trade-off.

Mixed precision is outside the initial project scope.

---

## 9. Correctness metrics

For a backend solution $\phi_b$ and a reference solution $\phi_r$, the relative $L^2$ error is

$$
E_2
=
\frac{
\|\phi_b-\phi_r\|_2
}{
\|\phi_r\|_2
}.
$$

The maximum absolute error is

$$
E_\infty
=
\max_{i,j}
|\phi_{b,i,j}-\phi_{r,i,j}|.
$$

Exact bitwise equality is not required across CPU and GPU backends because instruction ordering and fused multiply-add operations may differ.

---

## 10. Numerical verification

The discrete operator is verified independently of GPU performance.

The verification suite contains:

1. constant-field preservation;
2. pure diffusion;
3. pure advection for a single time step;
4. combined advection-diffusion;
5. negative velocity components;
6. anisotropic grids with $\Delta x\ne\Delta y$;
7. non-square domains;
8. non-power-of-two sizes;
9. deterministic random fields;
10. multi-step error propagation.

Pure centered advection with Forward Euler is not used as a long-time stability benchmark.

It is used only as an operator-correctness test.

---

## 11. Spatial convergence

For spatial verification, an analytical smooth field is used, for example

$$
\phi(x,y)
=
\sin(k_xx)\cos(k_yy).
$$

The exact continuous operator

$$
\mathcal{L}\phi
=
-u\phi_x
-v\phi_y
+
D\nabla^2\phi
$$

is compared with its discrete approximation $\mathcal{L}_h\phi$.

The centered finite-difference operator is expected to show approximately

$$
E(h)=O(h^2).
$$

The observed order is

$$
p
=
\log_2
\left(
\frac{E(h)}
{E(h/2)}
\right).
$$

---

## 12. Temporal convergence

Temporal convergence is measured using a stable analytical diffusion problem.

For

$$
u=v=0
$$

and

$$
\phi(x,y,0)
=
\sin(\pi x)\sin(\pi y),
$$

the exact solution is

$$
\phi(x,y,t)
=
\sin(\pi x)\sin(\pi y)
\exp(-2D\pi^2t).
$$

Forward Euler is expected to show

$$
E(\Delta t)=O(\Delta t).
$$

---

## 13. Performance measurements

Performance results are reported only after correctness tests pass.

The principal grid sizes are

$$
128^2,\;
256^2,\;
512^2,\;
1024^2,\;
2048^2,\;
4096^2,
$$

with larger grids added when hardware memory allows.

The original Counterflow scale

$$
50\times50
$$

is also retained to demonstrate the small-problem regime.

Additional irregular sizes are used for robustness, but not necessarily for headline performance results.

---

## 14. Kernel-only timing

GPU kernel timing excludes:

* allocation;
* initialization;
* host-to-device transfers;
* device-to-host transfers;
* file I/O.

Inputs are already resident on the device before timing starts.

CUDA timing uses GPU events rather than host wall-clock timing.

---

## 15. End-to-end timing

A separate experiment measures

$$
t_{\mathrm{E2E}}
=
t_{\mathrm{H2D}}
+
t_{\mathrm{kernel}}
+
t_{\mathrm{D2H}}.
$$

Kernel-only and end-to-end timings must never be mixed in the same performance comparison.

---

## 16. Performance metrics

The following quantities are reported:

* latency;
* grid points per second;
* effective bandwidth;
* speed-up relative to NumPy;
* speed-up relative to C++ CPU;
* speed-up relative to PyTorch CUDA;
* achieved memory throughput;
* occupancy;
* warp efficiency;
* register usage;
* shared-memory usage.

Speed-up is an observed result, not a pass/fail correctness criterion.

---

## 17. CUDA optimization sequence

The CUDA study contains three principal implementations.

### C0 — Global-memory baseline

A correct implementation using global memory with naturally coalesced access along the contiguous $x$ direction.

### C1 — Launch-geometry tuning

C0 is benchmarked across different CUDA block geometries.

Examples include:

* `8 x 8`;
* `16 x 8`;
* `32 x 8`;
* `16 x 16`;
* `32 x 16`.

The best configuration is selected using both timing and profiling metrics.

### C2 — Shared-memory tiling

Each block loads a tile of $\phi$ with a one-cell halo.

For a block of size

$$
B_x\times B_y,
$$

the corresponding shared-memory tile has size

$$
(B_x+2)\times(B_y+2).
$$

C2 is compared with C1 to determine whether reduced global-memory traffic compensates for synchronization and resource overhead.

---

## 18. Triton implementation

The Triton backend implements the same discrete operator and memory layout.

Its goal is not necessarily to outperform CUDA C++.

The study compares:

* correctness;
* latency;
* effective bandwidth;
* implementation complexity;
* profiling results.

---

## 19. Profiling

Nsight Systems is used for whole-application behavior:

* CUDA API calls;
* memory transfers;
* synchronization;
* kernel-launch overhead;
* GPU timeline.

Nsight Compute is used for kernel-level analysis:

* occupancy;
* DRAM throughput;
* memory workload;
* warp behavior;
* register pressure;
* shared-memory behavior;
* stall reasons;
* Speed-of-Light metrics.

---

## 20. Scope

This repository studies one operation only:

> the explicit two-dimensional advection-diffusion stencil.

It does not implement:

* the complete Counterflow solver;
* pressure projection;
* Navier-Stokes integration;
* combustion chemistry;
* NVIDIA Modulus;
* cuDNN;
* full application deployment.

The purpose is to isolate and understand GPU performance at kernel level.

