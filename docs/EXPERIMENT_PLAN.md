| Experiment            | Sizes                 | Precision | Backends     | Main metric        |
| --------------------- | --------------------- | --------- | ------------ | ------------------ |
| Correctness           | irregular + small     | FP64      | all          | \(E_2,E_\infty\)   |
| Spatial convergence   | 33→513                | FP64      | NumPy/C++    | order \(p_x\)      |
| Temporal convergence  | dt→dt/8               | FP64      | NumPy/C++    | order \(p_t\)      |
| Multi-step robustness | 1→1000 steps          | FP64/32   | all          | error growth       |
| Scaling               | 128²→4096²+           | FP64      | all          | latency            |
| Precision             | 512²→4096²            | 32/64     | GPU          | accuracy/speed     |
| CUDA blocks           | large grids           | FP64      | C0/C1        | latency, occupancy |
| Shared memory         | large grids           | FP64      | C1/C2        | DRAM traffic       |
| Kernel-only           | all performance sizes | 32/64     | GPU          | latency/GCell/s    |
| E2E                   | selected sizes        | 32/64     | GPU          | H2D+kernel+D2H     |
| Nsight                | 1024²/4096²           | FP64      | C0/C1/C2     | hardware metrics   |
| Roofline              | large grids           | FP64      | best kernels | FLOP/byte          |

