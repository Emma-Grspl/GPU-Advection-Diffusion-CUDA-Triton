#pragma once

#include <cuda_runtime.h>

#include <cstddef>


namespace stencil
{

struct CudaParameters
{
    std::size_t nx;
    std::size_t ny;

    double dx;
    double dy;

    double diffusivity;
    double dt;
};


struct CudaLaunchConfig
{
    unsigned int block_x = 32;
    unsigned int block_y = 8;
};


void launch_advection_diffusion_fp64(
    const double* phi_old,
    const double* u,
    const double* v,
    double* phi_new,
    const CudaParameters& parameters,
    const CudaLaunchConfig& launch,
    cudaStream_t stream = nullptr
);

}  // namespace stencil
