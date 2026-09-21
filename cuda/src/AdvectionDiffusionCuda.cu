#include "stencil/AdvectionDiffusionCuda.hpp"
#include "stencil/CudaError.hpp"

#include <cuda_runtime.h>

#include <stdexcept>


namespace stencil
{
namespace
{

__global__
void advection_diffusion_fp64_kernel(
    const double* __restrict__ phi_old,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double* __restrict__ phi_new,
    const std::size_t nx,
    const std::size_t ny,
    const double inv_2dx,
    const double inv_2dy,
    const double inv_dx2,
    const double inv_dy2,
    const double diffusivity_dt,
    const double dt
)
{
    const std::size_t i =
        static_cast<std::size_t>(
            blockIdx.x
        )
        * blockDim.x
        + threadIdx.x;

    const std::size_t j =
        static_cast<std::size_t>(
            blockIdx.y
        )
        * blockDim.y
        + threadIdx.y;

    if (
        i >= nx
        || j >= ny
    )
    {
        return;
    }

    const std::size_t center =
        i + j * nx;

    // Preserve all boundary values exactly.
    if (
        i == 0
        || j == 0
        || i == nx - 1
        || j == ny - 1
    )
    {
        phi_new[center] =
            phi_old[center];

        return;
    }

    const std::size_t left =
        center - 1;

    const std::size_t right =
        center + 1;

    const std::size_t bottom =
        center - nx;

    const std::size_t top =
        center + nx;

    const double center_value =
        phi_old[center];

    const double dphi_dx =
        (
            phi_old[right]
            - phi_old[left]
        )
        * inv_2dx;

    const double dphi_dy =
        (
            phi_old[top]
            - phi_old[bottom]
        )
        * inv_2dy;

    const double laplacian =
        (
            phi_old[right]
            - 2.0 * center_value
            + phi_old[left]
        )
        * inv_dx2
        +
        (
            phi_old[top]
            - 2.0 * center_value
            + phi_old[bottom]
        )
        * inv_dy2;

    const double advection =
        u[center] * dphi_dx
        +
        v[center] * dphi_dy;

    phi_new[center] =
        center_value
        - dt * advection
        + diffusivity_dt
        * laplacian;
}


}  // namespace



__global__ void advection_diffusion_shared_fp64_kernel(
    const double* __restrict__ phi_old,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double* __restrict__ phi_new,
    const std::size_t nx,
    const std::size_t ny,
    const double dt,
    const double inv_2dx,
    const double inv_2dy,
    const double inv_dx2,
    const double inv_dy2,
    const double diffusivity_dt
)
{
    extern __shared__ double phi_tile[];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tx;

    const std::size_t j =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + ty;

    const unsigned int tile_width =
        blockDim.x + 2;

    const unsigned int tile_height =
        blockDim.y + 2;

    const unsigned int threads_per_block =
        blockDim.x * blockDim.y;

    const unsigned int linear_thread =
        ty * blockDim.x + tx;

    const unsigned int tile_elements =
        tile_width * tile_height;

    /*
     * Cooperative loading of the complete tile including
     * the one-cell halo.
     *
     * This is deliberately generic and also works for
     * partially filled edge blocks.
     */
    for (
        unsigned int index = linear_thread;
        index < tile_elements;
        index += threads_per_block
    )
    {
        const unsigned int sx =
            index % tile_width;

        const unsigned int sy =
            index / tile_width;

        const long long global_i =
            static_cast<long long>(blockIdx.x * blockDim.x)
            + static_cast<long long>(sx)
            - 1LL;

        const long long global_j =
            static_cast<long long>(blockIdx.y * blockDim.y)
            + static_cast<long long>(sy)
            - 1LL;

        double value = 0.0;

        if (
            global_i >= 0
            && global_j >= 0
            && global_i < static_cast<long long>(nx)
            && global_j < static_cast<long long>(ny)
        )
        {
            const std::size_t global_index =
                static_cast<std::size_t>(global_i)
                + static_cast<std::size_t>(global_j) * nx;

            value =
                phi_old[global_index];
        }

        phi_tile[index] =
            value;
    }

    __syncthreads();

    if (
        i >= nx
        || j >= ny
    )
    {
        return;
    }

    const std::size_t k =
        i + j * nx;

    if (
        i == 0
        || j == 0
        || i + 1 == nx
        || j + 1 == ny
    )
    {
        phi_new[k] =
            phi_old[k];

        return;
    }

    const unsigned int local_x =
        tx + 1;

    const unsigned int local_y =
        ty + 1;

    const unsigned int center =
        local_y * tile_width
        + local_x;

    const double phi_center =
        phi_tile[center];

    const double phi_left =
        phi_tile[center - 1];

    const double phi_right =
        phi_tile[center + 1];

    const double phi_bottom =
        phi_tile[center - tile_width];

    const double phi_top =
        phi_tile[center + tile_width];

    const double dphi_dx =
        (
            phi_right
            - phi_left
        ) * inv_2dx;

    const double dphi_dy =
        (
            phi_top
            - phi_bottom
        ) * inv_2dy;

    const double laplacian =
        (
            phi_right
            - 2.0 * phi_center
            + phi_left
        ) * inv_dx2
        +
        (
            phi_top
            - 2.0 * phi_center
            + phi_bottom
        ) * inv_dy2;

    phi_new[k] =
        phi_center
        - dt
            * (
                u[k] * dphi_dx
                + v[k] * dphi_dy
            )
        + diffusivity_dt
            * laplacian;
}


void launch_advection_diffusion_fp64(
    const double* phi_old,
    const double* u,
    const double* v,
    double* phi_new,
    const CudaParameters& parameters,
    const CudaLaunchConfig& launch,
    const cudaStream_t stream
)
{
    if (
        phi_old == nullptr
        || u == nullptr
        || v == nullptr
        || phi_new == nullptr
    )
    {
        throw std::invalid_argument(
            "CUDA device pointers must not be null."
        );
    }

    if (
        phi_old == phi_new
    )
    {
        throw std::invalid_argument(
            "In-place CUDA stencil updates are not allowed."
        );
    }

    if (
        parameters.nx < 3
        || parameters.ny < 3
    )
    {
        throw std::invalid_argument(
            "nx and ny must both be at least 3."
        );
    }

    if (
        parameters.dx <= 0.0
        || parameters.dy <= 0.0
    )
    {
        throw std::invalid_argument(
            "dx and dy must be strictly positive."
        );
    }

    if (
        parameters.diffusivity < 0.0
    )
    {
        throw std::invalid_argument(
            "diffusivity cannot be negative."
        );
    }

    if (
        parameters.dt <= 0.0
    )
    {
        throw std::invalid_argument(
            "dt must be strictly positive."
        );
    }

    if (
        launch.block_x == 0
        || launch.block_y == 0
    )
    {
        throw std::invalid_argument(
            "CUDA block dimensions must be non-zero."
        );
    }

    const double inv_2dx =
        1.0
        / (
            2.0
            * parameters.dx
        );

    const double inv_2dy =
        1.0
        / (
            2.0
            * parameters.dy
        );

    const double inv_dx2 =
        1.0
        / (
            parameters.dx
            * parameters.dx
        );

    const double inv_dy2 =
        1.0
        / (
            parameters.dy
            * parameters.dy
        );

    const double diffusivity_dt =
        parameters.diffusivity
        * parameters.dt;

    const dim3 block(
        launch.block_x,
        launch.block_y,
        1
    );

    const dim3 grid(
        static_cast<unsigned int>(
            (
                parameters.nx
                + launch.block_x
                - 1
            )
            / launch.block_x
        ),
        static_cast<unsigned int>(
            (
                parameters.ny
                + launch.block_y
                - 1
            )
            / launch.block_y
        ),
        1
    );

    advection_diffusion_fp64_kernel
        <<<grid, block, 0, stream>>>(
            phi_old,
            u,
            v,
            phi_new,
            parameters.nx,
            parameters.ny,
            inv_2dx,
            inv_2dy,
            inv_dx2,
            inv_dy2,
            diffusivity_dt,
            parameters.dt
        );

    STENCIL_CUDA_CHECK(
        cudaGetLastError()
    );
}



void launch_advection_diffusion_shared_fp64(
    const double* phi_old,
    const double* u,
    const double* v,
    double* phi_new,
    const CudaParameters& parameters,
    const CudaLaunchConfig& launch_config,
    cudaStream_t stream
)
{
    if (
        phi_old == nullptr
        || u == nullptr
        || v == nullptr
        || phi_new == nullptr
    )
    {
        throw std::invalid_argument(
            "CUDA stencil received a null pointer."
        );
    }

    if (phi_old == phi_new)
    {
        throw std::invalid_argument(
            "CUDA stencil requires separate old/new buffers."
        );
    }

    if (
        parameters.nx < 3
        || parameters.ny < 3
        || parameters.dx <= 0.0
        || parameters.dy <= 0.0
        || parameters.diffusivity < 0.0
        || parameters.dt <= 0.0
    )
    {
        throw std::invalid_argument(
            "Invalid CUDA stencil parameters."
        );
    }

    if (
        launch_config.block_x == 0
        || launch_config.block_y == 0
        || launch_config.block_x
            * launch_config.block_y
            > 1024
    )
    {
        throw std::invalid_argument(
            "Invalid CUDA block dimensions."
        );
    }

    const double inv_2dx =
        1.0 / (2.0 * parameters.dx);

    const double inv_2dy =
        1.0 / (2.0 * parameters.dy);

    const double inv_dx2 =
        1.0
        / (
            parameters.dx
            * parameters.dx
        );

    const double inv_dy2 =
        1.0
        / (
            parameters.dy
            * parameters.dy
        );

    const double diffusivity_dt =
        parameters.diffusivity
        * parameters.dt;

    const dim3 block(
        launch_config.block_x,
        launch_config.block_y
    );

    const dim3 grid(
        static_cast<unsigned int>(
            (
                parameters.nx
                + launch_config.block_x
                - 1
            )
            / launch_config.block_x
        ),
        static_cast<unsigned int>(
            (
                parameters.ny
                + launch_config.block_y
                - 1
            )
            / launch_config.block_y
        )
    );

    const std::size_t shared_bytes =
        static_cast<std::size_t>(
            launch_config.block_x + 2
        )
        * static_cast<std::size_t>(
            launch_config.block_y + 2
        )
        * sizeof(double);

    advection_diffusion_shared_fp64_kernel
        <<<grid, block, shared_bytes, stream>>>(
            phi_old,
            u,
            v,
            phi_new,
            parameters.nx,
            parameters.ny,
            parameters.dt,
            inv_2dx,
            inv_2dy,
            inv_dx2,
            inv_dy2,
            diffusivity_dt
        );

    STENCIL_CUDA_CHECK(
        cudaGetLastError()
    );
}


}  // namespace stencil
