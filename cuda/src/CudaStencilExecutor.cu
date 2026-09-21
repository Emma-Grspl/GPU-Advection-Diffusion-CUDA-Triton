#include "stencil/CudaStencilExecutor.hpp"
#include "stencil/CudaError.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>


namespace stencil
{

CudaStencilExecutor::CudaStencilExecutor(
    const std::size_t nx,
    const std::size_t ny
)
    : nx_(nx),
      ny_(ny),
      size_(
          nx > 0
          && ny
             > std::numeric_limits<std::size_t>::max()
                 / nx
              ? throw std::overflow_error(
                    "nx * ny overflows size_t."
                )
              : nx * ny
      ),
      phi_a_(size_),
      phi_b_(size_),
      u_(size_),
      v_(size_)
{
    if (
        nx_ < 3
        || ny_ < 3
    )
    {
        throw std::invalid_argument(
            "nx and ny must both be at least 3."
        );
    }
}


void CudaStencilExecutor::upload(
    const std::span<const double> phi,
    const std::span<const double> u,
    const std::span<const double> v
)
{
    phi_a_.upload(
        phi
    );

    u_.upload(
        u
    );

    v_.upload(
        v
    );

    current_is_a_ = true;
}


void CudaStencilExecutor::advance(
    const CudaParameters& parameters,
    const std::size_t steps,
    const CudaLaunchConfig& launch
)
{
    if (
        parameters.nx != nx_
        || parameters.ny != ny_
    )
    {
        throw std::invalid_argument(
            "Executor grid size and CUDA parameters differ."
        );
    }

    if (steps == 0)
    {
        throw std::invalid_argument(
            "steps must be >= 1."
        );
    }

    for (
        std::size_t step = 0;
        step < steps;
        ++step
    )
    {
        launch_advection_diffusion_fp64(
            current_phi(),
            u_.data(),
            v_.data(),
            next_phi(),
            parameters,
            launch
        );

        current_is_a_ =
            !current_is_a_;
    }
}


void CudaStencilExecutor::advance_shared(
    const CudaParameters& parameters,
    const std::size_t steps,
    const CudaLaunchConfig& launch
)
{
    if (
        parameters.nx != nx_
        || parameters.ny != ny_
    )
    {
        throw std::invalid_argument(
            "Executor grid size and CUDA parameters differ."
        );
    }

    if (steps == 0)
    {
        throw std::invalid_argument(
            "steps must be >= 1."
        );
    }

    for (
        std::size_t step = 0;
        step < steps;
        ++step
    )
    {
        launch_advection_diffusion_shared_fp64(
            current_phi(),
            u_.data(),
            v_.data(),
            next_phi(),
            parameters,
            launch
        );

        current_is_a_ =
            !current_is_a_;
    }
}


void CudaStencilExecutor::synchronize() const
{
    STENCIL_CUDA_CHECK(
        cudaDeviceSynchronize()
    );
}


void CudaStencilExecutor::download(
    const std::span<double> output
) const
{
    if (output.size() != size_)
    {
        throw std::invalid_argument(
            "Output size mismatch."
        );
    }

    if (current_is_a_)
    {
        phi_a_.download(
            output
        );
    }
    else
    {
        phi_b_.download(
            output
        );
    }
}


double*
CudaStencilExecutor::current_phi() noexcept
{
    return (
        current_is_a_
        ? phi_a_.data()
        : phi_b_.data()
    );
}


const double*
CudaStencilExecutor::current_phi() const noexcept
{
    return (
        current_is_a_
        ? phi_a_.data()
        : phi_b_.data()
    );
}


double*
CudaStencilExecutor::next_phi() noexcept
{
    return (
        current_is_a_
        ? phi_b_.data()
        : phi_a_.data()
    );
}


}  // namespace stencil
