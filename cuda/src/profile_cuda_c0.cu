#include "stencil/CudaStencilExecutor.hpp"

#include <cstddef>
#include <iostream>
#include <vector>


int main()
{
    try
    {
        constexpr std::size_t nx = 4096;
        constexpr std::size_t ny = 4096;
        constexpr std::size_t size = nx * ny;

        constexpr double dx =
            1.0 / static_cast<double>(nx - 1);

        constexpr double dy =
            1.0 / static_cast<double>(ny - 1);

        constexpr double diffusivity =
            1.0e-3;

        constexpr double dt =
            1.0e-8;

        std::vector<double> phi(
            size,
            1.0
        );

        std::vector<double> u(
            size,
            0.5
        );

        std::vector<double> v(
            size,
            -0.2
        );

        std::vector<double> output(
            size
        );

        stencil::CudaStencilExecutor executor(
            nx,
            ny
        );

        executor.upload(
            phi,
            u,
            v
        );

        const stencil::CudaParameters parameters{
            .nx = nx,
            .ny = ny,
            .dx = dx,
            .dy = dy,
            .diffusivity = diffusivity,
            .dt = dt,
        };

        const stencil::CudaLaunchConfig launch{
            .block_x = 32,
            .block_y = 8,
        };

        // Warm up the CUDA context and caches.
        executor.advance(
            parameters,
            10,
            launch
        );

        executor.synchronize();

        // One target launch for Nsight Compute.
        executor.advance(
            parameters,
            1,
            launch
        );

        executor.synchronize();

        executor.download(
            output
        );

        // Prevent the result from being completely unused.
        std::cout
            << "center="
            << output[
                (nx / 2)
                + (ny / 2) * nx
            ]
            << '\n';

        return 0;
    }
    catch (
        const std::exception& exception
    )
    {
        std::cerr
            << "ERROR: "
            << exception.what()
            << '\n';

        return 1;
    }
}
