#include "stencil/CudaStencilExecutor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>


int main()
{
    try
    {
        constexpr std::size_t nx =
            37;

        constexpr std::size_t ny =
            51;

        const std::size_t size =
            nx * ny;

        std::vector<double> phi(size);
        std::vector<double> u(size);
        std::vector<double> v(size);

        for (
            std::size_t j = 0;
            j < ny;
            ++j
        )
        {
            for (
                std::size_t i = 0;
                i < nx;
                ++i
            )
            {
                const std::size_t k =
                    i + j * nx;

                const double x =
                    static_cast<double>(i)
                    / static_cast<double>(nx - 1);

                const double y =
                    static_cast<double>(j)
                    / static_cast<double>(ny - 1);

                phi[k] =
                    std::sin(3.1 * x)
                    * std::cos(2.7 * y)
                    + 0.2 * x * y;

                u[k] =
                    0.4
                    + 0.1 * y;

                v[k] =
                    -0.2
                    + 0.05 * x;
            }
        }

        const stencil::CudaParameters parameters{
            .nx = nx,
            .ny = ny,
            .dx = 0.7 / static_cast<double>(nx - 1),
            .dy = 1.3 / static_cast<double>(ny - 1),
            .diffusivity = 1.0e-3,
            .dt = 1.0e-5,
        };

        const stencil::CudaLaunchConfig launch{
            .block_x = 32,
            .block_y = 8,
        };

        const std::vector<std::size_t> step_counts{
            1,
            10,
            1000,
        };

        bool success =
            true;

        std::cout
            << std::setprecision(16);

        for (
            const std::size_t steps
            : step_counts
        )
        {
            stencil::CudaStencilExecutor c0(
                nx,
                ny
            );

            stencil::CudaStencilExecutor c2(
                nx,
                ny
            );

            c0.upload(
                phi,
                u,
                v
            );

            c2.upload(
                phi,
                u,
                v
            );

            c0.advance(
                parameters,
                steps,
                launch
            );

            c2.advance_shared(
                parameters,
                steps,
                launch
            );

            c0.synchronize();
            c2.synchronize();

            std::vector<double> ref(size);
            std::vector<double> shared(size);

            c0.download(ref);
            c2.download(shared);

            double error_squared =
                0.0;

            double reference_squared =
                0.0;

            double max_abs =
                0.0;

            for (
                std::size_t k = 0;
                k < size;
                ++k
            )
            {
                const double difference =
                    shared[k]
                    - ref[k];

                error_squared +=
                    difference
                    * difference;

                reference_squared +=
                    ref[k]
                    * ref[k];

                max_abs =
                    std::max(
                        max_abs,
                        std::abs(difference)
                    );
            }

            const double relative_l2 =
                std::sqrt(
                    error_squared
                    / reference_squared
                );

            const bool pass =
                max_abs <= 1.0e-12
                && relative_l2 <= 1.0e-12;

            success =
                success && pass;

            std::cout
                << (pass ? "[PASS] " : "[FAIL] ")
                << "steps="
                << steps
                << " rel_L2="
                << relative_l2
                << " Linf="
                << max_abs
                << '\n';
        }

        std::cout
            << '\n'
            << "CUDA C2 SHARED VALIDATION: "
            << (success ? "PASS" : "FAIL")
            << '\n';

        return success
            ? 0
            : 1;
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
