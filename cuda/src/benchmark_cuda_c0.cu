#include "stencil/AdvectionDiffusionCuda.hpp"
#include "stencil/CudaError.hpp"
#include "stencil/CudaStencilExecutor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>


namespace
{

using Clock =
    std::chrono::steady_clock;


double milliseconds(
    const Clock::time_point start,
    const Clock::time_point stop
)
{
    return std::chrono::duration<
        double,
        std::milli
    >(
        stop - start
    ).count();
}


double median(
    std::vector<double> values
)
{
    if (values.empty())
    {
        throw std::invalid_argument(
            "Cannot compute median of empty data."
        );
    }

    std::sort(
        values.begin(),
        values.end()
    );

    const std::size_t middle =
        values.size() / 2;

    if (values.size() % 2 == 1)
    {
        return values[middle];
    }

    return 0.5 * (
        values[middle - 1]
        + values[middle]
    );
}


void initialize_fields(
    const std::size_t nx,
    const std::size_t ny,
    std::vector<double>& phi,
    std::vector<double>& u,
    std::vector<double>& v
)
{
    constexpr double pi =
        3.141592653589793238462643383279502884;

    for (
        std::size_t j = 0;
        j < ny;
        ++j
    )
    {
        const double y =
            static_cast<double>(j)
            / static_cast<double>(ny - 1);

        for (
            std::size_t i = 0;
            i < nx;
            ++i
        )
        {
            const double x =
                static_cast<double>(i)
                / static_cast<double>(nx - 1);

            const std::size_t k =
                i + j * nx;

            const double dx1 =
                x - 0.30;

            const double dy1 =
                y - 0.60;

            const double dx2 =
                x - 0.75;

            const double dy2 =
                y - 0.25;

            phi[k] =
                std::exp(
                    -40.0
                    * (
                        dx1 * dx1
                        + dy1 * dy1
                    )
                )
                + 0.25
                * std::exp(
                    -60.0
                    * (
                        dx2 * dx2
                        + dy2 * dy2
                    )
                );

            u[k] =
                0.50
                + 0.10
                * std::sin(
                    2.0
                    * pi
                    * y
                );

            v[k] =
                -0.20
                + 0.10
                * std::cos(
                    2.0
                    * pi
                    * x
                );
        }
    }
}


void benchmark_size(
    const std::size_t nx,
    const std::size_t ny,
    const std::size_t warmup_steps,
    const std::size_t timed_steps,
    const std::size_t repeats
)
{
    const std::size_t size =
        nx * ny;

    std::vector<double> phi(
        size
    );

    std::vector<double> u(
        size
    );

    std::vector<double> v(
        size
    );

    std::vector<double> output(
        size
    );

    initialize_fields(
        nx,
        ny,
        phi,
        u,
        v
    );

    const double dx =
        1.0
        / static_cast<double>(
            nx - 1
        );

    const double dy =
        1.0
        / static_cast<double>(
            ny - 1
        );

    // Tiny dt keeps every tested grid comfortably
    // inside the explicit stability regime.
    constexpr double diffusivity =
        1.0e-3;

    constexpr double dt =
        1.0e-8;

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

    const auto allocation_start =
        Clock::now();

    stencil::CudaStencilExecutor executor(
        nx,
        ny
    );

    const auto allocation_stop =
        Clock::now();

    const auto upload_start =
        Clock::now();

    executor.upload(
        phi,
        u,
        v
    );

    const auto upload_stop =
        Clock::now();

    executor.advance(
        parameters,
        warmup_steps,
        launch
    );

    executor.synchronize();

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};

    STENCIL_CUDA_CHECK(
        cudaEventCreate(
            &start_event
        )
    );

    STENCIL_CUDA_CHECK(
        cudaEventCreate(
            &stop_event
        )
    );

    std::vector<double> total_kernel_ms;
    total_kernel_ms.reserve(
        repeats
    );

    for (
        std::size_t repeat = 0;
        repeat < repeats;
        ++repeat
    )
    {
        STENCIL_CUDA_CHECK(
            cudaEventRecord(
                start_event
            )
        );

        executor.advance(
            parameters,
            timed_steps,
            launch
        );

        STENCIL_CUDA_CHECK(
            cudaEventRecord(
                stop_event
            )
        );

        STENCIL_CUDA_CHECK(
            cudaEventSynchronize(
                stop_event
            )
        );

        float elapsed_ms =
            0.0F;

        STENCIL_CUDA_CHECK(
            cudaEventElapsedTime(
                &elapsed_ms,
                start_event,
                stop_event
            )
        );

        total_kernel_ms.push_back(
            static_cast<double>(
                elapsed_ms
            )
        );
    }

    STENCIL_CUDA_CHECK(
        cudaEventDestroy(
            start_event
        )
    );

    STENCIL_CUDA_CHECK(
        cudaEventDestroy(
            stop_event
        )
    );

    const auto download_start =
        Clock::now();

    executor.download(
        output
    );

    const auto download_stop =
        Clock::now();

    const double median_total_ms =
        median(
            total_kernel_ms
        );

    const double median_step_ms =
        median_total_ms
        / static_cast<double>(
            timed_steps
        );

    const double seconds_per_step =
        median_step_ms
        * 1.0e-3;

    const double cells =
        static_cast<double>(
            nx
        )
        * static_cast<double>(
            ny
        );

    const double gcell_per_s =
        cells
        / seconds_per_step
        / 1.0e9;

    // Effective algorithmic traffic for C0:
    //
    // 5 phi loads
    // + 1 u load
    // + 1 v load
    // + 1 phi_new store
    //
    // = 8 doubles = 64 bytes per output cell.
    //
    // This is NOT measured DRAM traffic. Cache reuse can
    // reduce the actual physical traffic seen by HBM.
    constexpr double bytes_per_cell =
        64.0;

    const double effective_bandwidth_gbs =
        gcell_per_s
        * bytes_per_cell;

    std::cout
        << nx
        << ','
        << ny
        << ','
        << warmup_steps
        << ','
        << timed_steps
        << ','
        << repeats
        << ','
        << milliseconds(
            allocation_start,
            allocation_stop
        )
        << ','
        << milliseconds(
            upload_start,
            upload_stop
        )
        << ','
        << milliseconds(
            download_start,
            download_stop
        )
        << ','
        << median_total_ms
        << ','
        << median_step_ms
        << ','
        << gcell_per_s
        << ','
        << effective_bandwidth_gbs
        << '\n';
}


}  // namespace


int main()
{
    try
    {
        constexpr std::size_t warmup_steps =
            50;

        constexpr std::size_t timed_steps =
            500;

        constexpr std::size_t repeats =
            7;

        const std::vector<std::size_t> sizes{
            50,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        };

        std::cout
            << std::setprecision(10);

        std::cout
            << "nx,ny,warmup_steps,timed_steps,repeats,"
            << "allocation_ms,h2d_ms,d2h_ms,"
            << "kernel_total_median_ms,"
            << "kernel_step_median_ms,"
            << "gcell_per_s,"
            << "effective_bandwidth_gbs\n";

        for (
            const std::size_t n
            : sizes
        )
        {
            benchmark_size(
                n,
                n,
                warmup_steps,
                timed_steps,
                repeats
            );
        }

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
