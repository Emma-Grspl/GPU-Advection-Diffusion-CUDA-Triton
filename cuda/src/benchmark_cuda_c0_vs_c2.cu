#include "stencil/CudaError.hpp"
#include "stencil/CudaStencilExecutor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>


namespace
{

double median(
    std::vector<double> values
)
{
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


double mean(
    const std::vector<double>& values
)
{
    return std::accumulate(
        values.begin(),
        values.end(),
        0.0
    ) / static_cast<double>(
        values.size()
    );
}


double stddev(
    const std::vector<double>& values
)
{
    const double avg =
        mean(values);

    double accumulator =
        0.0;

    for (const double value : values)
    {
        const double delta =
            value - avg;

        accumulator +=
            delta * delta;
    }

    return std::sqrt(
        accumulator
        / static_cast<double>(
            values.size()
        )
    );
}


double time_c0(
    stencil::CudaStencilExecutor& executor,
    const stencil::CudaParameters& parameters,
    const stencil::CudaLaunchConfig& launch,
    const std::size_t steps,
    cudaEvent_t start_event,
    cudaEvent_t stop_event
)
{
    STENCIL_CUDA_CHECK(
        cudaEventRecord(
            start_event
        )
    );

    executor.advance(
        parameters,
        steps,
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

    return
        static_cast<double>(
            elapsed_ms
        )
        / static_cast<double>(
            steps
        );
}


double time_c2(
    stencil::CudaStencilExecutor& executor,
    const stencil::CudaParameters& parameters,
    const stencil::CudaLaunchConfig& launch,
    const std::size_t steps,
    cudaEvent_t start_event,
    cudaEvent_t stop_event
)
{
    STENCIL_CUDA_CHECK(
        cudaEventRecord(
            start_event
        )
    );

    executor.advance_shared(
        parameters,
        steps,
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

    return
        static_cast<double>(
            elapsed_ms
        )
        / static_cast<double>(
            steps
        );
}


void benchmark_size(
    const std::size_t nx,
    const std::size_t ny
)
{
    constexpr std::size_t warmup_steps =
        100;

    constexpr std::size_t timed_steps =
        500;

    constexpr std::size_t rounds =
        7;

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

            phi[k] =
                std::sin(
                    3.1 * x
                )
                * std::cos(
                    2.7 * y
                )
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
        .dx = 1.0 / static_cast<double>(nx - 1),
        .dy = 1.0 / static_cast<double>(ny - 1),
        .diffusivity = 1.0e-3,
        .dt = 1.0e-8,
    };

    const stencil::CudaLaunchConfig launch{
        .block_x = 32,
        .block_y = 8,
    };

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
        warmup_steps,
        launch
    );

    c2.advance_shared(
        parameters,
        warmup_steps,
        launch
    );

    c0.synchronize();
    c2.synchronize();

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

    std::vector<double> c0_times;
    std::vector<double> c2_times;

    c0_times.reserve(
        rounds
    );

    c2_times.reserve(
        rounds
    );

    /*
     * Alternate execution order to reduce
     * temperature / clock / ordering bias.
     */
    for (
        std::size_t round = 0;
        round < rounds;
        ++round
    )
    {
        if (round % 2 == 0)
        {
            c0_times.push_back(
                time_c0(
                    c0,
                    parameters,
                    launch,
                    timed_steps,
                    start_event,
                    stop_event
                )
            );

            c2_times.push_back(
                time_c2(
                    c2,
                    parameters,
                    launch,
                    timed_steps,
                    start_event,
                    stop_event
                )
            );
        }
        else
        {
            c2_times.push_back(
                time_c2(
                    c2,
                    parameters,
                    launch,
                    timed_steps,
                    start_event,
                    stop_event
                )
            );

            c0_times.push_back(
                time_c0(
                    c0,
                    parameters,
                    launch,
                    timed_steps,
                    start_event,
                    stop_event
                )
            );
        }
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

    const double c0_median_ms =
        median(
            c0_times
        );

    const double c2_median_ms =
        median(
            c2_times
        );

    const double cells =
        static_cast<double>(
            nx
        )
        * static_cast<double>(
            ny
        );

    const double c0_gcell =
        cells
        / (
            c0_median_ms
            * 1.0e-3
        )
        / 1.0e9;

    const double c2_gcell =
        cells
        / (
            c2_median_ms
            * 1.0e-3
        )
        / 1.0e9;

    const double speedup =
        c0_median_ms
        / c2_median_ms;

    const double percent_change =
        (
            speedup
            - 1.0
        )
        * 100.0;

    std::cout
        << nx
        << ','
        << ny
        << ','
        << c0_median_ms * 1000.0
        << ','
        << c2_median_ms * 1000.0
        << ','
        << stddev(c0_times) * 1000.0
        << ','
        << stddev(c2_times) * 1000.0
        << ','
        << c0_gcell
        << ','
        << c2_gcell
        << ','
        << speedup
        << ','
        << percent_change
        << '\n';
}


} // namespace


int main()
{
    try
    {
        STENCIL_CUDA_CHECK(
            cudaFree(
                nullptr
            )
        );

        const std::vector<std::size_t> sizes{
            512,
            1024,
            2048,
            4096,
        };

        std::cout
            << std::setprecision(10);

        std::cout
            << "nx,ny,"
            << "c0_median_us,"
            << "c2_median_us,"
            << "c0_std_us,"
            << "c2_std_us,"
            << "c0_gcell_per_s,"
            << "c2_gcell_per_s,"
            << "c2_speedup,"
            << "c2_percent_change\n";

        for (
            const std::size_t n
            : sizes
        )
        {
            benchmark_size(
                n,
                n
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
