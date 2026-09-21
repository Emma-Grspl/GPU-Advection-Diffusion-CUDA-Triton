#include "stencil/CudaError.hpp"
#include "stencil/CudaStencilExecutor.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>


namespace
{

struct BlockConfig
{
    unsigned int x;
    unsigned int y;
};


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


double standard_deviation(
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

    const std::vector<BlockConfig> blocks{
        {8, 8},
        {16, 8},
        {32, 8},
        {16, 16},
        {32, 16},
    };

    const std::size_t number_of_blocks =
        blocks.size();

    const std::size_t size =
        nx * ny;

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

    const stencil::CudaParameters parameters{
        .nx = nx,
        .ny = ny,
        .dx = 1.0 / static_cast<double>(nx - 1),
        .dy = 1.0 / static_cast<double>(ny - 1),
        .diffusivity = 1.0e-3,
        .dt = 1.0e-8,
    };

    stencil::CudaStencilExecutor executor(
        nx,
        ny
    );

    executor.upload(
        phi,
        u,
        v
    );

    // Global warm-up using the original C0 launch geometry.
    executor.advance(
        parameters,
        warmup_steps,
        stencil::CudaLaunchConfig{
            .block_x = 32,
            .block_y = 8,
        }
    );

    executor.synchronize();

    std::vector<std::vector<double>>
        samples(
            number_of_blocks
        );

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

    /*
     * Cyclic ordering:
     *
     * round 0: A B C D E
     * round 1: B C D E A
     * round 2: C D E A B
     * ...
     *
     * Therefore each block geometry occupies each
     * position in the execution order.
     */
    for (
        std::size_t round = 0;
        round < rounds;
        ++round
    )
    {
        for (
            std::size_t position = 0;
            position < number_of_blocks;
            ++position
        )
        {
            const std::size_t config_index =
                (
                    position
                    + round
                )
                % number_of_blocks;

            const BlockConfig block =
                blocks[config_index];

            const stencil::CudaLaunchConfig launch{
                .block_x = block.x,
                .block_y = block.y,
            };

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

            samples[config_index].push_back(
                static_cast<double>(
                    elapsed_ms
                )
                / static_cast<double>(
                    timed_steps
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

    for (
        std::size_t config_index = 0;
        config_index < number_of_blocks;
        ++config_index
    )
    {
        const BlockConfig block =
            blocks[config_index];

        const double median_ms =
            median(
                samples[config_index]
            );

        const double mean_ms =
            mean(
                samples[config_index]
            );

        const double std_ms =
            standard_deviation(
                samples[config_index]
            );

        const double cells =
            static_cast<double>(nx)
            * static_cast<double>(ny);

        const double gcell_per_s =
            cells
            / (
                median_ms
                * 1.0e-3
            )
            / 1.0e9;

        std::cout
            << nx
            << ','
            << ny
            << ','
            << block.x
            << ','
            << block.y
            << ','
            << block.x * block.y
            << ','
            << rounds
            << ','
            << median_ms * 1000.0
            << ','
            << mean_ms * 1000.0
            << ','
            << std_ms * 1000.0
            << ','
            << gcell_per_s
            << '\n';
    }
}


} // namespace


int main()
{
    try
    {
        // Force CUDA context initialization outside measurements.
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
            << "block_x,block_y,"
            << "threads_per_block,"
            << "rounds,"
            << "median_step_us,"
            << "mean_step_us,"
            << "std_step_us,"
            << "gcell_per_s\n";

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
