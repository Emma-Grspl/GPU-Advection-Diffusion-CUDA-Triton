#include "stencil/AdvectionDiffusion.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>


namespace
{

using Clock =
    std::chrono::steady_clock;


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
    )
    / static_cast<double>(
        values.size()
    );
}


double stddev(
    const std::vector<double>& values
)
{
    const double avg =
        mean(values);

    double sum =
        0.0;

    for (const double value : values)
    {
        const double delta =
            value - avg;

        sum +=
            delta * delta;
    }

    return std::sqrt(
        sum
        / static_cast<double>(
            values.size()
        )
    );
}


void initialize(
    const std::size_t nx,
    const std::size_t ny,
    std::vector<double>& phi,
    std::vector<double>& u,
    std::vector<double>& v
)
{
    constexpr double pi =
        3.14159265358979323846;

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


void benchmark(
    const std::size_t nx,
    const std::size_t ny,
    const std::size_t timed_steps
)
{
    constexpr std::size_t warmup_steps =
        3;

    constexpr std::size_t repeats =
        7;

    const std::size_t size =
        nx * ny;

    std::vector<double> initial(size);
    std::vector<double> u(size);
    std::vector<double> v(size);

    initialize(
        nx,
        ny,
        initial,
        u,
        v
    );

    const stencil::Parameters parameters{
        .nx = nx,
        .ny = ny,
        .dx = 1.0 / static_cast<double>(nx - 1),
        .dy = 1.0 / static_cast<double>(ny - 1),
        .diffusivity = 1.0e-3,
        .dt = 1.0e-8,
    };

    std::vector<double> samples;

    samples.reserve(
        repeats
    );

    for (
        std::size_t repeat = 0;
        repeat < repeats;
        ++repeat
    )
    {
        std::vector<double> current =
            initial;

        std::vector<double> next(
            size
        );

        for (
            std::size_t step = 0;
            step < warmup_steps;
            ++step
        )
        {
            stencil::advance_scalar(
                current,
                u,
                v,
                next,
                parameters
            );

            current.swap(
                next
            );
        }

        const auto start =
            Clock::now();

        for (
            std::size_t step = 0;
            step < timed_steps;
            ++step
        )
        {
            stencil::advance_scalar(
                current,
                u,
                v,
                next,
                parameters
            );

            current.swap(
                next
            );
        }

        const auto stop =
            Clock::now();

        const double elapsed_us =
            std::chrono::duration<
                double,
                std::micro
            >(
                stop - start
            ).count();

        samples.push_back(
            elapsed_us
            / static_cast<double>(
                timed_steps
            )
        );
    }

    const double median_us =
        median(
            samples
        );

    const double mean_us =
        mean(
            samples
        );

    const double std_us =
        stddev(
            samples
        );

    const double cells =
        static_cast<double>(
            nx
        )
        * static_cast<double>(
            ny
        );

    const double gcell_per_s =
        cells
        / (
            median_us
            * 1.0e-6
        )
        / 1.0e9;

    std::cout
        << nx
        << ','
        << ny
        << ','
        << timed_steps
        << ','
        << median_us
        << ','
        << mean_us
        << ','
        << std_us
        << ','
        << gcell_per_s
        << '\n';
}


} // namespace


int main()
{
    std::cout
        << std::setprecision(10);

    std::cout
        << "nx,ny,timed_steps,"
        << "median_step_us,"
        << "mean_step_us,"
        << "std_step_us,"
        << "gcell_per_s\n";

    benchmark(512, 512, 50);
    benchmark(1024, 1024, 20);
    benchmark(2048, 2048, 5);
    benchmark(4096, 4096, 2);

    return 0;
}
