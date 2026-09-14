#include "stencil/AdvectionDiffusion.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>


int main()
{
    constexpr std::size_t nx = 13;
    constexpr std::size_t ny = 11;

    const std::size_t size =
        nx * ny;

    std::vector<double> phi_old(
        size,
        0.0
    );

    std::vector<double> u(
        size,
        0.0
    );

    std::vector<double> v(
        size,
        0.0
    );

    std::vector<double> phi_new(
        size,
        0.0
    );

    // --------------------------------------------------------
    // Deterministic input fields.
    //
    // The scalar contains:
    // - x dependence,
    // - y dependence,
    // - an xy cross term,
    // - x² and y² curvature.
    //
    // Therefore both advection and diffusion contribute.
    // --------------------------------------------------------

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
                stencil::flat_index(
                    i,
                    j,
                    nx
                );

            const double x_index =
                static_cast<double>(i);

            const double y_index =
                static_cast<double>(j);

            phi_old[k] =
                0.25
                + 0.01 * x_index
                + 0.02 * y_index
                + 0.001
                    * x_index
                    * y_index
                + 0.0003
                    * x_index
                    * x_index
                - 0.0002
                    * y_index
                    * y_index;

            u[k] =
                0.30
                + 0.02 * x_index
                - 0.01 * y_index;

            v[k] =
                -0.20
                + 0.005 * x_index
                + 0.015 * y_index;
        }
    }

    const stencil::Parameters parameters{
        .nx = nx,
        .ny = ny,
        .dx = 0.013,
        .dy = 0.021,
        .diffusivity = 0.004,
        .dt = 0.0002,
    };

    stencil::advance_scalar(
        phi_old,
        u,
        v,
        phi_new,
        parameters
    );

    // max_digits10 guarantees enough decimal digits
    // to reconstruct an IEEE-754 double exactly.
    std::cout
        << std::setprecision(17);

    std::cout
        << "k,phi_new\n";

    for (
        std::size_t k = 0;
        k < size;
        ++k
    )
    {
        std::cout
            << k
            << ','
            << phi_new[k]
            << '\n';
    }

    return 0;
}
