#include "stencil/AdvectionDiffusion.hpp"

#include <algorithm>
#include <stdexcept>


namespace stencil
{
namespace
{

void validate_inputs(
    const std::span<const double> phi_old,
    const std::span<const double> u,
    const std::span<const double> v,
    const std::span<double> phi_new,
    const Parameters& parameters
)
{
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

    if (parameters.diffusivity < 0.0)
    {
        throw std::invalid_argument(
            "diffusivity cannot be negative."
        );
    }

    if (parameters.dt <= 0.0)
    {
        throw std::invalid_argument(
            "dt must be strictly positive."
        );
    }

    const std::size_t expected_size =
        parameters.nx * parameters.ny;

    if (
        phi_old.size() != expected_size
        || u.size() != expected_size
        || v.size() != expected_size
        || phi_new.size() != expected_size
    )
    {
        throw std::invalid_argument(
            "Input and output buffers must have size nx * ny."
        );
    }

    if (phi_old.data() == phi_new.data())
    {
        throw std::invalid_argument(
            "phi_old and phi_new must be distinct buffers."
        );
    }
}

}  // namespace


void advance_scalar(
    const std::span<const double> phi_old,
    const std::span<const double> u,
    const std::span<const double> v,
    const std::span<double> phi_new,
    const Parameters& parameters
)
{
    validate_inputs(
        phi_old,
        u,
        v,
        phi_new,
        parameters
    );

    std::copy(
        phi_old.begin(),
        phi_old.end(),
        phi_new.begin()
    );

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

    for (
        std::size_t j = 1;
        j < parameters.ny - 1;
        ++j
    )
    {
        for (
            std::size_t i = 1;
            i < parameters.nx - 1;
            ++i
        )
        {
            const std::size_t center =
                flat_index(
                    i,
                    j,
                    parameters.nx
                );

            const std::size_t left =
                flat_index(
                    i - 1,
                    j,
                    parameters.nx
                );

            const std::size_t right =
                flat_index(
                    i + 1,
                    j,
                    parameters.nx
                );

            const std::size_t bottom =
                flat_index(
                    i,
                    j - 1,
                    parameters.nx
                );

            const std::size_t top =
                flat_index(
                    i,
                    j + 1,
                    parameters.nx
                );

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
                    - 2.0 * phi_old[center]
                    + phi_old[left]
                )
                * inv_dx2
                +
                (
                    phi_old[top]
                    - 2.0 * phi_old[center]
                    + phi_old[bottom]
                )
                * inv_dy2;

            const double advection =
                u[center] * dphi_dx
                +
                v[center] * dphi_dy;

            phi_new[center] =
                phi_old[center]
                - parameters.dt * advection
                + parameters.diffusivity
                * parameters.dt
                * laplacian;
        }
    }
}

}  // namespace stencil
