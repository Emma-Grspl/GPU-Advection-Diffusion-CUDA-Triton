#include "stencil/AdvectionDiffusion.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>


namespace
{

void require(
    const bool condition,
    const std::string_view message
)
{
    if (!condition)
    {
        throw std::runtime_error(
            std::string(message)
        );
    }
}


void require_close(
    const double actual,
    const double expected,
    const double tolerance,
    const std::string_view message
)
{
    if (
        std::abs(actual - expected)
        > tolerance
    )
    {
        throw std::runtime_error(
            std::string(message)
        );
    }
}


void test_flat_index()
{
    constexpr std::size_t nx = 11;

    require(
        stencil::flat_index(0, 0, nx) == 0,
        "flat_index failed at (0, 0)"
    );

    require(
        stencil::flat_index(1, 0, nx) == 1,
        "x must be contiguous"
    );

    require(
        stencil::flat_index(0, 1, nx) == nx,
        "y stride must equal nx"
    );

    require(
        stencil::flat_index(7, 3, nx)
            == 7 + 3 * nx,
        "general flat_index failed"
    );
}


void test_constant_field()
{
    constexpr std::size_t nx = 17;
    constexpr std::size_t ny = 13;

    const std::size_t size =
        nx * ny;

    std::vector<double> phi(
        size,
        3.25
    );

    std::vector<double> u(
        size,
        0.7
    );

    std::vector<double> v(
        size,
        -0.2
    );

    std::vector<double> result(
        size,
        0.0
    );

    const stencil::Parameters parameters{
        .nx = nx,
        .ny = ny,
        .dx = 0.01,
        .dy = 0.015,
        .diffusivity = 1.0e-4,
        .dt = 1.0e-4,
    };

    stencil::advance_scalar(
        phi,
        u,
        v,
        result,
        parameters
    );

    for (
        std::size_t k = 0;
        k < size;
        ++k
    )
    {
        require_close(
            result[k],
            phi[k],
            0.0,
            "constant field was modified"
        );
    }
}


void test_boundaries_are_preserved()
{
    constexpr std::size_t nx = 9;
    constexpr std::size_t ny = 7;

    const std::size_t size =
        nx * ny;

    std::vector<double> phi(size);
    std::vector<double> u(size);
    std::vector<double> v(size);
    std::vector<double> result(
        size,
        0.0
    );

    for (
        std::size_t k = 0;
        k < size;
        ++k
    )
    {
        phi[k] =
            0.1
            * static_cast<double>(k);

        u[k] =
            0.2
            + 0.01
            * static_cast<double>(k);

        v[k] =
            -0.3
            + 0.005
            * static_cast<double>(k);
    }

    const stencil::Parameters parameters{
        .nx = nx,
        .ny = ny,
        .dx = 0.1,
        .dy = 0.2,
        .diffusivity = 0.01,
        .dt = 0.001,
    };

    stencil::advance_scalar(
        phi,
        u,
        v,
        result,
        parameters
    );

    for (
        std::size_t i = 0;
        i < nx;
        ++i
    )
    {
        const auto bottom =
            stencil::flat_index(
                i,
                0,
                nx
            );

        const auto top =
            stencil::flat_index(
                i,
                ny - 1,
                nx
            );

        require_close(
            result[bottom],
            phi[bottom],
            0.0,
            "bottom boundary modified"
        );

        require_close(
            result[top],
            phi[top],
            0.0,
            "top boundary modified"
        );
    }

    for (
        std::size_t j = 0;
        j < ny;
        ++j
    )
    {
        const auto left =
            stencil::flat_index(
                0,
                j,
                nx
            );

        const auto right =
            stencil::flat_index(
                nx - 1,
                j,
                nx
            );

        require_close(
            result[left],
            phi[left],
            0.0,
            "left boundary modified"
        );

        require_close(
            result[right],
            phi[right],
            0.0,
            "right boundary modified"
        );
    }
}


void test_single_cell_formula()
{
    constexpr std::size_t nx = 5;
    constexpr std::size_t ny = 5;

    const std::size_t size =
        nx * ny;

    std::vector<double> phi(
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

    std::vector<double> result(
        size,
        0.0
    );

    constexpr std::size_t i = 2;
    constexpr std::size_t j = 2;

    const auto center =
        stencil::flat_index(
            i,
            j,
            nx
        );

    const auto left =
        stencil::flat_index(
            i - 1,
            j,
            nx
        );

    const auto right =
        stencil::flat_index(
            i + 1,
            j,
            nx
        );

    const auto bottom =
        stencil::flat_index(
            i,
            j - 1,
            nx
        );

    const auto top =
        stencil::flat_index(
            i,
            j + 1,
            nx
        );

    phi[center] = 4.0;
    phi[left] = 1.0;
    phi[right] = 7.0;
    phi[bottom] = 2.0;
    phi[top] = 8.0;

    u[center] = 0.5;
    v[center] = -0.25;

    const stencil::Parameters parameters{
        .nx = nx,
        .ny = ny,
        .dx = 0.2,
        .dy = 0.4,
        .diffusivity = 0.03,
        .dt = 0.01,
    };

    const double dphi_dx =
        (7.0 - 1.0)
        / (2.0 * parameters.dx);

    const double dphi_dy =
        (8.0 - 2.0)
        / (2.0 * parameters.dy);

    const double laplacian =
        (
            7.0
            - 2.0 * 4.0
            + 1.0
        )
        / (
            parameters.dx
            * parameters.dx
        )
        +
        (
            8.0
            - 2.0 * 4.0
            + 2.0
        )
        / (
            parameters.dy
            * parameters.dy
        );

    const double expected =
        4.0
        - parameters.dt
        * (
            0.5 * dphi_dx
            - 0.25 * dphi_dy
        )
        + parameters.diffusivity
        * parameters.dt
        * laplacian;

    stencil::advance_scalar(
        phi,
        u,
        v,
        result,
        parameters
    );

    require_close(
        result[center],
        expected,
        1.0e-14,
        "single-cell stencil formula failed"
    );
}


void test_in_place_update_is_rejected()
{
    constexpr std::size_t nx = 5;
    constexpr std::size_t ny = 5;

    std::vector<double> phi(
        nx * ny,
        1.0
    );

    std::vector<double> u(
        nx * ny,
        0.0
    );

    std::vector<double> v(
        nx * ny,
        0.0
    );

    const stencil::Parameters parameters{
        .nx = nx,
        .ny = ny,
        .dx = 0.1,
        .dy = 0.1,
        .diffusivity = 0.01,
        .dt = 0.001,
    };

    bool exception_thrown = false;

    try
    {
        stencil::advance_scalar(
            phi,
            u,
            v,
            phi,
            parameters
        );
    }
    catch (
        const std::invalid_argument&
    )
    {
        exception_thrown = true;
    }

    require(
        exception_thrown,
        "in-place update must be rejected"
    );
}

}  // namespace


int main()
{
    try
    {
        test_flat_index();
        test_constant_field();
        test_boundaries_are_preserved();
        test_single_cell_formula();
        test_in_place_update_is_rejected();

        std::cout
            << "All CPU reference tests passed.\n";

        return 0;
    }
    catch (
        const std::exception& exception
    )
    {
        std::cerr
            << "TEST FAILURE: "
            << exception.what()
            << '\n';

        return 1;
    }
}
