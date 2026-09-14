#pragma once

#include <cstddef>
#include <span>


namespace stencil
{

struct Parameters
{
    std::size_t nx;
    std::size_t ny;

    double dx;
    double dy;

    double diffusivity;
    double dt;
};


[[nodiscard]]
constexpr std::size_t flat_index(
    const std::size_t i,
    const std::size_t j,
    const std::size_t nx
) noexcept
{
    return i + j * nx;
}


void advance_scalar(
    std::span<const double> phi_old,
    std::span<const double> u,
    std::span<const double> v,
    std::span<double> phi_new,
    const Parameters& parameters
);

}  // namespace stencil
