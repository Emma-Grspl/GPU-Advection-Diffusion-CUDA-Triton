#pragma once

#include "stencil/AdvectionDiffusionCuda.hpp"
#include "stencil/DeviceBuffer.hpp"

#include <cstddef>
#include <span>


namespace stencil
{

class CudaStencilExecutor
{
public:
    CudaStencilExecutor(
        std::size_t nx,
        std::size_t ny
    );


    void upload(
        std::span<const double> phi,
        std::span<const double> u,
        std::span<const double> v
    );


    void advance(
        const CudaParameters& parameters,
        std::size_t steps,
        const CudaLaunchConfig& launch = {}
    );


    void synchronize() const;


    void download(
        std::span<double> output
    ) const;


private:
    [[nodiscard]]
    double* current_phi() noexcept;


    [[nodiscard]]
    const double* current_phi() const noexcept;


    [[nodiscard]]
    double* next_phi() noexcept;


    std::size_t nx_;
    std::size_t ny_;
    std::size_t size_;

    DeviceBuffer<double> phi_a_;
    DeviceBuffer<double> phi_b_;

    DeviceBuffer<double> u_;
    DeviceBuffer<double> v_;

    bool current_is_a_ = true;
};

}  // namespace stencil
