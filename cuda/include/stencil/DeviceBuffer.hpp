#pragma once

#include "stencil/CudaError.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>


namespace stencil
{

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() noexcept = default;


    explicit DeviceBuffer(
        const std::size_t size
    )
        : size_(size)
    {
        if (size_ == 0)
        {
            return;
        }

        STENCIL_CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(
                    &data_
                ),
                size_ * sizeof(T)
            )
        );
    }


    ~DeviceBuffer() noexcept
    {
        reset();
    }


    DeviceBuffer(
        const DeviceBuffer&
    ) = delete;


    DeviceBuffer& operator=(
        const DeviceBuffer&
    ) = delete;


    DeviceBuffer(
        DeviceBuffer&& other
    ) noexcept
        : data_(other.data_),
          size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }


    DeviceBuffer& operator=(
        DeviceBuffer&& other
    ) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        reset();

        data_ = other.data_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;

        return *this;
    }


    [[nodiscard]]
    T* data() noexcept
    {
        return data_;
    }


    [[nodiscard]]
    const T* data() const noexcept
    {
        return data_;
    }


    [[nodiscard]]
    std::size_t size() const noexcept
    {
        return size_;
    }


    void upload(
        const std::span<const T> host
    )
    {
        check_size(
            host.size()
        );

        STENCIL_CUDA_CHECK(
            cudaMemcpy(
                data_,
                host.data(),
                size_ * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );
    }


    void download(
        const std::span<T> host
    ) const
    {
        check_size(
            host.size()
        );

        STENCIL_CUDA_CHECK(
            cudaMemcpy(
                host.data(),
                data_,
                size_ * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );
    }


private:
    void check_size(
        const std::size_t candidate
    ) const
    {
        if (candidate != size_)
        {
            throw std::invalid_argument(
                "Host/device buffer size mismatch."
            );
        }
    }


    void reset() noexcept
    {
        if (data_ != nullptr)
        {
            // A destructor must never throw.
            cudaFree(
                data_
            );

            data_ = nullptr;
            size_ = 0;
        }
    }


    T* data_ = nullptr;

    std::size_t size_ = 0;
};

}  // namespace stencil
