#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>


namespace stencil
{

inline void check_cuda(
    const cudaError_t error,
    const char* expression,
    const char* file,
    const int line
)
{
    if (error == cudaSuccess)
    {
        return;
    }

    std::ostringstream message;

    message
        << "CUDA error at "
        << file
        << ':'
        << line
        << "\nExpression: "
        << expression
        << "\nError: "
        << cudaGetErrorString(error);

    throw std::runtime_error(
        message.str()
    );
}

}  // namespace stencil


#define STENCIL_CUDA_CHECK(expression) \
    ::stencil::check_cuda(             \
        (expression),                  \
        #expression,                   \
        __FILE__,                      \
        __LINE__                       \
    )
