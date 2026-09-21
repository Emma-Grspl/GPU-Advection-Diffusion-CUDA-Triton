cmake_minimum_required(VERSION 3.20)

project(
    GPUAdvectionDiffusionCUDATriton
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ------------------------------------------------------------
# CPU reference implementation
# ------------------------------------------------------------

add_library(
    stencil_cpu
    STATIC
    cpu/src/AdvectionDiffusion.cpp
)

target_include_directories(
    stencil_cpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu/include
)

if(
    CMAKE_CXX_COMPILER_ID
    MATCHES "GNU|Clang|AppleClang"
)
    target_compile_options(
        stencil_cpu
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

enable_testing()

add_executable(
    test_cpu_reference
    tests/test_cpu_reference.cpp
)

target_link_libraries(
    test_cpu_reference
    PRIVATE
        stencil_cpu
)

add_test(
    NAME cpu_reference
    COMMAND test_cpu_reference
)


# ------------------------------------------------------------
# Deterministic CPU reference case
# ------------------------------------------------------------

add_executable(
    cpu_reference_case
    cpu/src/reference_case.cpp
)

target_link_libraries(
    cpu_reference_case
    PRIVATE
        stencil_cpu
)


# ------------------------------------------------------------
# Generic CPU validation runner
# ------------------------------------------------------------

add_executable(
    cpu_stencil_cli
    cpu/src/stencil_cli.cpp
)

target_link_libraries(
    cpu_stencil_cli
    PRIVATE
        stencil_cpu
)


# ------------------------------------------------------------
# Optional CUDA backend
# ------------------------------------------------------------

option(
    STENCIL_ENABLE_CUDA
    "Build the CUDA stencil backend"
    OFF
)

if(STENCIL_ENABLE_CUDA)
    include(CheckLanguage)

    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(
            FATAL_ERROR
            "STENCIL_ENABLE_CUDA=ON but no CUDA compiler was found."
        )
    endif()

    enable_language(CUDA)

    find_package(
        CUDAToolkit
        REQUIRED
    )

    set(
        CMAKE_CUDA_STANDARD
        20
    )

    set(
        CMAKE_CUDA_STANDARD_REQUIRED
        ON
    )

    add_library(
        stencil_cuda
        STATIC
        cuda/src/AdvectionDiffusionCuda.cu
        cuda/src/CudaStencilExecutor.cu
    )

    target_include_directories(
        stencil_cuda
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/cuda/include
    )

    target_link_libraries(
        stencil_cuda
        PUBLIC
            CUDA::cudart
    )

    set_target_properties(
        stencil_cuda
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(
        cuda_stencil_cli
        cuda/src/stencil_cuda_cli.cu
    )

    target_link_libraries(
        cuda_stencil_cli
        PRIVATE
            stencil_cuda
    )

    add_executable(
        benchmark_cuda_c0
        cuda/src/benchmark_cuda_c0.cu
    )

    target_link_libraries(
        benchmark_cuda_c0
        PRIVATE
            stencil_cuda
    )

    add_executable(
        profile_cuda_c0
        cuda/src/profile_cuda_c0.cu
    )

    target_link_libraries(
        profile_cuda_c0
        PRIVATE
            stencil_cuda
    )

    add_executable(
        benchmark_cuda_blocks
        cuda/src/benchmark_cuda_blocks.cu
    )

    target_link_libraries(
        benchmark_cuda_blocks
        PRIVATE
            stencil_cuda
    )

    add_executable(
        validate_cuda_shared
        cuda/src/validate_cuda_shared.cu
    )

    target_link_libraries(
        validate_cuda_shared
        PRIVATE
            stencil_cuda
    )

    add_executable(
        benchmark_cuda_c0_vs_c2
        cuda/src/benchmark_cuda_c0_vs_c2.cu
    )

    target_link_libraries(
        benchmark_cuda_c0_vs_c2
        PRIVATE
            stencil_cuda
    )

    add_executable(
        profile_cuda_c2
        cuda/src/profile_cuda_c2.cu
    )

    target_link_libraries(
        profile_cuda_c2
        PRIVATE
            stencil_cuda
    )

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(
            STATUS
            "CMAKE_CUDA_ARCHITECTURES was not set; "
            "the CUDA compiler default will be used."
        )
    endif()
endif()
