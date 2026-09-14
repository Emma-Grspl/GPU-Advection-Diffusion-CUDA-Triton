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
