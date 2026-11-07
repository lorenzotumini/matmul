option(MATMUL_ENABLE_CUDA "Build the optional CUDA exercise" OFF)
if(NOT MATMUL_ENABLE_CUDA)
    return()
endif()

if(WIN32 AND NOT MSVC)
    message(FATAL_ERROR "CUDA on Windows: use nob --cuda (clang-cl), or an MSVC-compatible C compiler.")
endif()

# CUDA 12.x works with VS 2022. Prefer that installation over a newer VS
# when nvcc initializes its host environment. Respect an explicit user choice.
if(WIN32 AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(_cuda_vswhere vswhere.exe
        PATHS "$ENV{ProgramFiles\(x86\)}/Microsoft Visual Studio/Installer")
    if(_cuda_vswhere)
        execute_process(COMMAND "${_cuda_vswhere}" -latest -version "[17.0,18.0)"
            -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64
            -property installationPath
            OUTPUT_VARIABLE _cuda_vs OUTPUT_STRIP_TRAILING_WHITESPACE)
        file(GLOB _cuda_hosts "${_cuda_vs}/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe")
        list(SORT _cuda_hosts COMPARE NATURAL ORDER DESCENDING)
        if(_cuda_hosts)
            list(GET _cuda_hosts 0 _cuda_host)
            set(CMAKE_CUDA_HOST_COMPILER "${_cuda_host}" CACHE FILEPATH "nvcc host compiler")
        endif()
    endif()
endif()

# Pascal MX330 in this laptop. Override when configuring for another GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures (MX330: 61)")
endif()
if(DEFINED CMAKE_CUDA_COMPILER AND NOT CMAKE_CUDA_COMPILER)
    unset(CMAKE_CUDA_COMPILER CACHE)
endif()
# Explicit opt-in: report compiler detection errors instead of silently skipping
# the exercise. Enable directly so the selected host compiler is preserved.
enable_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    message(FATAL_ERROR "The first CUDA exercise uses NVIDIA nvcc.")
endif()

set(MATMUL_CUDA_IMPLEMENTATIONS cuda_naive cuda_tiled cuda_register_tiled cuda_prefetch cuda_combined)
add_library(matmul_cuda STATIC src/cuda_support.cu src/cuda_naive.cu src/cuda_tiled.cu
    src/cuda_register_tiled.cu src/cuda_prefetch.cu src/cuda_combined.cu)
target_include_directories(matmul_cuda PRIVATE include)
set_target_properties(matmul_cuda PROPERTIES CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Static)
target_compile_options(matmul_cuda PRIVATE --generate-line-info)
# matmul was created before CUDA was enabled. Its static-library dependency
# now carries CUDA code, so initialize the property also on that existing target.
set_property(TARGET matmul PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_link_libraries(matmul PRIVATE matmul_cuda)
target_compile_definitions(matmul PUBLIC MATMUL_HAVE_CUDA=1)

option(MATMUL_ENABLE_CUBLAS "Build the optional cuBLAS FP32 reference" ON)
if(MATMUL_ENABLE_CUBLAS)
    find_package(CUDAToolkit QUIET)
    if(TARGET CUDA::cublas)
        target_sources(matmul_cuda PRIVATE src/cublas.cu)
        target_link_libraries(matmul_cuda PRIVATE CUDA::cublas)
        target_compile_definitions(matmul_cuda PRIVATE MATMUL_HAVE_CUBLAS=1)
        target_compile_definitions(matmul PUBLIC MATMUL_HAVE_CUBLAS=1)
        list(APPEND MATMUL_CUDA_IMPLEMENTATIONS cublas)
    else()
        message(STATUS "cuBLAS not found; keeping the learning CUDA backends")
    endif()
endif()
