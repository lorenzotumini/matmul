option(MATMUL_ENABLE_OPENCL "Build the optional Intel GPU OpenCL exercise" OFF)
if(NOT MATMUL_ENABLE_OPENCL)
    return()
endif()

# FindOpenCL can use CUDA's vendor-neutral headers/import library on Windows.
# The system ICD loader selects the Intel runtime, independently of that SDK.
find_package(OpenCL 1.2 REQUIRED)
add_library(matmul_opencl STATIC src/opencl_support.c)
target_include_directories(matmul_opencl PRIVATE include)
target_link_libraries(matmul_opencl PRIVATE OpenCL::OpenCL)
target_compile_definitions(matmul_opencl PRIVATE
    MATMUL_OPENCL_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/opencl_naive.cl"
    MATMUL_OPENCL_TILED_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/opencl_tiled.cl")
if(MSVC)
    # Use portable C17 stdio rather than the MS CRT's optional fopen_s API.
    target_compile_definitions(matmul_opencl PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
target_link_libraries(matmul PRIVATE matmul_opencl)
target_compile_definitions(matmul PUBLIC MATMUL_HAVE_OPENCL=1)
