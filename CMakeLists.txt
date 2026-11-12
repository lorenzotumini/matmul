cmake_minimum_required(VERSION 3.21)
project(matmul_lab VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(matmul STATIC src/registry.c
            src/cpu_features.c
            src/cpu_combined.c
            src/packed.c
            src/naive.c
		    src/loop_interchange.c
            src/blocked.c
            src/simd_sse.c)
target_include_directories(matmul PUBLIC include)

# AVX is optional and applies only to specialized backends. Feature detection
# and all baseline kernels must remain executable without AVX.
option(MATMUL_ENABLE_AVX "Build optional AVX backends" ON)
include(CheckCSourceCompiles)
include(CheckCCompilerFlag)
check_c_source_compiles("
#if !defined(__x86_64__) && !defined(_M_X64)
#error Not an x86-64 target
#endif
int main(void) { return 0; }
" MATMUL_TARGET_X86_64)
if(MATMUL_ENABLE_AVX AND MATMUL_TARGET_X86_64)
    if(MSVC)
        set(matmul_avx_flag /arch:AVX)
    elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        set(matmul_avx_flag -mavx)
    endif()
    if(DEFINED matmul_avx_flag)
        check_c_compiler_flag("${matmul_avx_flag}" MATMUL_COMPILER_HAS_AVX)
    endif()
endif()

# Keep OpenMP compile options on its own backend; only the runtime link
# dependency propagates to executables using the static libraries.
option(MATMUL_ENABLE_OPENMP "Build the optional OpenMP exercise" ON)
if(MATMUL_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS C)
    if(OpenMP_C_FOUND)
        add_library(matmul_openmp STATIC src/openmp.c)
        target_include_directories(matmul_openmp PRIVATE include)
        target_link_libraries(matmul_openmp PRIVATE OpenMP::OpenMP_C)
        target_link_libraries(matmul PRIVATE matmul_openmp)
        target_compile_definitions(matmul PRIVATE MATMUL_HAVE_OPENMP=1)
    endif()
endif()

include(cmake/CPUVariants.cmake)
include(cmake/OpenBLASBackend.cmake)
include(cmake/CUDABackend.cmake)
include(cmake/OpenCLBackend.cmake)

# Support code is separate from the kernels you will study.
add_library(lab_support STATIC tools/lab.c tools/gpu_bench.c)
target_include_directories(lab_support PUBLIC tools)
target_link_libraries(lab_support PUBLIC matmul)
if(NOT WIN32)
    target_link_libraries(lab_support PUBLIC m)
endif()

add_executable(matmul_bench tools/bench.c)
target_link_libraries(matmul_bench PRIVATE lab_support)
target_compile_definitions(matmul_bench PRIVATE
    LAB_COMPILER="${CMAKE_C_COMPILER_ID}-${CMAKE_C_COMPILER_VERSION}"
    LAB_BUILD="$<CONFIG>")

include(CTest)
if(BUILD_TESTING)
    add_executable(matmul_tests tests/test_matmul.c)
    target_include_directories(matmul_tests PRIVATE src)
    target_link_libraries(matmul_tests PRIVATE lab_support)
    add_test(NAME harness COMMAND matmul_tests --harness)
    set(_gpu_skip_args "")
    if(TARGET matmul_cuda)
        foreach(_implementation IN LISTS MATMUL_CUDA_IMPLEMENTATIONS)
            list(APPEND _gpu_skip_args --skip "${_implementation}")
            add_test(NAME ${_implementation} COMMAND matmul_tests --impl ${_implementation})
            set_tests_properties(${_implementation} PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()
        add_executable(matmul_cuda_timing_tests tests/test_cuda_timing.c)
        target_link_libraries(matmul_cuda_timing_tests PRIVATE lab_support)
        add_test(NAME cuda_timing COMMAND matmul_cuda_timing_tests)
        set_tests_properties(cuda_timing PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    if(TARGET matmul_opencl)
        set(_opencl_implementations opencl_naive opencl_tiled opencl_register_tiled)
        if(MATMUL_ENABLE_CLBLAST)
            list(APPEND _opencl_implementations clblast)
        endif()
        foreach(_implementation IN LISTS _opencl_implementations)
            list(APPEND _gpu_skip_args --skip ${_implementation})
            add_test(NAME ${_implementation} COMMAND matmul_tests --impl ${_implementation})
            set_tests_properties(${_implementation} PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()
        add_executable(matmul_opencl_timing_tests tests/test_opencl_timing.c)
        target_link_libraries(matmul_opencl_timing_tests PRIVATE lab_support)
        add_test(NAME opencl_timing COMMAND matmul_opencl_timing_tests)
        set_tests_properties(opencl_timing PROPERTIES SKIP_RETURN_CODE 77)
        add_test(NAME opencl_tiled_timing COMMAND matmul_opencl_timing_tests --tiled)
        set_tests_properties(opencl_tiled_timing PROPERTIES SKIP_RETURN_CODE 77)
        add_test(NAME opencl_register_timing COMMAND matmul_opencl_timing_tests --register)
        set_tests_properties(opencl_register_timing PROPERTIES SKIP_RETURN_CODE 77)
        if(MATMUL_ENABLE_CLBLAST)
            add_test(NAME clblast_timing COMMAND matmul_opencl_timing_tests --clblast)
            set_tests_properties(clblast_timing PROPERTIES SKIP_RETURN_CODE 77)
        endif()
    endif()
    add_test(NAME kernels COMMAND matmul_tests ${_gpu_skip_args})
    add_test(NAME bench_help COMMAND matmul_bench --help)
    add_test(NAME bench_bad_dimension COMMAND matmul_bench --m -1)
    add_test(NAME profile_requires_impl COMMAND matmul_bench --profile)
    add_test(NAME profile_rejects_csv COMMAND matmul_bench --profile --impl naive --csv)
    add_test(NAME workload_validation COMMAND matmul_bench --profile --impl naive
        --m 7 --n 9 --k 15 --validate-only)
    set_tests_properties(workload_validation PROPERTIES PASS_REGULAR_EXPRESSION "Validated naive")
    add_test(NAME validation_rejects_bad_repeats COMMAND matmul_bench --profile --impl naive
        --repeats 0 --validate-only)
    add_test(NAME bench_bad_timing COMMAND matmul_bench --timing unknown)
    add_test(NAME kernel_timing_requires_gpu_impl COMMAND matmul_bench --timing kernel --impl naive)
    add_test(NAME kernel_timing_rejects_profile COMMAND matmul_bench --timing kernel --profile --impl naive)
    set_tests_properties(bench_bad_dimension profile_requires_impl profile_rejects_csv
        validation_rejects_bad_repeats bench_bad_timing kernel_timing_requires_gpu_impl
        kernel_timing_rejects_profile
        PROPERTIES WILL_FAIL TRUE)
endif()

if(MATMUL_OPENBLAS_RUNTIME_FILES)
    set(_openblas_executables matmul_bench)
    if(BUILD_TESTING)
        list(APPEND _openblas_executables matmul_tests)
        if(TARGET matmul_cuda_timing_tests)
            list(APPEND _openblas_executables matmul_cuda_timing_tests)
        endif()
        if(TARGET matmul_opencl_timing_tests)
            list(APPEND _openblas_executables matmul_opencl_timing_tests)
        endif()
    endif()
    # All runner executables share one output directory. One staging owner
    # avoids concurrent POST_BUILD copies of the same DLL on Windows.
    add_custom_target(matmul_openblas_runtime
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:matmul_bench>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MATMUL_OPENBLAS_RUNTIME_FILES} "$<TARGET_FILE_DIR:matmul_bench>"
        VERBATIM)
    foreach(_target IN LISTS _openblas_executables)
        add_dependencies(${_target} matmul_openblas_runtime)
    endforeach()
endif()

set(lab_targets matmul lab_support matmul_bench ${MATMUL_CPU_VARIANT_TARGETS})
if(TARGET matmul_openmp)
    list(APPEND lab_targets matmul_openmp)
endif()
if(TARGET matmul_openblas)
    list(APPEND lab_targets matmul_openblas)
endif()
if(TARGET matmul_opencl)
    list(APPEND lab_targets matmul_opencl)
endif()
if(BUILD_TESTING)
    list(APPEND lab_targets matmul_tests)
    if(TARGET matmul_cuda_timing_tests)
        list(APPEND lab_targets matmul_cuda_timing_tests)
    endif()
    if(TARGET matmul_opencl_timing_tests)
        list(APPEND lab_targets matmul_opencl_timing_tests)
    endif()
endif()
foreach(target IN LISTS lab_targets)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(${target} PRIVATE
            "$<$<CONFIG:RelWithDebInfo>:-fno-omit-frame-pointer>")
    endif()
endforeach()
# Deliberately no global AVX, OpenMP, native-CPU, or fast-math flags.
# Add optional capabilities per target when you reach those lessons.
