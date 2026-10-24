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

# Support code is separate from the kernels you will study.
add_library(lab_support STATIC tools/lab.c)
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
    add_test(NAME kernels COMMAND matmul_tests)
    add_test(NAME bench_help COMMAND matmul_bench --help)
    add_test(NAME bench_bad_dimension COMMAND matmul_bench --m -1)
    add_test(NAME profile_requires_impl COMMAND matmul_bench --profile)
    add_test(NAME profile_rejects_csv COMMAND matmul_bench --profile --impl naive --csv)
    add_test(NAME workload_validation COMMAND matmul_bench --profile --impl naive
        --m 7 --n 9 --k 15 --validate-only)
    set_tests_properties(workload_validation PROPERTIES PASS_REGULAR_EXPRESSION "Validated naive")
    add_test(NAME validation_rejects_bad_repeats COMMAND matmul_bench --profile --impl naive
        --repeats 0 --validate-only)
    set_tests_properties(bench_bad_dimension profile_requires_impl profile_rejects_csv
        validation_rejects_bad_repeats
        PROPERTIES WILL_FAIL TRUE)
endif()

if(MATMUL_OPENBLAS_RUNTIME_FILES)
    set(_openblas_executables matmul_bench)
    if(BUILD_TESTING)
        list(APPEND _openblas_executables matmul_tests)
    endif()
    foreach(_target IN LISTS _openblas_executables)
        add_custom_command(TARGET ${_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${MATMUL_OPENBLAS_RUNTIME_FILES} "$<TARGET_FILE_DIR:${_target}>"
            VERBATIM)
    endforeach()
endif()

set(lab_targets matmul lab_support matmul_bench ${MATMUL_CPU_VARIANT_TARGETS})
if(TARGET matmul_openmp)
    list(APPEND lab_targets matmul_openmp)
endif()
if(TARGET matmul_openblas)
    list(APPEND lab_targets matmul_openblas)
endif()
if(BUILD_TESTING)
    list(APPEND lab_targets matmul_tests)
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
