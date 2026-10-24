# OpenBLAS is a comparison backend, not a prerequisite for the learning kernels.
option(MATMUL_ENABLE_OPENBLAS "Build an optional OpenBLAS SGEMM comparison" ON)
set(OPENBLAS_ROOT "" CACHE PATH "OpenBLAS installation prefix")
if(NOT MATMUL_ENABLE_OPENBLAS)
    return()
endif()

set(_openblas_hints "${OPENBLAS_ROOT}" "$ENV{OPENBLAS_ROOT}")
if(MINGW)
    list(APPEND _openblas_hints "${PROJECT_SOURCE_DIR}/build/deps/openblas/ucrt64")
    get_filename_component(_compiler_bin "${CMAKE_C_COMPILER}" DIRECTORY)
    get_filename_component(_compiler_prefix "${_compiler_bin}" DIRECTORY)
    list(APPEND _openblas_hints "${_compiler_prefix}")
endif()
find_path(MATMUL_OPENBLAS_INCLUDE_DIR cblas.h HINTS ${_openblas_hints}
    PATH_SUFFIXES include/openblas include)
if(MINGW)
    # Prefer the DLL import library, avoiding static Fortran/runtime dependencies.
    find_library(MATMUL_OPENBLAS_LIBRARY NAMES libopenblas.dll.a openblas
        HINTS ${_openblas_hints} PATH_SUFFIXES lib lib64)
else()
    find_library(MATMUL_OPENBLAS_LIBRARY NAMES openblas
        HINTS ${_openblas_hints} PATH_SUFFIXES lib lib64)
endif()

if(MATMUL_OPENBLAS_INCLUDE_DIR AND MATMUL_OPENBLAS_LIBRARY)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES "${MATMUL_OPENBLAS_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${MATMUL_OPENBLAS_LIBRARY}")
    check_c_source_compiles([=[
#include <cblas.h>
int main(void) {
    _Static_assert(sizeof(blasint) == 4 || sizeof(blasint) == 8, "OpenBLAS integer ABI");
    float a = 1, b = 1, c = 0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, 1, 1, &a, 1, &b, 1, 0, &c, 1);
    return openblas_get_num_threads() + openblas_get_parallel();
}
]=] MATMUL_OPENBLAS_LINKS)
    cmake_pop_check_state()
    if(MATMUL_OPENBLAS_LINKS)
        add_library(matmul_openblas STATIC src/openblas.c)
        target_include_directories(matmul_openblas PRIVATE include)
        target_include_directories(matmul_openblas SYSTEM PRIVATE "${MATMUL_OPENBLAS_INCLUDE_DIR}")
        target_link_libraries(matmul_openblas PRIVATE "${MATMUL_OPENBLAS_LIBRARY}")
        target_link_libraries(matmul PRIVATE matmul_openblas)
        # The runner uses this definition to report library metadata outside timing.
        target_compile_definitions(matmul PUBLIC MATMUL_HAVE_OPENBLAS=1)

        if(WIN32)
            get_filename_component(_openblas_libdir "${MATMUL_OPENBLAS_LIBRARY}" DIRECTORY)
            get_filename_component(_openblas_prefix "${_openblas_libdir}" DIRECTORY)
            # Stage only named runtime dependencies, never an entire toolchain bin directory.
            foreach(_name libopenblas.dll openblas.dll libgfortran-5.dll libquadmath-0.dll
                          libgcc_s_seh-1.dll libgomp-1.dll libwinpthread-1.dll)
                unset(_openblas_dll CACHE)
                find_file(_openblas_dll NAMES "${_name}"
                    PATHS "${_openblas_prefix}/bin" "${_openblas_prefix}/lib"
                          "${_compiler_bin}" ENV PATH NO_DEFAULT_PATH)
                if(_openblas_dll)
                    list(APPEND MATMUL_OPENBLAS_RUNTIME_FILES "${_openblas_dll}")
                endif()
            endforeach()
            unset(_openblas_dll CACHE)
        endif()
    endif()
endif()
