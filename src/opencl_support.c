#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "opencl_bench.h"
#include <stdint.h>
#include <limits.h>
#ifdef MATMUL_HAVE_CLBLAST
#include <clblast_c.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void checked(cl_int error, const char *operation) {
    if (error != CL_SUCCESS) {
        fprintf(stderr, "OpenCL %s failed (%d).\n", operation, (int)error);
        exit(EXIT_FAILURE);
    }
}
#define CL_CHECK(operation) checked((operation), #operation)

/* One cached runtime per process, used from one host thread. Compilation and
 * driver initialization occur at the first validation call, before sampling.
 */
static struct {
    bool searched;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program[5];
    cl_kernel kernel[5];
    char name[256], driver[256];
    cl_bool unified_memory;
} runtime;

bool matmul_opencl_available(void) {
    if (runtime.searched) return runtime.device != NULL;
    runtime.searched = true;
    cl_uint count = 0;
    /* No installed ICD/platform is an unavailable backend, not a CPU fallback. */
    if (clGetPlatformIDs(0, NULL, &count) != CL_SUCCESS || !count) return false;
    cl_platform_id *platforms = malloc(count * sizeof *platforms);
    if (!platforms) return false;
    CL_CHECK(clGetPlatformIDs(count, platforms, NULL));
    for (cl_uint i = 0; i < count && !runtime.device; ++i) {
        cl_uint devices_count = 0;
        cl_int error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &devices_count);
        if (error == CL_DEVICE_NOT_FOUND) continue;
        CL_CHECK(error);
        cl_device_id *devices = malloc(devices_count * sizeof *devices);
        if (!devices) break;
        CL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, devices_count, devices, NULL));
        for (cl_uint j = 0; j < devices_count; ++j) {
            cl_uint vendor;
            cl_bool available, compiler;
            CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR_ID, sizeof vendor, &vendor, NULL));
            CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_AVAILABLE, sizeof available, &available, NULL));
            CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_COMPILER_AVAILABLE, sizeof compiler, &compiler, NULL));
            if (vendor == 0x8086 && available && compiler) {
                runtime.device = devices[j];
                break;
            }
        }
        free(devices);
    }
    free(platforms);
    if (!runtime.device) return false;
    CL_CHECK(clGetDeviceInfo(runtime.device, CL_DEVICE_NAME, sizeof runtime.name, runtime.name, NULL));
    CL_CHECK(clGetDeviceInfo(runtime.device, CL_DRIVER_VERSION, sizeof runtime.driver, runtime.driver, NULL));
    CL_CHECK(clGetDeviceInfo(runtime.device, CL_DEVICE_HOST_UNIFIED_MEMORY,
                            sizeof runtime.unified_memory, &runtime.unified_memory, NULL));
    return true;
}

const char *matmul_opencl_device_name(void) {
    return matmul_opencl_available() ? runtime.name : "unavailable";
}
const char *matmul_opencl_driver_version(void) {
    return matmul_opencl_available() ? runtime.driver : "unavailable";
}
bool matmul_opencl_host_unified_memory(void) {
    return matmul_opencl_available() && runtime.unified_memory == CL_TRUE;
}

static void shutdown_runtime(void) {
    for (int i = 0; i < 5; ++i) {
        if (runtime.kernel[i]) clReleaseKernel(runtime.kernel[i]);
        if (runtime.program[i]) clReleaseProgram(runtime.program[i]);
    }
    clReleaseCommandQueue(runtime.queue);
    clReleaseContext(runtime.context);
}

static void initialize_runtime(void) {
    if (!matmul_opencl_available()) {
        fprintf(stderr, "OpenCL: no available Intel GPU/compiler.\n");
        exit(EXIT_FAILURE);
    }
    cl_int error;
    if (!runtime.context) {
        runtime.context = clCreateContext(NULL, 1, &runtime.device, NULL, NULL, &error);
        CL_CHECK(error);
        runtime.queue = clCreateCommandQueue(runtime.context, runtime.device, CL_QUEUE_PROFILING_ENABLE, &error);
        CL_CHECK(error);
        if (atexit(shutdown_runtime) != 0) {
            fprintf(stderr, "Cannot register OpenCL cleanup.\n");
            exit(EXIT_FAILURE);
        }
    }
}

/* Fixed offline-tuned choices, shared by host and prepared timing paths.
 * 0=naive, 2=register baseline, 3=32x64 prefetched, 4=64x64 prefetched.
 */
static int combined_variant(size_t m, size_t n, size_t k) {
    double work = (double)m * (double)n * (double)k;
    if (work < 32.0*32*32) return 0;
    if (work < 128.0*128*128 || m < 32 || n < 32 || k < 16 ||
        (m < 128 && n < 512)) return 2;
    if (m >= 128 && m < 512 && n >= 128 && n < 512 &&
        m*n >= 256*256 && k >= 128) return 4;
    return 3;
}
const char *matmul_opencl_combined_strategy(size_t m, size_t n, size_t k) {
    switch (combined_variant(m,n,k)) {
    case 0: return "naive";
    case 2: return "register-32x32";
    case 4: return "prefetch-64x64,wg16x8,register8x4";
    default: return "prefetch-32x64,wg16x8,register4x4";
    }
}

static void initialize(int variant) {
    initialize_runtime();
    if (runtime.kernel[variant]) return;
    const char *path = variant >= 3 ? MATMUL_OPENCL_COMBINED_SOURCE_PATH :
                       variant == 2 ? MATMUL_OPENCL_REGISTER_SOURCE_PATH :
                       variant == 1 ? MATMUL_OPENCL_TILED_SOURCE_PATH : MATMUL_OPENCL_SOURCE_PATH;
    if (!matmul_opencl_available()) {
        fprintf(stderr, "OpenCL: no available Intel GPU with a compiler.\n");
        exit(EXIT_FAILURE);
    }
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "Cannot read OpenCL exercise: %s\n", path);
        exit(EXIT_FAILURE);
    }
    long length = ftell(file);
    if (length <= 0 || length > 1024*1024 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Invalid OpenCL source size.\n");
        exit(EXIT_FAILURE);
    }
    char *source = malloc((size_t)length + 1);
    if (!source || fread(source, 1, (size_t)length, file) != (size_t)length) {
        fprintf(stderr, "Cannot load OpenCL source.\n");
        exit(EXIT_FAILURE);
    }
    fclose(file);
    source[length] = '\0';
    cl_int error;
    const char *text = source;
    runtime.program[variant] = clCreateProgramWithSource(runtime.context, 1, &text, NULL, &error);
    CL_CHECK(error);
    free(source);
    const char *options = variant == 3 ?
        "-cl-std=CL1.2 -DWX=16 -DWY=8 -DRM=4 -DRN=4 -DPREFETCH=1" : variant == 4 ?
        "-cl-std=CL1.2 -DWX=16 -DWY=8 -DRM=8 -DRN=4 -DPREFETCH=1" : "-cl-std=CL1.2";
    error = clBuildProgram(runtime.program[variant], 1, &runtime.device, options, NULL, NULL);
    if (error != CL_SUCCESS) {
        size_t bytes = 0;
        CL_CHECK(clGetProgramBuildInfo(runtime.program[variant], runtime.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &bytes));
        char *log = calloc(bytes + 1, 1);
        if (log) {
            CL_CHECK(clGetProgramBuildInfo(runtime.program[variant], runtime.device, CL_PROGRAM_BUILD_LOG, bytes, log, NULL));
            fprintf(stderr, "OpenCL compiler log (%s):\n%s\n", path, log);
            free(log);
        }
        CL_CHECK(error);
    }
    runtime.kernel[variant] = clCreateKernel(runtime.program[variant], variant >= 3 ? "matmul_opencl_combined" : variant == 2 ? "matmul_opencl_register_tiled" : variant == 1 ? "matmul_opencl_tiled" : "matmul_opencl_naive", &error);
    CL_CHECK(error);

}

struct MatmulOpenCLBenchmark {
    cl_ulong m, n, k;
    size_t global[2], local[2], c_bytes;
    bool tiled, library;
    cl_kernel kernel;
    cl_mem a, b, c;
};

static size_t bytes_for(size_t count) {
    if (count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "OpenCL buffer size overflows size_t.\n");
        exit(EXIT_FAILURE);
    }
    return count * sizeof(float);
}

bool matmul_opencl_kernel_timing_supported(MatmulFn implementation) {
    return implementation == matmul_opencl_naive || implementation == matmul_opencl_tiled ||
           implementation == matmul_opencl_register_tiled || implementation == matmul_opencl_combined
#ifdef MATMUL_HAVE_CLBLAST
           || implementation == matmul_clblast
#endif
           ;
}

#ifdef MATMUL_HAVE_CLBLAST
const char *matmul_clblast_build_options(void) {
    static bool prepared;
    if (!prepared && matmul_opencl_available()) {
        const char *user = getenv("CLBLAST_BUILD_OPTIONS");
        /* This old Intel compiler advertises __builtin_assume but crashes in
         * its SPIR-V writer. Disabling optional hints preserves arithmetic.
         * CLBlast exposes build options through the PROCESS environment only.
         */
        const char *fix = strcmp(runtime.driver, "30.0.100.9864") == 0 ?
            " -D__has_builtin(x)=0 " : " ";
        size_t bytes = (user ? strlen(user) : 0) + strlen(fix) + 1;
        char *options = malloc(bytes);
        if (!options) { fprintf(stderr, "Cannot prepare CLBlast build options.\n"); exit(EXIT_FAILURE); }
        snprintf(options, bytes, "%s%s", user ? user : "", fix);
        /* A trailing space is required by CLBlast 1.7.0's option concatenation. */
#ifdef _WIN32
        int error = _putenv_s("CLBLAST_BUILD_OPTIONS", options);
#else
        int error = setenv("CLBLAST_BUILD_OPTIONS", options, 1);
#endif
        free(options);
        if (error) { fprintf(stderr, "Cannot set process-local CLBlast build options.\n"); exit(EXIT_FAILURE); }
        prepared = true;
    }
    const char *options = getenv("CLBLAST_BUILD_OPTIONS");
    return options ? options : "";
}
#endif

MatmulOpenCLBenchmark *matmul_opencl_benchmark_create(MatmulFn implementation,
    size_t m, size_t n, size_t k, const float *a, const float *b) {
    if (!matmul_opencl_kernel_timing_supported(implementation) || !m || !n || !k) return NULL;
    MatmulOpenCLBenchmark *device = malloc(sizeof *device);
    if (!device) return NULL;
    device->library = false;
#ifdef MATMUL_HAVE_CLBLAST
    device->library = implementation == matmul_clblast;
    if (device->library) (void)matmul_clblast_build_options();
#endif
    int variant = implementation == matmul_opencl_combined ? combined_variant(m,n,k) :
                  implementation == matmul_opencl_register_tiled ? 2 :
                  implementation == matmul_opencl_tiled ? 1 : 0;
    device->tiled = !device->library && variant != 0;
    initialize_runtime();
    device->kernel = NULL;
    if (!device->library) {
        initialize(variant);
        device->kernel = runtime.kernel[variant];
    } else if (m > INT_MAX || n > INT_MAX || k > INT_MAX ||
               m*n > INT_MAX || m*k > INT_MAX || k*n > INT_MAX) {
        fprintf(stderr, "CLBlast dimensions/element counts exceed 32-bit kernel indexing.\n");
        exit(EXIT_FAILURE);
    }
    device->m = (cl_ulong)m; device->n = (cl_ulong)n; device->k = (cl_ulong)k;
    device->local[0] = 16; device->local[1] = variant >= 3 ? 8 : 16;
    device->global[0] = n; device->global[1] = m;
    if (device->tiled) {
        size_t tile_rows = variant == 4 ? 64 : variant >= 2 ? 32 : 16;
        size_t tile_cols = variant >= 3 ? 64 : variant == 2 ? 32 : 16;
        if (m > SIZE_MAX - (tile_rows - 1) || n > SIZE_MAX - (tile_cols - 1)) {
            fprintf(stderr, "OpenCL output grid overflows size_t.\n"); exit(EXIT_FAILURE);
        }
        device->global[0] = ((n + tile_cols - 1) / tile_cols) * device->local[0];
        device->global[1] = ((m + tile_rows - 1) / tile_rows) * device->local[1];
        size_t limit;
        CL_CHECK(clGetKernelWorkGroupInfo(device->kernel, runtime.device,
            CL_KERNEL_WORK_GROUP_SIZE, sizeof limit, &limit, NULL));
        if (limit < device->local[0] * device->local[1]) {
            fprintf(stderr, "OpenCL selected work-group exceeds the kernel limit.\n");
            exit(EXIT_FAILURE);
        }
    }
    device->c_bytes = bytes_for(m*n);
    size_t a_bytes = bytes_for(m*k), b_bytes = bytes_for(k*n);
    cl_int error;
    device->a = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY, a_bytes, NULL, &error);
    CL_CHECK(error);
    device->b = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY, b_bytes, NULL, &error);
    CL_CHECK(error);
    device->c = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, device->c_bytes, NULL, &error);
    CL_CHECK(error);
    CL_CHECK(clEnqueueWriteBuffer(runtime.queue, device->a, CL_TRUE, 0, a_bytes, a, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(runtime.queue, device->b, CL_TRUE, 0, b_bytes, b, 0, NULL, NULL));
    return device;
}

static void poison(MatmulOpenCLBenchmark *device) {
    const cl_uint nan_bits = UINT32_MAX;
    CL_CHECK(clEnqueueFillBuffer(runtime.queue, device->c, &nan_bits, sizeof nan_bits,
                               0, device->c_bytes, 0, NULL, NULL));
}

static void launch(MatmulOpenCLBenchmark *device, cl_event *event) {
#ifdef MATMUL_HAVE_CLBLAST
    if (device->library) {
        CLBlastStatusCode status = CLBlastSgemm(CLBlastLayoutRowMajor,
            CLBlastTransposeNo, CLBlastTransposeNo,
            (size_t)device->m, (size_t)device->n, (size_t)device->k, 1.0f,
            device->a, 0, (size_t)device->k, device->b, 0, (size_t)device->n,
            0.0f, device->c, 0, (size_t)device->n, &runtime.queue, event);
        if (status != CLBlastSuccess) {
            fprintf(stderr, "CLBlast SGEMM failed (%d).\n", (int)status);
            exit(EXIT_FAILURE);
        }
        return;
    }
#endif
    CL_CHECK(clSetKernelArg(device->kernel, 0, sizeof device->m, &device->m));
    CL_CHECK(clSetKernelArg(device->kernel, 1, sizeof device->n, &device->n));
    CL_CHECK(clSetKernelArg(device->kernel, 2, sizeof device->k, &device->k));
    CL_CHECK(clSetKernelArg(device->kernel, 3, sizeof device->a, &device->a));
    CL_CHECK(clSetKernelArg(device->kernel, 4, sizeof device->b, &device->b));
    CL_CHECK(clSetKernelArg(device->kernel, 5, sizeof device->c, &device->c));
    /* Global counts are work-items, not CUDA block counts. Each context records
     * its selected geometry; only the naive path leaves group sizes to the driver.
     */
    CL_CHECK(clEnqueueNDRangeKernel(runtime.queue, device->kernel, 2, NULL,
                                   device->global, device->tiled ? device->local : NULL, 0, NULL, event));
}

double matmul_opencl_benchmark_sample(MatmulOpenCLBenchmark *device, size_t batch) {
    if (!batch) {
        fprintf(stderr, "OpenCL kernel timing requires a positive batch.\n");
        exit(EXIT_FAILURE);
    }
    poison(device);
    cl_event first = NULL, last = NULL;
    if (device->library) {
        /* A returned CLBlast event may describe only its final internal kernel.
         * Enclose the ENTIRE library operation, including pre/postprocessing.
         */
        CL_CHECK(clEnqueueMarkerWithWaitList(runtime.queue, 0, NULL, &first));
        for (size_t i = 0; i < batch; ++i) launch(device, NULL);
        CL_CHECK(clEnqueueMarkerWithWaitList(runtime.queue, 0, NULL, &last));
    } else {
        for (size_t i = 0; i < batch; ++i)
            launch(device, i == 0 ? &first : i == batch - 1 ? &last : NULL);
        if (batch == 1) last = first;
    }
    CL_CHECK(clWaitForEvents(1, &last));
    cl_ulong start, end;
    CL_CHECK(clGetEventProfilingInfo(first, device->library ? CL_PROFILING_COMMAND_END : CL_PROFILING_COMMAND_START, sizeof start, &start, NULL));
    CL_CHECK(clGetEventProfilingInfo(last, CL_PROFILING_COMMAND_END, sizeof end, &end, NULL));
    CL_CHECK(clReleaseEvent(first));
    if (last != first) CL_CHECK(clReleaseEvent(last));
    if (end <= start) {
        fprintf(stderr, "OpenCL event duration is not positive; increase --batch.\n");
        exit(EXIT_FAILURE);
    }
    return (double)(end - start) * 1e-9 / (double)batch;
}

void matmul_opencl_benchmark_read(MatmulOpenCLBenchmark *device, float *c) {
    CL_CHECK(clEnqueueReadBuffer(runtime.queue, device->c, CL_TRUE, 0, device->c_bytes,
                                c, 0, NULL, NULL));
}
void matmul_opencl_benchmark_destroy(MatmulOpenCLBenchmark *device) {
    if (!device) return;
    CL_CHECK(clReleaseMemObject(device->c));
    CL_CHECK(clReleaseMemObject(device->b));
    CL_CHECK(clReleaseMemObject(device->a));
    free(device);
}

static void matmul_opencl(MatmulFn implementation, size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    if (!m || !n) return;
    if (!k) {
        for (size_t i = 0; i < m*n; ++i) c[i] = 0.0f;
        return;
    }
    MatmulOpenCLBenchmark *device = matmul_opencl_benchmark_create(implementation, m, n, k, a, b);
    if (!device) { fprintf(stderr, "OpenCL host allocation failed.\n"); exit(EXIT_FAILURE); }
    poison(device);
    launch(device, NULL);
    CL_CHECK(clFinish(runtime.queue));
    matmul_opencl_benchmark_read(device, c);
    matmul_opencl_benchmark_destroy(device);
}

void matmul_opencl_naive(size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    matmul_opencl(matmul_opencl_naive, m, n, k, a, b, c);
}
void matmul_opencl_tiled(size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    matmul_opencl(matmul_opencl_tiled, m, n, k, a, b, c);
}

void matmul_opencl_register_tiled(size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    matmul_opencl(matmul_opencl_register_tiled, m, n, k, a, b, c);
}
#ifdef MATMUL_HAVE_CLBLAST
void matmul_clblast(size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    matmul_opencl(matmul_clblast, m, n, k, a, b, c);
}
const char *matmul_clblast_version(void) { return "1.7.0"; }
#endif

void matmul_opencl_combined(size_t m, size_t n, size_t k,
    const float *a, const float *b, float *c) {
    matmul_opencl(matmul_opencl_combined, m, n, k, a, b, c);
}
