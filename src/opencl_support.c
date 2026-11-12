#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "opencl_bench.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
    cl_program program[2];
    cl_kernel kernel[2];
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
    for (int i = 0; i < 2; ++i) {
        if (runtime.kernel[i]) clReleaseKernel(runtime.kernel[i]);
        if (runtime.program[i]) clReleaseProgram(runtime.program[i]);
    }
    clReleaseCommandQueue(runtime.queue);
    clReleaseContext(runtime.context);
}

static void initialize(int variant) {
    if (runtime.kernel[variant]) return;
    const char *path = variant ? MATMUL_OPENCL_TILED_SOURCE_PATH : MATMUL_OPENCL_SOURCE_PATH;
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
    const char *text = source;
    runtime.program[variant] = clCreateProgramWithSource(runtime.context, 1, &text, NULL, &error);
    CL_CHECK(error);
    free(source);
    error = clBuildProgram(runtime.program[variant], 1, &runtime.device, "-cl-std=CL1.2", NULL, NULL);
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
    runtime.kernel[variant] = clCreateKernel(runtime.program[variant], variant ? "matmul_opencl_tiled" : "matmul_opencl_naive", &error);
    CL_CHECK(error);

}

struct MatmulOpenCLBenchmark {
    cl_ulong m, n, k;
    size_t global[2], local[2], c_bytes;
    bool tiled;
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
    return implementation == matmul_opencl_naive || implementation == matmul_opencl_tiled;
}

MatmulOpenCLBenchmark *matmul_opencl_benchmark_create(MatmulFn implementation,
    size_t m, size_t n, size_t k, const float *a, const float *b) {
    if (!matmul_opencl_kernel_timing_supported(implementation) || !m || !n || !k) return NULL;
    MatmulOpenCLBenchmark *device = malloc(sizeof *device);
    if (!device) return NULL;
    device->tiled = implementation == matmul_opencl_tiled;
    initialize(device->tiled ? 1 : 0);
    device->kernel = runtime.kernel[device->tiled ? 1 : 0];
    device->m = (cl_ulong)m; device->n = (cl_ulong)n; device->k = (cl_ulong)k;
    device->local[0] = 16; device->local[1] = 16;
    device->global[0] = n; device->global[1] = m;
    if (device->tiled) {
        /* ceil(size/16)*16, with overflow checks before rounding. */
        if (m > SIZE_MAX - 15 || n > SIZE_MAX - 15) {
            fprintf(stderr, "OpenCL tiled global size overflows size_t.\n");
            exit(EXIT_FAILURE);
        }
        device->global[0] = (n + 15) / 16 * 16;
        device->global[1] = (m + 15) / 16 * 16;
        size_t limit;
        CL_CHECK(clGetKernelWorkGroupInfo(device->kernel, runtime.device,
            CL_KERNEL_WORK_GROUP_SIZE, sizeof limit, &limit, NULL));
        if (limit < 256) {
            fprintf(stderr, "OpenCL tiled kernel requires a 16x16 work-group.\n");
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
    CL_CHECK(clSetKernelArg(device->kernel, 0, sizeof device->m, &device->m));
    CL_CHECK(clSetKernelArg(device->kernel, 1, sizeof device->n, &device->n));
    CL_CHECK(clSetKernelArg(device->kernel, 2, sizeof device->k, &device->k));
    CL_CHECK(clSetKernelArg(device->kernel, 3, sizeof device->a, &device->a));
    CL_CHECK(clSetKernelArg(device->kernel, 4, sizeof device->b, &device->b));
    CL_CHECK(clSetKernelArg(device->kernel, 5, sizeof device->c, &device->c));
    /* Global counts are individual work-items, not CUDA block counts. The naive
     * path lets the driver choose groups; tiling uses explicit 16x16 groups.
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
    for (size_t i = 0; i < batch; ++i)
        launch(device, i == 0 ? &first : i == batch - 1 ? &last : NULL);
    if (batch == 1) last = first;
    CL_CHECK(clWaitForEvents(1, &last));
    cl_ulong start, end;
    CL_CHECK(clGetEventProfilingInfo(first, CL_PROFILING_COMMAND_START, sizeof start, &start, NULL));
    CL_CHECK(clGetEventProfilingInfo(last, CL_PROFILING_COMMAND_END, sizeof end, &end, NULL));
    CL_CHECK(clReleaseEvent(first));
    if (batch > 1) CL_CHECK(clReleaseEvent(last));
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
