# 17 — OpenCL and integrated GPU computing

OpenCL separates a host program from kernels executed on a selected device.
Platforms expose devices; a context owns resources for them, and a command
queue orders operations. An explicit GPU selection avoids accidentally
benchmarking a CPU runtime or a different graphics adapter.

An ND-range describes individual work-items. In a two-dimensional matrix
kernel, one axis can represent columns and the other rows. A work-item can
compute one output independently. Work-groups partition the global range;
their size is a separate choice from the total output dimensions.

OpenCL C is a device language with address-space qualifiers. A `__global`
pointer refers to a device buffer; `__local` storage is shared within a
work-group. These names describe accessibility, not whether memory chips
are physically separate from CPU RAM. Runtime compilation builds source
for the chosen device, so compiler diagnostics and startup cost matter.

Integrated GPUs can share physical RAM with the CPU. Appropriate allocation
and ownership rules can avoid copies, but shared RAM alone does not make
explicit buffer copies or synchronization free. CPU and GPU also compete
for memory bandwidth, and their power/thermal budgets can interact.

Separate full host-array latency from prepared device execution time.
On a profiling-enabled in-order queue, the first kernel's start and last
kernel's end bound a device-timeline interval. It excludes initial copies
and includes gaps between launches in a batch. Compilation and runtime
initialization belong outside steady-state samples; first-use latency is
a separate question. A portable API does not guarantee portable performance.

See Khronos' [ND-range reference](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clEnqueueNDRangeKernel.html)
and [event profiling reference](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetEventProfilingInfo.html),
and Intel's [shared-memory discussion](https://www.intel.com/content/www/us/en/docs/opencl-sdk/developer-guide-processor-graphics/2019-4/memory-hierarchy.html).
