# 14 — GPU memory access layout

The set of values computed by a block does not uniquely determine which
thread owns each output. Different ownership mappings can perform the same
arithmetic with different memory-access patterns.

Coalescing concerns the addresses requested together by neighboring lanes
for an individual global-memory instruction. Contiguous, suitably aligned
addresses generally require fewer memory transactions than strided ones.
Two separate instructions that collectively cover contiguous elements do not
necessarily have the same cost as contiguous accesses within each instruction.
Warp shape, alignment, partial outputs and generated instructions matter.

A thread's register tile can be a group of nonadjacent output coordinates.
Separating those coordinates can let neighboring lanes access consecutive
columns while preserving per-thread reuse and accumulator count. Shared-memory
reads and output stores must consistently follow the chosen ownership.

Shared memory is divided into banks. Requests for different addresses in
the same bank can serialize; multiple lanes reading the same address can
use a broadcast. Output remapping may change these patterns even when the
shared arrays themselves are unchanged. Coalescing global accesses and
avoiding shared-memory bank conflicts are related layout concerns, with
different hardware rules.

Keeping block size, tile dimensions and arithmetic fixed makes ownership
remapping a useful controlled experiment. Inspect or measure the result:
source-level access patterns suggest candidates but do not prove a speedup
or identify the dominant bottleneck.

See NVIDIA's [CUDA best practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
for coalesced access and shared-memory banks.
