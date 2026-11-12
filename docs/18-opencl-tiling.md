# 18 — Work-group local-memory tiling

Local memory lets work-items in one work-group reuse inputs cooperatively.
A square output tile loads matching A and B tiles once per reduction step,
then reuses each value across several outputs. Sharing physical system RAM
with the CPU does not remove this reuse benefit.

Global work-item counts and local group dimensions are separate choices.
Rounding the global range up permits uniform groups at boundaries.
Out-of-range work-items still participate: guarded loads supply zeros and
only valid outputs are stored. Reduction tails require zero padding too.

A local-memory barrier makes cooperative loads visible before consumption.
A second barrier protects readers before the next tile overwrites storage.
Every work-item must encounter each required group barrier, including those
with no valid output. An early return can break cooperation.

Tile size trades reuse against storage, register pressure, group-size limits
and boundary waste. A portable API does not guarantee portable performance;
compare device time and full-call latency across representative shapes.

See the [OpenCL barrier reference](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/barrier.html).
