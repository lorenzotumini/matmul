# 19 — Register tiling in OpenCL

A work-item can own a small rectangle of outputs, keeping several independent
sums in registers. A 2x2 group combines two A values and two B values into
four products at each reduction position. This adds private reuse to the
work-group reuse already supplied by local memory.

Work-group size, output tile size and reduction depth are independent choices.
Keeping the number of work-items fixed while increasing output ownership
requires larger local input tiles and more cooperative loads per work-item.
Loading coordinates need not follow the work-item's output coordinates.

Each output needs an independent boundary check. Edge work-items still
participate in loads and barriers; missing input values contribute zeros.
Larger tiles improve input reuse but increase local-memory/register demand
and reduce the number of groups available for scheduling. Portable source
does not guarantee identical register allocation or performance across GPUs.
