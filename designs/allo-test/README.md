# Allo ASIC-flow test

This design is a small 4x4 `int32` streaming systolic GEMM. Dedicated loader
kernels are the sole readers of the flattened A and B memories, a 4x4 mapped
kernel contains the compute PEs, and a dedicated store kernel is the sole writer
of flattened C. All communication between them uses point-to-point streams.
This makes the example legal for Vitis dataflow while remaining representative
of an ordinary accelerator architecture and useful for repeated-PE detection,
RTL capture, and manifest forwarding into the commercial ASIC flow.

The `allo-asic-compilation` node imports `allo_design.py` and calls:

```python
build(project, target, mode, configs)
```

The default graph targets Vitis HLS C synthesis, then passes the synthesized
Verilog and ASIC manifests through `sv2v-rtl-allo`.

`allo_design.py` also defines a deterministic `workload()` hook containing the
logical A and B inputs and expected C output. Running the file directly executes
that transaction with Allo's dataflow simulator and checks the result. The same
hook is intended to supply vectors to the future RTL/VCS testbench emitter.
