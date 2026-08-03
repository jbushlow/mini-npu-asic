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

The default graph targets Vitis HLS C synthesis, normalizes the generated RTL,
selects proven classes meeting `min_macro_reuse`, and runs the complete selected
batch through isolated parallel DC, Innovus, and PrimeTime/Library Compiler
workers, with batched Calibre DRC/LVS after PNR. Stage 1 publishes checked
Verilog, Liberty, DB, LEF, GDS, SPEF, SDF, SDC, reports, and a hardened-macro
registry. Per-macro VCS and power analysis are intentionally omitted.

Stage 2 validates and assembles the canonical macro instances, then runs the
new full-chip Design Compiler node with macro DBs as link-only libraries. The
default `macro_clock_period` is 8 ns and the full-chip `clock_period` is 10 ns;
preflight requires the chip period to be no smaller than the macro period.
Four-state VCS verification remains a planned insertion before full-chip
synthesis once automatic workload/testbench emission is defined.

`allo_design.py` also defines a deterministic `workload()` hook containing the
logical A and B inputs and expected C output. Running the file directly executes
that transaction with Allo's dataflow simulator and checks the result. The same
hook is intended to supply vectors to the future RTL/VCS testbench emitter.
