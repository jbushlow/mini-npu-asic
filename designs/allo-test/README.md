# Allo ASIC-flow test

This design is a small 4x4 `int32` systolic GEMM. Its mapped kernel includes
loader, compute, drain, and empty boundary positions, making it useful for
repeated-PE detection, Vitis RTL capture, and manifest forwarding into the
commercial ASIC flow.

The `allo-asic-compilation` node imports `allo_design.py` and calls:

```python
build(project, target, mode, configs)
```

The default graph targets Vitis HLS C synthesis, then passes the synthesized
Verilog and ASIC manifests through `sv2v-rtl-allo`.
