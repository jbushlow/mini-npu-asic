# Catapult blocking EVA ASIC flow

This design uses
`../allo-eva-rebuild/eva_sb_syscredit_rtprime_skid_leanalu.py` and the complete
two-stage ASIC graph, with the Allo Catapult backend selected in `csyn` mode.
Catapult uses the Nangate 45 nm behavioral library and an 8 ns HLS clock. The
current source has 32 workload elements per lane so Catapult can realize the
top arrays as synchronous memory interfaces; the manifest and testbench match
the resulting per-argument port families exactly.

Create a build in the usual way, using this directory's construct:

```bash
mflowgen run --design ./designs/allo-eva-catapult
```

The first node is `allo-asic-compilation`; it should emit Catapult's
self-contained `concat_rtl.v` together with the pre-HLS and enriched ASIC
manifest files.
