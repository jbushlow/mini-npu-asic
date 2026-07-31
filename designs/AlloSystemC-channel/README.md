# Allo SystemC channel ASIC smoke test

This design packages the Catapult-generated `hello_channel` RTL from
`systemC_rtl/rtl.v` for the commercial MiniNPU ASIC flow. The top-level module
accepts eight 32-bit ready/valid transfers, forwards them through two HLS
processes, and asserts `done` after all eight outputs have been accepted.

`HelloChannelTb.sv` is self-checking. It sends boundary-value and patterned
data, applies deterministic output backpressure, checks ordering and data
integrity, and waits for `done`.

To construct the flow in an environment with mflowgen and the commercial tools:

```sh
mkdir -p build/AlloSystemC-channel
cd build/AlloSystemC-channel
mflowgen run --design ../../designs/AlloSystemC-channel
```

The design intentionally has no ORFS directory; `construct-commercial-pnr.py`
uses the repository's full `cadence-innovus-pnr` node.
