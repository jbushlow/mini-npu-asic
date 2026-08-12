# Allo MiniNPU v2 ASIC design

This design runs the rebuilt MiniNPU v2 through the complete commercial
two-stage ASIC flow. The Allo source remains in
`../../allo-rebuild/mininpu_v2.py`; `allo_design.py` loads it and exposes the
standard `build(project, target, mode, configs)` entrypoint.

The selected configuration contains a 5 x 5 MXU compute array and five mapped
VPU lanes. Stage 1 selects equivalent semantic PE classes with reuse count at
least two, hardens one representative of each class, and publishes its logical
and physical views. Stage 2 substitutes those views, places the macro instances
using manifest topology, synthesizes and routes the remaining standard-cell
logic, and runs full-chip merge, DRC, LVS, and summary reporting.

Important graph settings are:

- Allo backend: Vitis HLS C synthesis
- top module and design name: `mininpu_v2`
- full-chip clock period: 10 ns
- macro/HLS clock period: 8 ns
- minimum macro reuse: 2
- repeated anonymous HLS submodule hardening: disabled
- Google Sheets summary: enabled and required, inherited from `allo-test`

Create the mflowgen build directory outside this source directory and point it
at `construct-commercial.py`. The terminal `allo-asic-flow-summary` node waits
for Stage 1, Stage 2, GDS merge, DRC, and LVS before appending its spreadsheet
row.

After the Allo compilation and macro-plan nodes complete, inspect
`asic-manifest-final.json` and `macro-plan.json` to confirm that the five VPU
lanes form one semantic macro class before launching the commercial stages.
