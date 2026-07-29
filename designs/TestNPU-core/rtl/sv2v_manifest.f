# Packages
pkg/npu_config_pkg.sv
pkg/npu_isa_pkg.sv
pkg/npu_modes_pkg.sv
pkg/tile_agu_pkg.sv

# Shared utility cells
common/

# Core compute plane
core/dmu/
core/mxu/
core/sequencer/
core/spad/
core/top/
core/vpu/fpu/
core/vpu/

# Excludes
!core/mxu/systolic_tb_wrap.sv
