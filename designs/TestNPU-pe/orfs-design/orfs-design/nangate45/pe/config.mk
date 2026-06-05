export DESIGN_NAME = pe
export PLATFORM    = nangate45

# add all required verilog files
export VERILOG_FILES = \
  ./designs/src/$(DESIGN_NAME)/fp16_mul_fp32.sv \
  ./designs/src/$(DESIGN_NAME)/fp32_add.sv \
  ./designs/src/$(DESIGN_NAME)/pe.sv

export SDC_FILE      = ./designs/$(PLATFORM)/$(DESIGN_NAME)/constraint.sdc
export ABC_AREA      = 1

# Adders degrade GCD
export ADDER_MAP_FILE :=

export CORE_UTILIZATION ?= 55
export PLACE_DENSITY_LB_ADDON = 0.20
export TNS_END_PERCENT        = 100
export REMOVE_CELLS_FOR_EQY   = TAPCELL*

export SYNTH_ARGS = -noshare