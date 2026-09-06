# cocotb unit verification for VX_packbf16_arith (packbf16.mul/add).
#
# Runs against the in-tree RTL with the build-generated config on the
# include path; VX_CFG_XLEN is fixed to 32 (one packed word = two bf16
# lanes, matching the instruction semantics).
#
# The RTL is warning-clean under `verilator --lint-only -Wall`, so no
# -Wno-fatal escape hatch is needed; Verilator stays strict here.
SIM ?= verilator
TOPLEVEL := VX_packbf16_arith
VORTEX_ROOT := $(realpath ../../..)
VERILOG_SOURCES := $(VORTEX_ROOT)/hw/rtl/core/VX_packbf16_arith.sv
EXTRA_ARGS += -I$(VORTEX_ROOT)/hw/rtl -I$(VORTEX_ROOT)/build/hw \
              +define+VX_CFG_XLEN=32 --timing
MODULE := test_bf16
# cocotb 2.x: include Makefile.sim (which pulls in the per-simulator
# makefile providing the results.xml rule); Makefile.inc alone is a no-op
# for user makefiles and leaves the regression target without a recipe.
include $(shell cocotb-config --makefiles)/Makefile.sim
