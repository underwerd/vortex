# cocotb unit verification for VX_math_lane (ex2/tanh/sigmoid datapath).
#
# Streams operands through the 16-stage enable-driven pipeline and compares
# every result bit-exactly against math_rtl_model.py (the executable image
# of the frozen approx scheme).
SIM ?= verilator
TOPLEVEL := VX_math_lane
VORTEX_ROOT := $(realpath ../../..)
VERILOG_SOURCES := $(VORTEX_ROOT)/hw/rtl/VX_gpu_pkg.sv \
                   $(VORTEX_ROOT)/hw/rtl/fpu/VX_math_lane.sv
EXTRA_ARGS += -I$(VORTEX_ROOT)/hw/rtl -I$(VORTEX_ROOT)/hw/rtl/fpu \
              -I$(VORTEX_ROOT)/hw/rtl/libs -I$(VORTEX_ROOT)/build/hw \
              +define+VX_CFG_XLEN=32 +define+VX_CFG_XLEN_32 \
              +define+VX_CFG_FLEN=32 +define+VX_CFG_FLEN_32 --timing \
              -Wno-SELRANGE -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC --public-flat-rw
MODULE := test_math_lane
include $(shell cocotb-config --makefiles)/Makefile.sim
