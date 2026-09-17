#!/usr/bin/env python3
# cocotb runner for the VX_tm_pr standalone unit test (small-module-first).
import os
from cocotb_tools.runner import get_runner

HERE = os.path.dirname(os.path.abspath(__file__))
VX   = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # third_party/vortex
WORK = "/u/cpu-arc/lwb529826/ai-for-rtl/work/tm_cocotb"

SOURCES = [
    os.path.join(VX, "hw/rtl/libs/VX_priority_encoder.sv"),
    os.path.join(VX, "hw/unittest/task_manager/VX_tm_pkg.sv"),
    os.path.join(VX, "hw/unittest/task_manager/VX_tm_pr.sv"),
]
BUILD_ARGS = [
    "-Wno-fatal", "--assert",
    "-DSIMULATION", "-DVX_CFG_XLEN=32",
    "-I" + os.path.join(VX, "build/hw"),
    "-I" + os.path.join(VX, "hw/rtl"),
    "-y", os.path.join(VX, "hw/rtl/libs"),
    "-y", os.path.join(VX, "hw/rtl"),
]

def main():
    os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ.get("PYTHONPATH", "")
    runner = get_runner("verilator")
    runner.build(
        sources=SOURCES,
        hdl_toplevel="VX_tm_pr",
        build_dir=os.path.join(WORK, "build_pr"),
        build_args=BUILD_ARGS,
        always=True,
    )
    runner.test(
        hdl_toplevel="VX_tm_pr",
        test_module="test_tm_pr",
        build_dir=os.path.join(WORK, "build_pr"),
        results_xml=os.path.join(WORK, "results_pr.xml"),
    )

if __name__ == "__main__":
    main()
