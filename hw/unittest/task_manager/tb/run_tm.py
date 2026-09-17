#!/usr/bin/env python3
# cocotb runner for the integrated VX_task_manager (T1-T9 full-lifecycle).
import os
from cocotb_tools.runner import get_runner

HERE = os.path.dirname(os.path.abspath(__file__))
VX   = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # third_party/vortex
WORK = "/u/cpu-arc/lwb529826/ai-for-rtl/work/tm_cocotb"
TM   = os.path.join(VX, "hw/unittest/task_manager")

SOURCES = [
    os.path.join(VX, "hw/rtl/libs/VX_priority_encoder.sv"),
    os.path.join(TM, "VX_tm_pkg.sv"),
    os.path.join(TM, "VX_tm_tsm.sv"),
    os.path.join(TM, "VX_tm_pqm.sv"),
    os.path.join(TM, "VX_tm_pr.sv"),
    os.path.join(TM, "VX_task_manager.sv"),
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
        hdl_toplevel="VX_task_manager",
        build_dir=os.path.join(WORK, "build_tm"),
        build_args=BUILD_ARGS,
        always=True,
    )
    runner.test(
        hdl_toplevel="VX_task_manager",
        test_module="test_task_manager",
        build_dir=os.path.join(WORK, "build_tm"),
        results_xml=os.path.join(WORK, "results_tm.xml"),
    )

if __name__ == "__main__":
    main()
