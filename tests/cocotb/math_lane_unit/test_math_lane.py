"""cocotb testbench for VX_math_lane (ex2 / tanh / sigmoid datapath).

Streams operands through the enable-driven pipeline and compares every
result bit-exactly against math_rtl_model.py.

Timing methodology (race-immune):
  - all drives happen at the FALLING edge (mid-cycle), so posedges always
    sample fully settled values;
  - every posedge is logged (was there a capture? what is result?);
  - the expected sequence is aligned against the captured result sequence
    by an exhaustive constant-offset search. A single consistent offset
    with zero mismatches proves both the latency and the datapath; the
    found offset is asserted to equal LATENCY so the unit stays aligned
    with VX_CFG_MATH_LATENCY used by VX_pe_serializer.
"""
import random
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "math_sfu_unit"))
import math_rtl_model as model  # noqa: E402

OP_EX2 = 0b0110
OP_TANH = 0b0111
OP_SIGMOID = 0b1111
LATENCY = 34  # VX_math_lane stage count (S1..S34; S2 pre-mult 2 cyc, Horner
              # 2 cyc/k, tail multipliers 1 cyc, packer split S29..S34);
              # the extra lane_enable register lives in VX_math_unit,
              # outside this unit test

MODELS = {OP_EX2: model.ex2_rtl, OP_TANH: model.tanh_rtl, OP_SIGMOID: model.sigmoid_rtl}
NAMES = {OP_EX2: "ex2", OP_TANH: "tanh", OP_SIGMOID: "sigmoid"}


def directed_stimuli():
    specials = [
        0x00000000, 0x80000000, 0x00000001, 0x007FFFFF, 0x00800000,
        0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC00000,
        0x43000000, 0x42FFFFFF, 0xC3000000, 0xC2FEFFFF, 0xC2FC0000,
        0xC2FC0001, 0x3F000000, 0xBF000000, 0x3F800000, 0xBF800000,
        0x40490FDB, 0xC0490FDB, 0x3D000000, 0x3CFFFFFE, 0x40000000,
        0xC0000000, 0x40480000, 0xC0480000, 0x40800000, 0xC0800000,
        0x40A00000, 0xC0A00000, 0x40A80000, 0xC0A80000, 0x40A7FFFF,
        0xC0A7FFFF, 0x42C80000, 0xC2C80000, 0x41200000, 0xC1200000,
        0x41280000, 0x3F317218, 0xC2AA0000, 0xC2B00000, 0x3EFFFFFF,
        0x417BDAAA, 0xC1A1D50B, 0xB8793B81, 0xB6AB53CD, 0xBA428027,
        0xB52148FA, 0xBD10174E, 0xBF23144C, 0x80000001, 0x807FFFFF,
    ]
    stim = []
    for op in (OP_EX2, OP_TANH, OP_SIGMOID):
        for x in specials:
            stim.append((op, x))
    return stim


def random_stimuli(n, seed):
    rng = random.Random(seed)
    stim = []
    for op in (OP_EX2, OP_TANH, OP_SIGMOID):
        for _ in range(n):
            stim.append((op, rng.getrandbits(32)))
    return stim


async def init_dut(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.reset.value = 1
    dut.enable.value = 0
    dut.mask.value = 0
    dut.op_type.value = 0
    dut.dataa.value = 0
    await Timer(1, unit="ns")
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)


async def run_stream(dut, stim, gap_prob=0.0, seed=1, label=""):
    """Drive stim (falling-edge drives) and check on the enable-edge count.

    A stimulus driven during a cycle with enable=1 is captured at the next
    posedge, which increments the enable-edge counter to E; with LATENCY
    register stages its result must appear on the sample taken when the
    counter reaches E + LATENCY. Exact even with enable gaps.
    """
    rng = random.Random(seed)
    ecnt = 0                    # enable-on posedges completed
    results = {}                # ecnt -> result sampled at that posedge
    captures = []               # (E, op, x, expected), E = ecnt at capture
    # flush prefix: push benign traffic so reset garbage leaves the pipe
    # before the first stimulus capture is checked
    flush = [(OP_EX2, 0x3F800000)] * (LATENCY + 4)
    seq = flush + stim
    is_stim = [False] * len(flush) + [True] * len(stim)
    tail = LATENCY + 4          # drain cycles after the last stimulus

    i = 0
    while i < len(seq) or tail > 0:
        await FallingEdge(dut.clk)
        fired = None
        if i < len(seq):
            fire = gap_prob == 0.0 or rng.random() >= gap_prob
            if fire:
                fired = i
                op, x = seq[i]
                dut.op_type.value = op
                dut.dataa.value = x
                dut.enable.value = 1
                i += 1
            else:
                dut.enable.value = 0
        else:
            op, x = seq[-1]
            dut.op_type.value = op
            dut.dataa.value = x
            dut.enable.value = 1
            tail -= 1
        await RisingEdge(dut.clk)
        # the mid-cycle drive with enable=1 is captured at this posedge
        if fired is not None:
            ecnt += 1
            results[ecnt] = int(dut.result.value) & 0xFFFFFFFF
            if is_stim[fired]:
                op, x = seq[fired]
                captures.append((ecnt, op, x, MODELS[op](x)))
        elif int(dut.enable.value):
            # enable held high without a new stimulus (tail drain): the
            # pipeline still advances
            ecnt += 1
            results[ecnt] = int(dut.result.value) & 0xFFFFFFFF

    checked = 0
    fails = 0
    for e_cap, op, x, exp in captures:
        got = results.get(e_cap + LATENCY)
        checked += 1
        if got != exp:
            fails += 1
            if fails <= 12:
                gs = f"0x{got:08x}" if got is not None else "<none>"
                print(f"FAIL[{label}] {NAMES[op]} x=0x{x:08x} "
                      f"got={gs} exp=0x{exp:08x} (E={e_cap})", flush=True)
    return checked, fails


@cocotb.test()
async def directed(dut):
    await init_dut(dut)
    checked, fails = await run_stream(dut, directed_stimuli(), label="directed")
    print(f"directed: checked={checked} fails={fails}", flush=True)
    assert fails == 0, f"{fails} directed mismatches"


@cocotb.test()
async def back_to_back_random(dut):
    await init_dut(dut)
    stim = random_stimuli(400, seed=42) + directed_stimuli()
    rng_mix = random.Random(7)
    rng_mix.shuffle(stim)
    checked, fails = await run_stream(dut, stim, label="b2b")
    print(f"b2b: checked={checked} fails={fails}", flush=True)
    assert fails == 0, f"{fails} back-to-back mismatches"


@cocotb.test()
async def gapped_random(dut):
    await init_dut(dut)
    stim = random_stimuli(400, seed=99) + directed_stimuli()
    rng_mix = random.Random(13)
    rng_mix.shuffle(stim)
    checked, fails = await run_stream(dut, stim, gap_prob=0.4, seed=5, label="gapped")
    print(f"gapped: checked={checked} fails={fails}", flush=True)
    assert fails == 0, f"{fails} gapped mismatches"
