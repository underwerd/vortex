"""cocotb unit testbench for VX_packbf16_arith (packbf16.mul/add).

Drives the packed-word DUT (two VX_bf16_op lanes, NUM_LANES=1) with
directed specials and random operand pairs, both operations, and
compares every result word bit-exactly against the integer-exact
golden model. Combinational DUT: drive inputs, settle one delta,
sample.

The runner (harness V3 step) parses the summary JSON this testbench
writes to results.json to assemble the numeric report.

Failure classes are counted so the report can say *what kind* of
numerical defect the RTL has, not just that it has one.
"""
import json
import os
import random
import sys
from pathlib import Path

import cocotb
from cocotb.triggers import Timer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_model import pack_word  # noqa: E402

N_RANDOM = int(os.environ.get("PACKBF16_RANDOM_PAIRS", "1024"))
RESULT_PATH = os.environ.get(
    "PACKBF16_RESULT_JSON",
    str(Path(__file__).resolve().parent / "results.json"),
)


def _classify(a: int, b: int) -> str:
    def klass(x):
        e, m = (x >> 7) & 0xFF, x & 0x7F
        if e == 0xFF:
            return "nan" if m else "inf"
        if e == 0:
            return "zero" if m == 0 else "denormal"
        return "normal"
    ka, kb = klass(a), klass(b)
    for special in ("nan", "inf"):
        if special in (ka, kb):
            return special
    if "denormal" in (ka, kb):
        return "denormal"
    if "zero" in (ka, kb):
        return "zero"
    return "normal"


def _make_word_pairs(items):
    """Pack a list of (bf16_a, bf16_b) operand pairs into (rs1, rs2)
    32-bit word pairs: two bf16 pairs drive the hi and lo lanes of one
    packed-word operation, halving the drive count."""
    if len(items) % 2:
        items = items + [items[-1]]
    out = []
    for i in range(0, len(items), 2):
        a1, b1 = items[i]
        a2, b2 = items[i + 1]
        out.append((((a1 & 0xFFFF) << 16) | (a2 & 0xFFFF),
                    ((b1 & 0xFFFF) << 16) | (b2 & 0xFFFF)))
    return out


async def _run_batch(dut, word_pairs, is_add: bool, stats: dict) -> list:
    """Drive (rs1, rs2) word pairs; returns per-lane mismatch dicts."""
    mismatches = []
    dut.is_add.value = int(is_add)
    op = "add" if is_add else "mul"
    for rs1, rs2 in word_pairs:
        a_hi, a_lo = (rs1 >> 16) & 0xFFFF, rs1 & 0xFFFF
        b_hi, b_lo = (rs2 >> 16) & 0xFFFF, rs2 & 0xFFFF
        dut.rs1.value = rs1 & 0xFFFFFFFF
        dut.rs2.value = rs2 & 0xFFFFFFFF
        await Timer(1, unit="ns")       # settle the combinational path
        got = int(dut.result.value) & 0xFFFFFFFF
        exp = pack_word(a_hi, a_lo, b_hi, b_lo, is_add)
        if got != exp:
            for lane, la, lb, g16, e16 in (
                ("hi", a_hi, b_hi, (got >> 16) & 0xFFFF, (exp >> 16) & 0xFFFF),
                ("lo", a_lo, b_lo, got & 0xFFFF, exp & 0xFFFF),
            ):
                if g16 != e16:
                    cls = _classify(la, lb)
                    mismatches.append({
                        "op": op, "lane": lane,
                        "a": hex(la), "b": hex(lb),
                        "got": hex(g16), "expected": hex(e16), "class": cls,
                    })
                    stats[(op, cls)] = stats.get((op, cls), 0) + 1
    return mismatches


@cocotb.test()
async def directed_specials_both_ops(dut):
    from golden_model import directed_specials
    sp = directed_specials()
    items = [(a, b) for a in sp for b in sp]
    word_pairs = _make_word_pairs(items)
    stats: dict = {}
    mism = []
    for is_add in (False, True):
        mism += await _run_batch(dut, word_pairs, is_add, stats)
    summary = {
        "test": "directed_specials",
        "operand_pairs": len(items),
        "checked": len(items) * 2,
        "mismatches": len(mism),
        "by_class": {f"{op}/{cls}": n for (op, cls), n in sorted(stats.items())},
        "examples": mism[:12],
    }
    dut._log.info("directed specials: %d checks, %d mismatches, classes=%s",
                  summary["checked"], summary["mismatches"], summary["by_class"])
    assert not mism, (
        "RTL deviates from golden on directed specials: %s" % summary["by_class"])


@cocotb.test()
async def random_pairs_both_ops(dut):
    rng = random.Random(20260818)
    word_pairs = [(rng.getrandbits(32), rng.getrandbits(32))
                  for _ in range(N_RANDOM)]
    stats: dict = {}
    mism = []
    for is_add in (False, True):
        mism += await _run_batch(dut, word_pairs, is_add, stats)
    summary = {
        "test": "random_pairs",
        "word_pairs": len(word_pairs),
        "checked": len(word_pairs) * 2 * 2,
        "mismatches": len(mism),
        "by_class": {f"{op}/{cls}": n for (op, cls), n in sorted(stats.items())},
        "examples": mism[:12],
    }
    dut._log.info("random pairs: %d checks, %d mismatches, classes=%s",
                  summary["checked"], summary["mismatches"], summary["by_class"])
    assert not mism, (
        "RTL deviates from golden on random pairs: %s" % summary["by_class"])


@cocotb.test()
async def write_results_json(dut):
    """Aggregate the two batches into the runner-consumed summary.

    Runs the batches again internally (combinational DUT, cheap) so
    the summary file exists even when the assert-based tests fail.
    """
    from golden_model import directed_specials
    sp = directed_specials()
    special_items = [(a, b) for a in sp for b in sp]
    special_pairs = _make_word_pairs(special_items)
    rng = random.Random(20260818)
    random_pairs = [(rng.getrandbits(32), rng.getrandbits(32))
                    for _ in range(N_RANDOM)]
    stats: dict = {}
    special_mism = []
    random_mism = []
    for is_add in (False, True):
        special_mism += await _run_batch(dut, special_pairs, is_add, stats)
        random_mism += await _run_batch(dut, random_pairs, is_add, stats)
    specials_passed = not special_mism
    document = {
        "dut": "VX_packbf16_arith(NUM_LANES=1)",
        "golden": "integer-exact bf16 golden (golden_model.py)",
        "specials_checked": len(special_items) * 2,
        "specials_passed": specials_passed,
        "random_pairs": N_RANDOM,
        "checks_total": (len(special_items) + N_RANDOM * 2) * 2,
        "mismatches_total": len(special_mism) + len(random_mism),
        "by_class": {f"{op}/{cls}": n for (op, cls), n in sorted(stats.items())},
        "examples": (special_mism + random_mism)[:12],
    }
    Path(RESULT_PATH).write_text(json.dumps(document, indent=1),
                                 encoding="utf-8")
    dut._log.info("results.json written: %d/%d mismatches",
                  document["mismatches_total"], document["checks_total"])
