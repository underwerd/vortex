# cocotb unit test for VX_tm_tsm (genTask face + Free-ID FIFO + seed/drain).
# Small-module-first: this exercises the TSM in isolation against a Python
# reference model of the genTask protocol, before the PQM/PR/top exist.
import collections
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer

NUM_TASKS = 256


class TsmModel:
    """Cycle-abstract model of the genTask result (granted mask, per-lane id,
    drain_stop). The RTL scans 32 lanes over 32 cycles; the result is what
    matters here, not the intermediate cycles."""

    def __init__(self, num_tasks=NUM_TASKS):
        self.free = collections.deque(range(num_tasks))
        self.seed = 0
        self.expected = 0
        self.draining = False

    def cfg_ld(self, expected):
        self.expected = expected
        self.seed = 0
        self.draining = False

    def gen(self, mask):
        granted = 0
        ids = [0] * 32
        if not self.draining:
            for lane in range(32):
                if (mask >> lane) & 1 and self.free and self.seed < self.expected:
                    tid = self.free.popleft()
                    ids[lane] = tid
                    granted |= (1 << lane)
                    self.seed += 1
            if self.expected != 0 and self.seed >= self.expected:
                self.draining = True
        drain_stop = 1 if (self.expected != 0 and self.seed >= self.expected) else 0
        return granted, ids, drain_stop

    def free_count(self):
        return len(self.free)


async def _idle_inputs(dut):
    dut.gen_req_valid.value = 0
    dut.gen_req_warp_id.value = 0
    dut.gen_req_mask.value = 0
    dut.cfg_ld.value = 0
    dut.cfg_expected_seeds.value = 0
    dut.free_push.value = 0
    dut.free_push_id.value = 0


async def reset_dut(dut):
    dut.reset.value = 1
    await _idle_inputs(dut)
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.reset.value = 0
    # the Free-ID FIFO refills 0..NUM_TASKS-1 after reset; wait for pool_ready
    for _ in range(NUM_TASKS + 50):
        await FallingEdge(dut.clk)
        if int(dut.pool_ready.value) == 1:
            return
    raise AssertionError("pool_ready never asserted after reset")


async def do_cfg(dut, model, expected):
    await FallingEdge(dut.clk)
    dut.cfg_expected_seeds.value = expected
    dut.cfg_ld.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.cfg_ld.value = 0
    model.cfg_ld(expected)


async def do_gen(dut, warp, mask):
    """Drive one genTask request to a SINGLE acceptance, then wait for its rsp.

    Assert valid, let ready settle in the same cycle, and deassert right after
    the one rising edge that captures — holding valid an extra cycle would
    double-capture into the pending table (two responses for one request)."""
    await FallingEdge(dut.clk)
    dut.gen_req_valid.value = 1
    dut.gen_req_warp_id.value = warp
    dut.gen_req_mask.value = mask
    await Timer(1, unit="ns")
    while int(dut.gen_req_ready.value) != 1:
        await FallingEdge(dut.clk)
        await Timer(1, unit="ns")
    await RisingEdge(dut.clk)          # this edge captures (valid && ready held)
    dut.gen_req_valid.value = 0
    for _ in range(200):
        await FallingEdge(dut.clk)
        if int(dut.gen_rsp_valid.value) == 1:
            return (int(dut.gen_rsp_warp_id.value),
                    int(dut.gen_rsp_granted.value),
                    int(dut.gen_rsp_task_id.value),
                    int(dut.gen_rsp_drain_stop.value))
    raise AssertionError("gen_rsp_valid never pulsed")


def granted_lane_ids(task_id_256, granted):
    return {lane: (task_id_256 >> (lane * 8)) & 0xFF
            for lane in range(32) if (granted >> lane) & 1}


@cocotb.test()
async def test_pool_ready(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    assert int(dut.pool_ready.value) == 1
    assert int(dut.free_count.value) == NUM_TASKS, \
        f"free_count={int(dut.free_count.value)} expected {NUM_TASKS}"
    assert int(dut.draining.value) == 0


@cocotb.test()
async def test_gen_full_grant(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    model = TsmModel()
    await do_cfg(dut, model, expected=100000)

    warp, mask = 2, 0xFFFFFFFF
    g_warp, g_granted, g_taskid, g_stop = await do_gen(dut, warp, mask)
    m_granted, m_ids, m_stop = model.gen(mask)
    assert g_warp == warp, f"warp RTL={g_warp} exp={warp}"
    assert g_granted == m_granted, f"granted RTL={g_granted:#x} ref={m_granted:#x}"
    assert g_stop == m_stop, f"drain_stop RTL={g_stop} ref={m_stop}"
    rtl_ids = granted_lane_ids(g_taskid, g_granted)
    ref_ids = {l: m_ids[l] for l in range(32) if (m_granted >> l) & 1}
    assert rtl_ids == ref_ids, f"task_ids RTL={rtl_ids} ref={ref_ids}"
    assert int(dut.free_count.value) == model.free_count(), \
        f"free_count RTL={int(dut.free_count.value)} ref={model.free_count()}"


@cocotb.test()
async def test_gen_partial_then_drain(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    model = TsmModel()
    # cap at 5: an 8-lane mask grants only lanes 0..4 and trips drain_stop
    await do_cfg(dut, model, expected=5)

    g_warp, g_granted, g_taskid, g_stop = await do_gen(dut, 1, 0xFF)
    m_granted, m_ids, m_stop = model.gen(0xFF)
    assert g_granted == m_granted == 0x1F, \
        f"partial granted RTL={g_granted:#x} ref={m_granted:#x} exp=0x1f"
    assert g_stop == m_stop == 1, f"drain_stop RTL={g_stop} ref={m_stop} exp=1"
    assert granted_lane_ids(g_taskid, g_granted) == {l: m_ids[l] for l in range(5)}

    # now draining: a further request must complete with granted=0, drain_stop=1
    g_warp, g_granted, g_taskid, g_stop = await do_gen(dut, 3, 0xFF)
    m_granted, m_ids, m_stop = model.gen(0xFF)
    assert g_granted == m_granted == 0, f"drained granted RTL={g_granted:#x} exp=0"
    assert g_stop == m_stop == 1, f"drained drain_stop RTL={g_stop} exp=1"
    assert int(dut.draining.value) == 1


@cocotb.test()
async def test_gen_stream_random(dut):
    """Seeded random masks/expected, several requests, checked vs the model."""
    import random
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    model = TsmModel()
    rng = random.Random(0xC0FFEE)
    await do_cfg(dut, model, expected=40)
    for i in range(6):
        warp = rng.randrange(8)
        mask = rng.getrandbits(32)
        g_warp, g_granted, g_taskid, g_stop = await do_gen(dut, warp, mask)
        m_granted, m_ids, m_stop = model.gen(mask)
        assert g_warp == warp, f"i={i} warp RTL={g_warp} exp={warp}"
        assert g_granted == m_granted, \
            f"i={i} mask={mask:#x} granted RTL={g_granted:#x} ref={m_granted:#x}"
        assert g_stop == m_stop, f"i={i} drain_stop RTL={g_stop} ref={m_stop}"
        assert granted_lane_ids(g_taskid, g_granted) == \
            {l: m_ids[l] for l in range(32) if (m_granted >> l) & 1}, \
            f"i={i} task_ids mismatch RTL vs ref"
        assert int(dut.free_count.value) == model.free_count(), \
            f"i={i} free_count RTL={int(dut.free_count.value)} ref={model.free_count()}"
        if model.draining:
            break
