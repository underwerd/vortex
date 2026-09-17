# cocotb unit test for VX_tm_pqm (getWork face: phase queues + Directory +
# Packet Assembly + owner-migration requests). Small-module-first: exercises the
# PQM in isolation against a Python reference model, before the PR/top exist.
#
# The Owner table is not here; the PQM REQUESTS migrations (pqm_mig_*). The TB
# ties pqm_mig_grant=1 (accept all) and checks the request stream, then checks
# the assembled gw_rsp packet.
import collections
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer

NUM_BUCKETS = 64
PH_TRACE, PH_SHADE, PH_FINAL = 0, 1, 2
ROLE_TRACE, ROLE_SHADE, ROLE_FINAL = 0, 1, 2
# tm_role_owner(): TRACE->TRACE_WK(2), SHADE->SHADE_WK(4), FINALIZE->FINAL_WK(5)
ROLE_OWNER = {ROLE_TRACE: 2, ROLE_SHADE: 4, ROLE_FINAL: 5}


class PqmModel:
    def __init__(self, num_buckets=NUM_BUCKETS):
        self.trace = collections.deque()
        self.final = collections.deque()
        self.buckets = [collections.deque() for _ in range(num_buckets)]

    def enq(self, phase, tid, sid):
        if phase == PH_TRACE:
            self.trace.append(tid)
        elif phase == PH_FINAL:
            self.final.append(tid)
        else:
            self.buckets[sid].append(tid)

    def shade_cand(self):
        # prefer a full bucket (>=32), lowest index; else any non-empty, lowest
        for b in range(len(self.buckets)):
            if len(self.buckets[b]) >= 32:
                return b
        for b in range(len(self.buckets)):
            if len(self.buckets[b]) > 0:
                return b
        return None

    def serve(self, role):
        """Pop one packet for `role`. Returns (popped, shader_id, count)."""
        if role == ROLE_TRACE:
            q, sid = self.trace, 0
        elif role == ROLE_FINAL:
            q, sid = self.final, 0
        else:
            b = self.shade_cand()
            if b is None:
                return None
            q, sid = self.buckets[b], b
        if len(q) == 0:
            return None
        total = min(len(q), 32)
        popped = [q.popleft() for _ in range(total)]
        return popped, sid, total


def pack(popped):
    v = 0
    for lane, tid in enumerate(popped):
        v |= (tid & 0xFF) << (lane * 8)
    return v


async def _idle_inputs(dut):
    dut.pool_ready.value = 1
    dut.kernel_done.value = 0
    dut.rt_free_est.value = 128
    dut.pqm_mig_grant.value = 1
    dut.gw_req_valid.value = 0
    dut.gw_req_warp_id.value = 0
    dut.gw_req_role.value = 0
    dut.pqm_enq_valid.value = 0
    dut.pqm_enq_phase.value = 0
    dut.pqm_enq_task_id.value = 0
    dut.pqm_enq_shader_id.value = 0


async def reset_dut(dut):
    dut.reset.value = 1
    await _idle_inputs(dut)
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.reset.value = 0
    await FallingEdge(dut.clk)


async def do_enq(dut, items):
    """Push a list of (phase, task_id, shader_id), one per cycle (no ready)."""
    for phase, tid, sid in items:
        await FallingEdge(dut.clk)
        dut.pqm_enq_valid.value = 1
        dut.pqm_enq_phase.value = phase
        dut.pqm_enq_task_id.value = tid
        dut.pqm_enq_shader_id.value = sid
        await RisingEdge(dut.clk)
        dut.pqm_enq_valid.value = 0
    await FallingEdge(dut.clk)


async def do_gw(dut, warp, role):
    """Register one pending getWork to a SINGLE acceptance."""
    await FallingEdge(dut.clk)
    dut.gw_req_valid.value = 1
    dut.gw_req_warp_id.value = warp
    dut.gw_req_role.value = role
    await Timer(1, unit="ns")
    while int(dut.gw_req_ready.value) != 1:
        await FallingEdge(dut.clk)
        await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    dut.gw_req_valid.value = 0


async def wait_rsp(dut, rng=None, grant_p=1.0, timeout=4000):
    """Wait for gw_rsp_valid; capture the owner-migration request stream on the
    way. If rng is given, drive pqm_mig_grant randomly (stall the assembly)."""
    mig = []
    for _ in range(timeout):
        await FallingEdge(dut.clk)
        # drive grant from the Python-known value (never read it back in the
        # same timestep -- that returns the stale value, cocotb pitfall #1)
        g = 1 if (rng is None or rng.random() < grant_p) else 0
        dut.pqm_mig_grant.value = g
        await Timer(1, unit="ns")     # let grant + the comb mig_id settle
        if int(dut.pqm_mig_valid.value) == 1 and g == 1:
            mig.append((int(dut.pqm_mig_id.value), int(dut.pqm_mig_val.value)))
        if int(dut.gw_rsp_valid.value) == 1:
            dut.pqm_mig_grant.value = 1
            return {
                "warp": int(dut.gw_rsp_warp_id.value),
                "task_id": int(dut.gw_rsp_task_id.value),
                "shader_id": int(dut.gw_rsp_shader_id.value),
                "count": int(dut.gw_rsp_count.value),
                "kernel_done": int(dut.gw_rsp_kernel_done.value),
            }, mig
    raise AssertionError("gw_rsp_valid never pulsed")


def check_packet(rsp, mig, popped, sid, role, warp):
    assert rsp["kernel_done"] == 0, f"unexpected kernel_done in a work packet"
    assert rsp["warp"] == warp, f"warp RTL={rsp['warp']} exp={warp}"
    assert rsp["count"] == len(popped), \
        f"count RTL={rsp['count']} exp={len(popped)}"
    assert rsp["shader_id"] == sid, \
        f"shader_id RTL={rsp['shader_id']} exp={sid}"
    assert rsp["task_id"] == pack(popped), \
        f"task_id RTL={rsp['task_id']:#x} exp={pack(popped):#x}"
    exp_mig = [(tid, ROLE_OWNER[role]) for tid in popped]
    assert mig == exp_mig, f"mig stream RTL={mig} exp={exp_mig}"


@cocotb.test()
async def test_trace_basic(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    m = PqmModel()
    ids = [10, 11, 12, 13, 14]
    await do_enq(dut, [(PH_TRACE, t, 0) for t in ids])
    for t in ids:
        m.enq(PH_TRACE, t, 0)
    await do_gw(dut, warp=2, role=ROLE_TRACE)
    rsp, mig = await wait_rsp(dut)
    popped, sid, total = m.serve(ROLE_TRACE)
    check_packet(rsp, mig, popped, sid, ROLE_TRACE, warp=2)
    assert total == 5


@cocotb.test()
async def test_shade_basic(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    m = PqmModel()
    ids = [20, 21, 22]
    await do_enq(dut, [(PH_SHADE, t, 7) for t in ids])
    for t in ids:
        m.enq(PH_SHADE, t, 7)
    await do_gw(dut, warp=1, role=ROLE_SHADE)
    rsp, mig = await wait_rsp(dut)
    popped, sid, total = m.serve(ROLE_SHADE)
    check_packet(rsp, mig, popped, sid, ROLE_SHADE, warp=1)
    assert sid == 7 and total == 3


@cocotb.test()
async def test_shade_prefers_full(dut):
    """A bucket with >=32 must win the candidate scan over a smaller one."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    m = PqmModel()
    items = [(PH_SHADE, 100 + i, 5) for i in range(5)]           # small bucket 5
    items += [(PH_SHADE, 200 + i, 9) for i in range(40)]         # full bucket 9
    await do_enq(dut, items)
    for ph, t, s in items:
        m.enq(ph, t, s)
    await do_gw(dut, warp=3, role=ROLE_SHADE)
    rsp, mig = await wait_rsp(dut)
    popped, sid, total = m.serve(ROLE_SHADE)
    check_packet(rsp, mig, popped, sid, ROLE_SHADE, warp=3)
    assert sid == 9, f"candidate should be the full bucket 9, got {sid}"
    assert total == 32, f"a full bucket yields a 32-task packet, got {total}"


@cocotb.test()
async def test_finalize_basic(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    m = PqmModel()
    ids = [30, 31]
    await do_enq(dut, [(PH_FINAL, t, 0) for t in ids])
    for t in ids:
        m.enq(PH_FINAL, t, 0)
    await do_gw(dut, warp=0, role=ROLE_FINAL)
    rsp, mig = await wait_rsp(dut)
    popped, sid, total = m.serve(ROLE_FINAL)
    check_packet(rsp, mig, popped, sid, ROLE_FINAL, warp=0)
    assert total == 2


@cocotb.test()
async def test_kernel_done_drain(dut):
    """With kernel_done set, a pending getWork returns count=0/kernel_done=1
    even though its queue is empty (invariant 12 landing point two)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_gw(dut, warp=4, role=ROLE_TRACE)   # nothing enqueued: not serviceable
    await FallingEdge(dut.clk)
    dut.kernel_done.value = 1
    rsp, mig = await wait_rsp(dut)
    assert rsp["kernel_done"] == 1, "expected a kernel_done response"
    assert rsp["count"] == 0, f"done packet count RTL={rsp['count']} exp=0"
    assert rsp["warp"] == 4, f"warp RTL={rsp['warp']} exp=4"
    assert mig == [], f"no migration on a done packet, got {mig}"
    dut.kernel_done.value = 0


@cocotb.test()
async def test_trace_gated_by_free_est(dut):
    """A TRACE getWork must NOT be served while rt_free_est==0, then served
    once a slot frees."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    m = PqmModel()
    ids = [40, 41]
    await do_enq(dut, [(PH_TRACE, t, 0) for t in ids])
    for t in ids:
        m.enq(PH_TRACE, t, 0)
    dut.rt_free_est.value = 0
    await do_gw(dut, warp=5, role=ROLE_TRACE)
    # no response while gated
    for _ in range(50):
        await FallingEdge(dut.clk)
        assert int(dut.gw_rsp_valid.value) == 0, "served a TRACE packet with free_est=0"
    dut.rt_free_est.value = 128
    rsp, mig = await wait_rsp(dut)
    popped, sid, total = m.serve(ROLE_TRACE)
    check_packet(rsp, mig, popped, sid, ROLE_TRACE, warp=5)


@cocotb.test()
async def test_mig_grant_stall(dut):
    """Random grant stalling must not corrupt the packet: same tasks, same
    order, just assembled over more cycles."""
    import random
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    m = PqmModel()
    ids = list(range(50, 70))   # 20 tasks in a bucket
    await do_enq(dut, [(PH_SHADE, t, 3) for t in ids])
    for t in ids:
        m.enq(PH_SHADE, t, 3)
    await do_gw(dut, warp=6, role=ROLE_SHADE)
    rng = random.Random(0xBEEF)
    rsp, mig = await wait_rsp(dut, rng=rng, grant_p=0.5)
    popped, sid, total = m.serve(ROLE_SHADE)
    check_packet(rsp, mig, popped, sid, ROLE_SHADE, warp=6)
    assert total == 20


@cocotb.test()
async def test_random_stream(dut):
    """Seeded random: load a random queue mix, serve one getWork at a time
    (single pending entry, so the service PE has no ambiguity), check each."""
    import random
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    m = PqmModel()
    rng = random.Random(0xC0FFEE)
    tid = 0
    for it in range(25):
        # random enqueue burst
        items = []
        for _ in range(rng.randrange(1, 12)):
            phase = rng.choice([PH_TRACE, PH_SHADE, PH_FINAL])
            sid = rng.randrange(NUM_BUCKETS) if phase == PH_SHADE else 0
            items.append((phase, tid, sid))
            tid += 1
        if items:
            await do_enq(dut, items)
            for ph, t, s in items:
                m.enq(ph, t, s)
        # pick a role that is actually serviceable so a packet must come back
        role = rng.choice([ROLE_TRACE, ROLE_SHADE, ROLE_FINAL])
        have = {ROLE_TRACE: len(m.trace), ROLE_FINAL: len(m.final),
                ROLE_SHADE: sum(1 for b in m.buckets if b)}[role]
        if have == 0:
            continue
        warp = rng.randrange(8)
        await do_gw(dut, warp=warp, role=role)
        rsp, mig = await wait_rsp(dut, rng=rng, grant_p=0.85)
        popped, sid, total = m.serve(role)
        check_packet(rsp, mig, popped, sid, role, warp)
