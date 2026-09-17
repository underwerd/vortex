# cocotb unit test for VX_tm_pr (publishPhase + RT completion + TRACE admission
# + the commit pipeline that drains both into the PQM enqueue / release / TaskState
# write / owner-migration requests). Small-module-first: the Owner table and the
# PQM queues are NOT here; the TB ties pqm_enq_ready=1 and records the request
# streams, checking them against a Python model.
#
# A single tick() drives inputs at the FallingEdge, settles, samples every output
# event, then clocks -- so stimulus and event capture never race.
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer

CFG_RT_SLOTS = 128
PH_TRACE, PH_SHADE, PH_FINAL, PH_RELEASE = 0, 1, 2, 3


def pack(vals, w):
    v = 0
    m = (1 << w) - 1
    for i, x in enumerate(vals):
        v |= (x & m) << (i * w)
    return v


def build_completion(task, hit, shader, t, gen=0, bu=0, bv=0, flags=0):
    v = 0
    v |= (task & 0xFF) << 0
    v |= (gen & 0x1F) << 8
    v |= (hit & 1) << 13
    v |= (t & 0xFFFFFFFF) << 14
    v |= (bu & 0xFFFFFFFF) << 46
    v |= (bv & 0xFFFFFFFF) << 78
    v |= (shader & 0x3F) << 110
    v |= (flags & 0x3FFFF) << 116
    return v


def build_desc(task, ray, bounce, cont):
    return (task << 296) | (ray << 40) | (bounce << 32) | cont


class PrTB:
    def __init__(self, dut):
        self.dut = dut
        self.ev = {"enq": [], "rel": [], "cmig": [], "amig": [], "tswr": [], "desc": []}
        self.last = {}

    def _g(self, s):
        return int(getattr(self.dut, s).value)

    async def tick(self, **drv):
        dut = self.dut
        await FallingEdge(dut.clk)
        for k, val in drv.items():
            getattr(dut, k).value = val
        await Timer(1, unit="ns")
        self.last = {
            "pub_ready": self._g("pub_req_ready"),
            "cmpl_ready": self._g("rt_completion_ready"),
            "adm_ready": self._g("adm_req_ready"),
            "est": self._g("rt_free_est"),
        }
        if self._g("pqm_enq_valid") and self._g("pqm_enq_ready"):
            self.ev["enq"].append((self._g("pqm_enq_phase"),
                                   self._g("pqm_enq_task_id"),
                                   self._g("pqm_enq_shader_id")))
        if self._g("pr_rel_valid"):
            self.ev["rel"].append(self._g("pr_rel_id"))
        if self._g("pr_cmpl_mig_valid"):
            self.ev["cmig"].append(self._g("pr_cmpl_mig_id"))
        if self._g("pr_adm_mig_valid"):
            self.ev["amig"].append(self._g("pr_adm_mig_id"))
        if self._g("pr_ts_wr_valid"):
            self.ev["tswr"].append((self._g("pr_ts_wr_addr"), self._g("pr_ts_wr_data")))
        if self._g("rt_adm_desc_valid"):
            self.ev["desc"].append(self._g("rt_adm_desc"))
        await RisingEdge(dut.clk)
        return self.last


async def reset_dut(dut):
    dut.reset.value = 1
    dut.pub_req_valid.value = 0
    dut.pub_req_mask.value = 0
    dut.pub_req_task_id.value = 0
    dut.pub_req_shader_id.value = 0
    dut.pub_req_next_phase.value = 0
    dut.pub_req_src.value = 0
    dut.adm_req_valid.value = 0
    dut.adm_req_task_id.value = 0
    dut.adm_req_ray.value = 0
    dut.adm_req_bounce.value = 0
    dut.adm_req_cont.value = 0
    dut.rt_adm_accept_valid.value = 0
    dut.rt_adm_accept.value = 0
    dut.rt_adm_reject_valid.value = 0
    dut.rt_adm_reject.value = 0
    dut.rt_completion_valid.value = 0
    dut.rt_completion.value = 0
    dut.pqm_enq_ready.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.reset.value = 0
    await FallingEdge(dut.clk)


async def do_pub(tb, mask, tasks, shaders, phases, src=1):
    drv = dict(pub_req_valid=1, pub_req_mask=mask,
               pub_req_task_id=pack(tasks, 8), pub_req_shader_id=pack(shaders, 6),
               pub_req_next_phase=pack(phases, 2), pub_req_src=src)
    for _ in range(200):
        last = await tb.tick(**drv)
        if last["pub_ready"] == 1:
            break
    else:
        raise AssertionError("pub_req never accepted")
    await tb.tick(pub_req_valid=0)


async def do_cmpl(tb, task, hit, shader, t):
    payload = build_completion(task, hit, shader, t)
    for _ in range(200):
        last = await tb.tick(rt_completion_valid=1, rt_completion=payload)
        if last["cmpl_ready"] == 1:
            break
    else:
        raise AssertionError("completion never accepted")
    await tb.tick(rt_completion_valid=0)


async def do_adm_req(tb, task, ray, bounce, cont):
    """Drive the admission request to acceptance, then tick until the descriptor
    has been emitted (FSM left in A_WAIT)."""
    drv = dict(adm_req_valid=1, adm_req_task_id=task, adm_req_ray=ray,
               adm_req_bounce=bounce, adm_req_cont=cont)
    for _ in range(200):
        last = await tb.tick(**drv)
        if last["adm_ready"] == 1:
            break
    else:
        raise AssertionError("adm_req never accepted")
    await tb.tick(adm_req_valid=0)
    # A_SEND emits the descriptor 1-2 cycles later
    for _ in range(6):
        await tb.tick()
        if tb.ev["desc"]:
            return


async def adrain(tb, n):
    for _ in range(n):
        await tb.tick()


@cocotb.test()
async def test_publish_shade(dut):
    """A full 32-lane SHADE publish streams 32 enqueues in lane order."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    tb = PrTB(dut)
    tasks = list(range(32))
    shaders = [(i * 3) % 64 for i in range(32)]
    await do_pub(tb, 0xFFFFFFFF, tasks, shaders, [PH_SHADE] * 32)
    await adrain(tb, 100)
    exp = [(PH_SHADE, tasks[l], shaders[l]) for l in range(32)]
    assert tb.ev["enq"] == exp, f"enq mismatch:\nRTL={tb.ev['enq']}\nEXP={exp}"
    assert tb.ev["rel"] == []


@cocotb.test()
async def test_publish_mixed(dut):
    """Per-lane phases: TRACE/SHADE/FINALIZE enqueue, RELEASE retires."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    tb = PrTB(dut)
    tasks = [40 + i for i in range(8)]
    shaders = [i for i in range(8)]
    phases = [PH_TRACE, PH_SHADE, PH_FINAL, PH_RELEASE,
              PH_SHADE, PH_RELEASE, PH_TRACE, PH_FINAL]
    mask = 0xFF
    await do_pub(tb, mask, tasks + [0] * 24, shaders + [0] * 24, phases + [0] * 24)
    await adrain(tb, 80)
    exp_enq, exp_rel = [], []
    for l in range(8):
        if phases[l] == PH_RELEASE:
            exp_rel.append(tasks[l])
        else:
            exp_enq.append((phases[l], tasks[l], shaders[l]))
    assert tb.ev["enq"] == exp_enq, f"enq RTL={tb.ev['enq']} EXP={exp_enq}"
    assert tb.ev["rel"] == exp_rel, f"rel RTL={tb.ev['rel']} EXP={exp_rel}"


@cocotb.test()
async def test_completion_hit_miss(dut):
    """Two completions: hit -> SHADE bucket + TaskState write; miss -> FINALIZE.
    Each migrates owner RT_SLOT->R_COMMIT at accept and bumps the free estimate."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    tb = PrTB(dut)
    await do_cmpl(tb, task=7, hit=1, shader=9, t=0x1234)
    await do_cmpl(tb, task=8, hit=0, shader=0, t=0x0)
    await adrain(tb, 60)
    assert tb.ev["cmig"] == [7, 8], f"completion migrations RTL={tb.ev['cmig']}"
    # every completion writes its trace overlay in WB (the miss carries t=0)
    assert tb.ev["tswr"] == [(7 << 5, 0x1234), (8 << 5, 0x0)], \
        f"TaskState write RTL={tb.ev['tswr']}"
    exp_enq = [(PH_SHADE, 7, 9), (PH_FINAL, 8, 0)]
    assert tb.ev["enq"] == exp_enq, f"enq RTL={tb.ev['enq']} EXP={exp_enq}"


@cocotb.test()
async def test_admission_accept(dut):
    """adm_req emits the descriptor; accept migrates owner->RT_SLOT, est -1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    tb = PrTB(dut)
    ray = 0xDEADBEEFCAFEBABE0102030405060708
    await do_adm_req(tb, task=11, ray=ray, bounce=2, cont=0xABCD)
    assert tb.ev["desc"] == [build_desc(11, ray, 2, 0xABCD)], \
        f"desc RTL={tb.ev['desc']}"
    est_before = tb.last["est"]
    await tb.tick(rt_adm_accept_valid=1, rt_adm_accept=(11 << 7) | 3)
    await tb.tick(rt_adm_accept_valid=0)
    assert tb.ev["amig"] == [11], f"admission migration RTL={tb.ev['amig']}"
    assert tb.last["est"] == est_before - 1, \
        f"est should drop by 1 on accept: {tb.last['est']} vs {est_before}"


@cocotb.test()
async def test_admission_reject(dut):
    """A rejected task re-injects as TRACE_READY (no external port)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    tb = PrTB(dut)
    await do_adm_req(tb, task=22, ray=0x1234, bounce=0, cont=0)
    await tb.tick(rt_adm_reject_valid=1, rt_adm_reject=22)
    await tb.tick(rt_adm_reject_valid=0)
    await adrain(tb, 60)
    assert tb.ev["amig"] == [], "a reject must not migrate to RT_SLOT"
    assert (PH_TRACE, 22, 0) in tb.ev["enq"], \
        f"rejected task should re-enqueue as TRACE_READY, got {tb.ev['enq']}"


@cocotb.test()
async def test_free_est_saturation(dut):
    """The estimate saturates at CFG_RT_SLOTS (completions) and 0 (admits)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    tb = PrTB(dut)
    # already at max: extra completions must not push past CFG_RT_SLOTS
    await do_cmpl(tb, task=1, hit=1, shader=0, t=0)
    await adrain(tb, 20)
    assert tb.last["est"] == CFG_RT_SLOTS, f"est should saturate high: {tb.last['est']}"
    # drain to 0 with admission accepts
    for i in range(CFG_RT_SLOTS + 5):
        await do_adm_req(tb, task=(i & 0xFF), ray=0, bounce=0, cont=0)
        await tb.tick(rt_adm_accept_valid=1, rt_adm_accept=((i & 0xFF) << 7) | 0)
        await tb.tick(rt_adm_accept_valid=0)
    await adrain(tb, 10)
    assert tb.last["est"] == 0, f"est should saturate at 0: {tb.last['est']}"


@cocotb.test()
async def test_pub_backpressure(dut):
    """pub_req_ready drops once the Publish FIFO cannot hold another 32."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    tb = PrTB(dut)
    # stall the commit drain so the FIFO fills
    dut.pqm_enq_ready.value = 0
    seen_ready_drop = False
    for r in range(5):
        tasks = [(r * 32 + i) & 0xFF for i in range(32)]
        await do_pub(tb, 0xFFFFFFFF, tasks, [0] * 32, [PH_SHADE] * 32)
        # after each publish, probe ready
        last = await tb.tick()
        if last["pub_ready"] == 0:
            seen_ready_drop = True
            break
    assert seen_ready_drop, "pub_req_ready never deasserted while the FIFO filled"
    dut.pqm_enq_ready.value = 1


@cocotb.test()
async def test_random_mix(dut):
    """Random publishes + completions; every entry must route correctly. Order
    between the two sources is arbitration-dependent, so compare as multisets."""
    import random
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    tb = PrTB(dut)
    rng = random.Random(0xF00D)
    exp_enq, exp_rel, exp_cmig = [], [], []
    tid = 0
    for r in range(8):
        if rng.random() < 0.5:
            n = rng.randrange(1, 9)
            mask = (1 << n) - 1
            tasks, shaders, phases = [], [], []
            for l in range(32):
                t = (tid + l) & 0xFF
                ph = rng.choice([PH_TRACE, PH_SHADE, PH_FINAL, PH_RELEASE])
                sh = rng.randrange(64)
                tasks.append(t); shaders.append(sh); phases.append(ph)
                if (mask >> l) & 1:
                    if ph == PH_RELEASE:
                        exp_rel.append(t)
                    else:
                        exp_enq.append((ph, t, sh))
            tid += 32
            await do_pub(tb, mask, tasks, shaders, phases)
        else:
            n = rng.randrange(1, 4)
            for _ in range(n):
                t = tid & 0xFF; tid += 1
                hit = rng.randrange(2)
                sh = rng.randrange(64)
                await do_cmpl(tb, task=t, hit=hit, shader=sh, t=0x1000 + t)
                exp_cmig.append(t)
                exp_enq.append((PH_SHADE if hit else PH_FINAL, t, sh))
        await adrain(tb, 40)
    await adrain(tb, 200)
    assert sorted(tb.ev["enq"]) == sorted(exp_enq), \
        f"enq multiset mismatch:\nRTL={sorted(tb.ev['enq'])}\nEXP={sorted(exp_enq)}"
    assert sorted(tb.ev["rel"]) == sorted(exp_rel), \
        f"rel multiset RTL={sorted(tb.ev['rel'])} EXP={sorted(exp_rel)}"
    assert sorted(tb.ev["cmig"]) == sorted(exp_cmig), \
        f"cmig multiset RTL={sorted(tb.ev['cmig'])} EXP={sorted(exp_cmig)}"
