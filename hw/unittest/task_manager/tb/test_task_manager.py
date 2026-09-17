# cocotb integration test for the whole VX_task_manager (TSM + PQM + PR + shared
# state wired in the top). The TB plays every external agent: the init warp
# (genTask + publishPhase INIT + ts_wr), the role workers (getWork + publishPhase
# WORKER + TRACE admission), and an RT Unit BFM (accept/reject + completion +
# rt_rayslot_idle).
#
# test_t1_single_lifecycle walks ONE task through all 13 hops and checks the
# owner code at each. test_soak_multi runs many tasks with concurrent workers and
# a random RT BFM, then asserts kernel_done + the final audit (every owner FREE,
# live_count 0, queues empty).
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer

PH_TRACE, PH_SHADE, PH_FINAL, PH_RELEASE = 0, 1, 2, 3
ROLE_TRACE, ROLE_SHADE, ROLE_FINAL = 0, 1, 2
OWN_FREE, OWN_INIT, OWN_TRACE_WK, OWN_RT_SLOT = 0, 1, 2, 3
OWN_SHADE_WK, OWN_FINAL_WK, OWN_R_COMMIT = 4, 5, 6


def pack(vals, w):
    v, m = 0, (1 << w) - 1
    for i, x in enumerate(vals):
        v |= (x & m) << (i * w)
    return v


def build_completion(task, hit, shader, t, gen=0, bu=0, bv=0, flags=0):
    return ((task & 0xFF) << 0) | ((gen & 0x1F) << 8) | ((hit & 1) << 13) | \
           ((t & 0xFFFFFFFF) << 14) | ((bu & 0xFFFFFFFF) << 46) | \
           ((bv & 0xFFFFFFFF) << 78) | ((shader & 0x3F) << 110) | ((flags & 0x3FFFF) << 116)


# ── low-level face drivers (single-acceptance handshakes) ───────────────
async def wait_ready(dut, sig):
    await Timer(1, unit="ns")
    while int(getattr(dut, sig).value) != 1:
        await FallingEdge(dut.clk)
        await Timer(1, unit="ns")


async def reset_dut(dut):
    dut.reset.value = 1
    for s in ("cfg_ld", "cfg_expected_seeds", "gen_req_valid", "gen_req_warp_id",
              "gen_req_mask", "gw_req_valid", "gw_req_warp_id", "gw_req_role",
              "pub_req_valid", "pub_req_mask", "pub_req_task_id", "pub_req_shader_id",
              "pub_req_next_phase", "pub_req_src", "adm_req_valid", "adm_req_task_id",
              "adm_req_ray", "adm_req_bounce", "adm_req_cont", "rt_adm_accept_valid",
              "rt_adm_accept", "rt_adm_reject_valid", "rt_adm_reject",
              "rt_completion_valid", "rt_completion", "ts_wr_valid", "ts_wr_task_id",
              "ts_wr_word", "ts_wr_data", "ts_rd_valid", "ts_rd_task_id", "ts_rd_word"):
        getattr(dut, s).value = 0
    dut.rt_rayslot_idle.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.reset.value = 0
    # wait for the Free-ID FIFO refill
    for _ in range(400):
        await FallingEdge(dut.clk)
        if int(dut.sts_free_tasks.value) == 256:
            return
    raise AssertionError("Free-ID pool never became ready")


async def do_cfg(dut, expected):
    await FallingEdge(dut.clk)
    dut.cfg_expected_seeds.value = expected
    dut.cfg_ld.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.cfg_ld.value = 0


async def do_gen(dut, warp, mask):
    await FallingEdge(dut.clk)
    dut.gen_req_valid.value = 1
    dut.gen_req_warp_id.value = warp
    dut.gen_req_mask.value = mask
    await wait_ready(dut, "gen_req_ready")
    await RisingEdge(dut.clk)
    dut.gen_req_valid.value = 0
    for _ in range(300):
        await FallingEdge(dut.clk)
        if int(dut.gen_rsp_valid.value) == 1:
            return (int(dut.gen_rsp_warp_id.value), int(dut.gen_rsp_granted.value),
                    int(dut.gen_rsp_task_id.value), int(dut.gen_rsp_drain_stop.value))
    raise AssertionError("gen_rsp never pulsed")


async def do_pub(dut, mask, tasks, shaders, phases, src):
    await FallingEdge(dut.clk)
    dut.pub_req_valid.value = 1
    dut.pub_req_mask.value = mask
    dut.pub_req_task_id.value = pack(tasks, 8)
    dut.pub_req_shader_id.value = pack(shaders, 6)
    dut.pub_req_next_phase.value = pack(phases, 2)
    dut.pub_req_src.value = src
    await wait_ready(dut, "pub_req_ready")
    await RisingEdge(dut.clk)
    dut.pub_req_valid.value = 0


async def do_gw_issue(dut, warp, role):
    """Drive one getWork request to acceptance on the shared gw_req port (does
    NOT wait for the response -- a dispatcher routes that by warp_id)."""
    await FallingEdge(dut.clk)
    dut.gw_req_valid.value = 1
    dut.gw_req_warp_id.value = warp
    dut.gw_req_role.value = role
    await wait_ready(dut, "gw_req_ready")
    await RisingEdge(dut.clk)
    dut.gw_req_valid.value = 0


async def do_gw(dut, warp, role, timeout=8000):
    await do_gw_issue(dut, warp, role)
    for _ in range(timeout):
        await FallingEdge(dut.clk)
        if int(dut.gw_rsp_valid.value) == 1:
            return dict(warp=int(dut.gw_rsp_warp_id.value),
                        task_id=int(dut.gw_rsp_task_id.value),
                        shader=int(dut.gw_rsp_shader_id.value),
                        count=int(dut.gw_rsp_count.value),
                        done=int(dut.gw_rsp_kernel_done.value))
    raise AssertionError("gw_rsp never pulsed")


async def do_adm(dut, task, ray, bounce, cont):
    await FallingEdge(dut.clk)
    dut.adm_req_valid.value = 1
    dut.adm_req_task_id.value = task
    dut.adm_req_ray.value = ray
    dut.adm_req_bounce.value = bounce
    dut.adm_req_cont.value = cont
    await wait_ready(dut, "adm_req_ready")
    await RisingEdge(dut.clk)
    dut.adm_req_valid.value = 0
    for _ in range(60):
        await FallingEdge(dut.clk)
        if int(dut.rt_adm_desc_valid.value) == 1:
            return int(dut.rt_adm_desc.value)
    raise AssertionError("rt_adm_desc never pulsed")


async def do_ts_wr(dut, task, word, data):
    await FallingEdge(dut.clk)
    dut.ts_wr_valid.value = 1
    dut.ts_wr_task_id.value = task
    dut.ts_wr_word.value = word
    dut.ts_wr_data.value = data
    await wait_ready(dut, "ts_wr_ready")
    await RisingEdge(dut.clk)
    dut.ts_wr_valid.value = 0


async def do_ts_rd(dut, task, word):
    await FallingEdge(dut.clk)
    dut.ts_rd_valid.value = 1
    dut.ts_rd_task_id.value = task
    dut.ts_rd_word.value = word
    await RisingEdge(dut.clk)
    dut.ts_rd_valid.value = 0
    for _ in range(10):
        await FallingEdge(dut.clk)
        if int(dut.ts_rd_data_valid.value) == 1:
            return int(dut.ts_rd_data.value)
    raise AssertionError("ts_rd_data_valid never pulsed")


async def rt_accept(dut, task, slot):
    await FallingEdge(dut.clk)
    dut.rt_adm_accept_valid.value = 1
    dut.rt_adm_accept.value = (task << 7) | slot
    await RisingEdge(dut.clk)
    dut.rt_adm_accept_valid.value = 0


async def rt_reject(dut, task):
    await FallingEdge(dut.clk)
    dut.rt_adm_reject_valid.value = 1
    dut.rt_adm_reject.value = task
    await RisingEdge(dut.clk)
    dut.rt_adm_reject_valid.value = 0


async def rt_completion(dut, task, hit, shader, t):
    await FallingEdge(dut.clk)
    dut.rt_completion_valid.value = 1
    dut.rt_completion.value = build_completion(task, hit, shader, t)
    await wait_ready(dut, "rt_completion_ready")
    await RisingEdge(dut.clk)
    dut.rt_completion_valid.value = 0


def owner_of(dut, task):
    return (int(dut.owner.value) >> (task * 3)) & 0x7


def lane_ids(task_id_256, count):
    return [(task_id_256 >> (l * 8)) & 0xFF for l in range(count)]


# ═══════════════════════ T1: one task, all 13 hops ═══════════════════════
@cocotb.test()
async def test_t1_single_lifecycle(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_cfg(dut, expected=1)

    # 1. genTask: init creates task 0 (seed cap 1 -> drain_stop, draining)
    warp, granted, tid256, stop = await do_gen(dut, 0, 0x1)
    assert granted == 0x1 and stop == 1, f"gen granted={granted:#x} stop={stop}"
    t0 = tid256 & 0xFF
    assert t0 == 0, f"first free id should be 0, got {t0}"
    assert owner_of(dut, t0) == OWN_INIT, f"owner after gen={owner_of(dut, t0)}"
    assert int(dut.sts_live_count.value) == 1
    assert int(dut.sts_draining.value) == 1
    # draining is set but a task is still live: kernel_done must NOT fire early
    assert int(dut.sts_kernel_done.value) == 0, "kernel_done asserted while live_count=1"

    # 2. init writes TaskState, then publishes t0 -> TRACE_READY (fence contract)
    await do_ts_wr(dut, t0, 0, 0xA5A5A5A5)
    await do_pub(dut, 0x1, [t0], [0], [PH_TRACE], src=0)
    assert owner_of(dut, t0) == OWN_INIT, "queued task keeps the producer (INIT) owner"

    # 3. TRACE worker getWork -> packet {t0}, owner INIT->TRACE_WK
    rsp = await do_gw(dut, 1, ROLE_TRACE)
    assert rsp["count"] == 1 and lane_ids(rsp["task_id"], 1) == [t0], f"TRACE pkt={rsp}"
    assert owner_of(dut, t0) == OWN_TRACE_WK, f"owner after gw(TRACE)={owner_of(dut, t0)}"

    # 4. admission: TRACE worker sends t0 to the RT Unit
    dut.rt_rayslot_idle.value = 0
    desc = await do_adm(dut, t0, ray=0x1234, bounce=0, cont=0x7)
    assert (desc >> 296) & 0xFF == t0, "desc task_id mismatch"

    # 5. RT accepts -> owner TRACE_WK->RT_SLOT
    await rt_accept(dut, t0, slot=5)
    await FallingEdge(dut.clk)
    assert owner_of(dut, t0) == OWN_RT_SLOT, f"owner after accept={owner_of(dut, t0)}"

    # 6. RT completion (hit, shader 9) -> owner RT_SLOT->R_COMMIT, routed to SHADE
    await rt_completion(dut, t0, hit=1, shader=9, t=0x4000)
    await FallingEdge(dut.clk)
    assert owner_of(dut, t0) == OWN_R_COMMIT, f"owner after completion={owner_of(dut, t0)}"
    dut.rt_rayslot_idle.value = 1

    # 7. SHADE worker getWork -> packet {t0} from bucket 9, owner R_COMMIT->SHADE_WK
    rsp = await do_gw(dut, 2, ROLE_SHADE)
    assert rsp["count"] == 1 and rsp["shader"] == 9, f"SHADE pkt={rsp}"
    assert lane_ids(rsp["task_id"], 1) == [t0]
    assert owner_of(dut, t0) == OWN_SHADE_WK, f"owner after gw(SHADE)={owner_of(dut, t0)}"

    # 8. SHADE worker writes results, publishes t0 -> FINALIZE_READY
    await do_ts_wr(dut, t0, 1, 0xDEAD)
    await do_pub(dut, 0x1, [t0], [0], [PH_FINAL], src=1)
    assert owner_of(dut, t0) == OWN_SHADE_WK, "queued task keeps the producer owner"

    # 9. FINALIZE worker getWork -> packet {t0}, owner SHADE_WK->FINAL_WK
    rsp = await do_gw(dut, 3, ROLE_FINAL)
    assert rsp["count"] == 1 and lane_ids(rsp["task_id"], 1) == [t0], f"FINAL pkt={rsp}"
    assert owner_of(dut, t0) == OWN_FINAL_WK, f"owner after gw(FINAL)={owner_of(dut, t0)}"

    # 10. FINALIZE worker publishes RELEASE -> owner FREE, id recycled, live--
    await do_pub(dut, 0x1, [t0], [0], [PH_RELEASE], src=1)
    for _ in range(80):
        await FallingEdge(dut.clk)
        if owner_of(dut, t0) == OWN_FREE:
            break
    assert owner_of(dut, t0) == OWN_FREE, f"owner after release={owner_of(dut, t0)}"
    assert int(dut.sts_live_count.value) == 0, f"live={int(dut.sts_live_count.value)}"

    # 11. kernel_done: draining + live0 + queues empty + fifos empty + rayslot idle
    for _ in range(200):
        await FallingEdge(dut.clk)
        if int(dut.sts_kernel_done.value) == 1:
            break
    assert int(dut.sts_kernel_done.value) == 1, "kernel_done never asserted"
    # final audit
    assert int(dut.owner.value) == 0, "some owner is not FREE at the end"
    assert int(dut.sts_free_tasks.value) == 256, \
        f"free pool not fully returned: {int(dut.sts_free_tasks.value)}"


@cocotb.test()
async def test_taskstate_readback(dut):
    """The TaskState RAM is a VX_dp_ram (1W1R, registered read). Write several
    words of a task's 128-B entry through the external port and read them back."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await do_cfg(dut, expected=1)
    vals = {0: 0xA5A5A5A5, 1: 0xDEADBEEF, 17: 0x0BADF00D, 31: 0x12345678}
    for w, d in vals.items():
        await do_ts_wr(dut, 5, w, d)
    for w, d in vals.items():
        got = await do_ts_rd(dut, 5, w)
        assert got == d, f"task5 word{w}: RTL={got:#010x} exp={d:#010x}"
    # a different task's entry must be untouched (no address aliasing)
    assert await do_ts_rd(dut, 6, 0) == 0, "task6 word0 should still read 0"


# ═══════════════════════ soak: many tasks, concurrent agents ═══════════════════════
# Each physical face is a SINGLE port, so every agent that shares it (getWork:
# three role workers; publishPhase: init + shade + finalize) funnels through one
# driver coroutine with a request queue -- mirroring how the SM arbitrates warps
# onto one port. The RT BFM owns the disjoint rt_* signals and splits its
# desc-watcher from its completion driver so accepts never block completions.
from cocotb.queue import Queue
from cocotb.triggers import Event


class PubAgent:
    def __init__(self, dut):
        self.dut = dut
        self.q = Queue()
        cocotb.start_soon(self._driver())

    async def _driver(self):
        while True:
            (mask, tasks, shaders, phases, src), ev = await self.q.get()
            await do_pub(self.dut, mask, tasks, shaders, phases, src)
            ev.set()

    async def publish(self, mask, tasks, shaders, phases, src):
        ev = Event()
        await self.q.put(((mask, tasks, shaders, phases, src), ev))
        await ev.wait()


class GwAgent:
    """getWork: register on the shared gw_req port (serialized), then await this
    warp's response, which a dispatcher routes off the shared gw_rsp port."""

    def __init__(self, dut):
        self.dut = dut
        self.regq = Queue()
        self.waiters = {}
        self.responses = {}
        cocotb.start_soon(self._reg_driver())
        cocotb.start_soon(self._dispatcher())

    async def _reg_driver(self):
        while True:
            warp, role, ev = await self.regq.get()
            await do_gw_issue(self.dut, warp, role)
            ev.set()

    async def _dispatcher(self):
        dut = self.dut
        while True:
            await FallingEdge(dut.clk)
            if int(dut.gw_rsp_valid.value) == 1:
                warp = int(dut.gw_rsp_warp_id.value)
                rsp = dict(task_id=int(dut.gw_rsp_task_id.value),
                           shader=int(dut.gw_rsp_shader_id.value),
                           count=int(dut.gw_rsp_count.value),
                           done=int(dut.gw_rsp_kernel_done.value))
                w = self.waiters.get(warp)
                if w is not None and not w.is_set():
                    self.responses[warp] = rsp
                    w.set()

    async def getwork(self, warp, role):
        resp = Event()
        self.waiters[warp] = resp
        reg = Event()
        await self.regq.put((warp, role, reg))
        await reg.wait()
        await resp.wait()
        self.waiters.pop(warp, None)
        return self.responses.pop(warp)


class RtBfm:
    def __init__(self, dut, rng, reject_p=0.15, hit_p=0.7, nshaders=64):
        self.dut = dut
        self.rng = rng
        self.reject_p = reject_p
        self.hit_p = hit_p
        self.nshaders = nshaders
        self.inflight = 0
        self.slot = 0
        self.cq = Queue()
        cocotb.start_soon(self._desc_loop())
        cocotb.start_soon(self._completer())

    async def _desc_loop(self):
        dut = self.dut
        while True:
            await FallingEdge(dut.clk)
            if int(dut.rt_adm_desc_valid.value) == 1:
                task = (int(dut.rt_adm_desc.value) >> 296) & 0xFF
                if self.rng.random() < self.reject_p:
                    await rt_reject(dut, task)
                else:
                    self.inflight += 1
                    dut.rt_rayslot_idle.value = 0
                    await rt_accept(dut, task, self.slot & 0x7F)
                    self.slot += 1
                    await self.cq.put(task)

    async def _completer(self):
        dut = self.dut
        while True:
            task = await self.cq.get()
            for _ in range(self.rng.randrange(1, 6)):
                await FallingEdge(dut.clk)
            hit = 1 if self.rng.random() < self.hit_p else 0
            await rt_completion(dut, task, hit, self.rng.randrange(self.nshaders),
                                0x1000 + task)
            self.inflight -= 1
            if self.inflight == 0:
                dut.rt_rayslot_idle.value = 1


async def trace_worker(dut, gw, adm_tasks):
    while True:
        rsp = await gw.getwork(1, ROLE_TRACE)
        if rsp["done"]:
            return
        for t in lane_ids(rsp["task_id"], rsp["count"]):
            await do_adm(dut, t, ray=t, bounce=0, cont=0)


async def shade_worker(dut, gw, pub):
    while True:
        rsp = await gw.getwork(2, ROLE_SHADE)
        if rsp["done"]:
            return
        tasks = lane_ids(rsp["task_id"], rsp["count"])
        await pub.publish((1 << rsp["count"]) - 1, tasks, [0] * len(tasks),
                          [PH_FINAL] * len(tasks), 1)


async def final_worker(dut, gw, pub):
    while True:
        rsp = await gw.getwork(3, ROLE_FINAL)
        if rsp["done"]:
            return
        tasks = lane_ids(rsp["task_id"], rsp["count"])
        await pub.publish((1 << rsp["count"]) - 1, tasks, [0] * len(tasks),
                          [PH_RELEASE] * len(tasks), 1)


async def init_warp(dut, pub, n, rng):
    """Generate n seeds and publish each granted task to TRACE_READY."""
    produced = 0
    while produced < n:
        mask = rng.getrandbits(min(8, n - produced)) | 1
        _, granted, tid256, stop_flag = await do_gen(dut, 0, mask)
        ids = lane_ids(tid256, 32)
        got = [ids[l] for l in range(32) if (granted >> l) & 1]
        for t in got:
            await do_ts_wr(dut, t, 0, 0x1111 * (t + 1))
        if got:
            await pub.publish(granted, ids, [0] * 32,
                              [PH_TRACE if ((granted >> l) & 1) else 0 for l in range(32)],
                              src=0)
        produced += len(got)
        if stop_flag:
            break


@cocotb.test()
async def test_soak_multi(dut):
    import os
    import random
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    N         = int(os.environ.get("SOAK_N", "40"))
    seed      = int(os.environ.get("SOAK_SEED", "0x5EED"), 0)
    reject_p  = float(os.environ.get("SOAK_REJECT", "0.15"))
    hit_p     = float(os.environ.get("SOAK_HIT", "0.7"))
    nshaders  = int(os.environ.get("SOAK_SHADERS", "64"))
    await do_cfg(dut, expected=N)
    rng = random.Random(seed)
    cocotb.log.info(f"soak: N={N} seed={seed:#x} reject_p={reject_p} "
                    f"hit_p={hit_p} nshaders={nshaders}")

    pub = PubAgent(dut)
    gw = GwAgent(dut)
    bfm = RtBfm(dut, rng, reject_p=reject_p, hit_p=hit_p, nshaders=nshaders)
    cocotb.start_soon(trace_worker(dut, gw, None))
    cocotb.start_soon(shade_worker(dut, gw, pub))
    cocotb.start_soon(final_worker(dut, gw, pub))
    cocotb.start_soon(init_warp(dut, pub, N, rng))

    done = False
    for _ in range(200000):
        await FallingEdge(dut.clk)
        if int(dut.sts_kernel_done.value) == 1:
            done = True
            break
    assert done, f"kernel_done never asserted; live={int(dut.sts_live_count.value)} " \
                 f"free={int(dut.sts_free_tasks.value)} inflight={bfm.inflight}"
    # final audit (T9): every owner FREE, live 0, the whole pool returned
    await FallingEdge(dut.clk)
    assert int(dut.owner.value) == 0, "an owner is not FREE at kernel_done"
    assert int(dut.sts_live_count.value) == 0
    assert int(dut.sts_free_tasks.value) == 256, \
        f"free pool not fully returned: {int(dut.sts_free_tasks.value)}"

