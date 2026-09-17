// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// VX_rtu_core — socket-shared ray-traversal engine, and the sole WRITER of the
// hit window (see VX_rtu_bus_if). Rays stage on arrival, a pool of SLOTS
// traverses several of them at once, and the scheduler's shared front end
// switches between resident contexts whenever one parks on memory.
//
// Two structures:
//
//   STAGING  one entry per {src, wid} — per warp of every core this RTU serves.
//            An arm and its RAY beats land here UNCONDITIONALLY: a warp holds
//            one trace, so its entry is free by construction. That keeps the
//            arm's ready a constant 1, so a TRACE burst can never stall in the
//            in-order SFU while holding the issue lock. The staging RAM is the
//            SINGLE home of the incoming ray: a slot holds a POINTER to its
//            entry, never a copy, and a candidate record's object-ray words are
//            read back from these same rows at write-back.
//
//   COHORTS  RTU_NUM_SLOTS admission batches in flight (the config macro keeps
//            its historical name). Contexts are a global pool: NUM_CTX ids on
//            a free-list FIFO, allocated per ray. A cohort is a small
//            descriptor — the staging pointer, the warp identity, a record
//            space of NUM_LANES {cohort,pos} slots in the scheduler, and a
//            record-walk FSM. The drain sequencer claims a free cohort for a
//            staged ray, burst-reads its 8 beat rows (all lanes per row), then
//            seeds one ray per cycle into a free-list-allocated context — each
//            ray starts traversing as it lands, and its context returns to the
//            pool the cycle its walk terminates, cohort still running or not.
//
// The result record lives in the scheduler's window store, one field-row (all
// lanes) per word: the record write-back here is an address counter over those
// rows, plus the object-ray rows read from staging, with the status written
// LAST — writing it is what completes the warp's parked WAIT.

`include "VX_define.vh"

module VX_rtu_core import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `VX_CFG_NUM_THREADS,  // = one context per thread of a warp
    parameter NUM_WARPS = `VX_CFG_NUM_WARPS,    // warps a source core can have in flight
    parameter NUM_SRCS  = 1,   // cores this RTU serves
    parameter TAG_WIDTH = 1,
    parameter CACHE_DATA_SIZE = `VX_CFG_MEM_BLOCK_SIZE,
    parameter CACHE_TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    // window bus (this core is the master)
    VX_rtu_bus_if.slave  rtu_bus_if,

    // RTCache port
    VX_mem_bus_if.master cache_bus_if
);
    // The slot spans the walk writes must each be contiguous, or the base+index
    // addressing below silently targets the wrong slots.
    `STATIC_ASSERT((`VX_RT_OBJECT_RAY_ORIGIN == RTU_RES_BASE + RTU_RES_HIT),
        ("the object ray must abut the hit attributes"))
    `STATIC_ASSERT((`VX_RT_CB_HANDLE == RTU_RES_BASE + RTU_RES_CAND - 1),
        ("the candidate result slots must be one contiguous span"))

    localparam SRC_WIDTH = `UP(`CLOG2(NUM_SRCS));

    // Register the outgoing bus/cache interfaces at this module boundary so the
    // SLR-crossing seams launch/capture at flops (see VX_rtu_bus_slice).
    VX_rtu_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH),
        .SRC_WIDTH (SRC_WIDTH)
    ) rtu_bus_w ();

    VX_rtu_bus_slice #(
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (TAG_WIDTH),
        .SRC_WIDTH   (SRC_WIDTH),
        .ARM_OUT_BUF (0),  // arm_ready is a constant 1; nothing to register
        .REQ_OUT_BUF (0),  // req already registered upstream (unit/arb)
        .SLV_OUT_BUF (3)   // register our outgoing window accesses
    ) rtu_bus_reg (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (rtu_bus_if),
        .bus_out_if (rtu_bus_w)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (CACHE_DATA_SIZE),
        .TAG_WIDTH (CACHE_TAG_WIDTH)
    ) cache_bus_w ();

    VX_mem_bus_slice #(
        .DATA_SIZE   (CACHE_DATA_SIZE),
        .TAG_WIDTH   (CACHE_TAG_WIDTH),
        .REQ_OUT_BUF (3),  // register our outgoing RTCache request
        .RSP_OUT_BUF (0)   // response registered by the RTCache output
    ) cache_bus_reg (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (cache_bus_w),
        .bus_out_if (cache_bus_if)
    );
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam LINE_BITS = `VX_CFG_MEM_BLOCK_SIZE * 8;

    // ── the ray pool ──────────────────────────────────────────────────
    localparam NUM_COHORTS = `VX_CFG_RTU_NUM_SLOTS;
    localparam NUM_CTX     = `VX_CFG_RTU_NUM_CTX;
    localparam COHORT_W    = `LOG2UP(NUM_COHORTS);
    localparam CTX_TAG_W   = `LOG2UP(NUM_CTX);
    localparam LANE_W      = `LOG2UP(NUM_LANES);
    localparam FLW         = CTX_TAG_W + 1;   // free-list counter width

    `STATIC_ASSERT((NUM_CTX >= NUM_COHORTS * NUM_LANES),
        ("RTU_NUM_CTX must cover every cohort's records: NUM_CTX >= NUM_COHORTS * NUM_THREADS"))

    // MERGE_DEPTH — the MSHR file that merges duplicate node fetches. This core
    // issues one request per context per fetch (per-context tags), which is
    // exactly what MERGE_DEPTH=0 selects; a config that raised it would describe
    // a machine this core is not. Fail the build instead of diverging silently.
    `STATIC_ASSERT((`VX_CFG_RTU_MERGE_DEPTH == 0),
        ("VX_CFG_RTU_MERGE_DEPTH > 0 is not implemented: this core does not merge node fetches"))

    // ── ray staging: one entry per {src, wid} ─────────────────────────
    localparam NUM_STG   = NUM_SRCS * NUM_WARPS;
    localparam STG_IDX_W = `LOG2UP(NUM_STG);
    localparam RAY_IDX_W = `CLOG2(RTU_RAY_BEATS);

    // A warp WAITs for its trace before arming another, and staging is keyed by
    // {src, wid}, so at most NUM_STG traces can ever be waiting for a cohort.
    // Cohorts past that are unreachable; the coupling is invisible unless a
    // config that breaks it fails the build.
    `STATIC_ASSERT((NUM_COHORTS <= NUM_STG),
        ("RTU_NUM_SLOTS must not exceed NUM_SRCS * NUM_WARPS: a warp holds at most one trace in flight"))

    reg [NUM_STG-1:0]                stg_armed;   // the arm's scalars are here
    reg [NUM_STG-1:0]                stg_full;    // all RTU_RAY_BEATS words are here
    reg [NUM_STG-1:0][RAY_IDX_W-1:0] stg_beat;    // ray beats landed so far

    // ── per-cohort descriptors ────────────────────────────────────────
    localparam [2:0] T_IDLE   = 3'd0,  // free
                     T_FILL   = 3'd1,  // the drain sequencer is streaming its rays
                     T_BUSY   = 3'd2,  // traversing
                     T_WRITE  = 3'd3,  // writing the record back (terminal | candidate)
                     T_CBWAIT = 3'd4,  // candidate returned; await the CONTINUE's t
                     T_CBATTR = 3'd5,  // ... and its hitAttribute (CONT beat 1)
                     T_RESUME = 3'd6,  // release this cohort's yield barrier
                     T_RWAIT  = 3'd7;  // await the resume commit -> terminal record
    reg [NUM_COHORTS-1:0][2:0]           tstate;

    reg [NUM_COHORTS-1:0][NUM_LANES-1:0] req_mask;
    reg [NUM_COHORTS-1:0][TAG_WIDTH-1:0] req_tag;
    reg [NUM_COHORTS-1:0][STG_IDX_W-1:0] req_stg;      // the staging entry it launched from
    reg [NUM_COHORTS-1:0][NW_WIDTH-1:0]  req_wid;      // its warp, for the window write
    reg [NUM_COHORTS-1:0][31:0]          req_payload;  // warp-uniform; staged for the callbacks
    reg [NUM_COHORTS-1:0]                is_cand;
    reg [NUM_COHORTS-1:0][NUM_LANES-1:0][RTU_CB_ACTION_BITS-1:0] cont_action;

    // ── the scheduler ─────────────────────────────────────────────────
    wire                       ss_valid;
    wire [COHORT_W-1:0]        ss_cohort;
    wire [NUM_LANES-1:0]       ss_mask;
    wire [15:0]                ss_flags, ss_cull;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] ss_scene;
    wire                       rw_valid;
    wire [CTX_TAG_W-1:0]       rw_ctx;
    wire [COHORT_W-1:0]        rw_cohort;
    wire [LANE_W-1:0]          rw_pos;
    wire [RTU_RAY_BEATS*32-1:0] rw_data;
    wire                       ctx_done_valid;
    wire [CTX_TAG_W-1:0]       ctx_done_id;

    wire [NUM_COHORTS-1:0]     sch_busy, sch_done, sch_yield;
    wire [NUM_CTX-1:0]         sch_hit, sch_yld, sch_attrv;
    wire [NUM_CTX-1:0][RTU_CB_TYPE_BITS-1:0] sch_cbtype;
    wire [NUM_COHORTS-1:0]     sch_resume;
    wire [NUM_CTX-1:0][RTU_CB_ACTION_BITS-1:0] sch_action;

    localparam WS_ADDRW = COHORT_W + RTU_WS_WORD_BITS;
    localparam ROW_BITS = NUM_LANES * 32;
    wire                 ww_valid;
    wire [WS_ADDRW-1:0]  ww_addr;
    wire [NUM_LANES-1:0] ww_wren;
    wire [ROW_BITS-1:0]  ww_data;
    wire                 wr_valid;
    wire [WS_ADDRW-1:0]  wr_addr;
    wire [ROW_BITS-1:0]  wr_data;

    wire                              m_req_valid, m_req_ready, m_rsp_valid, m_rsp_ready;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] m_req_addr;
    wire [CTX_TAG_W-1:0]              m_req_tag, m_rsp_tag;
    wire [LINE_BITS-1:0]              m_rsp_data;

    `UNUSED_VAR (sch_busy)

    VX_rtu_scheduler #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_COHORTS (NUM_COHORTS),
        .NUM_LANES   (NUM_LANES),
        .NUM_CTX     (NUM_CTX)
    ) scheduler (
        .clk                (clk),
        .reset              (reset),
        .cohort_start_valid (ss_valid),
        .cohort_start_cohort(ss_cohort),
        .cohort_start_mask  (ss_mask),
        .cohort_start_flags (ss_flags),
        .cohort_start_cull  (ss_cull),
        .cohort_start_scene (ss_scene),
        .ray_wr_valid       (rw_valid),
        .ray_wr_ctx         (rw_ctx),
        .ray_wr_cohort      (rw_cohort),
        .ray_wr_pos         (rw_pos),
        .ray_wr_data        (rw_data),
        .ctx_done_valid     (ctx_done_valid),
        .ctx_done_id        (ctx_done_id),
        .busy               (sch_busy),
        .done               (sch_done),
        .yield              (sch_yield),
        .hit_bits           (sch_hit),
        .yld_bits           (sch_yld),
        .cb_types           (sch_cbtype),
        .attr_vld           (sch_attrv),
        .resume             (sch_resume),
        .action             (sch_action),
        .win_wr_valid       (ww_valid),
        .win_wr_addr        (ww_addr),
        .win_wr_wren        (ww_wren),
        .win_wr_data        (ww_data),
        .win_rd_valid       (wr_valid),
        .win_rd_addr        (wr_addr),
        .win_rd_data        (wr_data),
        .mem_req_valid      (m_req_valid),
        .mem_req_addr       (m_req_addr),
        .mem_req_tag        (m_req_tag),
        .mem_req_ready      (m_req_ready),
        .mem_rsp_valid      (m_rsp_valid),
        .mem_rsp_data       (m_rsp_data),
        .mem_rsp_tag        (m_rsp_tag),
        .mem_rsp_ready      (m_rsp_ready)
    );

    // ── RTCache port: node/leaf fetch ─────────────────────────────────
    localparam RTU_LINE_SIZE  = `VX_CFG_MEM_BLOCK_SIZE;
    localparam RTU_LINE_ADDRW = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(RTU_LINE_SIZE);
    `UNUSED_VAR (m_req_addr[`CLOG2(RTU_LINE_SIZE)-1:0])

    `STATIC_ASSERT(CTX_TAG_W <= $bits(cache_bus_w.req_data.tag.value),
        ("rtu fetch tag (%0d bits) does not fit the rtcache tag field", CTX_TAG_W))

    assign cache_bus_w.req_valid        = m_req_valid;
    assign cache_bus_w.req_data.rw      = 1'b0;
    assign cache_bus_w.req_data.addr    = m_req_addr[`VX_CFG_MEM_ADDR_WIDTH-1 -: RTU_LINE_ADDRW];
    assign cache_bus_w.req_data.data    = '0;
    assign cache_bus_w.req_data.byteen  = {RTU_LINE_SIZE{1'b1}};
    assign cache_bus_w.req_data.tag.uuid  = '0;
    assign cache_bus_w.req_data.tag.value = $bits(cache_bus_w.req_data.tag.value)'(m_req_tag);
    assign cache_bus_w.req_data.attr    = '0;
    assign m_req_ready = cache_bus_w.req_ready;

    assign m_rsp_valid = cache_bus_w.rsp_valid;
    assign m_rsp_data  = cache_bus_w.rsp_data.data;
    assign m_rsp_tag   = CTX_TAG_W'(cache_bus_w.rsp_data.tag.value);
    assign cache_bus_w.rsp_ready = m_rsp_ready;
    `UNUSED_VAR (cache_bus_w.rsp_data.tag.uuid)

    // ── the req channel: RAY beats, and the CONTINUE ──────────────────
    // Always ready: everything arriving here is something this core is already
    // waiting for, so it can never back-pressure.
    assign rtu_bus_w.req_ready = 1'b1;
    wire req_fire = rtu_bus_w.req_valid;
    wire is_ray   = req_fire && (rtu_bus_w.req_data.kind == RTU_REQ_RAY);
    wire is_cont  = req_fire && (rtu_bus_w.req_data.kind == RTU_REQ_CONT);

    wire [STG_IDX_W-1:0] req_stg_idx = STG_IDX_W'({rtu_bus_w.req_data.src, rtu_bus_w.req_data.wid});

    // ── the arm channel: ALWAYS ready ─────────────────────────────────
    assign rtu_bus_w.arm_ready = 1'b1;
    wire arm_fire = rtu_bus_w.arm_valid;
    wire [STG_IDX_W-1:0] arm_stg_idx = STG_IDX_W'({rtu_bus_w.arm_data.src, rtu_bus_w.arm_data.wid});

    // The ABI: a warp WAITs for its trace before arming another. Nothing enforces
    // it, and a violation would overwrite a live entry — corruption, not a hang.
    `RUNTIME_ASSERT(~(arm_fire && stg_armed[arm_stg_idx]),
        ("%t: *** %s: staging entry %0d armed a second trace without waiting for the first",
            $time, INSTANCE_ID, arm_stg_idx))

    `RUNTIME_ASSERT(~(is_ray && stg_full[req_stg_idx]),
        ("%t: *** %s: RAY beat for staging entry %0d whose ray is already complete",
            $time, INSTANCE_ID, req_stg_idx))

    // ── the staging RAMs ──────────────────────────────────────────────
    // The beat RAM is the single home of the per-lane ray words: the fill
    // engine reads them out to launch, and a candidate record's object-ray and
    // t_max words are read straight back at write-back. Neither reader can
    // collide with a write to the same row (the fill waits for stg_full; the
    // write-back's warp is parked in WAIT and sends no beats).
    wire                       stg_rd;
    wire [STG_IDX_W+RAY_IDX_W-1:0] stg_raddr;
    wire [NUM_LANES-1:0][31:0] stg_rdata;

    VX_dp_ram #(
        .DATAW    (NUM_LANES * 32),
        .SIZE     (NUM_STG * RTU_RAY_BEATS),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) stg_ray_ram (
        .clk   (clk),
        .reset (reset),
        .read  (stg_rd),
        .write (is_ray),
        .wren  (1'b1),
        .waddr ({req_stg_idx, stg_beat[req_stg_idx]}),
        .wdata (rtu_bus_w.req_data.data),
        .raddr (stg_raddr),
        .rdata (stg_rdata)
    );

    // warp-uniform scalars of a staged trace (read once, at the claim)
    localparam SCLW = NUM_LANES + TAG_WIDTH + 32 + `VX_CFG_MEM_ADDR_WIDTH + 16 + 16;
    wire [SCLW-1:0] stg_scl_rdata;
    wire [COHORT_W-1:0]  drain_cohort_pick;
    wire [STG_IDX_W-1:0] drain_stg_pick;
    VX_dp_ram #(
        .DATAW    (SCLW),
        .SIZE     (NUM_STG),
        .LUTRAM   (1),
        .OUT_REG  (0),
        .RDW_MODE ("W")
    ) stg_scl_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (arm_fire),
        .wren  (1'b1),
        .waddr (arm_stg_idx),
        .wdata ({rtu_bus_w.arm_data.mask,
                 rtu_bus_w.arm_data.tag,
                 rtu_bus_w.arm_data.payload_ptr,
                 rtu_bus_w.arm_data.scene_base,
                 rtu_bus_w.arm_data.flags,
                 rtu_bus_w.arm_data.cull_mask}),
        .raddr (drain_stg_pick),
        .rdata (stg_scl_rdata)
    );
    wire [NUM_LANES-1:0] scl_mask;
    wire [TAG_WIDTH-1:0] scl_tag;
    wire [31:0]          scl_payload;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] scl_scene;
    wire [15:0]          scl_flags, scl_cull;
    assign {scl_mask, scl_tag, scl_payload, scl_scene, scl_flags, scl_cull} = stg_scl_rdata;

    // ── the ctx free pool: an occupancy mask, not a FIFO ──────────────
    // A ctx is free when its mask bit is set; release (the scheduler's ctx_done)
    // sets it, a drain seed clears the one it allocates. Allocation is a
    // first-fit pick off the mask, so the pool costs NUM_CTX bits + a counter
    // instead of NUM_CTX id-registers + a wide read mux.
    reg [NUM_CTX-1:0]    ctx_free;
    reg [FLW-1:0]        fl_cnt;       // free count (spares a NUM_CTX-wide popcount)
    wire [CTX_TAG_W-1:0] alloc_ctx;
    wire                 alloc_valid;
    VX_priority_encoder #(
        .N (NUM_CTX)
    ) ctx_alloc_pe (
        .data_in    (ctx_free),
        `UNUSED_PIN (onehot_out),
        .index_out  (alloc_ctx),
        .valid_out  (alloc_valid)
    );

    // ── the drain sequencer ───────────────────────────────────────────
    // One staged trace at a time streams into the pool: the cohort latches the
    // warp-uniform descriptor, then each active lane's 8 beat-rows are read from
    // staging and assembled into its ray, seeded into a free-pool ctx — that
    // write IS the ray's launch, so early rays traverse while later lanes still
    // drain. No full-trace assembly buffer: the rows are re-read per lane, which
    // costs launch cycles but not a NUM_LANES × RTU_RAY_BEATS register file.
    wire [NUM_STG-1:0] stg_ready = stg_armed & stg_full;

    wire drain_stg_valid;
    VX_priority_encoder #(
        .N (NUM_STG)
    ) stg_picker (
        .data_in    (stg_ready),
        `UNUSED_PIN (onehot_out),
        .index_out  (drain_stg_pick),
        .valid_out  (drain_stg_valid)
    );

    wire [NUM_COHORTS-1:0] cohort_free;
    for (genvar s = 0; s < NUM_COHORTS; ++s) begin : g_cohort_free
        assign cohort_free[s] = (tstate[s] == T_IDLE);
    end
    wire drain_cohort_valid;
    VX_priority_encoder #(
        .N (NUM_COHORTS)
    ) cohort_picker (
        .data_in    (cohort_free),
        `UNUSED_PIN (onehot_out),
        .index_out  (drain_cohort_pick),
        .valid_out  (drain_cohort_valid)
    );

    localparam D_IDLE = 1'b0, D_RUN = 1'b1;
    reg                  dstate;
    reg [COHORT_W-1:0]   drain_cohort;
    reg [STG_IDX_W-1:0]  drain_stg;
    reg [NUM_LANES-1:0]  drain_mask;
    reg [LANE_W-1:0]     drain_lane;
    reg [RAY_IDX_W-1:0]  fill_beat;      // next staging row to read for this lane
    reg                  fill_cap_vld;   // a row read is in flight (OUT_REG=1)
    reg [RAY_IDX_W-1:0]  fill_cap_beat;  // its capture index
    reg [6:0][31:0]      fill_asm;       // beats 0..6; beat 7 rides the seed write

    // Admission reserves the cohort's whole ctx allocation up front, so a seed
    // can never run dry mid-cohort (the alloc_valid guard is belt-and-braces).
    wire [LANE_W:0] stg_popcnt = (LANE_W+1)'($countones(scl_mask));
    wire drain_launch = (dstate == D_IDLE) && drain_stg_valid && drain_cohort_valid
                     && (fl_cnt >= FLW'(stg_popcnt));

    // write-back has priority on the staging read port; the drain fills gaps
    wire wb_stg_req;
    wire [RAY_IDX_W-1:0] wb_stg_beat;
    wire [STG_IDX_W-1:0] wb_stg_idx;

    wire seed_fire = fill_cap_vld && (fill_cap_beat == RAY_IDX_W'(RTU_RAY_BEATS - 1))
                  && alloc_valid;
    wire fill_issue = (dstate == D_RUN) && !wb_stg_req && drain_mask[drain_lane]
                   && !(fill_cap_vld && (fill_cap_beat == RAY_IDX_W'(RTU_RAY_BEATS - 1)));
    wire fill_skip      = (dstate == D_RUN) && !drain_mask[drain_lane];
    wire fill_last_lane = (drain_lane == LANE_W'(NUM_LANES - 1));

    assign stg_rd    = wb_stg_req || fill_issue;
    assign stg_raddr = wb_stg_req ? {wb_stg_idx, wb_stg_beat} : {drain_stg, fill_beat};

    // a ray's launch: the assembled 256-bit ray, beat 7 straight off the RAM read
    wire [31:0] fill_word = stg_rdata[drain_lane];
    assign rw_valid  = seed_fire;
    assign rw_ctx    = alloc_ctx;
    assign rw_cohort = drain_cohort;
    assign rw_pos    = drain_lane;
    assign rw_data   = {fill_word, fill_asm};

    // cohort claim: latch the descriptor, arm the scheduler's record space
    assign ss_valid  = drain_launch;
    assign ss_cohort = drain_cohort_pick;
    assign ss_mask   = scl_mask;
    assign ss_flags  = scl_flags;
    assign ss_cull   = scl_cull;
    assign ss_scene  = scl_scene;

    // ── the CONTINUE: route it to the cohort whose candidate it answers ─
    wire [NUM_COHORTS-1:0] cont_hit_cohort;
    for (genvar s = 0; s < NUM_COHORTS; ++s) begin : g_cont_hit
        assign cont_hit_cohort[s] = ((tstate[s] == T_CBWAIT) || (tstate[s] == T_CBATTR))
                               && (req_stg[s] == req_stg_idx);
    end
    wire got_cont = is_cont && (| cont_hit_cohort);
    `RUNTIME_ASSERT(~(is_cont && ~got_cont),
        ("%t: *** %s: CONTINUE from staging entry %0d with no parked candidate",
            $time, INSTANCE_ID, req_stg_idx))
    `UNUSED_VAR (got_cont)

    wire [COHORT_W-1:0] cont_cohort;
    VX_priority_encoder #(
        .N (NUM_COHORTS)
    ) cont_cohort_pe (
        .data_in    (cont_hit_cohort),
        `UNUSED_PIN (onehot_out),
        .index_out  (cont_cohort),
        `UNUSED_PIN (valid_out)
    );
    wire cont_beat0 = is_cont && (tstate[cont_cohort] == T_CBWAIT);
    wire cont_beat1 = is_cont && (tstate[cont_cohort] == T_CBATTR);

    // the CONTINUE's data rows land straight in the window store
    assign ww_valid = cont_beat0 || cont_beat1;
    assign ww_addr  = {cont_cohort, cont_beat0 ? RTU_WS_WORD_BITS'(RTU_WS_CONT_T)
                                             : RTU_WS_WORD_BITS'(RTU_WS_CONT_ATTR)};
    assign ww_wren  = cont_beat0 ? {NUM_LANES{1'b1}}
                                 : sch_yld[32'(cont_cohort)*NUM_LANES +: NUM_LANES];
    assign ww_data  = rtu_bus_w.req_data.data;

    for (genvar s = 0; s < NUM_COHORTS; ++s) begin : g_resume
        assign sch_resume[s] = (tstate[s] == T_RESUME);
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane
            assign sch_action[s * NUM_LANES + i] = cont_action[s][i];
        end
    end

    // ── window-write grant (sticky for a whole record) ────────────────
    wire [NUM_COHORTS-1:0] want_win;
    for (genvar s = 0; s < NUM_COHORTS; ++s) begin : g_want_win
        assign want_win[s] = (tstate[s] == T_WRITE);
    end

    reg              win_lock;
    reg [COHORT_W-1:0] win_owner_r;

    wire [COHORT_W-1:0] win_grant;
    wire              win_grant_valid;
    VX_priority_encoder #(
        .N (NUM_COHORTS)
    ) win_picker (
        .data_in    (want_win),
        `UNUSED_PIN (onehot_out),
        .index_out  (win_grant),
        .valid_out  (win_grant_valid)
    );

    // ws = the cohort whose record is being written back
    wire [COHORT_W-1:0] ws = win_owner_r;

    // ── the record write-back walk ────────────────────────────────────
    // One word per step: read the field-row(s), then present the beat. The
    // sources per word index (matching the window-slot record layout):
    //   0..6   hit attrs — terminal: hit row (miss lanes: t_max / 0);
    //          candidate: yld row for candidate lanes, hit row for the rest
    //   7..12  object ray (candidate only) — the staging RAM's beat rows
    //   13,14  cb_type (flag flops), sbt (window row)
    //   15     cb_handle (0)
    //   then   payload (descriptor), hitAttribute (window row), status LAST
    localparam [2:0] WB_IDLE = 3'd0,
                     WB_RD1  = 3'd1,
                     WB_CAP1 = 3'd2,
                     WB_RD2  = 3'd3,
                     WB_CAP2 = 3'd4,
                     WB_SEND = 3'd5;
    reg [2:0]               wb_state;
    reg [RTU_IDX_BITS-1:0]  wr_idx;
    reg [ROW_BITS-1:0]      wb_d1, wb_d2;
    reg [NUM_LANES-1:0][31:0] wb_ds;

    wire [RTU_IDX_BITS-1:0] n_attrs = is_cand[ws] ? RTU_IDX_BITS'(RTU_RES_CAND)
                                                  : RTU_IDX_BITS'(RTU_RES_HIT);
    wire wr_payload = (wr_idx == n_attrs);
    wire wr_attr    = (wr_idx == (n_attrs + RTU_IDX_BITS'(1)));
    wire wr_status  = (wr_idx == (n_attrs + RTU_IDX_BITS'(2)));
    wire wr_hitspan = (wr_idx < RTU_IDX_BITS'(RTU_RES_HIT));
    wire wr_objray  = is_cand[ws] && !wr_hitspan && (wr_idx < RTU_IDX_BITS'(13));
    wire wr_cbtype  = is_cand[ws] && (wr_idx == RTU_IDX_BITS'(13));
    wire wr_sbt     = is_cand[ws] && (wr_idx == RTU_IDX_BITS'(14));

    // reads this word needs
    wire wb_need_w1 = (wb_state == WB_RD1)
                   && (wr_hitspan || wr_sbt || wr_attr) && !wr_status && !wr_payload;
    wire wb_need_w2 = is_cand[ws] && wr_hitspan;                    // the hit row
    wire wb_need_st = (wr_hitspan && (wr_idx == '0)) || wr_objray;  // t_max / object ray

    assign wr_valid = ((wb_state == WB_RD1) && wb_need_w1)
                   || ((wb_state == WB_RD2));
    assign wr_addr  = (wb_state == WB_RD2)
                    ? {ws, RTU_WS_WORD_BITS'(RTU_WS_HIT_BASE) + RTU_WS_WORD_BITS'(wr_idx)}
                    : wr_hitspan
                        ? {ws, (is_cand[ws] ? RTU_WS_WORD_BITS'(RTU_WS_YLD_BASE)
                                            : RTU_WS_WORD_BITS'(RTU_WS_HIT_BASE))
                               + RTU_WS_WORD_BITS'(wr_idx)}
                    : wr_sbt  ? {ws, RTU_WS_WORD_BITS'(RTU_WS_YLD_SBT)}
                              : {ws, RTU_WS_WORD_BITS'(RTU_WS_RES_ATTR)};

    assign wb_stg_req  = (wb_state == WB_RD1) && wb_need_st;
    assign wb_stg_idx  = req_stg[ws];
    assign wb_stg_beat = wr_objray ? RAY_IDX_W'(wr_idx - RTU_IDX_BITS'(RTU_RES_HIT))
                                   : RAY_IDX_W'(RTU_RAY_BEATS - 1);   // t_max

    // per-lane word assembly (from the captured rows + the hot flags)
    wire [CTX_TAG_W-1:0] wctx [NUM_LANES];
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_wctx
        assign wctx[i] = CTX_TAG_W'((32'(ws) * NUM_LANES) + i);
    end

    reg [NUM_LANES-1:0][31:0] status_word;
    always @(*) begin
        for (integer i = 0; i < NUM_LANES; ++i) begin
            if (!is_cand[ws]) begin
                status_word[i] = sch_hit[wctx[i]] ? 32'(`VX_RT_STS_DONE_HIT)
                                                  : 32'(`VX_RT_STS_DONE_MISS);
            end else if (sch_yld[wctx[i]]) begin
                status_word[i] = (sch_cbtype[wctx[i]] == RTU_CB_TYPE_BITS'(`VX_RT_CB_TYPE_PROC))
                               ? 32'(`VX_RT_STS_YIELD_PROC)
                               : 32'(`VX_RT_STS_YIELD_ANYHIT);
            end else begin
                status_word[i] = 32'(`VX_RT_STS_PENDING);
            end
        end
    end

    reg [NUM_LANES-1:0][31:0] win_word;
    always @(*) begin
        for (integer i = 0; i < NUM_LANES; ++i) begin
            if (wr_status) begin
                win_word[i] = status_word[i];
            end else if (wr_payload) begin
                win_word[i] = req_payload[ws];
            end else if (wr_attr) begin
                win_word[i] = sch_attrv[wctx[i]] ? wb_d1[i*32 +: 32] : 32'd0;
            end else if (wr_hitspan) begin
                if (is_cand[ws] && sch_yld[wctx[i]]) begin
                    win_word[i] = wb_d1[i*32 +: 32];              // the candidate row
                end else if (sch_hit[wctx[i]]) begin
                    win_word[i] = is_cand[ws] ? wb_d2[i*32 +: 32] // the hit row
                                              : wb_d1[i*32 +: 32];
                end else begin
                    // no committed hit: t reads back the ray's own t_max
                    win_word[i] = (wr_idx == '0) ? wb_ds[i] : 32'd0;
                end
            end else if (wr_objray) begin
                win_word[i] = wb_ds[i];
            end else if (wr_cbtype) begin
                win_word[i] = {{(32-RTU_CB_TYPE_BITS){1'b0}}, sch_cbtype[wctx[i]]};
            end else if (wr_sbt) begin
                win_word[i] = wb_d1[i*32 +: 32] & 32'hff;
            end else begin
                win_word[i] = 32'd0;   // cb_handle
            end
        end
    end

    wire win_send = (wb_state == WB_SEND);
    assign rtu_bus_w.win_valid        = win_send;
    assign rtu_bus_w.win_data.is_cand = is_cand[ws];
    assign rtu_bus_w.win_data.wid     = req_wid[ws];
    assign rtu_bus_w.win_data.tag     = req_tag[ws];
    assign rtu_bus_w.win_data.data    = win_word;
    // A candidate's attributes only exist for its yielding lanes; its status,
    // and every whole-trace word, cover each active lane.
    wire wr_uniform = wr_status || wr_payload || wr_attr;
    wire [NUM_LANES-1:0] wr_cb_mask = sch_yld[32'(ws)*NUM_LANES +: NUM_LANES];
    assign rtu_bus_w.win_data.mask    = (is_cand[ws] && !wr_uniform) ? wr_cb_mask : req_mask[ws];
    assign rtu_bus_w.win_data.slot    =
          wr_status  ? RTU_SLOT_BITS'(RTU_STATUS_SLOT)
        : wr_payload ? RTU_SLOT_BITS'(RTU_PAYLOAD_SLOT)
        : wr_attr    ? RTU_SLOT_BITS'(RTU_ATTR_SLOT)
                     : RTU_SLOT_BITS'(RTU_RES_BASE) + RTU_SLOT_BITS'(wr_idx);

    wire win_fire = rtu_bus_w.win_valid && rtu_bus_w.win_ready;

    // ── control ───────────────────────────────────────────────────────
    integer i;
    always_ff @(posedge clk) begin
        if (reset) begin
            tstate       <= '0;   // T_IDLE
            stg_armed    <= '0;
            stg_full     <= '0;
            stg_beat     <= '0;
            dstate       <= D_IDLE;
            fill_cap_vld <= 1'b0;
            ctx_free     <= '1;
            fl_cnt       <= FLW'(NUM_CTX);
            win_lock     <= 1'b0;
            wb_state     <= WB_IDLE;
        end else begin
            // ── an arm claims its warp's staging entry ────────────────
            if (arm_fire) begin
                // NOT a beat-counter reset: beats may have overtaken this arm.
                stg_armed[arm_stg_idx] <= 1'b1;
            end

            // ── a RAY beat lands in its owner's staging entry ─────────
            if (is_ray) begin
                stg_beat[req_stg_idx] <= stg_beat[req_stg_idx] + RAY_IDX_W'(1);
                if (stg_beat[req_stg_idx] == RAY_IDX_W'(RTU_RAY_BEATS - 1)) begin
                    stg_full[req_stg_idx] <= 1'b1;
                    stg_beat[req_stg_idx] <= '0;
                end
            end

            // ── the ctx free pool ─────────────────────────────────────
            // release sets a ctx free, a seed clears the one it allocated
            if (ctx_done_valid) ctx_free[ctx_done_id] <= 1'b1;
            if (seed_fire)      ctx_free[alloc_ctx]   <= 1'b0;
            case ({ctx_done_valid, seed_fire})
                2'b10:   fl_cnt <= fl_cnt + FLW'(1);
                2'b01:   fl_cnt <= fl_cnt - FLW'(1);
                default: ;   // 2'b00 idle, 2'b11 set+clear cancels
            endcase

            // ── the drain sequencer ───────────────────────────────────
            if (drain_launch) begin
                // Claim the entry NOW: stg_ready drops this cycle so the next
                // launch cannot pick the same ray. stg_armed stays up until the
                // terminal record: it guards the entry for the whole trace.
                stg_full[drain_stg_pick]  <= 1'b0;
                drain_stg                 <= drain_stg_pick;
                drain_cohort              <= drain_cohort_pick;
                drain_mask                <= scl_mask;
                drain_lane                <= '0;
                fill_beat                 <= '0;
                fill_cap_vld              <= 1'b0;
                tstate[drain_cohort_pick] <= T_FILL;
                req_mask[drain_cohort_pick]    <= scl_mask;
                req_tag[drain_cohort_pick]     <= scl_tag;
                req_stg[drain_cohort_pick]     <= drain_stg_pick;
                req_wid[drain_cohort_pick]     <= drain_stg_pick[NW_WIDTH-1:0];  // stg = {src, wid}
                req_payload[drain_cohort_pick] <= scl_payload;
                dstate <= D_RUN;
            end else if (dstate == D_RUN) begin
                fill_cap_vld <= 1'b0;
                if (fill_issue) begin
                    fill_beat     <= (fill_beat == RAY_IDX_W'(RTU_RAY_BEATS - 1))
                                   ? fill_beat : (fill_beat + RAY_IDX_W'(1));
                    fill_cap_vld  <= 1'b1;
                    fill_cap_beat <= fill_beat;
                end
                if (fill_cap_vld && (fill_cap_beat != RAY_IDX_W'(RTU_RAY_BEATS - 1))) begin
                    fill_asm[fill_cap_beat[2:0]] <= fill_word;
                end
                if (fill_skip || (seed_fire && !fill_last_lane)) begin
                    drain_lane <= drain_lane + LANE_W'(1);
                    fill_beat  <= '0;
                end
                if ((fill_skip && fill_last_lane) || (seed_fire && fill_last_lane)) begin
                    tstate[drain_cohort] <= T_BUSY;
                    dstate <= D_IDLE;
                end
            end

            // ── the CONTINUE beats ────────────────────────────────────
            if (cont_beat0) begin
                for (i = 0; i < NUM_LANES; ++i) begin
                    cont_action[cont_cohort][i] <= rtu_bus_w.req_data.cb_action[i];
                end
                tstate[cont_cohort] <= T_CBATTR;
            end else if (cont_beat1) begin
                tstate[cont_cohort] <= T_RESUME;
            end

            // ── the per-cohort record FSMs ────────────────────────────
            for (integer s = 0; s < NUM_COHORTS; s = s + 1) begin
                case (tstate[s])
                T_BUSY: begin
                    // yield takes priority: the walk paused with a candidate
                    if (sch_yield[s]) begin
                        is_cand[s] <= 1'b1;
                        tstate[s]  <= T_WRITE;
                    end else if (sch_done[s]) begin
                        is_cand[s] <= 1'b0;
                        tstate[s]  <= T_WRITE;
                    end
                end
                T_RESUME: begin
                    tstate[s] <= T_RWAIT;
                end
                T_RWAIT: begin
                    if (sch_done[s]) begin
                        is_cand[s] <= 1'b0;
                        tstate[s]  <= T_WRITE;
                    end
                end
                default:;
                endcase
            end

            // ── the write-back walk ───────────────────────────────────
            case (wb_state)
            WB_IDLE: begin
                if (!win_lock && win_grant_valid) begin
                    win_lock    <= 1'b1;
                    win_owner_r <= win_grant;
                    wr_idx      <= '0;
                    wb_state    <= WB_RD1;
                end
            end
            WB_RD1: begin
                // the reads (if any) were issued this cycle
                if (wb_need_w1 || wb_need_st) begin
                    wb_state <= WB_CAP1;
                end else begin
                    wb_state <= WB_SEND;
                end
            end
            WB_CAP1: begin
                wb_d1 <= wr_data;
                wb_ds <= stg_rdata;
                if (wb_need_w2) begin
                    wb_state <= WB_RD2;
                end else begin
                    wb_state <= WB_SEND;
                end
            end
            WB_RD2: begin
                wb_state <= WB_CAP2;
            end
            WB_CAP2: begin
                wb_d2    <= wr_data;
                wb_state <= WB_SEND;
            end
            WB_SEND: begin
                if (win_fire) begin
                    if (wr_status) begin
                        // the record is whole; the channel is free
                        win_lock <= 1'b0;
                        wb_state <= WB_IDLE;
                        if (is_cand[ws]) begin
                            tstate[ws] <= T_CBWAIT;
                        end else begin
                            // a TERMINAL record ends the trace: release the
                            // warp's staging entry so it may arm again
                            stg_armed[req_stg[ws]] <= 1'b0;
                            tstate[ws] <= T_IDLE;
                        end
                    end else begin
                        wr_idx   <= wr_idx + RTU_IDX_BITS'(1);
                        wb_state <= WB_RD1;
                    end
                end
            end
            default: begin
                wb_state <= WB_IDLE;
            end
            endcase
        end
    end

// ── RTU occupancy counters ────────────────────────────────────────────────
`ifdef DBG_RTU_OCC
    longint unsigned occ_total, occ_busy, occ_write, occ_cb, occ_fill;
    longint unsigned occ_idle_ray_waiting, occ_idle_starved, occ_traces;
    // Per-cycle cohort concurrency: how many cohorts are simultaneously non-IDLE.
    // This is the direct answer to "can all NUM_COHORTS admission batches be in
    // flight at once" -- occ_hist[k] counts cycles with exactly k cohorts occupied
    // and occ_max is the peak. A cohort is live whenever tstate[s] != T_IDLE,
    // which spans FILL/BUSY/WRITE and every callback/resume wait state.
    longint unsigned occ_hist [0:NUM_COHORTS];
    longint unsigned occ_conc_sum, occ_max;
    // pool-side instrumentation for the decoupled (cohort + free-list) RTU:
    // avg free ctx depth, drain-sequencer active cycles, and seed stalls (an
    // active lane blocked on an empty pool -- must stay 0 under admission reserve).
    // occ_seed counts ray launches, occ_release counts ctx_done releases: they
    // must be equal at the end (every ray executes and retires exactly once);
    // release > seed would mean a ctx is spuriously re-woken after exec_done.
    longint unsigned occ_fl_sum, occ_drain, occ_seed_stall, occ_seed, occ_release;
    always @(posedge clk) begin
        if (reset) begin
            occ_total <= 0; occ_busy <= 0; occ_write <= 0; occ_cb <= 0;
            occ_fill <= 0; occ_idle_ray_waiting <= 0; occ_idle_starved <= 0;
            occ_traces <= 0; occ_conc_sum <= 0; occ_max <= 0;
            occ_fl_sum <= 0; occ_drain <= 0; occ_seed_stall <= 0;
            occ_seed <= 0; occ_release <= 0;
            for (integer k = 0; k <= NUM_COHORTS; k = k + 1) occ_hist[k] <= 0;
        end else begin
            // One non-blocking assignment per counter per cycle: `<=` inside the
            // cohort loop collapses to the last cohort's write, which would
            // silently degrade every per-state counter to "cycles with >=1 cohort
            // in state".
            longint unsigned conc, n_busy, n_write, n_cb, n_fill, n_iwait, n_istarve;
            conc = 0; n_busy = 0; n_write = 0; n_cb = 0; n_fill = 0;
            n_iwait = 0; n_istarve = 0;
            for (integer s = 0; s < NUM_COHORTS; s = s + 1) begin
                case (tstate[s])
                T_BUSY:   n_busy  = n_busy + 1;
                T_WRITE:  n_write = n_write + 1;
                T_FILL:   n_fill  = n_fill + 1;
                T_CBWAIT, T_CBATTR, T_RESUME, T_RWAIT: n_cb = n_cb + 1;
                T_IDLE: begin
                    if ((| stg_ready) || (dstate != D_IDLE)) n_iwait   = n_iwait + 1;
                    else                                     n_istarve = n_istarve + 1;
                end
                default:;
                endcase
                if (tstate[s] != T_IDLE) conc = conc + 1;
            end
            occ_total            <= occ_total + 1;
            occ_busy             <= occ_busy + n_busy;
            occ_write            <= occ_write + n_write;
            occ_cb               <= occ_cb + n_cb;
            occ_fill             <= occ_fill + n_fill;
            occ_idle_ray_waiting <= occ_idle_ray_waiting + n_iwait;
            occ_idle_starved     <= occ_idle_starved + n_istarve;
            if (ss_valid) begin
                occ_traces <= occ_traces + 1;
            end
            occ_conc_sum   <= occ_conc_sum + conc;
            occ_hist[conc] <= occ_hist[conc] + 1;
            if (conc > occ_max) occ_max <= conc;
            occ_fl_sum     <= occ_fl_sum + fl_cnt;
            if (dstate != D_IDLE) occ_drain <= occ_drain + 1;
            // a seed blocked on an empty pool: must stay 0 under admission reserve
            if ((dstate == D_RUN) && fill_cap_vld
             && (fill_cap_beat == RAY_IDX_W'(RTU_RAY_BEATS - 1))
             && drain_mask[drain_lane] && !alloc_valid)
                occ_seed_stall <= occ_seed_stall + 1;
            if (rw_valid)       occ_seed    <= occ_seed + 1;
            if (ctx_done_valid) occ_release <= occ_release + 1;
        end
    end
    always @(posedge clk) begin
        if (!reset && (occ_total % 5000 == 4999)) begin
            $display("RTU-OCC @%0d: traces=%0d busy=%0d write=%0d cb=%0d fill=%0d idle_raywait=%0d idle_starved=%0d | CONC max=%0d avg=%0d.%02d hist0..%0d=%0d/%0d/%0d/%0d/%0d | POOL avg_free=%0d.%02d/%0d drain_cyc=%0d seed_stall=%0d seed=%0d release=%0d",
                occ_total, occ_traces, occ_busy, occ_write, occ_cb, occ_fill,
                occ_idle_ray_waiting, occ_idle_starved,
                occ_max, occ_conc_sum / occ_total, (occ_conc_sum * 100 / occ_total) % 100,
                NUM_COHORTS, occ_hist[0], occ_hist[1], occ_hist[2], occ_hist[3], occ_hist[4],
                occ_fl_sum / occ_total, (occ_fl_sum * 100 / occ_total) % 100, NUM_CTX,
                occ_drain, occ_seed_stall, occ_seed, occ_release);
        end
    end
`endif

endmodule
