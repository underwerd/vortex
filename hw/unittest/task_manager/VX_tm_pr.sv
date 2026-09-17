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

// VX_tm_pr — TaskManager Phase-Router sub-block: publishPhase + TRACE admission
// + RT completion, and the commit pipeline that drains both into the PQM queues.
//
// Three ingest paths converge on one commit pipeline:
//   publishPhase  a 32-lane request streams 1 entry/cycle into the Publish FIFO;
//   RT completion accepted into the Completion FIFO (this cycle is the RT-side
//                 RaySlot release point: owner RT_SLOT->R_COMMIT, free-est +1);
//   admission     a TRACE task is sent to the RT Unit; on reject it re-injects
//                 itself into the Publish FIFO as TRACE_READY (no external port).
// The commit pipeline pulls one entry/cycle (completion has priority), writes
// TaskState for completion entries (WB), then PUBLISHes: enqueue to the target
// queue, or — for next_phase RELEASE — retire the task (owner->FREE, the id
// returns to the Free-ID FIFO, live_count drops).
//
// The Owner table and live_count live in the top; this block only REQUESTS the
// migrations (pr_adm_mig / pr_cmpl_mig / pr_rel) and they name task_ids disjoint
// from every other writer (assertion A1), so the top folds them into its single
// arbitrated write port.

`include "VX_define.vh"

module VX_tm_pr
    import VX_tm_pkg::*;
#(
    parameter `STRING INSTANCE_ID  = "",
    parameter NUM_TASKS        = 256,
    parameter NUM_BUCKETS      = 64,
    parameter CMPL_FIFO_DEPTH  = 16,
    parameter PUB_FIFO_DEPTH   = 64,
    parameter COMPLETION_W     = 134,
    parameter CFG_RT_SLOTS     = 128,
    parameter TASK_STATE_BYTES = 128
) (
    input wire clk,
    input wire reset,

    // ── publishPhase face (worker / init -> PR) ───────────────────────
    input  wire         pub_req_valid,
    output wire         pub_req_ready,
    input  wire [31:0]  pub_req_mask,
    input  wire [255:0] pub_req_task_id,
    input  wire [191:0] pub_req_shader_id,
    input  wire [63:0]  pub_req_next_phase,
    input  wire [1:0]   pub_req_src,

    // ── TRACE admission face (TRACE worker -> PR) ─────────────────────
    input  wire         adm_req_valid,
    output wire         adm_req_ready,
    input  wire [7:0]   adm_req_task_id,
    input  wire [255:0] adm_req_ray,
    input  wire [7:0]   adm_req_bounce,
    input  wire [31:0]  adm_req_cont,

    // ── RT face (<-> RT Unit) ─────────────────────────────────────────
    output reg          rt_adm_desc_valid,
    output reg  [303:0] rt_adm_desc,        // {task_id 8, ray 256, bounce 8, cont 32}
    input  wire         rt_adm_accept_valid,
    input  wire [14:0]  rt_adm_accept,      // {task_id 8, slot_id 7}
    input  wire         rt_adm_reject_valid,
    input  wire [7:0]   rt_adm_reject,      // {task_id}
    input  wire         rt_completion_valid,
    output wire         rt_completion_ready,
    input  wire [COMPLETION_W-1:0] rt_completion,

    // ── to the PQM: queue enqueue + the TRACE free-slot gate ──────────
    output wire         pqm_enq_valid,
    output wire [1:0]   pqm_enq_phase,
    output wire [`CLOG2(NUM_TASKS)-1:0]   pqm_enq_task_id,
    output wire [`CLOG2(NUM_BUCKETS)-1:0] pqm_enq_shader_id,
    input  wire         pqm_enq_ready,      // target queue/bucket can take it
    output wire [`CLOG2(CFG_RT_SLOTS+1)-1:0] rt_free_est,

    // ── to the top: owner migrations + TaskState write (disjoint ids, A1) ──
    output wire         pr_adm_mig_valid,   // TRACE_WK -> RT_SLOT (admission accept)
    output wire [`CLOG2(NUM_TASKS)-1:0] pr_adm_mig_id,
    output wire         pr_cmpl_mig_valid,  // RT_SLOT -> R_COMMIT (completion accept)
    output wire [`CLOG2(NUM_TASKS)-1:0] pr_cmpl_mig_id,
    output wire         pr_rel_valid,       // -> FREE (release): top frees id, live--
    output wire [`CLOG2(NUM_TASKS)-1:0] pr_rel_id,
    output wire         pr_ts_wr_valid,
    output wire [`CLOG2(NUM_TASKS)+`CLOG2(TASK_STATE_BYTES/4)-1:0] pr_ts_wr_addr,
    output wire [31:0]  pr_ts_wr_data,

    // ── to the top: kernel_done term + status ─────────────────────────
    output wire pr_fifos_empty,
    output wire [`CLOG2(CMPL_FIFO_DEPTH+1)-1:0] sts_cmpl_fifo_lvl,
    output wire [`CLOG2(PUB_FIFO_DEPTH+1)-1:0]  sts_pub_fifo_lvl
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam TASK_ID_W  = `CLOG2(NUM_TASKS);            // 8
    localparam BUCKET_W   = `CLOG2(NUM_BUCKETS);          // 6
    localparam CMPLLVL_W  = `CLOG2(CMPL_FIFO_DEPTH + 1);  // 5
    localparam PUBLVL_W   = `CLOG2(PUB_FIFO_DEPTH + 1);   // 7
    localparam RTEST_W    = `CLOG2(CFG_RT_SLOTS + 1);     // 8
    localparam TS_WORDS   = TASK_STATE_BYTES / 4;         // 32
    localparam TS_WORD_W  = `CLOG2(TS_WORDS);             // 5
    localparam TS_ADDR_W  = TASK_ID_W + TS_WORD_W;        // 13
    localparam PUBP_W     = `CLOG2(PUB_FIFO_DEPTH);       // 6
    localparam CMPLP_W    = `CLOG2(CMPL_FIFO_DEPTH);      // 4
    localparam PKT_MAX    = 32;

    // Publish FIFO entry: {src 2, phase 2, shader_id 6, task_id 8} = 18 b
    localparam PUB_ENT_W  = 2 + 2 + BUCKET_W + TASK_ID_W;

    // completion field offsets in the 134-b payload (hw_spec §2.6)
    localparam C_TASK_LO = 0,   C_TASK_HI = 7;     // task_id 8
    localparam C_HIT     = 13;                      // hit 1
    localparam C_T_LO    = 14,  C_T_HI    = 45;     // t 32
    localparam C_SH_LO   = 110, C_SH_HI   = 115;    // shader_id 6

    // ══════════════ Publish FIFO ══════════════
    reg [PUB_ENT_W-1:0] pub_fifo [0:PUB_FIFO_DEPTH-1];
    reg [PUBP_W-1:0]    pub_head, pub_tail;
    reg [PUBLVL_W-1:0]  pub_count;
    wire pub_avail = (pub_count != '0);
    wire [PUBLVL_W-1:0] pub_free = PUBLVL_W'(PUB_FIFO_DEPTH) - pub_count;
    wire [PUB_ENT_W-1:0] pub_head_ent = pub_fifo[pub_head];

    // ══════════════ Completion FIFO ══════════════
    reg [COMPLETION_W-1:0] cmpl_fifo [0:CMPL_FIFO_DEPTH-1];
    reg [CMPLP_W-1:0]      cmpl_head, cmpl_tail;
    reg [CMPLLVL_W-1:0]    cmpl_count;
    wire cmpl_avail = (cmpl_count != '0);
    assign rt_completion_ready = (cmpl_count < CMPLLVL_W'(CMPL_FIFO_DEPTH));
    wire cmpl_accept = rt_completion_valid && rt_completion_ready;
    wire [COMPLETION_W-1:0] cmpl_head_ent = cmpl_fifo[cmpl_head];
    wire [TASK_ID_W-1:0] cmpl_head_task = cmpl_head_ent[C_TASK_HI:C_TASK_LO];
    wire                 cmpl_head_hit  = cmpl_head_ent[C_HIT];
    wire [BUCKET_W-1:0]  cmpl_head_sh   = cmpl_head_ent[C_SH_HI:C_SH_LO];
    wire [31:0]          cmpl_head_t    = cmpl_head_ent[C_T_HI:C_T_LO];
    wire [TASK_ID_W-1:0] cmpl_acc_task  = rt_completion[C_TASK_HI:C_TASK_LO];

    // ══════════════ publishPhase ingest sequencer ══════════════
    localparam [1:0] P_IDLE = 2'd0, P_SCAN = 2'd1;
    reg [1:0]     pub_state;
    reg [5:0]     pub_lane;
    reg [31:0]    pub_mask_r;
    reg [255:0]   pub_task_r;
    reg [191:0]   pub_sh_r;
    reg [63:0]    pub_ph_r;
    reg [1:0]     pub_src_r;
    assign pub_req_ready = (pub_state == P_IDLE) && (pub_free >= PUBLVL_W'(PKT_MAX));
    wire pub_req_fire = pub_req_valid && pub_req_ready;

    // ══════════════ admission FSM + free-slot estimate ══════════════
    localparam [1:0] A_IDLE = 2'd0, A_SEND = 2'd1, A_WAIT = 2'd2, A_REINJ = 2'd3;
    reg [1:0]           adm_state;
    reg [TASK_ID_W-1:0] adm_task;
    reg [255:0]         adm_ray;
    reg [7:0]           adm_bounce;
    reg [31:0]          adm_cont;
    reg [RTEST_W-1:0]   rt_free_est_r;
    assign adm_req_ready = (adm_state == A_IDLE);

    wire adm_accept = (adm_state == A_WAIT) && rt_adm_accept_valid;
    wire adm_reject = (adm_state == A_WAIT) && rt_adm_reject_valid && !rt_adm_accept_valid;
    wire [TASK_ID_W-1:0] adm_accept_task = rt_adm_accept[14:7];

    // re-inject a rejected task as TRACE_READY: one Publish FIFO entry, pushed
    // only when the ingest sequencer is idle and there is room (single write port)
    wire rej_push = (adm_state == A_REINJ) && (pub_state == P_IDLE)
                 && (pub_free != '0);

    // Publish FIFO write source: the ingest scan, or a reject re-injection
    wire ing_push = (pub_state == P_SCAN) && pub_mask_r[pub_lane[4:0]];
    wire pub_push = ing_push || rej_push;
    wire [PUB_ENT_W-1:0] pub_push_ent = ing_push
        ? {pub_src_r,
           pub_ph_r[32'(pub_lane[4:0])*2 +: 2],
           pub_sh_r[32'(pub_lane[4:0])*6 +: BUCKET_W],
           pub_task_r[32'(pub_lane[4:0])*8 +: TASK_ID_W]}
        : {TM_SRC_WORKER, TM_PH_TRACE_READY, {BUCKET_W{1'b0}}, adm_task};

    // ══════════════ commit pipeline (2 registers: WB -> PUBLISH) ══════════════
    // entry: {is_cmpl 1, task_id 8, phase 2, shader_id 6, t 32}
    localparam CE_W = 1 + TASK_ID_W + 2 + BUCKET_W + 32;
    function automatic [CE_W-1:0] mk_ce(input is_cmpl, input [TASK_ID_W-1:0] tid,
                                        input [1:0] phase, input [BUCKET_W-1:0] sh,
                                        input [31:0] t);
        mk_ce = {is_cmpl, tid, phase, sh, t};
    endfunction

    reg               wb_v,  pub_v;
    reg [CE_W-1:0]    wb_e,  pub_e;
    // entry layout (MSB->LSB): {is_cmpl 1, task_id 8, phase 2, shader_id 6, t 32}
    wire                 pub_e_is_cmpl = pub_e[CE_W-1];
    wire [TASK_ID_W-1:0] pub_e_task  = pub_e[CE_W-2 -: TASK_ID_W];
    wire [1:0]           pub_e_phase = pub_e[32+BUCKET_W +: 2];
    wire [BUCKET_W-1:0]  pub_e_sh    = pub_e[32 +: BUCKET_W];
    wire                 pub_is_rel  = pub_e_is_cmpl ? 1'b0 : (pub_e_phase == TM_PH_RELEASE);

    // commit input: completion has priority over publishPhase
    wire pull_cmpl = cmpl_avail;
    wire pull_pub  = !cmpl_avail && pub_avail;
    wire pull_any  = pull_cmpl || pull_pub;

    // PUBLISH stalls only when it must enqueue and the target queue is full
    wire pub_stall = pub_v && !pub_is_rel && !pqm_enq_ready;
    wire advance   = !pub_stall;

    // PUBLISH-side effects
    assign pqm_enq_valid     = pub_v && !pub_is_rel && pqm_enq_ready;
    assign pqm_enq_phase     = pub_e_phase;
    assign pqm_enq_task_id   = pub_e_task;
    assign pqm_enq_shader_id = pub_e_sh;
    assign pr_rel_valid      = pub_v && pub_is_rel;
    assign pr_rel_id         = pub_e_task;

    // WB-side effect: a completion writes its trace overlay to TaskState. One
    // representative word here; the full 128-B ABI field write is integration
    // detail (hw_spec D4). Issues once, as the entry leaves WB.
    wire wb_is_cmpl = wb_e[CE_W-1];
    wire [TASK_ID_W-1:0] wb_task = wb_e[CE_W-2 -: TASK_ID_W];
    wire [31:0]          wb_t    = wb_e[31:0];
    assign pr_ts_wr_valid = wb_v && wb_is_cmpl && advance;
    assign pr_ts_wr_addr  = {wb_task, {TS_WORD_W{1'b0}}};
    assign pr_ts_wr_data  = wb_t;

    // owner migrations
    assign pr_adm_mig_valid  = adm_accept;
    assign pr_adm_mig_id     = adm_accept_task;
    assign pr_cmpl_mig_valid = cmpl_accept;
    assign pr_cmpl_mig_id    = cmpl_acc_task;

    assign rt_free_est = rt_free_est_r;
    assign pr_fifos_empty = (pub_count == '0) && (cmpl_count == '0) && !wb_v && !pub_v;
    assign sts_cmpl_fifo_lvl = cmpl_count;
    assign sts_pub_fifo_lvl  = pub_count;

    // ══════════════ sequential ══════════════
    always_ff @(posedge clk) begin
        if (reset) begin
            pub_head    <= '0;
            pub_tail    <= '0;
            pub_count   <= '0;
            cmpl_head   <= '0;
            cmpl_tail   <= '0;
            cmpl_count  <= '0;
            pub_state   <= P_IDLE;
            pub_lane    <= '0;
            adm_state   <= A_IDLE;
            rt_adm_desc_valid <= 1'b0;
            rt_free_est_r <= RTEST_W'(CFG_RT_SLOTS);
            wb_v        <= 1'b0;
            pub_v       <= 1'b0;
        end else begin
            rt_adm_desc_valid <= 1'b0;

            // ── completion accept: push + owner RT_SLOT->R_COMMIT + est+1 ──
            if (cmpl_accept) begin
                cmpl_fifo[cmpl_tail] <= rt_completion;
                cmpl_tail <= cmpl_tail + CMPLP_W'(1);
            end

            // ── Publish FIFO write (ingest scan or reject re-inject) ──
            if (pub_push) begin
                pub_fifo[pub_tail] <= pub_push_ent;
                pub_tail <= pub_tail + PUBP_W'(1);
            end

            // ── publishPhase ingest sequencer ──
            case (pub_state)
            P_IDLE: begin
                pub_lane <= '0;
                if (pub_req_fire) begin
                    pub_mask_r <= pub_req_mask;
                    pub_task_r <= pub_req_task_id;
                    pub_sh_r   <= pub_req_shader_id;
                    pub_ph_r   <= pub_req_next_phase;
                    pub_src_r  <= pub_req_src;
                    pub_state  <= P_SCAN;
                end
            end
            P_SCAN: begin
                if (pub_lane == 6'(PKT_MAX - 1)) pub_state <= P_IDLE;
                else                             pub_lane  <= pub_lane + 6'd1;
            end
            default: pub_state <= P_IDLE;
            endcase

            // ── admission FSM ──
            case (adm_state)
            A_IDLE: begin
                if (adm_req_valid) begin
                    adm_task   <= adm_req_task_id;
                    adm_ray    <= adm_req_ray;
                    adm_bounce <= adm_req_bounce;
                    adm_cont   <= adm_req_cont;
                    adm_state  <= A_SEND;
                end
            end
            A_SEND: begin
                rt_adm_desc_valid <= 1'b1;
                rt_adm_desc <= {adm_task, adm_ray, adm_bounce, adm_cont};
                adm_state   <= A_WAIT;
            end
            A_WAIT: begin
                if (adm_accept)      adm_state <= A_IDLE;
                else if (adm_reject) adm_state <= A_REINJ;
            end
            A_REINJ: begin
                if (rej_push) adm_state <= A_IDLE;
            end
            default: adm_state <= A_IDLE;
            endcase

            // ── free-slot estimate: -1 on admission accept, +1 on completion ──
            case ({cmpl_accept, adm_accept})
                2'b10: if (rt_free_est_r != RTEST_W'(CFG_RT_SLOTS)) rt_free_est_r <= rt_free_est_r + RTEST_W'(1);
                2'b01: if (rt_free_est_r != '0)                     rt_free_est_r <= rt_free_est_r - RTEST_W'(1);
                default: ;
            endcase

            // ── commit pipeline ──
            if (advance) begin
                // PUBLISH <- WB
                pub_v <= wb_v;
                pub_e <= wb_e;
                // WB <- input (completion priority), pop the source FIFO
                wb_v <= pull_any;
                if (pull_cmpl) begin
                    wb_e <= mk_ce(1'b1, cmpl_head_task,
                                  cmpl_head_hit ? TM_PH_SHADE_READY : TM_PH_FINALIZE_READY,
                                  cmpl_head_sh, cmpl_head_t);
                    cmpl_head <= cmpl_head + CMPLP_W'(1);
                end else if (pull_pub) begin
                    // Publish FIFO entry layout: {src 2, phase 2, shader 6, task 8}
                    wb_e <= mk_ce(1'b0, pub_head_ent[TASK_ID_W-1:0],
                                  pub_head_ent[TASK_ID_W+BUCKET_W +: 2],
                                  pub_head_ent[TASK_ID_W +: BUCKET_W], 32'd0);
                    pub_head <= pub_head + PUBP_W'(1);
                end
            end

            // ── FIFO occupancy (push/pop may coincide) ──
            case ({pub_push, (advance && pull_pub)})
                2'b10:   pub_count <= pub_count + PUBLVL_W'(1);
                2'b01:   pub_count <= pub_count - PUBLVL_W'(1);
                default: ;
            endcase
            case ({cmpl_accept, (advance && pull_cmpl)})
                2'b10:   cmpl_count <= cmpl_count + CMPLLVL_W'(1);
                2'b01:   cmpl_count <= cmpl_count - CMPLLVL_W'(1);
                default: ;
            endcase
        end
    end

    `RUNTIME_ASSERT(~(cmpl_accept && (cmpl_count >= CMPLLVL_W'(CMPL_FIFO_DEPTH))),
        ("%t: *** %s: completion FIFO overflow", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(~(pub_push && (pub_count >= PUBLVL_W'(PUB_FIFO_DEPTH))),
        ("%t: *** %s: publish FIFO overflow", $time, INSTANCE_ID))

endmodule
