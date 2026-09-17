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

// VX_tm_pqm — TaskManager Phase-Queue-Manager sub-block: the getWork face.
//
// It owns the three phase queues that feed the three worker roles:
//   TRACE_READY / FINALIZE_READY  plain task-id FIFOs,
//   SHADE_READY                   NUM_BUCKETS per-shader FIFOs whose
//                                 head/tail/depth live in the Directory.
// The PR commit pipeline enqueues published tasks (pqm_enq_*); getWork
// dequeues up to 32 into a packet and emits one gw_rsp.
//
// The Owner table is NOT here — it lives in the top. As a task is popped its
// owner must migrate to the serving role's worker; this block REQUESTS that
// (pqm_mig_valid/id/val) and only advances the pop when the top's write
// arbiter grants it (pqm_mig_grant), so pop and owner-migration stay atomic.
// A stalled grant just stalls assembly (backpressure), never a lost task.

`include "VX_define.vh"

module VX_tm_pqm
    import VX_tm_pkg::*;
#(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_TASKS    = 256,
    parameter NUM_BUCKETS  = 64,
    parameter BUCKET_DEPTH = 64,
    parameter NUM_WORKERS  = 7,
    parameter CFG_RT_SLOTS = 128
) (
    input wire clk,
    input wire reset,

    // gating / global state (from the top: TSM pool_ready, PR free-slot est,
    // and the five-way kernel_done term)
    input wire pool_ready,
    input wire kernel_done,
    input wire [`CLOG2(CFG_RT_SLOTS+1)-1:0] rt_free_est,

    // getWork face (worker <-> PQM)
    input  wire         gw_req_valid,
    output wire         gw_req_ready,
    input  wire [2:0]   gw_req_warp_id,
    input  wire [1:0]   gw_req_role,
    output reg          gw_rsp_valid,
    output reg  [2:0]   gw_rsp_warp_id,
    output reg  [255:0] gw_rsp_task_id,
    output reg  [5:0]   gw_rsp_shader_id,
    output reg  [5:0]   gw_rsp_count,
    output reg          gw_rsp_kernel_done,

    // enqueue face (PR commit pipeline, stage 3 -> here): 1 task/cycle
    input  wire                          pqm_enq_valid,
    input  wire [1:0]                    pqm_enq_phase,
    input  wire [`CLOG2(NUM_TASKS)-1:0]  pqm_enq_task_id,
    input  wire [`CLOG2(NUM_BUCKETS)-1:0] pqm_enq_shader_id,
    output wire                          pqm_enq_ready,  // target queue can take it

    // owner-migration request to the top's arbitrated write port
    output wire                          pqm_mig_valid,
    output wire [`CLOG2(NUM_TASKS)-1:0]  pqm_mig_id,
    output wire [2:0]                    pqm_mig_val,
    input  wire                          pqm_mig_grant,

    // queue-occupancy term for the top's kernel_done
    output wire queues_empty
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam TASK_ID_W = `CLOG2(NUM_TASKS);          // 8
    localparam CNT_W     = `CLOG2(NUM_TASKS + 1);      // 9
    localparam BUCKET_W  = `CLOG2(NUM_BUCKETS);        // 6 = shader_id width
    localparam BDEPTH_W  = `CLOG2(BUCKET_DEPTH + 1);   // 7 (0..64)
    localparam BSLOT_W   = `CLOG2(BUCKET_DEPTH);       // 6 (bucket slot index)
    localparam WORKER_W  = `CLOG2(NUM_WORKERS);        // 3
    localparam RTEST_W   = `CLOG2(CFG_RT_SLOTS + 1);   // 8
    localparam PKT_MAX   = 32;                         // tasks per gw_rsp packet

    // ── SHADE buckets: a task-id store + a per-bucket directory ───────
    // depth!=0 IS the valid bit (no separate bk_valid to keep in sync).
    // The store is a VX_dp_ram (sync read) so synthesis maps it to a SRAM macro
    // instead of 32 Kbit of flops + a 4096:1 read mux; the look-ahead read
    // address (bk_raddr, below) makes bk_rdata the FWFT head — bit-identical to
    // the old combinational bucket_ram[{asm_bucket, bk_head[asm_bucket]}] read.
    wire [TASK_ID_W-1:0]                bk_rdata;
    reg [NUM_BUCKETS-1:0][BSLOT_W-1:0]  bk_head, bk_tail;
    reg [NUM_BUCKETS-1:0][BDEPTH_W-1:0] bk_depth;

    // ── TRACE_READY / FINALIZE_READY FIFOs (VX_fifo_queue: FWFT over SRAM) ──
    // data_out is the FWFT head (= old trace_fifo[trace_head]); empty/size are
    // registered, matching the old *_count timing. Instances are below (they
    // need the enq/deq strobes).
    wire [TASK_ID_W-1:0] trace_data_out, final_data_out;
    wire                 trace_empty,    final_empty;
    wire [CNT_W-1:0]     trace_size,     final_size;

    // ── pending getWork table ─────────────────────────────────────────
    reg [NUM_WORKERS-1:0]      gwv_valid;
    reg [NUM_WORKERS-1:0][2:0] gwv_warp;
    reg [NUM_WORKERS-1:0][1:0] gwv_role;
    wire [WORKER_W-1:0] gw_alloc_idx;
    wire                gw_alloc_valid;
    VX_priority_encoder #(.N (NUM_WORKERS)) gw_free_pe (
        .data_in    (~gwv_valid),
        `UNUSED_PIN (onehot_out),
        .index_out  (gw_alloc_idx),
        .valid_out  (gw_alloc_valid)
    );
    assign gw_req_ready = pool_ready && gw_alloc_valid;
    wire gw_req_fire = gw_req_valid && gw_req_ready;

    // ── queue occupancy ───────────────────────────────────────────────
    wire shade_any_nonempty = (|bk_depth);
    assign queues_empty = trace_empty && final_empty && !shade_any_nonempty;

    // Enqueue backpressure: the TRACE/FINALIZE FIFOs are NUM_TASKS deep so they
    // can never overflow (at most NUM_TASKS tasks are live); only a per-shader
    // SHADE bucket can fill, so the PR commit pipeline stalls on a full bucket.
    assign pqm_enq_ready = (pqm_enq_phase != TM_PH_SHADE_READY)
                        || (bk_depth[pqm_enq_shader_id] < BDEPTH_W'(BUCKET_DEPTH));

    // ── SHADE candidate select: prefer a full bucket (depth>=32), else any
    //    non-empty one. Suboptimal pick is allowed (hw_spec §4.2.3); a plain
    //    priority scan keeps it combinational and avoids a scan FSM.
    reg [NUM_BUCKETS-1:0] bk_full, bk_nonempty;
    always_comb begin
        for (int b = 0; b < NUM_BUCKETS; ++b) begin
            bk_full[b]     = (bk_depth[b] >= BDEPTH_W'(PKT_MAX));
            bk_nonempty[b] = (bk_depth[b] != '0);
        end
    end
    wire [BUCKET_W-1:0] bk_full_idx, bk_ne_idx;
    wire                bk_full_valid, bk_ne_valid;
    VX_priority_encoder #(.N (NUM_BUCKETS)) bk_full_pe (
        .data_in    (bk_full),
        `UNUSED_PIN (onehot_out),
        .index_out  (bk_full_idx),
        .valid_out  (bk_full_valid)
    );
    VX_priority_encoder #(.N (NUM_BUCKETS)) bk_ne_pe (
        .data_in    (bk_nonempty),
        `UNUSED_PIN (onehot_out),
        .index_out  (bk_ne_idx),
        .valid_out  (bk_ne_valid)
    );
    wire [BUCKET_W-1:0] shade_cand = bk_full_valid ? bk_full_idx : bk_ne_idx;

    // ── serviceability of each pending entry ──────────────────────────
    reg [NUM_WORKERS-1:0] gw_srv;
    always_comb begin
        for (int i = 0; i < NUM_WORKERS; ++i) begin
            gw_srv[i] = gwv_valid[i] && (
                ((gwv_role[i] == TM_ROLE_TRACE)    && !trace_empty && (rt_free_est != '0)) ||
                ((gwv_role[i] == TM_ROLE_FINALIZE) && !final_empty) ||
                ((gwv_role[i] == TM_ROLE_SHADE)    && shade_any_nonempty));
        end
    end
    // kernel_done drains every pending entry with a count=0 rsp; otherwise only
    // a serviceable one is picked.
    wire [NUM_WORKERS-1:0] gw_pick_mask = kernel_done ? gwv_valid : gw_srv;
    wire [WORKER_W-1:0] gw_pick_idx;
    wire                gw_pick_valid;
    VX_priority_encoder #(.N (NUM_WORKERS)) gw_srv_pe (
        .data_in    (gw_pick_mask),
        `UNUSED_PIN (onehot_out),
        .index_out  (gw_pick_idx),
        .valid_out  (gw_pick_valid)
    );

    // ── getWork service FSM ───────────────────────────────────────────
    localparam [1:0] W_IDLE = 2'd0, W_ASM = 2'd1, W_RSP = 2'd2, W_DONE = 2'd3;
    reg [1:0]           w_state;
    reg [WORKER_W-1:0]  asm_slot;
    reg [2:0]           asm_warp;
    reg [1:0]           asm_role;
    reg [BUCKET_W-1:0]  asm_bucket;
    reg [5:0]           asm_total;   // 1..32
    reg [5:0]           asm_lane;    // 0..31, the slot being filled
    reg [255:0]         asm_task;    // packet buffer (lane*8 +: 8)

    // the task popped this cycle is the head of the selected queue
    wire [TASK_ID_W-1:0] asm_id =
        (asm_role == TM_ROLE_SHADE)    ? bk_rdata :
        (asm_role == TM_ROLE_FINALIZE) ? final_data_out :
                                         trace_data_out;

    assign pqm_mig_valid = (w_state == W_ASM);
    assign pqm_mig_id    = asm_id;
    assign pqm_mig_val   = tm_role_owner(asm_role);
    wire   pop_fire      = (w_state == W_ASM) && pqm_mig_grant;

    // dequeue strobes (only this FSM pops; enqueue is the PR side)
    wire tr_deq = pop_fire && (asm_role == TM_ROLE_TRACE);
    wire fi_deq = pop_fire && (asm_role == TM_ROLE_FINALIZE);
    wire bk_deq = pop_fire && (asm_role == TM_ROLE_SHADE);

    // enqueue strobes
    wire tr_enq = pqm_enq_valid && (pqm_enq_phase == TM_PH_TRACE_READY);
    wire fi_enq = pqm_enq_valid && (pqm_enq_phase == TM_PH_FINALIZE_READY);
    wire bk_enq = pqm_enq_valid && (pqm_enq_phase == TM_PH_SHADE_READY);

    // ── bucket store: VX_dp_ram (sync read) + look-ahead FWFT address ──
    // W_IDLE presents the candidate bucket's head (the prime — W_IDLE is itself
    // the prime cycle, so the first pop fires in the first W_ASM cycle with no
    // bubble); W_ASM presents head+bk_deq so the registered read returns the
    // next head each cycle, sustaining 1 pop/cycle. A stalled pop re-presents
    // the same head. The write port is independent (true 1R1W), so a concurrent
    // enqueue to a different bucket is unaffected.
    wire [BUCKET_W+BSLOT_W-1:0] bk_raddr = (w_state == W_IDLE)
        ? {shade_cand, bk_head[shade_cand]}
        : {asm_bucket, bk_head[asm_bucket] + BSLOT_W'(bk_deq)};
    VX_dp_ram #(
        .DATAW    (TASK_ID_W),
        .SIZE     (NUM_BUCKETS * BUCKET_DEPTH),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) bucket_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (bk_enq),
        .wren  (1'b1),
        .waddr ({pqm_enq_shader_id, bk_tail[pqm_enq_shader_id]}),
        .wdata (pqm_enq_task_id),
        .raddr (bk_raddr),
        .rdata (bk_rdata)
    );

    // ── TRACE / FINALIZE queues: FWFT FIFOs over SRAM. Depth = NUM_TASKS so
    //    they can never overflow (at most NUM_TASKS tasks are live); `full` is
    //    unused. Enqueue and pop may coincide — the FIFO nets that internally.
    VX_fifo_queue #(
        .DATAW(TASK_ID_W), .DEPTH(NUM_TASKS), .OUT_REG(1),
        .ALM_FULL(NUM_TASKS-1), .ALM_EMPTY(1)
    ) trace_fifo (
        .clk(clk), .reset(reset),
        .push(tr_enq), .pop(tr_deq), .data_in(pqm_enq_task_id),
        .data_out(trace_data_out), .empty(trace_empty),
        `UNUSED_PIN (alm_empty), `UNUSED_PIN (full), `UNUSED_PIN (alm_full),
        .size(trace_size)
    );
    VX_fifo_queue #(
        .DATAW(TASK_ID_W), .DEPTH(NUM_TASKS), .OUT_REG(1),
        .ALM_FULL(NUM_TASKS-1), .ALM_EMPTY(1)
    ) final_fifo (
        .clk(clk), .reset(reset),
        .push(fi_enq), .pop(fi_deq), .data_in(pqm_enq_task_id),
        .data_out(final_data_out), .empty(final_empty),
        `UNUSED_PIN (alm_empty), `UNUSED_PIN (full), `UNUSED_PIN (alm_full),
        .size(final_size)
    );

    // packet size chosen at the W_IDLE decision
    function automatic [5:0] pkt_min32(input [CNT_W-1:0] n);
        pkt_min32 = (n >= CNT_W'(PKT_MAX)) ? 6'(PKT_MAX) : n[5:0];
    endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
            bk_head     <= '0;
            bk_tail     <= '0;
            bk_depth    <= '0;
            gwv_valid   <= '0;
            w_state     <= W_IDLE;
            asm_lane    <= '0;
            asm_total   <= '0;
            asm_task    <= '0;
            gw_rsp_valid <= 1'b0;
        end else begin
            gw_rsp_valid <= 1'b0;

            // ── pending getWork registration ──────────────────────────
            if (gw_req_fire) begin
                gwv_valid[gw_alloc_idx] <= 1'b1;
                gwv_warp[gw_alloc_idx]  <= gw_req_warp_id;
                gwv_role[gw_alloc_idx]  <= gw_req_role;
            end

            // ── enqueue: the FIFOs take tr_enq/fi_enq directly; only the
            //    bucket tail pointer is maintained here (its data write is the
            //    VX_dp_ram write port).
            if (bk_enq) begin
                bk_tail[pqm_enq_shader_id] <= bk_tail[pqm_enq_shader_id] + BSLOT_W'(1);
            end

            // ── dequeue (pop) on a granted migration: the FIFOs take
            //    tr_deq/fi_deq directly; only the bucket head is here.
            if (bk_deq) bk_head[asm_bucket] <= bk_head[asm_bucket] + BSLOT_W'(1);

            // ── bucket occupancy: enq and deq may name different buckets;
            //    guard the same-bucket case so bk_depth is never double-written.
            if (bk_enq && (!bk_deq || (pqm_enq_shader_id != asm_bucket)))
                bk_depth[pqm_enq_shader_id] <= bk_depth[pqm_enq_shader_id] + BDEPTH_W'(1);
            if (bk_deq && (!bk_enq || (pqm_enq_shader_id != asm_bucket)))
                bk_depth[asm_bucket] <= bk_depth[asm_bucket] - BDEPTH_W'(1);

            // ── service FSM ───────────────────────────────────────────
            case (w_state)
            W_IDLE: begin
                if (gw_pick_valid) begin
                    asm_slot   <= gw_pick_idx;
                    asm_warp   <= gwv_warp[gw_pick_idx];
                    asm_role   <= gwv_role[gw_pick_idx];
                    asm_lane   <= '0;
                    asm_task   <= '0;
                    if (kernel_done) begin
                        w_state <= W_DONE;
                    end else begin
                        case (gwv_role[gw_pick_idx])
                        TM_ROLE_SHADE: begin
                            asm_bucket <= shade_cand;
                            asm_total  <= pkt_min32({{(CNT_W-BDEPTH_W){1'b0}}, bk_depth[shade_cand]});
                        end
                        TM_ROLE_FINALIZE: asm_total <= pkt_min32(final_size);
                        default:          asm_total <= pkt_min32(trace_size);
                        endcase
                        w_state <= W_ASM;
                    end
                end
            end
            W_ASM: begin
                if (pop_fire) begin
                    asm_task[asm_lane*8 +: 8] <= asm_id;
                    if (asm_lane == asm_total - 6'd1) w_state <= W_RSP;
                    else                              asm_lane <= asm_lane + 6'd1;
                end
            end
            W_RSP: begin
                gw_rsp_valid       <= 1'b1;
                gw_rsp_warp_id     <= asm_warp;
                gw_rsp_task_id     <= asm_task;
                gw_rsp_shader_id   <= (asm_role == TM_ROLE_SHADE) ? asm_bucket : 6'd0;
                gw_rsp_count       <= asm_total;
                gw_rsp_kernel_done <= 1'b0;
                gwv_valid[asm_slot] <= 1'b0;
                w_state            <= W_IDLE;
            end
            W_DONE: begin
                gw_rsp_valid       <= 1'b1;
                gw_rsp_warp_id     <= asm_warp;
                gw_rsp_task_id     <= '0;
                gw_rsp_shader_id   <= 6'd0;
                gw_rsp_count       <= 6'd0;
                gw_rsp_kernel_done <= 1'b1;
                gwv_valid[asm_slot] <= 1'b0;
                w_state            <= W_IDLE;
            end
            default: w_state <= W_IDLE;
            endcase
        end
    end

    `RUNTIME_ASSERT(~(pop_fire && (asm_lane >= asm_total)),
        ("%t: *** %s: getWork assembled past the packet size", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(~(gw_rsp_valid && !gw_rsp_kernel_done && (gw_rsp_count == '0)),
        ("%t: *** %s: getWork returned an empty non-done packet", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(~(bk_enq && (bk_depth[pqm_enq_shader_id] >= BDEPTH_W'(BUCKET_DEPTH))),
        ("%t: *** %s: SHADE bucket %0d overflow", $time, INSTANCE_ID, pqm_enq_shader_id))

endmodule
