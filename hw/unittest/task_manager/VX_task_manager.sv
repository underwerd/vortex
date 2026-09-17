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

// VX_task_manager — standalone RayTask task manager (see hw_spec_task_manager).
//
// Self-contained: it does NOT touch the SM internals (no decode, no scoreboard,
// no SMEM/RF/CTA scheduler). The instruction face (genTask / getWork /
// publishPhase) is exposed as port-level valid/ready protocols driven by a
// testbench that plays the worker / init roles; "stalled on the scoreboard"
// means a response simply does not appear and the requester does not re-issue.
// TaskState lives in an internal RAM (deviation D1) instead of an SMEM kernel
// partition; the 128 B/entry ABI is unchanged.
//
// This top owns only the state the three sub-blocks share, and wires them:
//   TSM  (VX_tm_tsm)  genTask: free-ID allocation, seed/drain counting.
//   PQM  (VX_tm_pqm)  getWork: phase queues, directory, packet assembly.
//   PR   (VX_tm_pr)   publishPhase + admission + completion + commit pipeline.
// Shared state here: the Owner/Valid table, live_count, the TaskState RAM (and
// its PR>external write arbiter), the five-way kernel_done term, and status.
//
// The Owner table has ONE entry array but FIVE writers (TSM grant, PQM getWork
// migration, PR admission / completion / release). They only ever name pairwise
// disjoint task_ids in a cycle (assertion A1: a task is in exactly one place at
// once), so each writer applies to its own entry with no arbitration and no
// lost update.

`include "VX_define.vh"

module VX_task_manager
    import VX_tm_pkg::*;
#(
    parameter `STRING INSTANCE_ID    = "",
    parameter NUM_TASKS        = 256,   // task-id space
    parameter NUM_BUCKETS      = 64,    // shader buckets (dense shader_id)
    parameter BUCKET_DEPTH     = 64,    // per-bucket task-id FIFO depth
    parameter NUM_WORKERS      = 7,     // Role-CTA worker warps (pending getWork)
    parameter NUM_GEN_PROD     = 8,     // pending genTask producers
    parameter CMPL_FIFO_DEPTH  = 16,    // RT completion FIFO depth
    parameter PUB_FIFO_DEPTH   = 64,    // publish-request FIFO depth
    parameter COMPLETION_W     = 134,   // RT completion payload width
    parameter TASK_STATE_BYTES = 128,   // TaskState entry size (ABI, fixed)
    parameter CFG_RT_SLOTS     = 128,   // RT RaySlot capacity (free-slot estimate init)
    parameter MAX_WAIT_ROUNDS  = 16     // bucket age cap (unused: priority candidate)
) (
    input wire clk,
    input wire reset,

    // ── config ────────────────────────────────────────────────────────
    input wire [31:0] cfg_expected_seeds,  // initial tasks this SM must generate
    input wire        cfg_ld,              // pulse: latch + reset the drain logic

    // ── genTask face (init warp <-> TSM) ──────────────────────────────
    input  wire         gen_req_valid,
    output wire         gen_req_ready,     // pending genTask table not full
    input  wire [2:0]   gen_req_warp_id,
    input  wire [31:0]  gen_req_mask,
    output wire         gen_rsp_valid,     // exactly once per request
    output wire [2:0]   gen_rsp_warp_id,
    output wire [31:0]  gen_rsp_granted,   // subset of gen_req_mask
    output wire [255:0] gen_rsp_task_id,   // 32 x 8b, per lane
    output wire         gen_rsp_drain_stop,

    // ── getWork face (worker <-> PQM) ─────────────────────────────────
    input  wire         gw_req_valid,
    output wire         gw_req_ready,      // pending getWork table not full
    input  wire [2:0]   gw_req_warp_id,
    input  wire [1:0]   gw_req_role,
    output wire         gw_rsp_valid,      // no nack: absent unless kernel_done
    output wire [2:0]   gw_rsp_warp_id,
    output wire [255:0] gw_rsp_task_id,
    output wire [5:0]   gw_rsp_shader_id,
    output wire [5:0]   gw_rsp_count,
    output wire         gw_rsp_kernel_done,

    // ── publishPhase face (worker / init <-> PR) ──────────────────────
    input  wire         pub_req_valid,
    output wire         pub_req_ready,     // publish FIFO has >=32 free
    input  wire [31:0]  pub_req_mask,
    input  wire [255:0] pub_req_task_id,
    input  wire [191:0] pub_req_shader_id,
    input  wire [63:0]  pub_req_next_phase,
    input  wire [1:0]   pub_req_src,       // INIT / WORKER

    // ── TRACE admission face (TRACE worker -> PR) ─────────────────────
    input  wire         adm_req_valid,
    output wire         adm_req_ready,     // admission FSM in A_IDLE
    input  wire [7:0]   adm_req_task_id,
    input  wire [255:0] adm_req_ray,
    input  wire [7:0]   adm_req_bounce,
    input  wire [31:0]  adm_req_cont,

    // ── RT face (<-> RT Unit) ─────────────────────────────────────────
    output wire         rt_adm_desc_valid,
    output wire [303:0] rt_adm_desc,       // {task_id 8, ray 256, bounce 8, cont 32}
    input  wire         rt_adm_accept_valid,
    input  wire [14:0]  rt_adm_accept,     // {task_id 8, slot_id 7}
    input  wire         rt_adm_reject_valid,
    input  wire [7:0]   rt_adm_reject,     // {task_id}
    input  wire         rt_completion_valid,
    output wire         rt_completion_ready,  // completion FIFO not full
    input  wire [COMPLETION_W-1:0] rt_completion,
    input  wire         rt_rayslot_idle,

    // ── TaskState face (deviation D1: the internal RAM's external port) ─
    input  wire         ts_wr_valid,
    output wire         ts_wr_ready,
    input  wire [7:0]   ts_wr_task_id,
    input  wire [4:0]   ts_wr_word,
    input  wire [31:0]  ts_wr_data,
    input  wire         ts_rd_valid,
    input  wire [7:0]   ts_rd_task_id,
    input  wire [4:0]   ts_rd_word,
    output wire [31:0]  ts_rd_data,
    output wire         ts_rd_data_valid,

    // ── status / debug ────────────────────────────────────────────────
    output wire       sts_kernel_done,
    output wire       sts_draining,
    output wire [8:0] sts_live_count,
    output wire [8:0] sts_free_tasks,
    output wire [4:0] sts_cmpl_fifo_lvl,
    output wire [6:0] sts_pub_fifo_lvl
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (MAX_WAIT_ROUNDS)

    localparam TASK_ID_W  = `CLOG2(NUM_TASKS);            // 8
    localparam BUCKET_W   = `CLOG2(NUM_BUCKETS);          // 6
    localparam CNT_W      = `CLOG2(NUM_TASKS + 1);        // 9
    localparam RTEST_W    = `CLOG2(CFG_RT_SLOTS + 1);     // 8
    localparam TS_WORDS   = TASK_STATE_BYTES / 4;         // 32
    localparam TS_WORD_W  = `CLOG2(TS_WORDS);             // 5
    localparam TS_ADDR_W  = TASK_ID_W + TS_WORD_W;        // 13
    localparam CMPL_LVL_W = `CLOG2(CMPL_FIFO_DEPTH + 1);  // 5
    localparam PUB_LVL_W  = `CLOG2(PUB_FIFO_DEPTH + 1);   // 7

    // ═══════════════════════ shared state ═════════════════════════════
    reg [NUM_TASKS-1:0][2:0] owner;       // one entry per task, five writers
    reg [CNT_W-1:0]          live_count;  // tasks with owner != FREE

    // TaskState RAM: 1W1R, the PR completion write has strict priority over the
    // external (init/worker) write port. Mapped to VX_dp_ram so synthesis treats
    // it as a SRAM macro and not 256 Kbit of flip-flops: deviation D1 makes this
    // the SMEM kernel partition moved inside the module, returned to SMEM at
    // integration -- it is storage the TM owns, not TM control logic.
    wire [31:0] ts_rd_data_w;
    reg         ts_rd_valid_q;

    // ═══════════════════════ inter-submodule wires ════════════════════
    // TSM -> top / PQM
    wire                 tsm_grant_valid, tsm_live_inc, tsm_draining, tsm_pool_ready;
    wire [TASK_ID_W-1:0] tsm_grant_id;
    wire [CNT_W-1:0]     tsm_free_count;
    // PR -> TSM (FINALIZE release returns the id to the Free-ID FIFO)
    wire                 pr_rel_valid;
    wire [TASK_ID_W-1:0] pr_rel_id;
    // PR -> PQM (commit pipeline enqueue + free-slot gate)
    wire                 pqm_enq_valid, pqm_enq_ready;
    wire [1:0]           pqm_enq_phase;
    wire [TASK_ID_W-1:0] pqm_enq_task_id;
    wire [BUCKET_W-1:0]  pqm_enq_shader_id;
    wire [RTEST_W-1:0]   rt_free_est;
    // PQM -> top (getWork owner migration)
    wire                 pqm_mig_valid;
    wire [TASK_ID_W-1:0] pqm_mig_id;
    wire [2:0]           pqm_mig_val;
    // PR -> top (admission / completion owner migrations)
    wire                 pr_adm_mig_valid, pr_cmpl_mig_valid;
    wire [TASK_ID_W-1:0] pr_adm_mig_id, pr_cmpl_mig_id;
    // PR -> top (TaskState completion write)
    wire                 pr_ts_wr_valid;
    wire [TS_ADDR_W-1:0] pr_ts_wr_addr;
    wire [31:0]          pr_ts_wr_data;
    // occupancy terms for kernel_done
    wire                 pqm_queues_empty, pr_fifos_empty;

    wire kernel_done = tsm_draining && (live_count == '0)
                    && pqm_queues_empty && pr_fifos_empty && rt_rayslot_idle;

    // ═══════════════════════ TSM ══════════════════════════════════════
    VX_tm_tsm #(
        .INSTANCE_ID  (INSTANCE_ID),
        .NUM_TASKS    (NUM_TASKS),
        .NUM_GEN_PROD (NUM_GEN_PROD)
    ) tsm (
        .clk                (clk),
        .reset              (reset),
        .cfg_expected_seeds (cfg_expected_seeds),
        .cfg_ld             (cfg_ld),
        .gen_req_valid      (gen_req_valid),
        .gen_req_ready      (gen_req_ready),
        .gen_req_warp_id    (gen_req_warp_id),
        .gen_req_mask       (gen_req_mask),
        .gen_rsp_valid      (gen_rsp_valid),
        .gen_rsp_warp_id    (gen_rsp_warp_id),
        .gen_rsp_granted    (gen_rsp_granted),
        .gen_rsp_task_id    (gen_rsp_task_id),
        .gen_rsp_drain_stop (gen_rsp_drain_stop),
        .tsm_grant_valid    (tsm_grant_valid),
        .tsm_grant_id       (tsm_grant_id),
        .tsm_live_inc       (tsm_live_inc),
        .free_push          (pr_rel_valid),
        .free_push_id       (pr_rel_id),
        .draining           (tsm_draining),
        .pool_ready         (tsm_pool_ready),
        .free_count         (tsm_free_count),
        `UNUSED_PIN         (seed_count)
    );

    // ═══════════════════════ PQM ══════════════════════════════════════
    VX_tm_pqm #(
        .INSTANCE_ID  (INSTANCE_ID),
        .NUM_TASKS    (NUM_TASKS),
        .NUM_BUCKETS  (NUM_BUCKETS),
        .BUCKET_DEPTH (BUCKET_DEPTH),
        .NUM_WORKERS  (NUM_WORKERS),
        .CFG_RT_SLOTS (CFG_RT_SLOTS)
    ) pqm (
        .clk              (clk),
        .reset            (reset),
        .pool_ready       (tsm_pool_ready),
        .kernel_done      (kernel_done),
        .rt_free_est      (rt_free_est),
        .gw_req_valid     (gw_req_valid),
        .gw_req_ready     (gw_req_ready),
        .gw_req_warp_id   (gw_req_warp_id),
        .gw_req_role      (gw_req_role),
        .gw_rsp_valid     (gw_rsp_valid),
        .gw_rsp_warp_id   (gw_rsp_warp_id),
        .gw_rsp_task_id   (gw_rsp_task_id),
        .gw_rsp_shader_id (gw_rsp_shader_id),
        .gw_rsp_count     (gw_rsp_count),
        .gw_rsp_kernel_done (gw_rsp_kernel_done),
        .pqm_enq_valid    (pqm_enq_valid),
        .pqm_enq_phase    (pqm_enq_phase),
        .pqm_enq_task_id  (pqm_enq_task_id),
        .pqm_enq_shader_id (pqm_enq_shader_id),
        .pqm_enq_ready    (pqm_enq_ready),
        .pqm_mig_valid    (pqm_mig_valid),
        .pqm_mig_id       (pqm_mig_id),
        .pqm_mig_val      (pqm_mig_val),
        .pqm_mig_grant    (1'b1),          // multi-write owner array always takes it
        .queues_empty     (pqm_queues_empty)
    );

    // ═══════════════════════ PR ═══════════════════════════════════════
    VX_tm_pr #(
        .INSTANCE_ID      (INSTANCE_ID),
        .NUM_TASKS        (NUM_TASKS),
        .NUM_BUCKETS      (NUM_BUCKETS),
        .CMPL_FIFO_DEPTH  (CMPL_FIFO_DEPTH),
        .PUB_FIFO_DEPTH   (PUB_FIFO_DEPTH),
        .COMPLETION_W     (COMPLETION_W),
        .CFG_RT_SLOTS     (CFG_RT_SLOTS),
        .TASK_STATE_BYTES (TASK_STATE_BYTES)
    ) pr (
        .clk               (clk),
        .reset             (reset),
        .pub_req_valid     (pub_req_valid),
        .pub_req_ready     (pub_req_ready),
        .pub_req_mask      (pub_req_mask),
        .pub_req_task_id   (pub_req_task_id),
        .pub_req_shader_id (pub_req_shader_id),
        .pub_req_next_phase (pub_req_next_phase),
        .pub_req_src       (pub_req_src),
        .adm_req_valid     (adm_req_valid),
        .adm_req_ready     (adm_req_ready),
        .adm_req_task_id   (adm_req_task_id),
        .adm_req_ray       (adm_req_ray),
        .adm_req_bounce    (adm_req_bounce),
        .adm_req_cont      (adm_req_cont),
        .rt_adm_desc_valid (rt_adm_desc_valid),
        .rt_adm_desc       (rt_adm_desc),
        .rt_adm_accept_valid (rt_adm_accept_valid),
        .rt_adm_accept     (rt_adm_accept),
        .rt_adm_reject_valid (rt_adm_reject_valid),
        .rt_adm_reject     (rt_adm_reject),
        .rt_completion_valid (rt_completion_valid),
        .rt_completion_ready (rt_completion_ready),
        .rt_completion     (rt_completion),
        .pqm_enq_valid     (pqm_enq_valid),
        .pqm_enq_phase     (pqm_enq_phase),
        .pqm_enq_task_id   (pqm_enq_task_id),
        .pqm_enq_shader_id (pqm_enq_shader_id),
        .pqm_enq_ready     (pqm_enq_ready),
        .rt_free_est       (rt_free_est),
        .pr_adm_mig_valid  (pr_adm_mig_valid),
        .pr_adm_mig_id     (pr_adm_mig_id),
        .pr_cmpl_mig_valid (pr_cmpl_mig_valid),
        .pr_cmpl_mig_id    (pr_cmpl_mig_id),
        .pr_rel_valid      (pr_rel_valid),
        .pr_rel_id         (pr_rel_id),
        .pr_ts_wr_valid    (pr_ts_wr_valid),
        .pr_ts_wr_addr     (pr_ts_wr_addr),
        .pr_ts_wr_data     (pr_ts_wr_data),
        .pr_fifos_empty    (pr_fifos_empty),
        .sts_cmpl_fifo_lvl (sts_cmpl_fifo_lvl),
        .sts_pub_fifo_lvl  (sts_pub_fifo_lvl)
    );

    // ═══════════════════════ shared-state update ══════════════════════
    // Owner table: five disjoint writers (A1), each to its own entry.
    always_ff @(posedge clk) begin
        if (reset) begin
            owner      <= '0;   // all FREE
            live_count <= '0;
        end else begin
            if (tsm_grant_valid)   owner[tsm_grant_id]   <= TM_OWN_INIT;
            if (pqm_mig_valid)     owner[pqm_mig_id]     <= pqm_mig_val;
            if (pr_adm_mig_valid)  owner[pr_adm_mig_id]  <= TM_OWN_RT_SLOT;
            if (pr_cmpl_mig_valid) owner[pr_cmpl_mig_id] <= TM_OWN_R_COMMIT;
            if (pr_rel_valid)      owner[pr_rel_id]      <= TM_OWN_FREE;

            case ({tsm_live_inc, pr_rel_valid})
                2'b10:   live_count <= live_count + CNT_W'(1);
                2'b01:   live_count <= live_count - CNT_W'(1);
                default: ;
            endcase
        end
    end

    // TaskState RAM write port: PR completion write has strict priority; the
    // external (init/worker) writer holds while the PR is using the port.
    wire                 ts_ram_write = pr_ts_wr_valid || ts_wr_valid;
    wire [TS_ADDR_W-1:0] ts_ram_waddr = pr_ts_wr_valid ? pr_ts_wr_addr
                                                       : {ts_wr_task_id, ts_wr_word};
    wire [31:0]          ts_ram_wdata = pr_ts_wr_valid ? pr_ts_wr_data : ts_wr_data;
    VX_dp_ram #(
        .DATAW    (32),
        .SIZE     (NUM_TASKS * TS_WORDS),
        .OUT_REG  (1),
        .RDW_MODE ("W")
    ) ts_ram (
        .clk   (clk),
        .reset (reset),
        .read  (ts_rd_valid),
        .write (ts_ram_write),
        .wren  (1'b1),
        .waddr (ts_ram_waddr),
        .wdata (ts_ram_wdata),
        .raddr ({ts_rd_task_id, ts_rd_word}),
        .rdata (ts_rd_data_w)
    );
    always_ff @(posedge clk) begin
        if (reset) ts_rd_valid_q <= 1'b0;
        else       ts_rd_valid_q <= ts_rd_valid;
    end
    assign ts_rd_data       = ts_rd_data_w;
    assign ts_rd_data_valid = ts_rd_valid_q;
    assign ts_wr_ready      = !pr_ts_wr_valid;

    // ── status ────────────────────────────────────────────────────────
    assign sts_kernel_done = kernel_done;
    assign sts_draining    = tsm_draining;
    assign sts_live_count  = live_count;
    assign sts_free_tasks  = CNT_W'(tsm_free_count);

    // ── assertions ────────────────────────────────────────────────────
    // A1: the five owner writers name pairwise-disjoint task_ids in a cycle.
    `RUNTIME_ASSERT(~(tsm_grant_valid && pqm_mig_valid && (tsm_grant_id == pqm_mig_id)),
        ("%t: *** %s: A1 TSM grant and PQM migration collide on task %0d", $time, INSTANCE_ID, tsm_grant_id))
    `RUNTIME_ASSERT(~(tsm_grant_valid && pr_rel_valid && (tsm_grant_id == pr_rel_id)),
        ("%t: *** %s: A1 TSM grant and PR release collide on task %0d", $time, INSTANCE_ID, tsm_grant_id))
    `RUNTIME_ASSERT(~(pqm_mig_valid && pr_rel_valid && (pqm_mig_id == pr_rel_id)),
        ("%t: *** %s: A1 PQM migration and PR release collide on task %0d", $time, INSTANCE_ID, pqm_mig_id))
    `RUNTIME_ASSERT(~(pr_adm_mig_valid && pr_cmpl_mig_valid && (pr_adm_mig_id == pr_cmpl_mig_id)),
        ("%t: *** %s: A1 PR admission and completion collide on task %0d", $time, INSTANCE_ID, pr_adm_mig_id))
    // A2: a task is granted only when its owner is FREE.
    `RUNTIME_ASSERT(~(tsm_grant_valid && (owner[tsm_grant_id] != TM_OWN_FREE)),
        ("%t: *** %s: A2 genTask granted task %0d whose owner is not FREE", $time, INSTANCE_ID, tsm_grant_id))

`ifdef SIMULATION
    // A3: live_count tracks the non-FREE owner population exactly. Both are
    // updated together (a grant sets INIT and increments; a release clears and
    // decrements), so they can never diverge. Simulation-only: the per-entry
    // recount is far too wide to synthesize.
    reg [CNT_W-1:0] owner_nonfree;
    always_comb begin
        owner_nonfree = '0;
        for (int i = 0; i < NUM_TASKS; ++i)
            if (owner[i] != TM_OWN_FREE) owner_nonfree = owner_nonfree + CNT_W'(1);
    end
    `RUNTIME_ASSERT((live_count == owner_nonfree),
        ("%t: *** %s: A3 live_count %0d != non-FREE owners %0d", $time, INSTANCE_ID, live_count, owner_nonfree))

    // A4: kernel_done is monotonic -- once asserted it holds until cfg_ld starts
    // a new kernel (draining latches, and no new task can enter once draining).
    reg kernel_done_q;
    always_ff @(posedge clk) begin
        if (reset) kernel_done_q <= 1'b0;
        else       kernel_done_q <= kernel_done;
    end
    `RUNTIME_ASSERT(~(kernel_done_q && !kernel_done && !cfg_ld),
        ("%t: *** %s: A4 kernel_done dropped without cfg_ld", $time, INSTANCE_ID))
`endif

endmodule
