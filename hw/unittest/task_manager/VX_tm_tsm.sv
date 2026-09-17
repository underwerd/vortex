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

// VX_tm_tsm — TaskManager Task-State-Manager sub-block: the genTask face.
//
// Self-contained: it owns the Free-ID FIFO, the seed counter and the drain
// flag, so the per-lane allocation gate (seed_count < expected) closes entirely
// inside this module — no combinational loop through the top. The Owner table
// and live_count live in the top; this block only REQUESTS a grant
// (tsm_grant_valid/id) and a live increment, which the top folds into the
// single arbitrated owner-write port. FINALIZE release returns an id via
// free_push (driven by the PR sub-block through the top).

`include "VX_define.vh"

module VX_tm_tsm #(
    parameter `STRING INSTANCE_ID  = "",
    parameter NUM_TASKS    = 256,
    parameter NUM_GEN_PROD = 8
) (
    input wire clk,
    input wire reset,

    // config
    input wire [31:0] cfg_expected_seeds,
    input wire        cfg_ld,

    // genTask face (init warp <-> TSM)
    input  wire         gen_req_valid,
    output wire         gen_req_ready,
    input  wire [2:0]   gen_req_warp_id,
    input  wire [31:0]  gen_req_mask,
    output reg          gen_rsp_valid,
    output reg  [2:0]   gen_rsp_warp_id,
    output reg  [31:0]  gen_rsp_granted,
    output reg  [255:0] gen_rsp_task_id,
    output reg          gen_rsp_drain_stop,

    // grant request to the top's Owner table (top writes owner[id] = INIT)
    output wire                  tsm_grant_valid,
    output wire [`CLOG2(NUM_TASKS)-1:0] tsm_grant_id,
    output wire                  tsm_live_inc,

    // Free-ID return (FINALIZE release, via the top from PR)
    input wire                   free_push,
    input wire [`CLOG2(NUM_TASKS)-1:0] free_push_id,

    // status
    output wire        draining,
    output wire        pool_ready,
    output wire [`CLOG2(NUM_TASKS+1)-1:0] free_count,
    output wire [31:0] seed_count
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam TASK_ID_W = `CLOG2(NUM_TASKS);        // 8
    localparam CNT_W     = `CLOG2(NUM_TASKS + 1);    // 9
    localparam GENPROD_W = `CLOG2(NUM_GEN_PROD);     // 3

    // ── Free-ID FIFO: a VX_fifo_queue (FWFT over VX_dp_ram → SRAM macro) ──
    // Reset refills 0..NUM_TASKS-1; pool_ready gates every request until done.
    // data_out is the FWFT head (= old free_id_ram[free_head]); empty/size are
    // registered, matching the old free_count_r timing exactly.
    wire [TASK_ID_W-1:0] free_head_id;
    wire                 free_empty;
    wire [CNT_W-1:0]     free_size;
    reg [TASK_ID_W-1:0]  refill_id;
    reg                  pool_ready_r;

    // ── seed counter + drain flag ─────────────────────────────────────
    reg [31:0] seed_count_r;
    reg [31:0] expected_seeds;
    reg        draining_r;

    // ── pending genTask table ─────────────────────────────────────────
    reg [NUM_GEN_PROD-1:0]       pg_valid;
    reg [NUM_GEN_PROD-1:0][2:0]  pg_warp;
    reg [NUM_GEN_PROD-1:0][31:0] pg_mask;
    wire [GENPROD_W-1:0] pg_alloc_idx;
    wire                 pg_alloc_valid;
    VX_priority_encoder #(.N (NUM_GEN_PROD)) pg_free_pe (
        .data_in    (~pg_valid),
        `UNUSED_PIN (onehot_out),
        .index_out  (pg_alloc_idx),
        .valid_out  (pg_alloc_valid)
    );
    wire [GENPROD_W-1:0] pg_srv_idx;
    wire                 pg_srv_valid;
    VX_priority_encoder #(.N (NUM_GEN_PROD)) pg_srv_pe (
        .data_in    (pg_valid),
        `UNUSED_PIN (onehot_out),
        .index_out  (pg_srv_idx),
        .valid_out  (pg_srv_valid)
    );
    assign gen_req_ready = pool_ready_r && pg_alloc_valid;
    wire gen_req_fire = gen_req_valid && gen_req_ready;

    // ── allocation sequencer ──────────────────────────────────────────
    localparam [1:0] T_IDLE = 2'd0, T_SCAN = 2'd1, T_RSP = 2'd2, T_DRAIN = 2'd3;
    reg [1:0]          tsm_state;
    reg [GENPROD_W-1:0] tsm_slot;
    reg [5:0]          tsm_lane;     // 0..31
    reg [2:0]          tsm_warp;
    reg [31:0]         tsm_mask;
    reg [31:0]         tsm_granted;
    reg [255:0]        tsm_taskid;

    // a lane allocates when active, an id is free, and the seed cap is not hit
    wire tsm_alloc = (tsm_state == T_SCAN) && tsm_mask[tsm_lane[4:0]]
                  && !free_empty && (seed_count_r < expected_seeds);

    assign tsm_grant_valid = tsm_alloc;
    assign tsm_grant_id    = free_head_id;
    assign tsm_live_inc    = tsm_alloc;

    // Free-ID store: during refill push the ramp every cycle, afterward push
    // FINALIZE releases; pop one id per granted lane (tsm_alloc).
    wire                 free_push_sel = pool_ready_r ? free_push     : 1'b1;
    wire [TASK_ID_W-1:0] free_data_in  = pool_ready_r ? free_push_id  : refill_id;
    VX_fifo_queue #(
        .DATAW     (TASK_ID_W),
        .DEPTH     (NUM_TASKS),
        .OUT_REG   (1),
        .ALM_FULL  (NUM_TASKS - 1),
        .ALM_EMPTY (1)
    ) free_id_fifo (
        .clk       (clk),
        .reset     (reset),
        .push      (free_push_sel),
        .pop       (tsm_alloc),
        .data_in   (free_data_in),
        .data_out  (free_head_id),
        .empty     (free_empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        .size      (free_size)
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            refill_id    <= '0;
            pool_ready_r <= 1'b0;
            seed_count_r <= '0;
            expected_seeds <= '0;
            draining_r   <= 1'b0;
            pg_valid     <= '0;
            tsm_state    <= T_IDLE;
            tsm_lane     <= '0;
            tsm_granted  <= '0;
            tsm_taskid   <= '0;
            gen_rsp_valid <= 1'b0;
        end else begin
            gen_rsp_valid <= 1'b0;

            if (cfg_ld) begin
                expected_seeds <= cfg_expected_seeds;
                seed_count_r   <= '0;
                draining_r     <= 1'b0;
            end

            // ── Free-ID FIFO refill: the FIFO itself takes the push/pop; here
            //    only drive the ramp counter and latch pool_ready when done.
            if (!pool_ready_r) begin
                refill_id <= refill_id + TASK_ID_W'(1);
                if (refill_id == TASK_ID_W'(NUM_TASKS - 1)) pool_ready_r <= 1'b1;
            end

            // ── pending genTask registration ──────────────────────────
            if (gen_req_fire) begin
                pg_valid[pg_alloc_idx] <= 1'b1;
                pg_warp[pg_alloc_idx]  <= gen_req_warp_id;
                pg_mask[pg_alloc_idx]  <= gen_req_mask;
            end

            // ── allocation sequencer ──────────────────────────────────
            case (tsm_state)
            T_IDLE: begin
                tsm_granted <= '0;
                tsm_taskid  <= '0;
                tsm_lane    <= '0;
                if (pg_srv_valid) begin
                    tsm_slot  <= pg_srv_idx;
                    tsm_warp  <= pg_warp[pg_srv_idx];
                    tsm_mask  <= pg_mask[pg_srv_idx];
                    tsm_state <= draining_r ? T_DRAIN : T_SCAN;
                end
            end
            T_SCAN: begin
                if (tsm_alloc) begin
                    tsm_taskid[32'(tsm_lane[4:0])*8 +: 8] <= free_head_id;
                    tsm_granted[tsm_lane[4:0]]            <= 1'b1;
                    seed_count_r <= seed_count_r + 32'd1;
                end
                if (tsm_lane == 6'd31) tsm_state <= T_RSP;
                else                   tsm_lane  <= tsm_lane + 6'd1;
            end
            T_RSP: begin
                gen_rsp_valid      <= 1'b1;
                gen_rsp_warp_id    <= tsm_warp;
                gen_rsp_granted    <= tsm_granted;
                gen_rsp_task_id    <= tsm_taskid;
                // seed_count_r already includes this grant's last lane, so the
                // cap test is exact here; draining_r would lag by one cycle and
                // miss a cap hit on lane 31.
                gen_rsp_drain_stop <= (expected_seeds != '0)
                                   && (seed_count_r >= expected_seeds);
                pg_valid[tsm_slot] <= 1'b0;
                tsm_state          <= T_IDLE;
            end
            T_DRAIN: begin
                gen_rsp_valid      <= 1'b1;
                gen_rsp_warp_id    <= tsm_warp;
                gen_rsp_granted    <= '0;
                gen_rsp_task_id    <= '0;
                gen_rsp_drain_stop <= 1'b1;
                pg_valid[tsm_slot] <= 1'b0;
                tsm_state          <= T_IDLE;
            end
            default: tsm_state <= T_IDLE;
            endcase

            // draining latches when the seed count caps, and never clears until
            // cfg_ld: the allocator is gated by it, so no new task can enter and
            // kernel_done (in the top) stays monotonic.
            if ((seed_count_r >= expected_seeds) && (expected_seeds != '0)) begin
                draining_r <= 1'b1;
            end
        end
    end

    assign draining    = draining_r;
    assign pool_ready  = pool_ready_r;
    assign free_count  = free_size;
    assign seed_count  = seed_count_r;

    `RUNTIME_ASSERT(~(tsm_alloc && (seed_count_r >= expected_seeds)),
        ("%t: *** %s: genTask allocated past the seed cap", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(~(gen_rsp_valid && ((gen_rsp_granted & ~tsm_mask) != '0)),
        ("%t: *** %s: genTask granted lanes outside the request mask", $time, INSTANCE_ID))

endmodule
