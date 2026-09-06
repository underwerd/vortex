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

// Math-SFU execution unit: ex2.f32 / tanh.f32 / sigmoid.f32 (EXT_F-gated).
//
// Dedicated EX_MATH slot, structurally a slimmed-down VX_fpu_unit:
//   lane_dispatch -> tag_store + pe_serializer + VX_math_lane array -> gather
//
// Each warp is chopped into SIMD_WIDTH/NUM_MATH_LANES packets by
// VX_lane_dispatch (one packet per cycle); NUM_MATH_LANES == NUM_PES keeps the
// serializer in full-throughput passthrough mode, so back-to-back packets ride
// the fixed VX_CFG_MATH_LATENCY (= 35: the PE input register stage below +
// the 34 VX_math_lane stages) pipeline.
// No fflags: the approx-class instructions never raise FP exceptions.

`include "VX_fpu_define.vh"

module VX_math_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`VX_CFG_ISSUE_WIDTH]
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BLOCK_SIZE = 1;
    localparam NUM_LANES  = `VX_CFG_NUM_MATH_LANES;
    localparam TAG_WIDTH  = `LOG2UP(`VX_CFG_MATH_QUEUE_SIZE);
    localparam PARTIAL_BW = (BLOCK_SIZE != `VX_CFG_ISSUE_WIDTH) || (NUM_LANES != `VX_CFG_SIMD_WIDTH);

    VX_execute_if #(
        .data_t (math_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) lane_dispatch (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_result_if #(
        .data_t (math_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
        wire math_req_valid, math_req_ready;
        wire math_rsp_valid, math_rsp_ready;
        wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] math_rsp_result;

        math_header_t math_hdr, math_hdr_wb, math_hdr_store;

        // Math always writes back, so header.wb is always 1 at dispatch.
        always_comb begin
            math_hdr_store    = per_block_execute_if[block_idx].data.header;
            math_hdr_store.wb = 1'b0;
        end

        // Force wb=1 on the readback path.
        always_comb begin
            math_hdr_wb    = math_hdr;
            math_hdr_wb.wb = 1'b1;
        end

        wire [TAG_WIDTH-1:0] math_req_tag, math_rsp_tag;
        wire mdata_full;

        wire execute_fire = per_block_execute_if[block_idx].valid && per_block_execute_if[block_idx].ready;
        wire math_rsp_fire = math_rsp_valid && math_rsp_ready;

        VX_index_buffer #(
            .DATAW  ($bits(math_header_t)),
            .SIZE   (`VX_CFG_MATH_QUEUE_SIZE)
        ) tag_store (
            .clk          (clk),
            .reset        (reset),
            .acquire_en   (execute_fire),
            .write_addr   (math_req_tag),
            .write_data   (math_hdr_store),
            .read_data    (math_hdr),
            .read_addr    (math_rsp_tag),
            .release_en   (math_rsp_fire),
            .full         (mdata_full),
            `UNUSED_PIN (empty)
        );

        // submit math request
        assign math_req_valid = per_block_execute_if[block_idx].valid && ~mdata_full;
        assign per_block_execute_if[block_idx].ready = math_req_ready && ~mdata_full;

        wire pe_enable;
        wire [NUM_LANES-1:0] pe_mask;
        wire [NUM_LANES-1:0][31:0] pe_data_in;
        wire [NUM_LANES-1:0][31:0] pe_data_out;
        wire [INST_FPU_BITS-1:0] pe_shared;
        wire [NUM_LANES-1:0] mask_out;
        wire [NUM_LANES-1:0][31:0] lane_result;

        // NUM_LANES == NUM_PES: full-throughput passthrough serializer; the
        // LATENCY-deep shift register aligns valid/mask/tag with the
        // fixed-depth VX_math_lane pipeline.
        //
        // PE input register stage: lane_enable registers pe_enable before it
        // fans out to the lane's ~2300 enable-gated stage muxes — the
        // serializer drives pe_enable combinationally from its output
        // backpressure, and ABC does not buffer high-fanout port nets, so
        // the unregistered net collapsed slew in post-synth STA
        // (math-sfu-002 V4b R1). enable/mask/shared/data are registered
        // together on the same edge so their relative alignment is
        // preserved; the extra cycle is absorbed by VX_CFG_MATH_LATENCY
        // (serializer shift register depth). Registering only enable (data
        // left combinational) is WRONG: the lane would sample the next
        // packet's operand one cycle late — caught by rtlsim regression.
        VX_pe_serializer #(
            .NUM_LANES      (NUM_LANES),
            .NUM_PES        (NUM_LANES),
            .LATENCY        (`VX_CFG_MATH_LATENCY),
            .DATA_IN_WIDTH  (32),
            .DATA_OUT_WIDTH (32),
            .SHARED_WIDTH   (INST_FPU_BITS),
            .TAG_WIDTH      (TAG_WIDTH),
            .PE_REG         (0),
            .OUT_BUF        (2)
        ) pe_ser (
            .clk           (clk),
            .reset         (reset),
            .valid_in      (math_req_valid),
            .mask_in       (per_block_execute_if[block_idx].data.header.tmask),
            .data_in       (per_block_execute_if[block_idx].data.rs1_data),
            .shared_in     (per_block_execute_if[block_idx].data.op_type),
            .tag_in        (math_req_tag),
            .ready_in      (math_req_ready),
            .pe_enable     (pe_enable),
            .pe_mask_out   (pe_mask),
            .pe_data_out   (pe_data_in),
            .pe_shared_out (pe_shared),
            .pe_data_in    (pe_data_out),
            .valid_out     (math_rsp_valid),
            .mask_out      (mask_out),
            .data_out      (lane_result),
            .tag_out       (math_rsp_tag),
            .ready_out     (math_rsp_ready)
        );

        reg                     lane_enable;
        reg [NUM_LANES-1:0]     lane_mask;
        reg [INST_FPU_BITS-1:0] lane_shared;
        reg [NUM_LANES-1:0][31:0] lane_data;

        always @(posedge clk) begin
            if (reset) begin
                lane_enable <= 1'b0;
            end else begin
                lane_enable <= pe_enable;
            end
        end

        always @(posedge clk) begin
            lane_mask   <= pe_mask;
            lane_shared <= pe_shared;
            lane_data   <= pe_data_in;
        end

        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lanes
            VX_math_lane math_lane (
                .clk     (clk),
                .reset   (reset),
                .enable  (lane_enable),
                .mask    (lane_mask[i]),
                .op_type (lane_shared),
                .dataa   (lane_data[i]),
                .result  (pe_data_out[i])
            );
        end
        `UNUSED_VAR (mask_out)

        // box the f32 result into the XLEN-wide writeback word
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_pack
        `ifdef VX_CFG_FLEN_64
            // NaN-boxed for FLEN > 32
            assign math_rsp_result[i] = {32'hffffffff, lane_result[i]};
        `else
            assign math_rsp_result[i] = `VX_CFG_XLEN'(lane_result[i]);
        `endif
        end

        // send response
        VX_elastic_buffer #(
            .DATAW ($bits(math_header_t) + (NUM_LANES * `VX_CFG_XLEN)),
            .SIZE  (0)
        ) rsp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (math_rsp_valid),
            .ready_in  (math_rsp_ready),
            .data_in   ({math_hdr_wb, math_rsp_result}),
            .data_out  (per_block_result_if[block_idx].data),
            .valid_out (per_block_result_if[block_idx].valid),
            .ready_out (per_block_result_if[block_idx].ready)
        );
    end

    VX_lane_gather #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) lane_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule
