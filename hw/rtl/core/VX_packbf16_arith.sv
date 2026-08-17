// Packed BF16 (bfloat16) arithmetic: lane-wise mul and add on two
// bf16 values packed per 32-bit word (lane 0 in [15:0], lane 1 in [31:16]).
// bf16 = 1 sign + 8 exponent (bias 127) + 7 mantissa bits.
// Round-to-nearest-even (RNE). Special cases follow IEEE-754 spirit.

`include "VX_define.vh"

module VX_packbf16_arith #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1
) (
    input  wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] rs1,
    input  wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] rs2,
    input  wire                                     is_add,
    output reg  [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] result
);
    `UNUSED_SPARAM (INSTANCE_ID)

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lane
        wire [15:0] a_lo = rs1[i][15:0];
        wire [15:0] a_hi = rs1[i][31:16];
        wire [15:0] b_lo = rs2[i][15:0];
        wire [15:0] b_hi = rs2[i][31:16];

        wire [15:0] r_lo, r_hi;
        VX_bf16_op bf16_lo (.a(a_lo), .b(b_lo), .is_add(is_add), .out(r_lo));
        VX_bf16_op bf16_hi (.a(a_hi), .b(b_hi), .is_add(is_add), .out(r_hi));

        always @(*) begin
            result[i] = {r_hi, r_lo};
        end
    end
endmodule

// Single bf16 operation: mul or add with RNE rounding.
module VX_bf16_op (
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        is_add,
    output reg  [15:0] out
);

    // Decompose inputs
    wire a_sign = a[15], b_sign = b[15];
    wire [7:0] a_exp = a[14:7], b_exp = b[14:7];
    wire [6:0] a_man = a[6:0],   b_man = b[6:0];

    wire a_zero = (a_exp == 0) && (a_man == 0);
    wire b_zero = (b_exp == 0) && (b_man == 0);
    wire a_inf  = (a_exp == 8'hFF) && (a_man == 0);
    wire b_inf  = (b_exp == 8'hFF) && (b_man == 0);
    wire a_nan  = (a_exp == 8'hFF) && (a_man != 0);
    wire b_nan  = (b_exp == 8'hFF) && (b_man != 0);
    wire a_denorm = (a_exp == 0) && (a_man != 0);
    wire b_denorm = (b_exp == 0) && (b_man != 0);

    // Effective significand: for normals, prepend implicit 1; for denormals, no implicit 1
    wire [7:0] a_sig = a_denorm ? {1'b0, a_man} : {1'b1, a_man};
    wire [7:0] b_sig = b_denorm ? {1'b0, b_man} : {1'b1, b_man};

    // --- MULTIPLY PATH ---
    wire r_sign_mul = a_sign ^ b_sign;
    wire [15:0] exp_sum = {7'd0, a_exp} + {7'd0, b_exp} - 16'd127;
    wire [15:0] product = {8'd0, a_sig} * {8'd0, b_sig}; // up to 16 bits

    // Find leading 1 in product for normalization
    wire product_lz = ~product[15]; // 1 if leading bit is 0
    wire [15:0] product_norm = product_lz ? (product << 1) : product;
    wire [15:0] exp_mul = product_lz ? (exp_sum - 1) : exp_sum;

    // Rounding: keep top 8 bits (sign+exp+7 mantissa), round from bit 7
    wire [7:0] man_round_raw = product_norm[14:7];
    wire round_bit_mul = product_norm[6];
    wire sticky_mul = |product_norm[5:0];
    wire [7:0] man_rounded_mul = man_round_raw + {7'd0, round_bit_mul & (sticky_mul | man_round_raw[0])};

    // Check for mantissa overflow after rounding
    wire man_carry_mul = (man_rounded_mul == 8'h80); // overflow into implicit bit position
    wire [7:0] final_man_mul = man_carry_mul ? 8'h00 : man_rounded_mul[6:0];
    wire [8:0] final_exp_mul = man_carry_mul ? (exp_mul + 1) : exp_mul[8:0];

    // --- ADD PATH ---
    wire r_sign_add_raw = a_sign;
    // Determine which operand has larger magnitude for add
    wire [8:0] a_eff_exp = a_denorm ? 9'd1 : {1'b0, a_exp};
    wire [8:0] b_eff_exp = b_denorm ? 9'd1 : {1'b0, b_exp};

    wire a_gt_b = (a_eff_exp > b_eff_exp) ||
                  ((a_eff_exp == b_eff_exp) && (a_sig >= b_sig));

    wire [8:0]  big_exp  = a_gt_b ? a_eff_exp : b_eff_exp;
    wire [8:0]  sml_exp  = a_gt_b ? b_eff_exp : a_eff_exp;
    wire [7:0]  big_sig  = a_gt_b ? a_sig : b_sig;
    wire [7:0]  sml_sig  = a_gt_b ? b_sig : a_sig;
    wire        big_sign = a_gt_b ? a_sign : b_sign;
    wire        sml_sign = a_gt_b ? b_sign : a_sign;

    wire [8:0] exp_diff = big_exp - sml_exp;
    wire shift_clamped = (exp_diff > 9'd8) ? 1'b1 : 1'b0;
    wire [3:0] shift_amt = shift_clamped ? 4'd8 : exp_diff[3:0];

    // Align smaller significand
    wire [7:0] sml_aligned = big_sig >> 0; // placeholder
    wire [7:0] sml_shifted = sml_sig >> shift_amt;

    // Add or subtract based on signs
    wire do_subtract = big_sign ^ sml_sign;
    wire [8:0] sum = do_subtract ?
        ({1'b0, big_sig} - {1'b0, sml_shifted}) :
        ({1'b0, big_sig} + {1'b0, sml_shifted});

    wire r_sign_add = do_subtract ? big_sign : big_sign;
    // For subtraction where result could be zero
    wire sub_zero = do_subtract && (big_sig == sml_shifted);

    // Normalize addition result
    wire [8:0] sum_exp = big_exp;
    wire sum_overflow = sum[8]; // carry out of addition
    wire [8:0] norm_exp_add = sum_overflow ? (sum_exp + 1) : sum_exp;
    wire [7:0] norm_man_add = sum_overflow ? sum[7:1] : sum[6:0];

    // Normalize subtraction result (find leading 1)
    reg [3:0] lz_count;
    reg [7:0] norm_man_sub;
    reg [8:0] norm_exp_sub;
    always @(*) begin
        lz_count = 0;
        norm_man_sub = sum[6:0];
        norm_exp_sub = sum_exp;
        if (sum[6:0] != 0) begin
            if (!sum[6]) begin
                lz_count = 1;
                if (!sum[5]) begin lz_count = 2;
                if (!sum[4]) begin lz_count = 3;
                if (!sum[3]) begin lz_count = 4;
                if (!sum[2]) begin lz_count = 5;
                if (!sum[1]) begin lz_count = 6;
                if (!sum[0]) begin lz_count = 7;
                end end end end end end
            end
            norm_man_sub = (sum[6:0] << lz_count);
            norm_exp_sub = sum_exp - {5'd0, lz_count};
        end else begin
            norm_man_sub = 0;
            norm_exp_sub = 0;
        end
    end

    wire [8:0] final_exp_add = do_subtract ? norm_exp_sub : norm_exp_add;
    wire [6:0] final_man_add_raw = do_subtract ? norm_man_sub[6:0] : norm_man_add[6:0];

    // --- SELECT MUL or ADD result ---
    wire [7:0] r_exp  = is_add ? final_exp_add[7:0] : final_exp_mul[7:0];
    wire [6:0] r_man  = is_add ? final_man_add_raw  : final_man_mul[6:0];
    wire       r_sign = is_add ? (sub_zero ? 1'b0 : r_sign_add) : r_sign_mul;

    // --- SPECIAL CASES ---
    wire any_nan = a_nan || b_nan;
    wire inf_times_zero = is_add ? 1'b0 : ((a_inf && b_zero) || (b_inf && a_zero));
    wire inf_result = is_add ?
        (a_inf || b_inf) :
        (a_inf || b_inf || (final_exp_mul[8] && !final_exp_mul[7])); // overflow to inf

    always @(*) begin
        if (any_nan || inf_times_zero) begin
            out = 16'h7FC0; // qNaN
        end else if (inf_result) begin
            out = {r_sign, 8'hFF, 7'h00}; // infinity
        end else if (a_zero && b_zero && !is_add) begin
            out = 16'h0000; // 0 * 0 = 0
        end else if (a_zero || b_zero) begin
            if (is_add)
                out = a_zero ? b : a; // 0 + x = x
            else
                out = {r_sign, 15'd0}; // x * 0 = 0 (with sign)
        end else if (sub_zero) begin
            out = 16'h0000; // exact cancellation
        end else if (r_exp == 0 || r_exp[8]) begin
            // Underflow: denormal or zero
            out = {r_sign, 8'h00, r_man}; // simplified: flush to zero-ish
        end else if (r_exp == 8'hFF) begin
            out = {r_sign, 8'hFF, 7'h00}; // overflow to inf
        end else begin
            out = {r_sign, r_exp[7:0], r_man};
        end
    end
endmodule
