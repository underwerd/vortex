// Packed BF16 (bfloat16) arithmetic: lane-wise mul and add on two
// bf16 values packed per 32-bit word (lane 0 in [15:0], lane 1 in [31:16]).
// bf16 = 1 sign + 8 exponent (bias 127) + 7 mantissa bits.
// Single-step round-to-nearest-even, gradual denormals, IEEE-754 style
// special-case adjudication (see tests/cocotb/packbf16_unit/golden_model.py
// for the bit-exact reference this implementation is verified against).

`include "VX_define.vh"

module VX_packbf16_arith #(
    /* verilator lint_off UNUSEDPARAM */
    parameter `STRING INSTANCE_ID = "",
    /* verilator lint_on UNUSEDPARAM */
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
/* verilator lint_on DECLFILENAME */

// Single bf16 operation: mul or add with single-step RNE rounding.
//
// Formulation: every finite non-zero input is decomposed to an exact
// integer significand and power-of-two exponent (value = m * 2^e); the
// operation produces an exact integer result (multiply) or a 12-bit
// aligned sum with a sticky remainder (add), and one shared encoder
// rounds that value to bf16 exactly once (RNE, gradual denormals,
// overflow to infinity).
/* verilator lint_off DECLFILENAME */
module VX_bf16_op (
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        is_add,
    output reg  [15:0] out
);

    // Decompose inputs
    wire        a_sign = a[15], b_sign = b[15];
    wire [7:0]  a_exp  = a[14:7], b_exp = b[14:7];
    wire [6:0]  a_man  = a[6:0],  b_man = b[6:0];

    wire a_zero   = (a_exp == 8'd0) && (a_man == 7'd0);
    wire b_zero   = (b_exp == 8'd0) && (b_man == 7'd0);
    wire a_inf    = (a_exp == 8'hFF) && (a_man == 7'd0);
    wire b_inf    = (b_exp == 8'hFF) && (b_man == 7'd0);
    wire a_nan    = (a_exp == 8'hFF) && (a_man != 7'd0);
    wire b_nan    = (b_exp == 8'hFF) && (b_man != 7'd0);
    wire a_denorm = (a_exp == 8'd0) && (a_man != 7'd0);
    wire b_denorm = (b_exp == 8'd0) && (b_man != 7'd0);

    // Unified exact representation: value = m * 2^e.
    // Normal: m = 1.man (8b), e = exp - 134. Denormal: m = man (7b), e = -133
    // (man * 2^-133 == 0.man * 2^-126). Both share the 2^-134 significand scale.
    wire [7:0] a_m = a_denorm ? {1'b0, a_man} : {1'b1, a_man};
    wire [7:0] b_m = b_denorm ? {1'b0, b_man} : {1'b1, b_man};
    wire signed [9:0] a_e = a_denorm ? -10'sd133
                                       : ($signed({2'b00, a_exp}) - 10'sd134);
    wire signed [9:0] b_e = b_denorm ? -10'sd133
                                       : ($signed({2'b00, b_exp}) - 10'sd134);

    // --- MULTIPLY PATH: exact 16-bit product ---
    wire [15:0]          mul_m = a_m * b_m;           // exact, [1, 65025]
    wire signed [10:0]   mul_e = a_e + b_e;           // [-268, 242]

    // --- ADD PATH: align the smaller exponent, 4 guard bits + sticky ---
    // big/sml are chosen by magnitude, so e_gap >= 0 always.
    wire a_ge_b = (a_e > b_e) || ((a_e == b_e) && (a_m >= b_m));
    wire [7:0]         big_m = a_ge_b ? a_m : b_m;
    wire [7:0]         sml_m = a_ge_b ? b_m : a_m;
    wire signed [9:0]  big_e = a_ge_b ? a_e : b_e;
    wire signed [9:0]  sml_e = a_ge_b ? b_e : a_e;
    wire               eff_sub = a_sign ^ b_sign;

    wire [9:0] e_gap   = big_e - sml_e;               // 0..254
    wire       gap_big = (e_gap > 10'd11);
    wire [3:0] gap     = gap_big ? 4'd11 : e_gap[3:0];

    wire [11:0] sml_ext = {sml_m, 4'd0};              // 12-bit aligned grid
    wire [11:0] sml_shr = sml_ext >> gap;
    // Any bit shifted out of the grid (including the whole operand when
    // gap_big) makes the aligned value inexact.
    wire lost = gap_big || (|(sml_ext & ((12'd1 << gap) - 12'd1)));

    wire [12:0] big_ext = {1'b0, big_m, 4'd0};
    // Subtraction with truncation: the true difference is
    // big_ext - sml_shr - frac with frac in (0,1) when lost; representing
    // it as (big_ext - sml_shr - 1) + (1 - frac) keeps the remainder
    // positive so a single sticky bit stays correct for RNE.
    wire [12:0] add_sum = eff_sub
        ? (lost ? (big_ext - {1'b0, sml_shr} - 13'd1)
                : (big_ext - {1'b0, sml_shr}))
        : (big_ext + {1'b0, sml_shr});
    wire add_sticky = lost;

    // big_ext/sml_shr live on a grid whose LSB weight is 2^(big_e-4):
    // the 4 appended zeros extend 4 guard bits below big_m's LSB.
    wire signed [10:0] add_e = big_e - 11'sd4;
    wire add_sign = eff_sub ? (a_ge_b ? a_sign : b_sign) : a_sign;

    // --- SHARED ENCODER: round m * 2^e to bf16 exactly once ---
    function automatic [4:0] highbit(input [16:0] m);
        highbit = 5'd0;
        for (int i = 0; i <= 16; ++i)
            if (m[i]) highbit = i[4:0];
    endfunction

    function automatic [15:0] bf16_encode(
        input [16:0]         m,          // exact significand integer
        input signed [10:0]  e,          // value = m * 2^e
        input                sign,
        input                sticky_in   // extra inexact remainder below m
    );
        /* verilator lint_off UNUSEDSIGNAL */
        reg [4:0]          hb;
        reg signed [11:0]  e_unb;        // unbiased exponent of the MSB
        reg [4:0]          s;            // normalization shift amount
        reg [7:0]          sig8;
        reg                rnd;
        reg                stk;
        reg                carry;
        reg signed [11:0]  e_fin;
        reg [10:0]         kk_full;
        reg [7:0]          kk;           // denormal right shift, 1..133
        reg [18:0]         m_ext;
        reg [18:0]         dm_sh;
        reg [6:0]          dm_floor;
        reg                rnd_d, stk_d;
        reg [7:0]          dm_rne;
        /* verilator lint_on UNUSEDSIGNAL */
    begin
        if (m == 17'd0) begin
            // Exact cancellation (x + (-x) rounds to +0 under RNE).
            bf16_encode = 16'h0000;
        end else begin
            hb    = highbit(m);
            e_unb = $signed({{6{1'b0}}, hb[4:0]}) + $signed({e[10], e});
            if (e_unb >= -12'sd126) begin
                // Normal window: keep 8 significand bits, RNE the rest.
                if (hb >= 5'd7) begin
                    s = hb - 5'd7;                   // right shift 0..9
                    if (s == 5'd0) begin
                        sig8 = m[7:0];
                        rnd  = 1'b0;
                        stk  = sticky_in;
                    end else begin
                        sig8 = m[s+7 -: 8];
                        rnd  = m[s-1];
                        stk  = sticky_in
                               || (|(m & ((17'd1 << (s-1)) - 17'd1)));
                    end
                end else begin
                    // Massive cancellation left fewer than 8 significand
                    // bits: left-normalize (exact, no rounding needed);
                    // the sticky remainder stays meaningful below m.
                    s = 5'd7 - hb;                   // left shift 1..7
                    sig8 = {1'b0, m[6:0]} << s;    // fits 8 bits exactly
                    rnd  = 1'b0;
                    stk  = sticky_in;
                end
                // RNE increment; only a wrap 0xFF -> 0x00 means the
                // mantissa carried into the exponent field.
                carry = 1'b0;
                if (rnd && (stk || sig8[0])) begin
                    sig8  = sig8 + 8'd1;
                    carry = (sig8 == 8'd0);
                end
                e_fin = e_unb + (carry ? 12'sd1 : 12'sd0);
                if (e_fin > 12'sd127) begin
                    bf16_encode = {sign, 8'hFF, 7'h00};   // overflow -> inf
                end else begin
                    // sig8 carries the implicit leading 1; only the 7
                    // stored mantissa bits go into the encoding.
                    bf16_encode = {sign, e_fin[7:0] + 8'd127, sig8[6:0]};
                end
            end else begin
                // Denormal window: integer denormal mantissa is
                // dm = m * 2^(e+133); here e <= -134 always (for multiply
                // e = -268..-134 below the normal window cutoff given
                // m >= 128, for add e = -137), so kk = -(e+133) >= 1.
                kk_full = -(e + 11'sd133);
                // Any bit above 19 shifts means the value is far below
                // half the smallest denormal; clamp to the >19 branch.
                kk      = (kk_full > 11'd19) ? 8'd20 : kk_full[7:0];
                m_ext   = {2'b00, m};
                if (kk > 8'd19) begin
                    // Value is below half of the smallest denormal and
                    // cannot tie: rounds to zero.
                    dm_floor = 7'd0;
                    rnd_d    = 1'b0;
                    stk_d    = 1'b1;
                end else begin
                    dm_sh   = m_ext >> kk;
                    dm_floor = dm_sh[6:0];
                    rnd_d    = m_ext[kk-1];
                    stk_d    = sticky_in
                               || (|(m_ext & ((19'd1 << (kk-1)) - 19'd1)));
                end
                dm_rne = {1'b0, dm_floor}
                         + ((rnd_d && (stk_d || dm_floor[0])) ? 8'd1 : 8'd0);
                if (dm_rne == 8'd128) begin
                    // Rounded up into the smallest normal.
                    bf16_encode = {sign, 8'd1, 7'd0};
                end else begin
                    bf16_encode = {sign, 8'd0, dm_rne[6:0]};
                end
            end
        end
    end
    endfunction

    wire [16:0] m_mul = {1'b0, mul_m};
    wire [16:0] m_add = {4'd0, add_sum};

    // --- SPECIAL CASES + RESULT SELECT ---
    always @(*) begin
        if (a_nan || b_nan) begin
            out = 16'h7FC0;                            // qNaN
        end else if (!is_add) begin
            // multiply
            if ((a_inf && b_zero) || (b_inf && a_zero)) begin
                out = 16'h7FC0;                        // inf * 0 -> qNaN
            end else if (a_inf || b_inf) begin
                out = {a_sign ^ b_sign, 8'hFF, 7'h00}; // inf * finite
            end else if (a_zero || b_zero) begin
                out = {a_sign ^ b_sign, 15'd0};        // signed zero product
            end else begin
                out = bf16_encode(m_mul, mul_e, a_sign ^ b_sign, 1'b0);
            end
        end else begin
            // add
            if (a_inf && b_inf) begin
                out = (a_sign == b_sign) ? {a_sign, 8'hFF, 7'h00}
                                         : 16'h7FC0;   // inf - inf -> qNaN
            end else if (a_inf) begin
                out = a;
            end else if (b_inf) begin
                out = b;
            end else if (a_zero && b_zero) begin
                out = (a_sign == b_sign) ? {a_sign, 15'd0} : 16'h0000;
            end else if (a_zero) begin
                out = b;                               // x + 0 = x
            end else if (b_zero) begin
                out = a;
            end else begin
                out = bf16_encode(m_add, add_e, add_sign, add_sticky);
            end
        end
    end
endmodule
