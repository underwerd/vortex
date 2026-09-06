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

// Math-SFU lane datapath for ex2.f32 / tanh.f32 / sigmoid.f32.
//
// Bit-exact RTL image of the executable specification
// tests/cocotb/math_sfu_unit/math_rtl_model.py (which validates against the
// frozen approx_scheme over the full domain):
//
//   ex2.f32    : x = n + f split (n = rne(x), f in [-0.5, 0.5]); degree-4
//                2^f polynomial; exponent-field rebuild. x >= 128 -> +Inf,
//                x < -126 -> 0 (FTZ).
//   tanh.f32   : odd symmetry; |x| < 2^-7 pass-through; |x| >= 5.25 -> +-1.0;
//                segment 1 a*P1(a^2) (degree 4 in u = a^2) on [2^-7, 1);
//                segment 2 degree-10 polynomial on the centered variable
//                v = a*S + B in [-1, 1) — an exact rebasing of the frozen
//                segment-2 polynomial (the power basis on a amplifies s2.24
//                coefficient quantization by a^10 ~ 2^24 at the segment
//                edge; |v|^k <= 1 keeps it at 2^-25).
//   sigmoid.f32: x >= 10.5 -> 1.0; x >= 0 -> (1 + tanh(x/2)) / 2 reusing the
//                tanh segments on x/2; x < 0 -> u = 2^(x*log2e) via the ex2
//                polynomial, normalized to u = mu * 2^-ku with mu in
//                [0.5, 1), then a degree-7 1/(1+u) Horner with dynamic
//                right shift (Q + ku), output mu * p7 through a
//                normalizing packer.
//
// All coefficients are s2.24 (26-bit signed, RNE from the float64 frozen
// scheme). Horner accumulators are s4.24 (28-bit signed); products are
// truncated (shift right, no rounding) before the coefficient add, matching
// math_rtl_model.horner_trunc.
//
// Pipeline (34 stages, throughput 1 warp per cycle from the serializer's
// viewpoint; VX_CFG_MATH_LATENCY = lane stages + the math_unit PE input
// register = 35). Stage names below keep the pre-D17 numbering; since the
// S2 pre-multiply split (D17) and the tail-multiplier split (D19) each
// later name physically lands one or two cycles later than its number
// suggests:
//   S1  classify + f32->Q24 barrel shift + special decision
//   S2  pre-multiply (u = a^2 / v = a*S+B / y = x*log2e), two cycles:
//       a multiply half that registers the three products, then a
//       truncate/add/mux half
//   S3  rne integer split + sigmoid-negative underflow gate + array load k1
//   S4..S27  shared Horner array (load / multiply-add / bypass per mode;
//            every k step is two cycles: a multiply half that registers the
//            acc*px product, then an add half that shifts + adds the
//            coefficient; the k6 add half renormalizes (mu, ku) for the
//            sigmoid negative half and loads the p7 top coefficient)
//   S28 finalize tail (final products + f32-free ex2/tanh-p2 tails)
//   S29 packer front: leading-bit encoder only
//   S30 normalizer: normalization shift/mask + biased exponent
//   S31 packer back: RNE + assembly + t_mag select
//   S32 sigmoid (1+t)/2 fold + result mux + output
// The finalize used to be one stage; its ~8.3 ns of gate depth (tail
// multiplier -> normalizer -> fold in series) broke the 400 MHz target,
// so it is split across S28..S32 (the normalizer itself split after the
// fifth V4b run left ~4.4 ns inside it, the encoder/shift pair split again
// after the eighth run, and the Horner multiply got its own half-cycle
// after the ninth run still chained ~4.3 ns through one stage).
//
// Clock-enable scheme: every stage's registers are gated by `enable`.
// The stall semantics require it: the serializer's valid/mask/tag shift
// register freezes on a stall, so the lane data stages must freeze in
// lockstep or valid and data would drift apart. Two alternatives were
// tried and rejected in math-sfu-002: (a) free-running S2..S16 breaks the
// valid/data alignment on stalls; (b) duplicating the enable register per
// lane does not survive synthesis (yosys opt_merge / ABC structural
// hashing merge the replicas back into one net). The resulting ~2300-load
// enable net slew-collapses under ABC mapping, which inserts no fanout
// buffers; the repair lives at the netlist level instead — the harness
// synthesis backend post-processes the mapped netlist and inserts BUF_X
// trees on high-fanout nets (see project_runner/adapters/synthesis/
// fanout_buffer.py). Do not remove the stage gating, and do not move the
// buffering into this RTL.
//
// Array stage occupancy (k = 1..13 maps to S3..S15):
//   M_EX2 : k1 load c4, k2-k5 ma c3..c0, k6-k13 bypass
//   M_P1  : k1 load c4, k2-k5 ma c3..c0, k6-k13 bypass
//   M_P2  : k1-k2 bypass, k3 load c10, k4-k13 ma c9..c0
//   M_P7  : k1 load EX2 c4, k2-k5 ma EX2 c3..c0 (ex2 polynomial),
//           k6 renormalize + load p7 c7, k7-k13 ma p7 c6..c0 (shift Q+ku)

`include "VX_fpu_define.vh"

module VX_math_lane import VX_gpu_pkg::*; (
    input  wire clk,
    input  wire reset,

    input  wire enable,
    input  wire mask,

    input  wire [INST_FPU_BITS-1:0] op_type, // INST_FPU_EX2/TANH/SIGMOID
    input  wire [31:0]             dataa,    // f32 operand (rs1)

    output wire [31:0]             result
);
    `UNUSED_VAR (reset)

    `UNUSED_VAR (mask)

    ///////////////////////////////////////////////////////////////////////////
    // frozen-scheme coefficients, s2.24 (single source: math_rtl_model.py)
    ///////////////////////////////////////////////////////////////////////////

    // ex2: 2^f polynomial, degree 4 (index = coefficient of f^k)
    localparam logic signed [4:0][25:0] EX2_C = {
        26'h00279C8,    // c4 =  0.00967076
        26'h00E4DDB,    // c3 =  0.05587550
        26'h03D7F32,    // c2 =  0.24022212
        26'h0B170CA,    // c1 =  0.69312727
        26'h1000001     // c0 =  1.00000005
    };

    // tanh segment 1: P1(u), u = a^2, degree 4
    localparam logic signed [4:0][25:0] P1_C = {
        26'h0026CCB,    // c4 =  0.00947258
        26'h3F452BE,    // c3 = -0.04561245
        26'h0217D10,    // c2 =  0.13081455
        26'h3AABC9E,    // c1 = -0.33305943
        26'h0FFFFB0     // c0 =  0.99999521
    };

    // tanh segment 2: P2(v), v = a*S + B in [-1, 1), degree 10 (rebased)
    localparam logic signed [10:0][25:0] P2_C = {
        26'h001B8E8,    // c10 =  0.00672768
        26'h3FDF102,    // c9  = -0.00804126
        26'h3FD2CC6,    // c8  = -0.01103555
        26'h0060A75,    // c7  =  0.02359706
        26'h3F996C8,    // c6  = -0.02504302
        26'h009BB59,    // c5  =  0.03801497
        26'h3F2D29E,    // c4  = -0.05147374
        26'h00C9F06,    // c3  =  0.04930150
        26'h3F72779,    // c2  = -0.03455396
        26'h0042D34,    // c1  =  0.01631474
        26'h0FF036C     // c0  =  0.99614594
    };

    // sigmoid negative half: 1/(1+u) polynomial, degree 7
    localparam logic signed [7:0][25:0] P7_C = {
        26'h3F33938,    // c7 = -0.04990815
        26'h03F550C,    // c6 =  0.24739144
        26'h36F1EDD,    // c5 = -0.56593529
        26'h0D5C2DD,    // c4 =  0.83500463
        26'h309B142,    // c3 = -0.96213903
        26'h0FECE9A,    // c2 =  0.99534002
        26'h3001036,    // c1 = -0.99975262
        26'h0FFFFC9     // c0 =  0.99999675
    };

    localparam logic signed [25:0] LOG2E_C = 26'h1715476;    // 1.44269504
    localparam logic signed [25:0] SEG2_S   = 26'h0787878;    //  0.47058824 (1/2.125)
    localparam logic signed [25:0] SEG2_B   = 26'h2878788;    // -1.47058824 (-3.125/2.125)

    // array modes / instruction codes / special classes
    localparam logic [1:0] M_EX2 = 2'd0;
    localparam logic [1:0] M_P1  = 2'd1;
    localparam logic [1:0] M_P2  = 2'd2;
    localparam logic [1:0] M_P7  = 2'd3;

    localparam logic [1:0] I_EX2  = 2'd0;
    localparam logic [1:0] I_TANH = 2'd1;
    localparam logic [1:0] I_SIG  = 2'd2;

    localparam logic [1:0] SP_NONE   = 2'd0;   // evaluate in the array
    localparam logic [1:0] SP_BYPASS = 2'd1;   // fold t through the tail
    localparam logic [1:0] SP_CONST  = 2'd2;   // precomputed constant result

    localparam logic [31:0] F32_NAN  = 32'h7FC00000;
    localparam logic [31:0] F32_PINF = 32'h7F800000;
    localparam logic [31:0] F32_PONE = 32'h3F800000;
    localparam logic [31:0] F32_NONE = 32'hBF800000;

    // per-lane control riding the pipeline from S3 to the tail stage
    typedef struct packed {
        logic [1:0]         instr;
        logic               sign;
        logic [1:0]         mode;
        logic [1:0]         special;
        logic [31:0]        const_bits;
        logic [31:0]        bypass_bits; // t magnitude (f32 bits) for SP_BYPASS
        logic signed [27:0] a_q;         // |x| or x/2 quantized, P1 tail
        logic signed [8:0]  n;           // ex2 integer split, ex2 tail
        logic [24:0]        mu;          // sigmoid-negative significand (Q24)
        logic signed [7:0]  ku;          // sigmoid-negative scale exponent
    } math_ctrl_t;

    math_ctrl_t ctrl_r [12:0];          // S3 + k add halves (index 0 = S3)

    // D20: dedicated tail-multiplier operand copies, written on the same
    // edge as ctrl_r[12] (k13 add half). The struct-packed a_q/mu bits
    // collapsed slew driving the multiplier partial products (clk-to-Q
    // 0.20 ns, path ctrl_r[12] -> p1_prod_r still -0.51 ns in the
    // fifteenth run); dedicated FFs reproduce the passing Horner
    // multiply-half shape (plain register operand -> multiplier).
    logic signed [27:0] a_q_fin_r;
    logic [24:0] mu_fin_r;
    logic signed [27:0] acc_r    [12:0]; // Horner accumulators, k1..k13 out
    logic signed [26:0] poly_x_r [11:0];  // Horner multiplicand, k1..k12 out

    // Horner multiply-half registers (D14): every k step registers the
    // acc*px product plus its sideband, then shifts + adds in the second
    // half. The one-stage multiply->shift->add chain measured ~4.3 ns
    // post-synth (math-sfu-002 V4b, ninth run); the product register
    // breaks it and costs one extra cycle per k step. The ctrl sideband
    // rides m_ctrl so it advances two cycles per k step in lockstep with
    // the data (chaining ctrl_r through the add half alone ran it at one
    // cycle per step and drifted it off the data by one warp per stage).
    logic signed [53:0] m_prod  [13:1];
    logic signed [27:0] m_acc   [13:1];
    logic signed [26:0] m_px    [13:1];
    logic signed [25:0] m_coeff [13:1];
    logic [7:0]         m_shamt [13:1];
    logic signed [7:0]  m_ku    [13:1];
    logic [2:0]         m_flags [13:1];  // {ld, ma, is_norm}
    math_ctrl_t         m_ctrl  [13:1];

    ///////////////////////////////////////////////////////////////////////////
    // S1: classification, f32 -> Q24 barrel shift, special decision
    ///////////////////////////////////////////////////////////////////////////

    logic [1:0] s1_instr;
    logic [1:0] s1_mode;
    logic [1:0] s1_special;
    logic [31:0] s1_const, s1_bypass;
    logic s1_sign;

    wire [7:0] xexp = dataa[30:23];
    wire [22:0] xfrac = dataa[22:0];
    wire is_nan = (xexp == 8'hFF) && (xfrac != 0);
    wire is_inf = (xexp == 8'hFF) && (xfrac == 0);

    assign s1_sign = dataa[31];

    // sigmoid positive half evaluates tanh(x/2): halve the exponent exactly.
    wire sel_half = (op_type == INST_FPU_SIGMOID) && ~dataa[31];
    wire [7:0] mag_exp = (sel_half && (xexp == 0)) ? 8'd0
                       : sel_half ? (xexp - 8'd1)
                       : xexp;
    wire [22:0] mag_frac = xfrac;

    always @(*) begin
        s1_instr   = I_EX2;
        s1_mode    = M_EX2;
        s1_special = SP_NONE;
        s1_const   = F32_NAN;
        s1_bypass  = dataa;
        if (op_type == INST_FPU_EX2) begin
            s1_instr = I_EX2;
            if (is_nan) begin
                s1_special = SP_CONST; s1_const = F32_NAN;
            end else if (is_inf) begin
                s1_special = SP_CONST; s1_const = s1_sign ? 32'h00000000 : F32_PINF; // 2^-Inf = 0
            end else if (s1_sign && ((xexp >= 8'd134) || ((xexp == 8'd133) && (xfrac > 23'h7C0000)))) begin
                s1_special = SP_CONST; s1_const = 32'h00000000;                      // x < -126 (FTZ)
            end else if (xexp >= 8'd134) begin
                s1_special = SP_CONST; s1_const = F32_PINF;                          // x >= 128
            end else begin
                s1_mode = M_EX2;
            end
        end else if (op_type == INST_FPU_TANH) begin
            s1_instr = I_TANH;
            if (is_nan) begin
                s1_special = SP_CONST; s1_const = F32_NAN;
            end else if (is_inf) begin
                s1_special = SP_CONST; s1_const = s1_sign ? F32_NONE : F32_PONE;
            end else if (xexp < 8'd120) begin
                s1_special = SP_BYPASS; s1_bypass = dataa;                           // |x| < 2^-7
            end else if ((xexp > 8'd129) || ((xexp == 8'd129) && (xfrac >= 23'h280000))) begin
                s1_special = SP_CONST; s1_const = s1_sign ? F32_NONE : F32_PONE;     // |x| >= 5.25
            end else begin
                s1_mode = (xexp < 8'd127) ? M_P1 : M_P2;
            end
        end else begin // INST_FPU_SIGMOID
            s1_instr = I_SIG;
            if (is_nan) begin
                s1_special = SP_CONST; s1_const = F32_NAN;
            end else if (is_inf) begin
                s1_special = SP_CONST; s1_const = s1_sign ? 32'h00000000 : F32_PONE;
            end else if (~s1_sign && ((xexp > 8'd130) || ((xexp == 8'd130) && (xfrac >= 23'h280000)))) begin
                s1_special = SP_BYPASS; s1_bypass = F32_PONE;                        // x >= 10.5 -> (1+1)/2
            end else if (~s1_sign && (xexp < 8'd121)) begin
                s1_special = SP_BYPASS; s1_bypass = {1'b0, mag_exp, mag_frac};       // x/2 < 2^-7
            end else if (s1_sign && (xexp >= 8'd134)) begin
                s1_special = SP_CONST; s1_const = 32'h00000000;                      // x <= -128: u underflows
            end else begin
                s1_mode = s1_sign ? M_P7 : ((mag_exp < 8'd127) ? M_P1 : M_P2);
            end
        end
    end

    // barrel shift: |operand| -> Q24 (mant << (exp-126)); exp = 0 flushes to 0.
    wire [24:0] s1_mant = {1'b0, 1'b1, mag_frac};
    wire signed [8:0] s1_shamt = $signed({1'b0, mag_exp}) - 9'sd126; // [-126, 7] after gates
    wire [7:0] s1_shr = ~s1_shamt[7:0] + 8'd1;                       // -shamt when negative
    wire [32:0] s1_mag_r = {8'd0, s1_mant} >> s1_shr;
    wire [32:0] s1_mag_l = {8'd0, s1_mant} << s1_shamt[2:0];
    wire [32:0] s1_mag = (mag_exp == 0) ? 33'd0
                       : s1_shamt[8] ? s1_mag_r
                       : s1_mag_l;
    wire signed [32:0] s1_xq = s1_sign ? -$signed(s1_mag) : $signed(s1_mag);

    // S1 registers (valid for the S2 comb stage of the same warp)
    logic [1:0] instr_r;
    logic [1:0] mode_r;
    logic [1:0] special_r;
    logic [31:0] const_r, bypass_r;
    logic sign_r;
    logic signed [32:0] xq_r;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [32:0] mag_r;   // D18: |x| rides alongside xq so S2 multipliers
                          // take a plain registered operand (no negate)
    /* verilator lint_on UNUSEDSIGNAL */

    always @(posedge clk) begin
        if (enable) begin
            instr_r   <= s1_instr;
            mode_r    <= s1_mode;
            special_r <= s1_special;
            const_r   <= s1_const;
            bypass_r  <= s1_bypass;
            sign_r    <= s1_sign;
            xq_r      <= s1_xq;
            mag_r     <= s1_mag;
        end
    end

    // S2 sideband registers: S3 must observe the warp's S1 classification
    // one cycle later, after xq_r & co. have been overwritten by the next
    // warp — hence this second pipeline register.
    logic [1:0] instr_s2, mode_s2, special_s2;
    logic [31:0] const_s2, bypass_s2;
    logic sign_s2;
    logic signed [32:0] xq_s2;

    ///////////////////////////////////////////////////////////////////////////
    // S2: pre-multiplication (mode-dependent)
    ///////////////////////////////////////////////////////////////////////////

    // D18: a_q used to strip the sign here (sign_r ? -xq_r : xq_r), which
    // chained a 28-bit negate carry into the a^2 multiplier and still
    // measured -0.52 ns xq_r -> u_prod_r (math-sfu-002 V4b, thirteenth
    // run). |x| is now registered in S1 (mag_r), leaving the multiplier
    // on plain register outputs like the passing Horner multiply halves.
    wire signed [27:0] a_q = $signed(mag_r[27:0]);   // |x|, s4.24 (range-gated)

    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [55:0] u_prod = a_q * a_q;                             // M_P1: a^2
    wire signed [53:0] v_prod = a_q * SEG2_S;                          // M_P2: a*S
    wire signed [58:0] y_prod = $signed(xq_r) * LOG2E_C;               // M_P7: x*log2e
    /* verilator lint_on UNUSEDSIGNAL */

    // D17: the three pre-multipliers + truncates + mode mux still chained
    // 3.11 ns in one cycle (math-sfu-002 V4b, twelfth run, path
    // xq_r -> pre_x_r), so the products get their own register set and
    // the truncate/add/mux moves to the second half. Sidebands ride one
    // extra bank (s2a -> s2) so S3 still sees everything on one warp.
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [55:0] u_prod_r;
    logic signed [53:0] v_prod_r;
    logic signed [58:0] y_prod_r;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [1:0] instr_s2a, mode_s2a, special_s2a;
    logic [31:0] const_s2a, bypass_s2a;
    logic sign_s2a;
    logic signed [32:0] xq_s2a;

    always @(posedge clk) begin
        if (enable) begin
            u_prod_r  <= u_prod;
            v_prod_r  <= v_prod;
            y_prod_r  <= y_prod;
            instr_s2a <= instr_r;
            mode_s2a  <= mode_r;
            special_s2a <= special_r;
            const_s2a <= const_r;
            bypass_s2a <= bypass_r;
            sign_s2a  <= sign_r;
            xq_s2a    <= xq_r;
        end
    end

    wire [25:0] u_next = u_prod_r[49:24];
    wire signed [26:0] v_sum = v_prod_r[49:24] + 27'(SEG2_B);
    wire signed [32:0] y_next = y_prod_r[56:24];

    wire signed [32:0] pre_x = (mode_s2a == M_P1) ? {7'd0, u_next}
                          : (mode_s2a == M_P2) ? {v_sum[26], v_sum[26], v_sum[26],
                                                  v_sum[26], v_sum[26], v_sum[26], v_sum}
                          : (mode_s2a == M_P7) ? y_next
                          : xq_s2a;

    logic signed [32:0] pre_x_r;

    always @(posedge clk) begin
        if (enable) begin
            pre_x_r  <= pre_x;
            instr_s2 <= instr_s2a;
            mode_s2  <= mode_s2a;
            special_s2 <= special_s2a;
            const_s2 <= const_s2a;
            bypass_s2 <= bypass_s2a;
            sign_s2  <= sign_s2a;
            xq_s2    <= xq_s2a;
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // S3: rne integer split + underflow gate + array load (k1)
    ///////////////////////////////////////////////////////////////////////////

    wire signed [32:0] rne_in = (mode_s2 == M_EX2) ? xq_s2 : pre_x_r;

    wire signed [8:0] n_floor = rne_in[32:24];
    wire [23:0] rne_frac = rne_in[23:0];
    wire rne_up = (rne_frac > 24'h800000)
               || ((rne_frac == 24'h800000) && n_floor[0]);            // ties to even
    wire signed [8:0] n_comb = n_floor + (rne_up ? 9'sd1 : 9'sd0);
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [32:0] f_comb = rne_in - (n_comb <<< 24);
    /* verilator lint_on UNUSEDSIGNAL */

    // sigmoid negative half: u = 2^y underflows (FTZ per the frozen scheme)
    wire y_ftz = (mode_s2 == M_P7) && ($signed(pre_x_r) <= -33'sd2113929216); // y <= -126

    wire [1:0] s3_special = y_ftz ? SP_CONST : special_s2;
    wire [31:0] s3_const = y_ftz ? 32'h00000000 : const_s2;

    wire signed [26:0] poly_x_init = ((mode_s2 == M_EX2) || (mode_s2 == M_P7)) ? f_comb[26:0]
                                                                        : pre_x_r[26:0];

    wire signed [27:0] k1_acc = (mode_s2 == M_P1) ? 28'(P1_C[4]) : 28'(EX2_C[4]);

    always @(posedge clk) begin
        if (enable) begin
            ctrl_r[0] <= '{
                instr: instr_s2,
                sign: sign_s2,
                mode: mode_s2,
                special: s3_special,
                const_bits: s3_const,
                bypass_bits: bypass_s2,
                a_q: sign_s2 ? -xq_s2[27:0] : xq_s2[27:0],
                n: n_comb,
                mu: 25'd0,
                ku: -n_comb[7:0]
            };
            acc_r[0] <= k1_acc;
            poly_x_r[0] <= poly_x_init;
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // Horner array (k2..k13), two cycles per k step (D14 multiply split)
    ///////////////////////////////////////////////////////////////////////////

    for (genvar k = 2; k <= 13; ++k) begin : g_stage
        wire signed [27:0] acc_in = acc_r[k-2];
        wire signed [26:0] px_in = poly_x_r[k-2];
        wire [1:0] mode = ctrl_r[k-2].mode;
        wire signed [7:0] ku_in = ctrl_r[k-2].ku;

        // stage role
        wire ld_k = ((k == 3) && (mode == M_P2))               // P2 top load
                 || ((k == 6) && (mode == M_P7));              // p7 top load (renorm)
        wire ma_k = ((k <= 5) && ((mode == M_EX2) || (mode == M_P1) || (mode == M_P7)))
                 || ((k >= 4) && (mode == M_P2))
                 || ((k >= 7) && (mode == M_P7));

        // stage coefficient
        logic signed [25:0] coeff;
        always @(*) begin
            coeff = 26'sd0;
            if (ld_k) begin
                if (k == 3) begin
                    coeff = P2_C[10];
                end else begin
                    coeff = P7_C[7];
                end
            end else if (ma_k) begin
                /* verilator lint_off SELRANGE */
                // the literal out-of-range selects below sit in per-k dead
                // arms (e.g. EX2_C[5-k] with k=2 never evaluates: EX2 lanes
                // only multiply at k4..k5); kept for the one-line scheme
                case (mode)
                    M_EX2: coeff = EX2_C[5-k];
                    M_P1:  coeff = P1_C[5-k];
                    M_P2:  coeff = P2_C[13-k];
                    default: coeff = (k <= 5) ? EX2_C[5-k] : P7_C[13-k];
                endcase
                /* verilator lint_on SELRANGE */
            end
        end

        // multiply-truncate-add, split at the product register: the
        // multiply half registers prod + sideband, the add half shifts
        // and adds (shift is dynamic only for the p7 stages)
        wire [7:0] shamt_k = ((mode == M_P7) && (k >= 7)) ? (8'd24 + ku_in) : 8'd24;
        wire is_norm_k = (k == 6) && (mode == M_P7);
        wire signed [53:0] prod_k = $signed(acc_in) * $signed(px_in);

        always @(posedge clk) begin
            if (enable) begin
                m_prod[k]  <= prod_k;
                m_acc[k]   <= acc_in;
                m_px[k]    <= px_in;
                m_coeff[k] <= coeff;
                m_shamt[k] <= shamt_k;
                m_flags[k] <= {ld_k, ma_k, is_norm_k};
                m_ku[k]    <= ku_in;
                m_ctrl[k]  <= ctrl_r[k-2];
            end
        end

        wire m_ld   = m_flags[k][2];
        wire m_ma   = m_flags[k][1];
        wire m_norm = m_flags[k][0];
        /* verilator lint_off UNUSEDSIGNAL */
        wire signed [53:0] shifted_k = m_prod[k] >>> m_shamt[k];
        /* verilator lint_on UNUSEDSIGNAL */
        wire signed [27:0] acc_next = m_ld ? 28'(m_coeff[k])
                                      : m_ma ? (28'(shifted_k) + 28'(m_coeff[k]))
                                      : m_acc[k];

        // sigmoid-negative renormalization at k6: m in [0.707, 1.414] -> (mu, ku)
        wire m_ge1 = m_acc[k][24];
        wire [24:0] mu_comb = m_ge1 ? (m_acc[k][24:0] >> 1) : m_acc[k][24:0]; // mu = m/2 or m
        wire signed [7:0] ku_fix = m_ge1 ? (m_ku[k] - 8'sd1) : m_ku[k];

        wire signed [26:0] px_next = m_norm ? $signed({2'b00, mu_comb}) : m_px[k];

        always @(posedge clk) begin
            if (enable) begin
                acc_r[k-1] <= acc_next;
                if (k < 13) begin
                    poly_x_r[k-1] <= px_next;
                end
                ctrl_r[k-1] <= m_ctrl[k];
                if (m_norm) begin
                    ctrl_r[k-1].mu <= mu_comb;
                    ctrl_r[k-1].ku <= ku_fix;
                end
                if (k == 13) begin
                    a_q_fin_r <= m_ctrl[k].a_q;   // D20, == ctrl_r[12].a_q
                    mu_fin_r  <= m_ctrl[k].mu;    // D20, == ctrl_r[12].mu
                end
            end
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // S28..S32: finalize, split into five stages
    //
    // A single-stage finalize (tail multiplier -> f32_pack normalizer ->
    // sigmoid fold -> result mux) measures ~8.3 ns of gate depth at 400
    // MHz post-synth (math-sfu-002 V4b, third run, after the fanout
    // buffer tree removed the slew collapse), so the stage is split:
    //   S28 tail: final products + the f32-free ex2/tanh-p2 tails
    //   S29 pack front: leading-bit encoder only
    //   S30 normalizer: e0-controlled shift/mask + biased exponent
    //   S31 pack back: RNE + assembly + t_mag select
    //   S32 fold: sigmoid (1+t)/2 + result mux
    // Each stage keeps the enable gating; the extra cycles are
    // absorbed by VX_CFG_MATH_LATENCY.
    ///////////////////////////////////////////////////////////////////////////

    wire [1:0] fin_instr = ctrl_r[12].instr;
    wire fin_sign = ctrl_r[12].sign;
    wire [1:0] fin_mode = ctrl_r[12].mode;
    wire [1:0] fin_special = ctrl_r[12].special;
    wire [31:0] fin_const = ctrl_r[12].const_bits;
    wire [31:0] fin_bypass = ctrl_r[12].bypass_bits;
    wire signed [27:0] fin_a_q = a_q_fin_r;   // D20: dedicated copy (was ctrl_r[12].a_q)
    wire signed [8:0] fin_n = ctrl_r[12].n;
    wire [24:0] fin_mu = mu_fin_r;            // D20: dedicated copy (was ctrl_r[12].mu)
    wire signed [7:0] fin_ku = ctrl_r[12].ku;
    wire signed [27:0] poly_out = acc_r[12];

    // normalizing f32 packer: value = mant * 2^w, mant unsigned nonzero.
    // Rounds the 24-bit significand to f32 (RNE); Inf on overflow, FTZ below
    // the normal range (math_rtl_model.pack_f32).
    // The leading-bit position e is resolved by a fully parallel one-hot
    // encoder: lead[i] = mant[i] & ~(any higher bit set), then e is the
    // one-hot OR-sum of (i+1) (both tree-able by synthesis). The original
    // sequential priority loop synthesized to a 56-bit serial chain (post-
    // synth WNS -3.10ns, ltp length 185; math-sfu-002 V4b fourth run).
    // Same function, bit-exact.
    function automatic [31:0] f32_pack(input [55:0] mant, input signed [8:0] w);
        integer drop;
        logic [55:0] rem, half;
        logic [24:0] hi;
        logic signed [8:0] e;
        logic signed [8:0] expu;
        logic signed [9:0] biased;
        logic [55:0] lead;
        integer i;
        begin
            // each lead[i] reads mant only (no inter-lead dependency), so
            // synthesis builds the higher-bit reductions as trees
            lead[55] = mant[55];
            for (i = 54; i >= 0; --i) begin
                lead[i] = mant[i] & ~(|(mant >> (i + 1)));
            end
            // e = index of the topmost set bit + 1 (0 when mant == 0):
            // one-hot OR-sum of (i+1) over lead[i]
            e = 9'sd0;
            for (i = 0; i <= 55; ++i) begin
                e = e | (lead[i] ? 9'(i + 1) : 9'sd0);
            end
            if (e == 0) begin
                f32_pack = 32'h00000000;                              // mant == 0
            end else begin
                if (e > 24) begin
                    /* verilator lint_off WIDTHEXPAND */
                    drop = e - 24;
                    /* verilator lint_on WIDTHEXPAND */
                    hi = 25'(mant >> drop);                          // top 24 bits
                    rem = mant & ((56'd1 << drop) - 56'd1);
                    half = 56'd1 << (drop - 1);
                    if ((rem > half) || ((rem == half) && hi[0])) begin
                        hi = hi + 25'd1;
                        if (hi == 25'd16777216) begin                 // carry: 2^24
                            hi = 25'd8388608;                         // -> 2^23
                            e = e + 1;
                        end
                    end
                end else begin
                    hi = 25'(mant << (24 - e));                       // exact left fill
                end
                expu = w + 9'(e - 1);
                biased = $signed(expu) + 10'sd127;
                if (biased >= 10'sd255) begin
                    f32_pack = F32_PINF;
                end else if (biased <= 10'sd0) begin
                    f32_pack = 32'h00000000;                          // FTZ (exempt domain)
                end else begin
                    f32_pack = {1'b0, biased[7:0], hi[22:0]};
                end
            end
        end
    endfunction

    // shared packer input: P1 tail (a * p1, w = -48) or p7 tail (mu * p7,
    // w = -48 - ku); the pack mux itself sits on the fin2p cycle (D19)
    wire signed [55:0] p1_prod = $signed(fin_a_q) * $signed(poly_out);
    wire signed [52:0] p7_prod = $signed({1'b0, fin_mu}) * $signed(poly_out);
    wire signed [8:0] pack_w = (fin_mode == M_P7)
                             ? (-9'sd48 - $signed({fin_ku[7], fin_ku})) : -9'sd48;

    // tanh segment 2 tail: p2 in [0.76, 1) -> 1.xx * 2^-1 with RNE to f32
    wire p2_ge1 = (poly_out >= 28'sd16777216);
    /* verilator lint_off UNUSEDSIGNAL */
    wire [24:0] p2_f24 = p2_ge1 ? (poly_out[24:0] - 25'd16777216)
                              : ({poly_out[23:0], 1'b0} - 25'd16777216);
    /* verilator lint_on UNUSEDSIGNAL */
    wire [23:0] p2_frac = {1'b0, p2_f24[23:1]};
    wire p2_carry = p2_f24[0] & p2_frac[0];
    wire [23:0] p2_frac_r = p2_frac + {23'd0, p2_carry};
    wire p2_is_one = p2_ge1 || (p2_frac_r == 24'd8388608);
    /* verilator lint_off WIDTHEXPAND */
    wire [31:0] t_p2 = p2_is_one ? F32_PONE : {8'd126, p2_frac_r[22:0]};
    /* verilator lint_on WIDTHEXPAND */

    // ex2 tail: exponent-field rebuild (math_rtl_model._ex2_rebuild)
    wire m_ge1 = poly_out[24];
    /* verilator lint_off UNUSEDSIGNAL */
    wire [24:0] ex2_f24 = m_ge1 ? (poly_out[24:0] - 25'd16777216)
                               : ({poly_out[23:0], 1'b0} - 25'd16777216);
    /* verilator lint_on UNUSEDSIGNAL */
    wire [23:0] ex2_frac = {1'b0, ex2_f24[23:1]};
    wire ex2_round = ex2_f24[0] & ex2_frac[0];
    wire [23:0] ex2_frac_r = ex2_frac + {23'd0, ex2_round};
    wire ex2_carry = (ex2_frac_r == 24'h800000);
    wire signed [8:0] ex2_expu = (m_ge1 ? fin_n : (fin_n - 9'sd1))
                               + (ex2_carry ? 9'sd1 : 9'sd0);
    /* verilator lint_off WIDTHEXPAND */
    wire [31:0] ex2_out = (ex2_expu >= 9'sd128) ? F32_PINF
                                                : {ex2_expu[7:0] + 8'd127, ex2_frac_r[22:0]};
    /* verilator lint_on WIDTHEXPAND */

    // D19: the tail multipliers + tails + pack-mux chain straight off the
    // Horner output registers still measured -0.58 ns ctrl_r[12] ->
    // pack_mant_r (math-sfu-002 V4b, fourteenth run), so the products
    // and tails get their own register bank (fin2p) and the pack mux
    // moves to the next cycle. Every downstream bank keeps its edge and
    // just reads fin2p, so all sidebands stay aligned (D14 lesson).
    logic signed [55:0] p1_prod_r;
    logic signed [52:0] p7_prod_r;
    logic signed [8:0] pack_w_p;
    logic [31:0] t_p2_p, ex2_out_p;
    logic [1:0] fin2p_instr, fin2p_mode, fin2p_special;
    logic [31:0] fin2p_const, fin2p_bypass;
    logic fin2p_sign;

    always @(posedge clk) begin
        if (enable) begin
            p1_prod_r  <= p1_prod;
            p7_prod_r  <= p7_prod;
            pack_w_p   <= pack_w;
            t_p2_p     <= t_p2;
            ex2_out_p  <= ex2_out;
            fin2p_instr  <= fin_instr;
            fin2p_mode   <= fin_mode;
            fin2p_special <= fin_special;
            fin2p_const  <= fin_const;
            fin2p_bypass <= fin_bypass;
            fin2p_sign   <= fin_sign;
        end
    end

    wire [55:0] pack_mant = (fin2p_mode == M_P7) ? {3'd0, p7_prod_r} : p1_prod_r;

    // S28 registers: pack mux + f32-free tails + sideband (one cycle
    // after the product bank above)
    logic [55:0] pack_mant_r;
    logic signed [8:0] pack_w_r;
    logic [31:0] t_p2_r, ex2_out_r;
    logic [1:0] fin2_instr, fin2_mode, fin2_special;
    logic [31:0] fin2_const, fin2_bypass;
    logic fin2_sign;

    always @(posedge clk) begin
        if (enable) begin
            pack_mant_r <= pack_mant;
            pack_w_r    <= pack_w_p;
            t_p2_r      <= t_p2_p;
            ex2_out_r   <= ex2_out_p;
            fin2_instr  <= fin2p_instr;
            fin2_mode   <= fin2p_mode;
            fin2_special <= fin2p_special;
            fin2_const  <= fin2p_const;
            fin2_bypass <= fin2p_bypass;
            fin2_sign   <= fin2p_sign;
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // S29: leading-bit encoder stage (D13 split)
    //
    // Post-synth the encoder + normalization shift still chained ~4.2 ns
    // in one stage (math-sfu-002 V4b, eighth run), so the encoder gets
    // its own stage: S29 resolves e0, S30 does the e0-controlled shifts.
    // f32_pack above stays as the single-stage reference; the split
    // stages implement it exactly.
    ///////////////////////////////////////////////////////////////////////////

    // leading-bit position e0 (hierarchical one-hot encoder, D12):
    // 7 groups of 8 bits; group-level any/lead select, then a local
    // one-hot inside the leading group. Log-depth, replacing the 56
    // independent shift+reduce encoders that kept ~4.1ns of post-synth
    // depth in the packer front (math-sfu-002 V4b, sixth run).
    logic [6:0] grp_any, grp_lead;
    logic [6:0][7:0] loc_lead;
    logic [55:0] lead;
    logic [7:0] grp_bits;
    integer g, b;
    always @(*) begin
        // per-group: any-bit + local one-hot (byte-wide ORs only)
        for (g = 0; g < 7; ++g) begin
            grp_bits = pack_mant_r[g*8 +: 8];
            grp_any[g] = |grp_bits;
            loc_lead[g][7] = grp_bits[7];
            for (b = 6; b >= 0; --b) begin
                loc_lead[g][b] = grp_bits[b] & ~(|(grp_bits >> (b + 1)));
            end
        end
        // inter-group leading: grp_lead[g] = any(g) & no higher group set
        grp_lead[6] = grp_any[6];
        for (g = 5; g >= 0; --g) begin
            grp_lead[g] = grp_any[g] & ~(|(grp_any >> (g + 1)));
        end
        // assemble the 56-bit one-hot (only the leading group contributes)
        for (g = 0; g < 7; ++g) begin
            for (b = 0; b < 8; ++b) begin
                lead[g*8 + b] = grp_lead[g] & loc_lead[g][b];
            end
        end
    end
    logic signed [8:0] e0;
    integer i_e;
    always @(*) begin
        e0 = 9'sd0;
        for (i_e = 0; i_e <= 55; ++i_e) begin
            e0 = e0 | (lead[i_e] ? 9'(i_e + 1) : 9'sd0);
        end
    end

    // S29 registers: encoder result + packer inputs + sideband
    logic signed [8:0] s17_e0;
    logic [55:0] s17_mant;
    logic signed [8:0] s17_w;
    logic [31:0] s17_t_p2, s17_ex2;
    logic [1:0] fin3_instr, fin3_mode, fin3_special;
    logic [31:0] fin3_const, fin3_bypass;
    logic fin3_sign;

    always @(posedge clk) begin
        if (enable) begin
            s17_e0      <= e0;
            s17_mant    <= pack_mant_r;
            s17_w       <= pack_w_r;
            s17_t_p2    <= t_p2_r;
            s17_ex2     <= ex2_out_r;
            fin3_instr  <= fin2_instr;
            fin3_mode   <= fin2_mode;
            fin3_special <= fin2_special;
            fin3_const  <= fin2_const;
            fin3_bypass <= fin2_bypass;
            fin3_sign   <= fin2_sign;
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // S30: normalizer (e0-controlled shift/mask + biased exponent)
    ///////////////////////////////////////////////////////////////////////////

    // normalization split (f32_pack body), windowed form (D16): with
    // drop = e - 24 the rounding inputs are rnd = mant[drop-1] and
    // rem = mant[drop-1:0]; since drop <= 32 they live entirely in
    // mant[31:0], and hi = mant[drop+24:drop]. Left-shifting the
    // 32-bit low half by (32-drop) puts rnd at bit 31 and
    // mant[drop-2:0] at bits [30 : 32-drop] with zeros below, so the
    // RNE test becomes OR/AND reductions of one 32-bit barrel-shift
    // output (and its inverted twin for the all-ones case) -- no
    // 56-bit mask subtractors or variable bit-selects, which still
    // measured -1.11 ns in S30 (math-sfu-002 V4b, eleventh run).
    /* verilator lint_off WIDTHTRUNC */
    wire [5:0] drop = s17_e0 - 9'sd24;
    /* verilator lint_on WIDTHTRUNC */
    wire [24:0] hi_pre = (s17_e0 > 9'sd24) ? 25'(s17_mant >> drop)
                                           : 25'(s17_mant[24:0] << (6'd24 - s17_e0[5:0]));
    wire [31:0] win = 32'(s17_mant[31:0] << (6'd32 - drop));   // {rnd, mant[drop-2:0], 0s}
    wire [31:0] win_n = 32'(~s17_mant[31:0] << (6'd33 - drop)); // zeros iff low all ones
    wire rnd_bit = win[31];                                     // mant[drop-1]
    wire low_any = |win[30:0];                                  // any mant[drop-2:0]
    wire low_all = ~|win_n;                                     // all mant[drop-2:0]
    wire rne_pre = (s17_e0 > 9'sd24) && rnd_bit
                && (low_any || (low_all && hi_pre[0]));
    wire signed [9:0] biased = (s17_w + s17_e0 - 9'sd1) + 10'sd127;

    // S30 registers (names kept from the pre-D13 layout)
    logic [24:0] s17_hi;
    logic s17_rne_up;
    logic signed [9:0] s17_biased;
    logic s17_nonzero;
    logic [31:0] s18_t_p2, s18_ex2;
    logic [1:0] fin3b_instr, fin3b_mode, fin3b_special;
    logic [31:0] fin3b_const, fin3b_bypass;
    logic fin3b_sign;

    always @(posedge clk) begin
        if (enable) begin
            s17_hi      <= hi_pre;
            s17_rne_up  <= rne_pre;
            s17_biased  <= biased;
            s17_nonzero <= (s17_e0 != 9'sd0);
            s18_t_p2    <= s17_t_p2;
            s18_ex2     <= s17_ex2;
            fin3b_instr  <= fin3_instr;
            fin3b_mode   <= fin3_mode;
            fin3b_special <= fin3_special;
            fin3b_const  <= fin3_const;
            fin3b_bypass <= fin3_bypass;
            fin3b_sign   <= fin3_sign;
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // S31: packer back half (RNE + assembly) + t_mag select
    ///////////////////////////////////////////////////////////////////////////

    wire s18_rne_up = s17_rne_up;
    wire [24:0] hi_inc = s17_hi + 25'd1;
    wire s18_carry = s18_rne_up && (hi_inc == 25'd16777216);
    /* verilator lint_off UNUSEDSIGNAL */
    wire [24:0] hi_fin = s18_rne_up ? (s18_carry ? 25'd8388608 : hi_inc)
                                     : s17_hi;
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [9:0] biased_fin = s17_biased + (s18_carry ? 10'sd1 : 10'sd0);
    wire [31:0] pack_norm = (biased_fin >= 10'sd255) ? F32_PINF
                          : (biased_fin <= 10'sd0) ? 32'h00000000
                          : {1'b0, biased_fin[7:0], hi_fin[22:0]};
    wire [31:0] pack_out = s17_nonzero ? pack_norm : 32'h00000000;

    // t magnitude for the tanh-family tails; SP_BYPASS lanes take the
    // precomputed bypass value (e.g. x/2 for small sigmoid inputs)
    wire [31:0] t_arr = (fin3b_mode == M_P1) ? pack_out : s18_t_p2;
    wire [31:0] t_mag = (fin3b_special == SP_BYPASS) ? fin3b_bypass : t_arr;

    // sigmoid positive-half fold precompute (D22): shift amount, quotient
    // and remainder are derived from the same-cycle t_mag source and
    // ride the S31 bank, leaving S32 with the small tail (half-bit,
    // compare, RNE, pack, mux). Run 17 path t_mag_r[25] -> result_r[23]
    // was -0.08 ns with the full fold (two 24-bit barrel shifts + mask
    // subtractor + compare + RNE) inside S32.
    wire [7:0] texp_c = t_mag[30:23];
    wire [23:0] tmant_c = {1'b1, t_mag[22:0]};
    wire [8:0] t_shift_c = 9'sd127 - $signed({1'b0, texp_c});         // >= 1
    wire t_shift_ok_c = (t_shift_c < 9'd24);
    wire [23:0] t_quot_c = t_shift_ok_c ? (tmant_c >> t_shift_c[4:0]) : 24'd0;
    wire [23:0] t_rem_c = t_shift_ok_c
                        ? (tmant_c & ((24'd1 << t_shift_c[4:0]) - 24'd1)) : 24'd0;

    // S31 registers
    /* verilator lint_off UNUSEDSIGNAL */
    logic [31:0] t_mag_r, ex2_out_s18, pack_out_r, fin4_bypass;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [1:0] fin4_instr, fin4_mode, fin4_special;
    logic [31:0] fin4_const;
    logic fin4_sign;
    logic t_shift_ok_r;             // D22
    logic [4:0] t_shamt_r;          // D22
    logic [23:0] t_quot_r, t_rem_r; // D22

    always @(posedge clk) begin
        if (enable) begin
            t_mag_r     <= t_mag;
            pack_out_r  <= pack_out;
            ex2_out_s18 <= s18_ex2;
            fin4_bypass <= fin3b_bypass;
            fin4_instr  <= fin3b_instr;
            fin4_mode   <= fin3b_mode;
            fin4_special <= fin3b_special;
            fin4_const  <= fin3b_const;
            fin4_sign   <= fin3b_sign;
            t_shift_ok_r <= t_shift_ok_c;   // D22
            t_shamt_r    <= t_shift_c[4:0]; // D22
            t_quot_r     <= t_quot_c;       // D22
            t_rem_r      <= t_rem_c;        // D22
        end
    end

    ///////////////////////////////////////////////////////////////////////////
    // S32: sigmoid (1+t)/2 fold + result mux
    ///////////////////////////////////////////////////////////////////////////

    // sigmoid positive-half fold: (1 + t) / 2 (math_rtl_model._sigmoid_pos)
    // D22: t_quot_r/t_rem_r/t_shamt_r precomputed in the S31 bank, same
    // edge and source as t_mag_r, so the fold tail below is bit-exact.
    wire [23:0] t_half = 24'd1 << (t_shamt_r - 5'd1);
    wire [24:0] sig_val_pre = 25'd8388608 + {1'b0, t_quot_r};
    wire sig_rne = t_shift_ok_r
                 && ((t_rem_r > t_half) || ((t_rem_r == t_half) && sig_val_pre[0]));
    wire [24:0] sig_val = sig_val_pre + (sig_rne ? 25'd1 : 25'd0);
    /* verilator lint_off WIDTHEXPAND */
    wire [31:0] sig_half_out = (sig_val == 25'd16777216) ? F32_PONE
                                                        : {8'd126, sig_val[22:0]};
    /* verilator lint_on WIDTHEXPAND */

    // result mux
    logic [31:0] fin_res;
    always @(*) begin
        case (fin4_instr)
            I_EX2: begin
                fin_res = (fin4_special == SP_CONST) ? fin4_const : ex2_out_s18;
            end
            I_TANH: begin
                fin_res = (fin4_special == SP_CONST) ? fin4_const
                          : (fin4_special == SP_BYPASS) ? fin4_bypass
                          : {fin4_sign, t_mag_r[30:0]};
            end
            default: begin // I_SIG
                fin_res = (fin4_special == SP_CONST) ? fin4_const
                          : (fin4_mode == M_P7) ? pack_out_r
                          : sig_half_out;                             // SP_BYPASS folds t
            end
        endcase
    end

    logic [31:0] result_r;

    always @(posedge clk) begin
        if (enable) begin
            result_r <= fin_res;
        end
    end

    assign result = result_r;

endmodule
