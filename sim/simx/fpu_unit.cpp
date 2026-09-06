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

#include "fpu_unit.h"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include <rvfloats.h>
#include "debug.h"
#include "core.h"
#include "csr_unit.h"
#include "constants.h"

using namespace vortex;

namespace {
inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}
inline bool is_nan_boxed(uint64_t value) {
  return (uint32_t(value >> 32) == 0xffffffff);
}
inline int64_t check_boxing(int64_t a) {
  if (is_nan_boxed(a))
    return a;
  return nan_box(0x7fc00000); // NaN
}

///////////////////////////////////////////////////////////////////////////////
// Math-SFU bit-exact model (ex2.f32 / tanh.f32 / sigmoid.f32).
//
// C++ image of hw/rtl/fpu/VX_math_lane.sv (single source of truth: the
// s2.24 coefficient localparams below are copied verbatim from the RTL,
// which itself mirrors tests/cocotb/math_sfu_unit/math_rtl_model.py).
// All arithmetic reproduces the hardware datapath exactly: s4.24 Horner
// accumulators with truncate-after-multiply, s8.24 input quantization,
// RNE integer split, and the normalizing f32 packer with RNE mantissa.
// No fflags are produced (approx-class, matches RTL wr_xregs[XREG_0]=0).

namespace math_sfu {

constexpr int Q = 24;
constexpr uint32_t F32_NAN  = 0x7FC00000;
constexpr uint32_t F32_PINF = 0x7F800000;
constexpr uint32_t F32_PONE = 0x3F800000;
constexpr uint32_t F32_NONE = 0xBF800000;

// two's-complement wrap to `bits` width (Python model `s(v, bits)`)
inline int64_t swrap(int64_t v, int bits) {
  int64_t m = (int64_t(1) << bits) - 1;
  v &= m;
  if (v >> (bits - 1))
    v -= int64_t(1) << bits;
  return v;
}

// arithmetic right shift by an amount that may exceed 63: hardware funnel
// semantics sign-extend past the MSB, but C++ shift counts >= width are UB
// (x86 masks them mod 64). The p7 shift Q+ku reaches 24+126 = 150.
inline int64_t ashr_any(int64_t v, int64_t n) {
  return (n >= 63) ? (v < 0 ? -1 : 0) : (v >> n);
}

// sign-extend the RTL 26-bit s2.24 coefficient literals
constexpr int32_t sc26(uint32_t v) {
  return (v & 0x02000000u) ? int32_t(v | 0xFC000000u) : int32_t(v);
}

// coefficients c0..cN ascending (RTL packed arrays list cN first)
constexpr int32_t EX2_C[5] = {
  sc26(0x1000001), sc26(0x0B170CA), sc26(0x03D7F32), sc26(0x00E4DDB), sc26(0x00279C8)
};
constexpr int32_t P1_C[5] = {
  sc26(0x0FFFFB0), sc26(0x3AABC9E), sc26(0x0217D10), sc26(0x3F452BE), sc26(0x0026CCB)
};
constexpr int32_t P2_C[11] = {
  sc26(0x0FF036C), sc26(0x0042D34), sc26(0x3F72779), sc26(0x00C9F06),
  sc26(0x3F2D29E), sc26(0x009BB59), sc26(0x3F996C8), sc26(0x0060A75),
  sc26(0x3FD2CC6), sc26(0x3FDF102), sc26(0x001B8E8)
};
constexpr int32_t P7_C[8] = {
  sc26(0x0FFFFC9), sc26(0x3001036), sc26(0x0FECE9A), sc26(0x309B142),
  sc26(0x0D5C2DD), sc26(0x36F1EDD), sc26(0x03F550C), sc26(0x3F33938)
};
constexpr int32_t LOG2E_Q = sc26(0x1715476); // 1.44269504
constexpr int32_t SEG2_S  = sc26(0x0787878); //  0.47058824 (1/2.125)
constexpr int32_t SEG2_B  = sc26(0x2878788); // -1.47058824 (-3.125/2.125)

// Horner with hardware semantics: acc = trunc(acc * x) + c, s4.24 wrap
inline int64_t horner_trunc(const int32_t* c, int n, int64_t x_q) {
  int64_t acc = c[n - 1];
  for (int i = n - 2; i >= 0; --i)
    acc = swrap((acc * x_q >> Q) + c[i], 28);
  return acc;
}

// f32 bits -> s{int_bits}.24, truncating funnel shift; subnormals flush to 0
inline int64_t f32_to_q(uint32_t x_bits) {
  uint32_t exp  = (x_bits >> 23) & 0xFF;
  if (exp == 0)
    return 0;
  int64_t mant = int64_t(0x800000 | (x_bits & 0x7FFFFF)); // mant * 2^(exp-150)
  int shift = int(exp) - 126; // exp - 150 + Q
  int64_t v = (shift >= 0) ? (mant << shift)
              : (-shift >= 25) ? 0 : (mant >> -shift);
  return (x_bits >> 31) ? -v : v;
}

// round Q24 value to nearest integer, ties to even (floor-div semantics)
inline int64_t rne_to_int(int64_t x_q) {
  int64_t n = x_q >> Q;
  int64_t r = x_q - (n << Q);
  int64_t twice = 2 * r;
  if (twice > (int64_t(1) << Q) || (twice == (int64_t(1) << Q) && (n & 1)))
    ++n;
  return n;
}

// round 24-bit fraction to 23 bits (RNE); carry on reaching 2^23
inline uint32_t frac23(int64_t f24, int& carry) {
  int64_t frac = f24 >> 1;
  if ((f24 & 1) && (frac & 1))
    ++frac;
  carry = (frac == (int64_t(1) << 23));
  if (carry)
    return 0;
  return uint32_t(frac);
}

// normalizing f32 packer: value = mant * 2^w, mant unsigned nonzero;
// RNE on the dropped significand bits, Inf on overflow, FTZ below range
inline uint32_t pack_f32(uint32_t sign, int64_t mant, int w) {
  if (mant == 0)
    return sign << 31;
  uint64_t m = uint64_t(mant);
  int e = 64 - __builtin_clzll(m);
  if (e > 24) {
    int drop = e - 24;
    uint64_t hi = m >> drop;
    uint64_t rem = m & ((uint64_t(1) << drop) - 1);
    uint64_t half = uint64_t(1) << (drop - 1);
    if (rem > half || (rem == half && (hi & 1))) {
      ++hi;
      if (hi == (uint64_t(1) << 24)) // carry into the exponent
        return (sign << 31) | uint32_t((w + e - 1) + 1 + 127) << 23;
    }
    m = hi;
  } else if (e < 24) {
    m <<= 24 - e;
    e = 24;
  }
  int biased = w + e - 1 + 127;
  if (biased >= 255)
    return (sign << 31) | F32_PINF;
  if (biased <= 0)
    return sign << 31; // FTZ (exempt domain)
  return (sign << 31) | uint32_t(biased << 23) | uint32_t(m & 0x7FFFFF);
}

// 2^f polynomial, f in [-0.5, 0.5] Q24; output [0.707, 1.414] * 2^24
inline int64_t ex2_poly(int64_t f_q) {
  return horner_trunc(EX2_C, 5, f_q);
}

// pack m * 2^n (m Q24 in [0.707, 1.414]) into f32 exponent-field rebuild
inline uint32_t ex2_rebuild(int64_t m, int64_t n) {
  int carry;
  uint32_t frac;
  int64_t expu;
  if (m >= (int64_t(1) << Q)) {
    frac = frac23(m - (int64_t(1) << Q), carry);
    expu = n + carry;
  } else {
    frac = frac23(m * 2 - (int64_t(1) << Q), carry);
    expu = n - 1 + carry;
  }
  if (expu + 127 >= 255)
    return F32_PINF;
  return uint32_t((expu + 127) << 23) | frac;
}

inline uint32_t ex2(uint32_t x) {
  uint32_t sign = x >> 31;
  uint32_t exp  = (x >> 23) & 0xFF;
  uint32_t frac = x & 0x7FFFFF;
  if (exp == 0xFF)
    return frac ? F32_NAN : (sign ? 0u : F32_PINF); // 2^-Inf = +0
  if (sign && (exp >= 134 || (exp == 133 && frac > 0x7C0000)))
    return 0;                                       // x < -126 (FTZ)
  if (exp >= 134)                                   // x >= 128
    return F32_PINF;
  int64_t x_q = f32_to_q(x);                        // s8.24
  int64_t n = rne_to_int(x_q);
  return ex2_rebuild(ex2_poly(x_q - (n << Q)), n);
}

constexpr uint32_t TANH_BYPASS_EXP = 120;                 // |x| < 2^-7
constexpr uint32_t TANH_SAT_EXP = 129, TANH_SAT_FRAC = 0x280000; // |x| >= 5.25

// tanh(a) for finite non-negative f32 a, returned as positive f32 bits
inline uint32_t tanh_mag(uint32_t a) {
  uint32_t exp  = (a >> 23) & 0xFF;
  uint32_t frac = a & 0x7FFFFF;
  if (exp < TANH_BYPASS_EXP)
    return a;                                       // pass-through
  if (exp > TANH_SAT_EXP || (exp == TANH_SAT_EXP && frac >= TANH_SAT_FRAC))
    return F32_PONE;                                // saturate
  int64_t a_q = f32_to_q(a);                        // s4.24
  if (exp < 127) {                                  // [2^-7, 1): a * P1(a^2)
    int64_t u = a_q * a_q >> Q;
    int64_t p1 = horner_trunc(P1_C, 5, u);
    return pack_f32(0, a_q * p1, -2 * Q);           // normalize Q48 product
  }
  // segment 2: v = a*S + B in [-1, 1); pre-add intermediate wraps s3.24
  int64_t v_q = swrap((a_q * SEG2_S >> Q) + SEG2_B, 27);
  int64_t p2 = horner_trunc(P2_C, 11, v_q);
  if (p2 >= (int64_t(1) << Q))
    return F32_PONE;                                // clamp overshoot
  int carry;
  uint32_t f23 = frac23(p2 * 2 - (int64_t(1) << Q), carry);
  if (carry)
    return F32_PONE;                                // rounded up to 1.0
  return (126u << 23) | f23;
}

inline uint32_t tanh(uint32_t x) {
  uint32_t sign = x >> 31;
  uint32_t exp  = (x >> 23) & 0xFF;
  uint32_t frac = x & 0x7FFFFF;
  if (exp == 0xFF)
    return frac ? F32_NAN : (sign ? F32_NONE : F32_PONE); // tanh(+-Inf)
  uint32_t mag = x & 0x7FFFFFFF;
  uint32_t t = tanh_mag(mag);
  if (t == mag && exp < TANH_BYPASS_EXP)
    return x;                                       // pass-through keeps sign
  return (sign << 31) | t;
}

constexpr uint32_t SIG_SAT_EXP = 130, SIG_SAT_FRAC = 0x280000; // x >= 10.5

// (1 + tanh(x/2)) / 2 for finite x in [0, 10.5)
inline uint32_t sigmoid_pos(uint32_t x) {
  uint32_t xexp  = (x >> 23) & 0xFF;
  uint32_t xfrac = x & 0x7FFFFF;
  if (xexp == 0)
    return 0x3F000000;                              // x = 0/subnormal -> 0.5
  uint32_t half_bits = ((xexp - 1) << 23) | xfrac;  // x/2, exact
  uint32_t t = tanh_mag(half_bits);                 // t in [0, 1)
  uint32_t texp = (t >> 23) & 0xFF;
  if (texp == 0)
    return 0x3F000000;                              // t flushed -> 0.5
  uint64_t tmant = 0x800000 | (t & 0x7FFFFF);
  int shift = 127 - int(texp);                      // align t/2 to Q24, >= 1
  int64_t val = int64_t(1) << 23;                   // 0.5 in Q24
  if (shift < 24) {
    val += int64_t(tmant >> shift);
    uint64_t rem = tmant & ((uint64_t(1) << shift) - 1);
    uint64_t half_bit = uint64_t(1) << (shift - 1);
    if (rem > half_bit || (rem == half_bit && (val & 1))) {
      ++val;
      if (val == (int64_t(1) << 24))                // rounded up to 1.0
        return F32_PONE;
    }
  }
  // else: t/2 < 2^-24, absorbed by the Q24 RNE (val unchanged)
  return (126u << 23) | uint32_t(val - (int64_t(1) << 23));
}

inline uint32_t sigmoid(uint32_t x) {
  uint32_t sign = x >> 31;
  uint32_t exp  = (x >> 23) & 0xFF;
  uint32_t frac = x & 0x7FFFFF;
  if (exp == 0xFF)
    return frac ? F32_NAN : (sign ? 0u : F32_PONE);
  if (!sign) {
    if (exp > SIG_SAT_EXP || (exp == SIG_SAT_EXP && frac >= SIG_SAT_FRAC))
      return F32_PONE;                              // x >= 10.5
    return sigmoid_pos(x);
  }
  // negative half: u = 2^(x*log2e) via the ex2 core on fixed-point y
  if (exp >= 134)
    return 0;                                       // x <= -128: u underflows
  int64_t x_q = f32_to_q(x);                        // s8.24 (|x| < 128)
  int64_t y_q = swrap((x_q * LOG2E_Q) >> Q, 33);    // truncated product
  if (y_q <= -(int64_t(126) << Q))
    return 0;                                       // u underflow (FTZ)
  int64_t n = rne_to_int(y_q);
  int64_t m = ex2_poly(y_q - (n << Q));             // Q24 in [0.707, 1.414]
  int64_t mu, ku;
  if (m >= (int64_t(1) << Q)) {
    mu = m >> 1; ku = -n - 1;                       // u = (m/2) * 2^(n+1)
  } else {
    mu = m;    ku = -n;                             // u = m * 2^n
  }
  // p7 Horner with dynamic shift: trunc(acc * mu) >> (Q + ku) + c, s4.24
  int64_t acc = P7_C[7];
  for (int i = 6; i >= 0; --i)
    acc = swrap(ashr_any(acc * mu, Q + ku) + P7_C[i], 28);
  return pack_f32(0, mu * acc, -2 * Q - int(ku));
}

} // namespace math_sfu
}

FpuUnit::FpuUnit(const SimContext& ctx, const char* name, Core* core)
	: FuncUnit<VX_CFG_NUM_FPU_BLOCKS>(ctx, name, core)
{}

uint32_t FpuUnit::latency_of(const instr_trace_t* trace) const {
	auto fpu_type = std::get<FpuType>(trace->op_type);
	const uint32_t delay = 2;
	switch (fpu_type) {
	case FpuType::FCMP:
	case FpuType::FSGNJ:
	case FpuType::FCLASS:
	case FpuType::FMVXW:
	case FpuType::FMVWX:
	case FpuType::FMINMAX:
		return 2+delay;
	case FpuType::FADD:
	case FpuType::FSUB:
	case FpuType::FMUL:
	case FpuType::FMADD:
	case FpuType::FMSUB:
	case FpuType::FNMADD:
	case FpuType::FNMSUB:
		return VX_CFG_FMA_LATENCY+delay;
	case FpuType::FDIV:
		return VX_CFG_FDIV_LATENCY+delay;
	case FpuType::FSQRT:
		return VX_CFG_FSQRT_LATENCY+delay;
	case FpuType::F2I:
	case FpuType::I2F:
	case FpuType::F2F:
		return VX_CFG_FCVT_LATENCY+delay;
	case FpuType::EX2:
	case FpuType::TANH:
	case FpuType::SIGMOID:
		return VX_CFG_MATH_LATENCY+delay;
	default:
		std::abort();
	}
}

void FpuUnit::execute(instr_trace_t* trace) {
	auto& emu = core_->csr_unit();           // CSR/FCSR helpers
	// Use trace->tmask captured at issue, not the live warp.tmask; divergent
	// control flow may change the warp's tmask before this trace executes.
	auto& tmask = trace->tmask;
	auto& instr = *trace->instr_ptr;
	auto instrArgs = instr.get_args();
	auto fpuArgs = std::get<IntrFpuArgs>(instrArgs);
	auto fpu_type = std::get<FpuType>(trace->op_type);
	uint32_t wid = trace->wid;
	uint32_t num_threads = VX_CFG_NUM_THREADS;
	auto& rs1_data = trace->src_data[0];
	auto& rs2_data = trace->src_data[1];
	auto& rs3_data = trace->src_data[2];

	uint32_t thread_start = 0;
	for (; thread_start < num_threads; ++thread_start) {
		if (tmask.test(thread_start)) break;
	}

	trace->dst_data.assign(num_threads, reg_data_t{});
	auto& rd_data = trace->dst_data;

	switch (fpu_type) {
	case FpuType::FADD: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fadd_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FSUB: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fsub_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMUL: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fmul_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fmul_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FDIV: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fdiv_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fdiv_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FSQRT: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fsqrt_d(rs1_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fsqrt_s(check_boxing(rs1_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FSGNJ: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				switch (fpuArgs.frm) {
				case 0: rd_data[t].u64 = rv_fsgnj_d(rs1_data[t].u64, rs2_data[t].u64); break;
				case 1: rd_data[t].u64 = rv_fsgnjn_d(rs1_data[t].u64, rs2_data[t].u64); break;
				case 2: rd_data[t].u64 = rv_fsgnjx_d(rs1_data[t].u64, rs2_data[t].u64); break;
				}
			} else {
				switch (fpuArgs.frm) {
				case 0: rd_data[t].u64 = nan_box(rv_fsgnj_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64))); break;
				case 1: rd_data[t].u64 = nan_box(rv_fsgnjn_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64))); break;
				case 2: rd_data[t].u64 = nan_box(rv_fsgnjx_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64))); break;
				}
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMINMAX: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				if (fpuArgs.frm) rd_data[t].u64 = rv_fmax_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
				else             rd_data[t].u64 = rv_fmin_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
			} else {
				if (fpuArgs.frm) rd_data[t].u64 = nan_box(rv_fmax_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags));
				else             rd_data[t].u64 = nan_box(rv_fmin_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FCMP: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				switch (fpuArgs.frm) {
				case 0: rd_data[t].i = rv_fle_d(rs1_data[t].u64, rs2_data[t].u64, &fflags); break;
				case 1: rd_data[t].i = rv_flt_d(rs1_data[t].u64, rs2_data[t].u64, &fflags); break;
				case 2: rd_data[t].i = rv_feq_d(rs1_data[t].u64, rs2_data[t].u64, &fflags); break;
				}
			} else {
				switch (fpuArgs.frm) {
				case 0: rd_data[t].i = rv_fle_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags); break;
				case 1: rd_data[t].i = rv_flt_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags); break;
				case 2: rd_data[t].i = rv_feq_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags); break;
				}
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::F2I: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				switch (fpuArgs.cvt) {
				case 0: rd_data[t].i = sext((uint64_t)rv_ftoi_d(rs1_data[t].u64, frm, &fflags), 32); break;
				case 1: rd_data[t].i = sext((uint64_t)rv_ftou_d(rs1_data[t].u64, frm, &fflags), 32); break;
				case 2: rd_data[t].i = rv_ftol_d(rs1_data[t].u64, frm, &fflags); break;
				case 3: rd_data[t].i = rv_ftolu_d(rs1_data[t].u64, frm, &fflags); break;
				}
			} else {
				switch (fpuArgs.cvt) {
				case 0: rd_data[t].i = sext((uint64_t)rv_ftoi_s(check_boxing(rs1_data[t].u64), frm, &fflags), 32); break;
				case 1: rd_data[t].i = sext((uint64_t)rv_ftou_s(check_boxing(rs1_data[t].u64), frm, &fflags), 32); break;
				case 2: rd_data[t].i = rv_ftol_s(check_boxing(rs1_data[t].u64), frm, &fflags); break;
				case 3: rd_data[t].i = rv_ftolu_s(check_boxing(rs1_data[t].u64), frm, &fflags); break;
				}
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::I2F: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				switch (fpuArgs.cvt) {
				case 0: rd_data[t].u64 = rv_itof_d(rs1_data[t].i, frm, &fflags); break;
				case 1: rd_data[t].u64 = rv_utof_d(rs1_data[t].i, frm, &fflags); break;
				case 2: rd_data[t].u64 = rv_ltof_d(rs1_data[t].i, frm, &fflags); break;
				case 3: rd_data[t].u64 = rv_lutof_d(rs1_data[t].i, frm, &fflags); break;
				}
			} else {
				switch (fpuArgs.cvt) {
				case 0: rd_data[t].u64 = nan_box(rv_itof_s(rs1_data[t].i, frm, &fflags)); break;
				case 1: rd_data[t].u64 = nan_box(rv_utof_s(rs1_data[t].i, frm, &fflags)); break;
				case 2: rd_data[t].u64 = nan_box(rv_ltof_s(rs1_data[t].i, frm, &fflags)); break;
				case 3: rd_data[t].u64 = nan_box(rv_lutof_s(rs1_data[t].i, frm, &fflags)); break;
				}
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::F2F: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_ftod(check_boxing(rs1_data[t].u64), frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_dtof(rs1_data[t].u64, frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FCLASS: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].i = rv_fclss_d(rs1_data[t].u64);
			} else {
				rd_data[t].i = rv_fclss_s(check_boxing(rs1_data[t].u64));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMVXW: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rs1_data[t].u64;
			} else {
				uint32_t result = (uint32_t)rs1_data[t].u64;
				rd_data[t].i = sext((uint64_t)result, 32);
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMVWX: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rs1_data[t].i;
			} else {
				rd_data[t].u64 = nan_box((uint32_t)rs1_data[t].i);
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMADD: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fmadd_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fmadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FMSUB: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fmsub_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fmsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FNMADD: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fnmadd_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fnmadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::FNMSUB: {
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t frm = emu.get_fpu_rm(fpuArgs.frm, wid, t);
			uint32_t fflags = 0;
			if (fpuArgs.is_f64) {
				rd_data[t].u64 = rv_fnmsub_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
			} else {
				rd_data[t].u64 = nan_box(rv_fnmsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
			}
			emu.update_fcrs(fflags, wid, t);
		}
	} break;
	case FpuType::EX2:
	case FpuType::TANH:
	case FpuType::SIGMOID: {
		// Approx-class math-SFU ops: frm ignored, no fflags. The lane
		// consumes the raw low 32 bits of the source register (mirrors
		// VX_math_unit.sv DATA_IN_WIDTH=32, no nan-box check) and the
		// result is nan-boxed on FLEN=64 like every f32 write.
		for (uint32_t t = thread_start; t < num_threads; ++t) {
			if (!tmask.test(t)) continue;
			uint32_t a = uint32_t(rs1_data[t].u64);
			uint32_t r = (fpu_type == FpuType::EX2) ? math_sfu::ex2(a) :
			             (fpu_type == FpuType::TANH) ? math_sfu::tanh(a) :
			                                           math_sfu::sigmoid(a);
			rd_data[t].u64 = nan_box(r);
		}
	} break;
	default:
		std::abort();
	}
	DT(3, this->name() << " execute: op=" << fpu_type << ", " << *trace);
}

void FpuUnit::on_tick() {
	for (uint32_t b = 0; b < VX_CFG_NUM_FPU_BLOCKS; ++b) {
		auto& input = Inputs.at(b);
		if (input.empty())
			continue;
		auto& output = Outputs.at(b);
		if (output.full())
			continue; // stall
		auto trace = input.peek();
		this->execute(trace);
		uint32_t delay = this->latency_of(trace);
		output.send(trace, delay);
		input.pop();
	}
}
