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
  if (exp >= 134)                                   // x >= 128
    return F32_PINF;
  if (sign && (exp > 133 || (exp == 133 && frac > 0x7C0000)))
    return 0;                                       // x < -126 (FTZ)
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
