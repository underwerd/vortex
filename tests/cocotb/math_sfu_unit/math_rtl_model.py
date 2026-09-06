"""Bit-accurate fixed-point model of the VX_math_unit datapath (S4 spec).

Executable specification the RTL (fpu/VX_math_unit.sv) is translated from,
and the bit-level oracle for V4a cocotb tests. Mirrors the planned hardware:

  - coefficients: s2.24 (26-bit signed, RNE from the float64 frozen scheme)
  - Horner accumulators: s4.24 (28-bit signed); multiply, truncate (shift out
    low bits, no rounding), then add coefficient — matches RTL shift semantics
  - ex2: input quantized to s8.24 (truncation), RNE integer split n/f,
    degree-4 2^f polynomial, exponent-field rebuild; x >= 128 -> +Inf,
    x < -126 -> 0 (FTZ, S3 D1); |x| < 2^-24 quantizes to 0 -> 1.0
  - tanh: |x| < 2^-7 pass-through, |x| >= 5.25 saturates to +-1.0 (S3 D2),
    P1 deg-4 in u=a^2 on [2^-7,1), P2 deg-10 on the centered variable
    v=(a-3.125)/2.125 in [-1,1) — an exact rebasing of the frozen segment-2
    polynomial (power basis on a is fixed-point hostile: a^10 ~ 2^24 at the
    segment edge amplifies s2.24 coefficient quantization to ~0.5 output
    error, while |v|^k <= 1 keeps it at 2^-25); results packed through a
    normalizing f32 packer with RNE mantissa
  - sigmoid: x >= 10.5 -> 1.0; x >= 0 -> (1 + tanh(x/2)) / 2 (tanh core
    reuse, exact exponent halving); x < 0 -> u = 2^(x*log2e) via the ex2
    core with the s8.24 product y = x*log2e (truncated), y <= -126 -> 0
    (FTZ), else p7 deg-7 Horner on the normalized pair (mu, ku) with
    u = mu*2^-ku, mu in [0.5,1), and output u*p7 via the packer

Accuracy contract: same adjudication domain as approx_scheme.validate()
(golden normal outputs only, |truth| >= 2^-126; subnormal tails FTZ-exempt
per spec q4 ruling).

Run:  python3 math_rtl_model.py     (self-check vs golden_model)
"""

from __future__ import annotations

import math
import struct
import sys

sys.path.insert(0, ".")
from approx_scheme import (  # noqa: E402
    EX2_POLY, SIGMOID_P, TANH_SEG1_X2, TANH_SEG2, LOG2E,
)
from golden_model import (  # noqa: E402
    EX2_SPECIALS, SIGMOID_SPECIALS, TANH_SPECIALS,
    ex2_exact, sigmoid_exact, tanh_exact,
)

Q = 24
TARGET = 2.0 ** -13

F32_NAN = 0x7FC00000
F32_PINF = 0x7F800000
F32_ONE = 0x3F800000

# ---------------------------------------------------------------------------
# fixed-point helpers (plain Python ints, two's-complement semantics)
# ---------------------------------------------------------------------------

def s(v: int, bits: int) -> int:
    """Interpret the low `bits` of v as signed two's complement."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def q_coeff(c: float) -> int:
    """RNE float64 coefficient -> s2.24 (26-bit signed)."""
    v = round(c * (1 << Q))
    assert -(1 << 25) <= v < (1 << 25), f"coefficient {c} overflows s2.24"
    return v


# Frozen-scheme coefficients, fixed-point images (lists are c0..cN,
# ascending — same order as approx_scheme). Single source of truth for the
# SystemVerilog localparams in VX_math_unit.sv.
EX2_C = [q_coeff(c) for c in EX2_POLY]            # deg 4 in f
TANH_P1_C = [q_coeff(c) for c in TANH_SEG1_X2]    # deg 4 in u = a^2
SIG_P7_C = [q_coeff(c) for c in SIGMOID_P]        # deg 7 in u
LOG2E_Q = q_coeff(LOG2E)                           # s2.24

# Segment 2 of tanh runs on the centered variable v = (a - C)/R in [-1, 1),
# not on a directly. In the power basis on a the top term reaches a^10 ~ 2^24
# at the segment edge, amplifying s2.24 coefficient quantization noise to
# ~0.5 output error — unusable in fixed point. The rebasing below is exact
# (same polynomial, different basis), so the frozen-scheme float64 bounds
# carry over unchanged; |v|^k <= 1 keeps per-coefficient sensitivity at
# 2^-25 and the Horner intermediates within [0, 1.0] (measured peak 1.0).
TANH_SEG2_CCTR = 3.125                           # center of [1, 5.25)
TANH_SEG2_RHALF = 2.125                          # half-width
TANH_SEG2_S = 1.0 / TANH_SEG2_RHALF              # 0.470588...
TANH_SEG2_B = -TANH_SEG2_CCTR / TANH_SEG2_RHALF  # -1.470588...


def _rebasis(coeffs, c0, r):
    """Exact coefficients of p(c0 + r*v) in the v power basis (c0..cN)."""
    from math import comb
    out = [0.0] * len(coeffs)
    for k, c in enumerate(coeffs):
        for j in range(k + 1):
            out[j] += c * comb(k, j) * c0 ** (k - j) * r ** j
    return out


TANH_P2_C = [q_coeff(c) for c in _rebasis(TANH_SEG2, TANH_SEG2_CCTR, TANH_SEG2_RHALF)]  # deg 10 in v
TANH_SEG2_S_Q = q_coeff(TANH_SEG2_S)              # s2.24
TANH_SEG2_B_Q = q_coeff(TANH_SEG2_B)              # s2.24, |B| < 2


def horner_trunc(coeffs, x_q, acc_bits=28):
    """Horner with hardware semantics: acc = trunc(acc * x + c).

    coeffs is c0..cN ascending (same order as approx_scheme); the loop
    starts from the highest degree. The double-width product is shifted
    right by Q (dropping low bits, matching the RTL funnel shift) and
    wrapped to the accumulator width.
    """
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        acc = s((acc * x_q >> Q) + c, acc_bits)
    return acc


def f32_to_q(x_bits: int, int_bits: int) -> int:
    """f32 bit pattern -> s{int_bits}.24 fixed point, truncation semantics.

    Mirrors the RTL funnel shift: significand lands at Q24, bits below 2^-24
    are dropped. Zero/subnormal inputs flush to 0.
    """
    assert 0 <= x_bits < (1 << 32)
    sign = x_bits >> 31
    exp = (x_bits >> 23) & 0xFF
    frac = x_bits & 0x7FFFFF
    if exp == 0:
        return 0
    mant = (1 << 23) | frac                    # value = mant * 2^(exp-150)
    shift = exp - 150 + Q
    v = mant << shift if shift >= 0 else mant >> -shift
    # |v| stays below 2^31 (callers range-gate first); negate numerically
    total = int_bits + 1 + Q
    if sign:
        v = -v
    assert -(1 << (total - 1)) <= v < (1 << (total - 1))
    return v


def rne_to_int(x_q: int) -> int:
    """Round a Q24 fixed value to the nearest integer, ties to even."""
    n, r = divmod(x_q, 1 << Q)                 # floor division, both signs
    twice = 2 * r
    if twice > (1 << Q) or (twice == (1 << Q) and (n & 1)):
        n += 1
    return n


def pack_f32(sign: int, mant: int, w: int) -> int:
    """Assemble f32 bits from an unsigned integer significand.

    value = mant * 2^w  (w = LSB weight of mant). The significand is
    normalized to 24 bits with round-to-nearest-even on the dropped bits.
    Overflows to Inf; anything below the normal range flushes to zero
    (callers here never architecturally produce subnormals — those points
    sit in the FTZ-exempt adjudication domain).
    """
    if mant == 0:
        return sign << 31
    e = mant.bit_length()
    if e > 24:
        drop = e - 24
        hi = mant >> drop
        rem = mant & ((1 << drop) - 1)
        half = 1 << (drop - 1)
        if rem > half or (rem == half and (hi & 1)):
            hi += 1
            if hi == (1 << 24):                # carry into the exponent
                hi >>= 1
                return (sign << 31) | (((w + e - 1) + 1 + 127) << 23)
        mant = hi
    elif e < 24:
        mant <<= 24 - e                        # exact left shift, w unchanged
        e = 24
    expu = w + e - 1                           # unbiased exponent
    biased = expu + 127
    if biased >= 255:
        return (sign << 31) | 0x7F800000
    if biased <= 0:
        return sign << 31                      # FTZ (exempt domain)
    return (sign << 31) | (biased << 23) | (mant & 0x7FFFFF)


def frac23(f24: int) -> tuple[int, int]:
    """Round a 24-bit fractional part to 23 bits (RNE on the dropped LSB).
    Returns (frac, carry) — carry set when rounding reaches 2^23."""
    frac = f24 >> 1
    if (f24 & 1) and (frac & 1):              # tie -> round to even
        frac += 1
    if frac == (1 << 23):
        return 0, 1
    return frac, 0


def unpack(x_bits: int):
    return x_bits >> 31, (x_bits >> 23) & 0xFF, x_bits & 0x7FFFFF


# ---------------------------------------------------------------------------
# ex2 core: shared by ex2.f32 and the sigmoid negative half
# ---------------------------------------------------------------------------

def _ex2_poly(f_q: int) -> int:
    """2^f polynomial value in Q24, f in [-0.5, 0.5] (Q24). Output
    in [0.707, 1.414] * 2^24, s4.24."""
    return horner_trunc(EX2_C, f_q, acc_bits=28)


def _ex2_rebuild(m: int, n: int) -> int:
    """Pack m * 2^n (m Q24 in [0.707, 1.414]) into f32 bits. The 24-bit
    polynomial mantissa is rounded to the f32 23-bit fraction with RNE."""
    if m >= (1 << Q):
        f24 = m - (1 << Q)
        frac, carry = frac23(f24)
        expu = n + carry
    else:
        f24 = m * 2 - (1 << Q)
        frac, carry = frac23(f24)
        expu = n - 1 + carry
    if expu + 127 >= 255:
        return F32_PINF
    return (expu + 127) << 23 | frac


def ex2_rtl(x_bits: int) -> int:
    sign, exp, frac = unpack(x_bits)
    if exp == 0xFF:
        if frac:
            return F32_NAN
        return 0 if sign else F32_PINF         # 2^-Inf = +0
    if sign and (exp >= 134 or (exp == 133 and frac > 0x7C0000)):
        return 0                                # x < -126 (FTZ)
    if exp >= 134:                              # x >= 128
        return F32_PINF
    x_q = f32_to_q(x_bits, 8)                   # s8.24 (|x|<2^-24 -> 0)
    n = rne_to_int(x_q)
    f_q = x_q - (n << Q)
    return _ex2_rebuild(_ex2_poly(f_q), n)


# ---------------------------------------------------------------------------
# tanh core: shared by tanh.f32 and the sigmoid positive half
# ---------------------------------------------------------------------------

TANH_BYPASS_EXP = 120                          # |x| < 2^-7
TANH_SAT_EXP, TANH_SAT_FRAC = 129, 0x280000    # |x| >= 5.25 = 1.3125*2^2


def _tanh_mag(a_bits: int) -> int:
    """tanh(a) for finite non-negative f32 a (sign bit clear), as f32 bits
    of the magnitude (positive)."""
    _, exp, frac = unpack(a_bits)
    if exp < TANH_BYPASS_EXP:
        return a_bits                           # pass-through: tanh(a) = a
    if exp > TANH_SAT_EXP or (exp == TANH_SAT_EXP and frac >= TANH_SAT_FRAC):
        return F32_ONE                          # saturate
    a_q = f32_to_q(a_bits, 4)                   # s4.24
    if exp < 127:                               # [2^-7, 1): a * P1(a^2)
        u = (a_q * a_q) >> Q                    # u = a^2, Q24
        p1 = horner_trunc(TANH_P1_C, u, acc_bits=28)
        return pack_f32(0, a_q * p1, -2 * Q)    # normalize Q48 product
    # segment 2: v = a*S + B (truncated product), v in [-1, 1); the
    # pre-add intermediate a*S peaks at ~2.47, hence the s3.24 wrap
    v_q = s(((a_q * TANH_SEG2_S_Q) >> Q) + TANH_SEG2_B_Q, 27)
    p2 = horner_trunc(TANH_P2_C, v_q, acc_bits=28)
    if p2 >= (1 << Q):
        return F32_ONE                          # clamp overshoot
    # p2 in [0.76, 1): value below 1 -> shift left one, exponent -1
    frac, carry = frac23(p2 * 2 - (1 << Q))
    if carry:
        return F32_ONE                          # rounded up to exactly 1.0
    return (126 << 23) | frac


def tanh_rtl(x_bits: int) -> int:
    sign, exp, frac = unpack(x_bits)
    if exp == 0xFF:
        if frac:
            return F32_NAN
        return 0xBF800000 if sign else F32_ONE  # tanh(+-Inf) = +-1
    mag = _tanh_mag(x_bits & 0x7FFFFFFF)
    if mag == (x_bits & 0x7FFFFFFF) and exp < TANH_BYPASS_EXP:
        return x_bits                           # pass-through keeps the sign
    return (sign << 31) | mag


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

SIG_SAT_EXP, SIG_SAT_FRAC = 130, 0x280000      # x >= 10.5 = 1.3125*2^3


def _sigmoid_pos(x_bits: int) -> int:
    """(1 + tanh(x/2)) / 2 for finite x in [0, 10.5)."""
    _, xexp, xfrac = unpack(x_bits)
    if xexp == 0:
        return 0x3F000000                       # x = 0 (or subnormal) -> 0.5
    half = ((xexp - 1) << 23) | xfrac           # x/2, exact
    t = _tanh_mag(half)                         # t in [0, 1)
    _, texp, tfrac = unpack(t)
    tmant = (1 << 23) | tfrac                   # t = tmant * 2^(texp-150)
    if texp == 0:
        return 0x3F000000                       # t flushed -> 0.5
    shift = 127 - texp                          # align t/2 to Q24, >= 1
    val = 1 << 23                               # 0.5 in Q24
    if shift < 24:
        val += tmant >> shift
        rem = tmant & ((1 << shift) - 1)
        half_bit = 1 << (shift - 1)
        if rem > half_bit or (rem == half_bit and (val & 1)):
            val += 1
            if val == (1 << 24):                # (1+t)/2 rounded up to 1.0
                return F32_ONE
    # else: t/2 < 2^-24 — absorbed by RNE at Q24 (val unchanged)
    return (126 << 23) | (val - (1 << 23))


def sigmoid_rtl(x_bits: int) -> int:
    sign, exp, frac = unpack(x_bits)
    if exp == 0xFF:
        if frac:
            return F32_NAN
        return 0 if sign else F32_ONE           # sigmoid(-Inf)=0, (+Inf)=1
    if not sign:
        if exp > SIG_SAT_EXP or (exp == SIG_SAT_EXP and frac >= SIG_SAT_FRAC):
            return F32_ONE                      # x >= 10.5
        return _sigmoid_pos(x_bits)
    # negative half: u = 2^(x*log2e) via the ex2 core on a fixed-point y
    if exp >= 134:
        return 0                                # x <= -128: u underflows
    x_q = f32_to_q(x_bits, 8)                   # s8.24 (|x| < 128 here)
    y_q = s((x_q * LOG2E_Q) >> Q, 33)           # s8.24, truncated product
    if y_q <= -(126 << Q):
        return 0                                # u underflow (FTZ, scheme)
    n = rne_to_int(y_q)
    f_q = y_q - (n << Q)
    m = _ex2_poly(f_q)                          # Q24 in [0.707, 1.414]
    if m >= (1 << Q):                           # u = (m/2) * 2^(n+1)
        mu, ku = m >> 1, -n - 1                 # mu in [0.5, 0.707)
    else:                                       # u = m * 2^n
        mu, ku = m, -n                          # mu in [0.707, 1)
    # p7 Horner with dynamic shift: acc = trunc(acc * u) + c with
    # u = mu * 2^-ku. The Q48 product acc*mu carries the mu scale, so the
    # truncating shift is Q + ku (always >= 23 since ku >= -1: poly(0)
    # RNE-quantizes to 2^24 + 1, letting u overshoot 1 by 1 LSB at n = 0).
    acc = SIG_P7_C[-1]
    for c in reversed(SIG_P7_C[:-1]):
        acc = s((acc * mu >> (Q + ku)) + c, 28)
    # out = u * p7(u) = mu * p7 * 2^-ku
    return pack_f32(0, mu * acc, -2 * Q - ku)


# ---------------------------------------------------------------------------
# validation (same adjudication domain as approx_scheme.validate)
# ---------------------------------------------------------------------------

def _bits(x: float) -> int:
    return struct.unpack("I", struct.pack("f", x))[0]


def _from_bits(b: int) -> float:
    return struct.unpack("f", struct.pack("I", b & 0xFFFFFFFF))[0]


def validate(n_random: int = 100000, seed: int = 20260820) -> dict:
    import random
    rng = random.Random(seed)
    inputs = [_from_bits(rng.getrandbits(32)) for _ in range(n_random)]
    stats = {}
    for name, rtl, exact, specials in (
        ("ex2", ex2_rtl, ex2_exact, EX2_SPECIALS),
        ("tanh", tanh_rtl, tanh_exact, TANH_SPECIALS),
        ("sigmoid", sigmoid_rtl, sigmoid_exact, SIGMOID_SPECIALS),
    ):
        worst, worst_x, checked = 0.0, None, 0
        for x in inputs + specials:
            truth = exact(x)
            if truth != truth or truth in (0.0, math.inf, -math.inf):
                continue
            if abs(truth) < 2.0 ** -126:
                continue                        # FTZ tail exempt (spec q4)
            got = rtl(_bits(x))
            e = abs(_from_bits(got) - truth) / abs(truth)
            checked += 1
            if e > worst:
                worst, worst_x = e, x
        stats[name] = {"max_rel_err": worst, "worst_x": worst_x,
                       "checked": checked, "target": TARGET,
                       "pass": worst <= TARGET}
    return stats


if __name__ == "__main__":
    stats = validate()
    ok = True
    for name, st in stats.items():
        flag = "PASS" if st["pass"] else "FAIL"
        ok &= st["pass"]
        print(f"{name:8s} max_rel_err = {st['max_rel_err']:.3e} "
              f"(target {st['target']:.3e}) [{flag}]  checked {st['checked']} "
              f"worst_x = {st['worst_x']!r}")
    sys.exit(0 if ok else 1)
