"""Bit-exact bf16 golden reference for packbf16.mul / packbf16.add.

S3 artifact of task pack-float-debug-001: an independent reference
model, written with pure integer arithmetic so it shares no floating
point path (and no rounding accident) with the DUT or the simx
implementation it cross-checks.

Semantics (operator-adjudicated in spec-packbf16-debug q1-q4):
- round to nearest, even on ties (single rounding step, always);
- IEEE-754 gradual denormals (inputs and outputs);
- inf * 0 = NaN, inf + (-inf) = NaN;
- any NaN input or NaN result -> default qNaN 0x7FC0;
- signed zeros per IEEE (x + (-x) = +0, -0 + -0 = -0).

The core is ``encode(mant, exp)``: it takes an exact value
``mant * 2**exp`` (mant a nonzero Python integer of any width) and
produces its correctly-rounded bf16 encoding, including the
denormal range and overflow-to-infinity. mul and add compute the
exact product/sum as big integers and delegate all rounding to that
one function, so there is exactly one rounding step anywhere.
"""
from __future__ import annotations

QNAN = 0x7FC0
POS_INF = 0x7F80
NEG_INF = 0xFF80

_SPECIALS = None  # built lazily by directed_specials()


def _parts(x: int):
    return (x >> 15) & 1, (x >> 7) & 0xFF, x & 0x7F


def _is_nan(e, m):
    return e == 0xFF and m != 0


def _is_inf(e, m):
    return e == 0xFF and m == 0


def _is_zero(e, m):
    return e == 0 and m == 0


def encode(mant: int, exp: int, sign: int) -> int:
    """bf16 encoding of the exact value ``mant * 2**exp`` (mant > 0).

    One rounding step (RNE) from the exact value to the bf16 grid:
    normals, gradual denormals, underflow to signed zero, and
    overflow to signed infinity are all handled here.
    """
    assert mant > 0
    hb = mant.bit_length() - 1          # value = 2**(hb+exp) * 1.xxx
    abs_e = hb + exp                    # absolute exponent of the leading bit

    if abs_e >= -126:
        # Normal range (possibly after rounding up into it): keep the
        # top 8 significand bits (implicit 1 in bit 7).
        shift = hb - 7
        if shift > 0:
            sig = mant >> shift
            round_bit = (mant >> (shift - 1)) & 1
            sticky = (mant & ((1 << (shift - 1)) - 1)) != 0
        else:
            sig = mant << -shift        # cannot happen for exact inputs,
            round_bit = 0               # kept for mathematical completeness
            sticky = False
        if round_bit and (sticky or (sig & 1)):
            sig += 1                    # RNE carry
        if sig > 0xFF:                  # carry out of the significand
            sig >>= 1
            abs_e += 1
        if abs_e > 127:
            return (sign << 15) | (0xFF << 7)      # overflow -> inf
        if abs_e < -126:
            # rounded down into the denormal range (e.g. tie at the
            # smallest normal): fall through to denormal encoding.
            return _encode_denormal(mant, exp, sign)
        return (sign << 15) | ((abs_e + 127) << 7) | (sig & 0x7F)

    return _encode_denormal(mant, exp, sign)


def _encode_denormal(mant: int, exp: int, sign: int) -> int:
    """Denormal encoding: value = m * 2**-133, m in [0, 128)."""
    # want m = mant * 2**(exp + 133), correctly rounded to integer
    k = exp + 133
    if k >= 0:
        m = mant << k                   # exact integer, no rounding
        round_bit, sticky = 0, False
    else:
        s = -k
        m = mant >> s
        round_bit = (mant >> (s - 1)) & 1
        sticky = (mant & ((1 << (s - 1)) - 1)) != 0
    if round_bit and (sticky or (m & 1)):
        m += 1
    if m >= 0x80:
        # tie/carry up to the smallest normal (exp field 1, sig 0)
        return (sign << 15) | (1 << 7)
    if m == 0:
        return sign << 15               # underflow to signed zero
    return (sign << 15) | m


def bf16_mul(a: int, b: int) -> int:
    sa, ea, ma = _parts(a)
    sb, eb, mb = _parts(b)
    sign = sa ^ sb
    if _is_nan(ea, ma) or _is_nan(eb, mb):
        return QNAN
    if _is_inf(ea, ma) or _is_inf(eb, mb):
        if _is_zero(ea, ma) or _is_zero(eb, mb):
            return QNAN                 # inf * 0 (denormals are nonzero)
        return (sign << 15) | POS_INF
    if _is_zero(ea, ma) or _is_zero(eb, mb):
        return sign << 15               # signed zero product
    # exact value: sig * 2**(eff_e - 134), denormals use eff_e = 1
    ea_eff = ea if ea != 0 else 1
    eb_eff = eb if eb != 0 else 1
    sig_a = (ma | 0x80) if ea != 0 else ma
    sig_b = (mb | 0x80) if eb != 0 else mb
    # value = sig_a*2**(ea_eff-134) * sig_b*2**(eb_eff-134)
    return encode(sig_a * sig_b, ea_eff + eb_eff - 268, sign)


def bf16_add(a: int, b: int) -> int:
    sa, ea, ma = _parts(a)
    sb, eb, mb = _parts(b)
    a_nan, b_nan = _is_nan(ea, ma), _is_nan(eb, mb)
    a_inf, b_inf = _is_inf(ea, ma), _is_inf(eb, mb)
    a_zero, b_zero = _is_zero(ea, ma), _is_zero(eb, mb)
    if a_nan or b_nan:
        return QNAN
    if a_inf and b_inf:
        return QNAN if sa != sb else (sa << 15) | POS_INF
    if a_inf:
        return a
    if b_inf:
        return b
    if a_zero and b_zero:
        # +0 + +0 = +0, -0 + -0 = -0, mixed = +0
        return (sa << 15) if sa == sb else 0
    if a_zero:
        return b
    if b_zero:
        return a
    # exact integer sum on a common exponent
    ea_eff = ea if ea != 0 else 1
    eb_eff = eb if eb != 0 else 1
    sig_a = (ma | 0x80) if ea != 0 else ma     # value = sig * 2**(eff-134)
    sig_b = (mb | 0x80) if eb != 0 else mb
    e0 = min(ea_eff, eb_eff)
    big_a = sig_a << (ea_eff - e0)
    big_b = sig_b << (eb_eff - e0)
    if sa == sb:
        return encode(big_a + big_b, e0 - 134, sa)
    if big_a == big_b:
        return 0                            # x + (-x) = +0 (RNE)
    if big_a > big_b:
        return encode(big_a - big_b, e0 - 134, sa)
    return encode(big_b - big_a, e0 - 134, sb)


def pack_word(a_hi: int, a_lo: int, b_hi: int, b_lo: int,
              is_add: bool) -> int:
    """Golden for one 32-bit packed word (two lanes)."""
    if is_add:
        hi, lo = bf16_add(a_hi, b_hi), bf16_add(a_lo, b_lo)
    else:
        hi, lo = bf16_mul(a_hi, b_hi), bf16_mul(a_lo, b_lo)
    return (hi << 16) | lo


def directed_specials() -> list:
    """Directed special-value operands covering every adjudicated class."""
    return [
        0x0000,  # +0
        0x8000,  # -0
        0x7F80,  # +inf
        0xFF80,  # -inf
        0x7FC0,  # qNaN
        0x7F81,  # sNaN payload
        0x0001,  # smallest denormal
        0x007F,  # largest denormal
        0x0080,  # smallest normal (2^-126)
        0x3F80,  # 1.0
        0xBF80,  # -1.0
        0x4000,  # 2.0
        0x3F81,  # 1 + 2^-7 (tie-relevant mantissa)
        0x3F7F,  # just below 1.0
        0x7F7F,  # largest normal
        0xFF7F,  # -largest normal
        0x4300,  # 2^7-ish boundary for add shift clamp
        0x3B00,  # small normal (alignment boundary, d=8 clamp region)
    ]


def self_check() -> None:
    """Sanity invariants the model must satisfy before it is trusted."""
    assert bf16_mul(0x3F80, 0x4000) == 0x4000          # 1.0 * 2.0 = 2.0
    assert bf16_mul(0x3F80, 0xBF80) == 0xBF80          # 1.0 * -1.0 = -1.0
    assert bf16_mul(0x4000, 0x4000) == 0x4080          # 2.0 * 2.0 = 4.0
    assert bf16_add(0x3F80, 0x3F80) == 0x4000          # 1 + 1 = 2
    assert bf16_add(0x3F80, 0xBF80) == 0x0000          # 1 - 1 = +0
    assert bf16_add(0x8000, 0x8000) == 0x8000          # -0 + -0 = -0
    assert bf16_add(0x7F80, 0xFF80) == QNAN            # inf - inf = NaN
    assert bf16_mul(0x7F80, 0x0000) == QNAN            # inf * 0 = NaN
    assert bf16_mul(0x0001, 0x0001) == 0x0000          # dn*dn underflows
    assert bf16_mul(0x0001, 0x4000) == 0x0002          # dn * 2 = 2^-133
    assert bf16_add(0x0001, 0x0001) == 0x0002          # dn + dn exact
    assert bf16_mul(0x7F7F, 0x407F) == 0x7F80          # overflow -> inf
    assert bf16_add(0x3F81, 0x3F81) == 0x4001          # 2 + 2^-6 exact
    # RNE tie at the bf16 mantissa LSB: 1.0 + (1.5 + 2^-7) = 2.5 + 2^-8
    # is a tie (half ULP in [2,4)); 2.5 has even mantissa -> stays 2.5.
    assert bf16_add(0x3F80, 0x3FC1) == 0x4020
    # just above the tie rounds up to the next grid point (0x4022)
    assert bf16_add(0x3F80, 0x3FC3) == 0x4022


if __name__ == "__main__":
    self_check()
    print("golden self-check passed")
