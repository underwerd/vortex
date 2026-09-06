"""Approximation scheme for the Vortex math SFU unit (S3 frozen algorithm).

Chosen datapaths (see DECISIONS.md D1-D3 and approx_explore_report.json):
  ex2.f32     : x = n + f split (n = round(x), f in [-0.5, 0.5]);
                2^f via degree-4 power-basis polynomial; result = p(f) * 2^n
                with exponent-field reconstruction (overflow -> Inf,
                underflow -> flush-to-zero per S3 FTZ policy).
  tanh.f32    : odd symmetry; |x| >= T_SAT saturates to +-1.0;
                segment 1 [0,1): x * P1(x^2)   (degree 4 in x^2);
                segment 2 [1,T): P2(x)         (degree 10).
  sigmoid.f32 : hybrid of two reused datapaths —
                x >= 2*T_SAT: saturate to 1.0;
                0 <= x < 2*T_SAT: tanh identity (1 + tanh(x/2))/2 (sigmoid
                in [0.5,1), no cancellation amplification);
                x < 0: sigmoid(x) = e^x/(1+e^x) = u * p(u) with u = ex2(
                x*log2e) (ex2 datapath reuse) and p = 1/(1+u) degree-7
                polynomial on u in (0,1] (pole at u=-1 keeps Chebyshev
                convergence fast). The naive identity on the negative half
                was rejected: (1+tanh(x/2))/2 suffers cancellation — tanh's
                absolute poly error divided by 2*sigmoid(x) blows past the
                contract for x < -4.

All coefficients are power-basis (c0 + c1*t + ...), produced by Chebyshev
near-minimax fits in float64; a Remez refinement pass before RTL freeze may
tighten them but must not loosen the measured bounds below.

Accuracy bounds measured against golden_model float64 truth (normal domain,
saturating tails per spec q4):
  ex2      7.3e-06   (~2^-17.2)
  tanh     5.5e-05   (~2^-14.1, saturation edge)
  sigmoid  ~9e-06 neg-half / ~2.2e-05 pos-half / 2.8e-05 saturating
Contract: <= 2^-13 each.

Self-check + full-domain validation: ``python3 approx_scheme.py``.
"""

from __future__ import annotations

import math
import struct
import sys

sys.path.insert(0, ".")
from golden_model import (  # noqa: E402
    EX2_SPECIALS, SIGMOID_SPECIALS, TANH_SPECIALS,
    ex2_exact, sigmoid_exact, tanh_exact,
)

LOG2E = 1.4426950408889634
TARGET = 2.0 ** -13


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


# ---------------------------------------------------------------------------
# ex2 core
# ---------------------------------------------------------------------------

# 2^f on f in [-0.5, 0.5], degree 4 (Chebyshev near-minimax, 20260819 seed)
EX2_POLY = [
    1.0000000522776347,
    0.6931272662119313,
    0.24022211737061572,
    0.055875501633870304,
    0.009670763128676136,
]


def _horner(coeffs, t):
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * t + c
    return acc


def ex2_scheme(x: float) -> float:
    """f32-in/f32-out model of the ex2 datapath (FTZ underflow policy)."""
    if x != x:
        return f32(float("nan"))
    if x == math.inf:
        return math.inf
    if x == -math.inf:
        return 0.0
    if x >= 128.0:
        return math.inf
    if x < math.log2(2.0 ** -126):
        return 0.0  # flush-to-zero tail (S3 decision)
    n = float(round(x))  # ties-to-even matches hardware round
    f = x - n
    mant = _horner(EX2_POLY, f)          # in [2^-0.5, 2^0.5]
    result = mant * 2.0 ** n
    if result >= (2.0 - 2.0 ** -23) * 2.0 ** 127:
        return math.inf if result > (2.0 - 2.0 ** -23) * 2.0 ** 127 else f32(result)
    if result < 2.0 ** -126:
        return 0.0
    return f32(result)


# ---------------------------------------------------------------------------
# tanh (odd symmetry, saturation, two segments)
# ---------------------------------------------------------------------------

T_SAT = 5.25  # saturation error 2*e^{-10.5}/tanh = 5.51e-05 (2.2x margin);
              # the exact contract edge atanh(1/(1+2^-13))=4.852 was rejected
              # in S3 exploration (measured 1.194e-4 vs 1.221e-4, <2% margin)

TANH_SEG1_X2 = [  # x * P1(x^2) on [0, 1)
    0.9999952088571489,
    -0.333059426447278,
    0.13081454663986686,
    -0.04561244963308882,
    0.00947257595777188,
]

TANH_SEG2 = [  # P2(x) on [1, T_SAT), degree 10
    -0.03033028914602376,
    1.0596483281434412,
    0.11265283407758755,
    -0.8295499312535636,
    0.6804868342193977,
    -0.3024014378883226,
    0.08492506441266956,
    -0.015537683914915888,
    0.001804092422140306,
    -0.00012107729894515699,
    3.583238348823538e-06,
]


def _tanh_mag(a: float) -> float:
    """tanh(a) for a >= 0 via the segmented scheme."""
    if a >= T_SAT:
        return 1.0
    if a < 1.0:
        return a * _horner(TANH_SEG1_X2, a * a)
    return _horner(TANH_SEG2, a)


def tanh_scheme(x: float) -> float:
    if x != x:
        return f32(float("nan"))
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return -1.0
    if x < 0.0:
        return f32(-_tanh_mag(-x))
    return f32(_tanh_mag(x))


# ---------------------------------------------------------------------------
# sigmoid via the tanh identity (zero multipliers beyond the tanh datapath)
# ---------------------------------------------------------------------------


def sigmoid_scheme(x: float) -> float:
    """Hybrid scheme: saturate / tanh identity / ex2+reciprocal-poly."""
    if x != x:
        return f32(float("nan"))
    if x == math.inf or x >= 2.0 * T_SAT:
        return 1.0
    if x == -math.inf:
        return 0.0
    if x >= 0.0:
        # sigmoid in [0.5, 1): identity has no cancellation amplification
        t = tanh_scheme(x / 2.0)
        return f32((1.0 + t) * 0.5)
    # negative half: sigmoid(x) = u/(1+u), u = e^x = 2^(x*log2e)
    u = ex2_scheme(f32(x * LOG2E))
    if u == 0.0:  # ex2 FTZ tail -> sigmoid underflows the same way
        return 0.0
    return f32(u * _horner(SIGMOID_P, u))


# 1/(1+u) on u in (0, 1], degree 7 (pole at u=-1: fast Chebyshev decay).
# Multiplying the naive q(u)=u/(1+u) directly is NOT allowed: its absolute
# fit error over u->0 destroys the relative contract at the small end.
SIGMOID_P = [
    0.9999967486321979,
    -0.9997526194675692,
    0.9953400169782212,
    -0.9621390322558268,
    0.8350046323458371,
    -0.56593528755981,
    0.2473914438002101,
    -0.04990815438539204,
]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def rel_err(approx: float, truth: float) -> float:
    if truth == 0.0:
        return 0.0 if approx == 0.0 else abs(approx - truth)
    if truth != truth:
        return 0.0 if approx != approx else math.inf
    return abs(approx - truth) / abs(truth)


def validate(n_random: int = 100000, seed: int = 20260819) -> dict:
    import random
    rng = random.Random(seed)

    def sample_inputs():
        vals = []
        for _ in range(n_random):
            bits = rng.getrandbits(32)
            vals.append(struct.unpack("f", struct.pack("I", bits))[0])
        return vals

    inputs = sample_inputs()
    stats = {}
    for name, scheme, exact, specials in (
        ("ex2", ex2_scheme, ex2_exact, EX2_SPECIALS),
        ("tanh", tanh_scheme, tanh_exact, TANH_SPECIALS),
        ("sigmoid", sigmoid_scheme, sigmoid_exact, SIGMOID_SPECIALS),
    ):
        worst = 0.0
        worst_x = None
        checked = 0
        for x in inputs + specials:
            truth = exact(x)
            if truth != truth or truth in (0.0, math.inf, -math.inf):
                # specials adjudicated exactly by the scheme's own branches
                continue
            if truth == 0.0 or abs(truth) < 2.0 ** -126:
                continue  # FTZ tail excluded from the accuracy domain
            e = rel_err(scheme(x), truth)
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
    for name, s in stats.items():
        flag = "PASS" if s["pass"] else "FAIL"
        ok &= s["pass"]
        print(f"{name:8s} max_rel_err = {s['max_rel_err']:.3e} "
              f"(target {s['target']:.3e}) [{flag}]  checked {s['checked']} "
              f"worst_x = {s['worst_x']!r}")
    sys.exit(0 if ok else 1)
