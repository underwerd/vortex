"""Golden reference model for the Vortex math SFU instructions.

Semantics source: tasks/math-sfu/charter.md (2026-08-19 rulings) and the
S1 spec (math-sfu-spec-v2).  The golden computes each function in float64
and adjudicates IEEE-754 special values; two views are exported:

  * ``*_exact``   — float64 mathematical truth (accuracy-benchmark basis)
  * ``*_golden``  — truth rounded to f32 via a single RNE step (output-
                    comparison basis for system-level host checks)

Underflow policy (S3 decision, recorded in DECISIONS.md): results whose
magnitude would fall below the f32 normal range are saturated to zero
(flush-to-zero tail); the accuracy contract cites the normal domain
per spec open-question q4.  Inputs that are NaN propagate a quiet NaN.

Self-check: ``python3 golden_model.py`` runs the invariant suite.
"""

from __future__ import annotations

import math
import struct
import sys

LOG2E = 1.4426950408889634  # log2(e)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def f32(value: float) -> float:
    """Round a float64 to float32 precision (RNE) and back to float64."""
    return struct.unpack("f", struct.pack("f", value))[0]


def is_nan(x: float) -> bool:
    return x != x


def _nan() -> float:
    return f32(float("nan"))


# ---------------------------------------------------------------------------
# ex2.f32 : 2^x
# ---------------------------------------------------------------------------

F32_MAX_EXP = 127                 # largest normal exponent
F32_MIN_NORMAL = 2.0 ** -126      # flush-to-zero threshold (S3 policy)
F32_MAX = f32((2.0 - 2.0 ** -23) * 2.0 ** 127)


def ex2_exact(x: float) -> float:
    """Float64 truth for 2^x with IEEE-style adjudication.

    Returns +Inf on overflow, +0 on underflow/flush, quiet NaN for NaN.
    """
    if is_nan(x):
        return float("nan")
    if x == math.inf:
        return math.inf
    if x == -math.inf:
        return 0.0
    if x >= 128.0:  # 2^128 overflows f32 even with the largest mantissa
        return math.inf
    if x < math.log2(F32_MIN_NORMAL):  # below FTZ threshold
        return 0.0
    return 2.0 ** x


def ex2_golden(x: float) -> float:
    return f32(ex2_exact(x))


# ---------------------------------------------------------------------------
# tanh.f32
# ---------------------------------------------------------------------------

def tanh_exact(x: float) -> float:
    if is_nan(x):
        return float("nan")
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return -1.0
    return math.tanh(x)


def tanh_golden(x: float) -> float:
    return f32(tanh_exact(x))


# ---------------------------------------------------------------------------
# sigmoid.f32 : 1 / (1 + e^-x)
# ---------------------------------------------------------------------------

def sigmoid_exact(x: float) -> float:
    if is_nan(x):
        return float("nan")
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    if x >= 89.0:   # e^-x vanishes well below f32 precision
        return 1.0
    if x <= -104.0:  # e^x vanishes below FTZ threshold
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_golden(x: float) -> float:
    return f32(sigmoid_exact(x))


# ---------------------------------------------------------------------------
# directed specials (shared by unit tests and system regressions)
# ---------------------------------------------------------------------------

EX2_SPECIALS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5,
    126.0, 126.999999, 127.0, 127.9999, 128.0, 1000.0,
    -126.0, -126.9999, -127.0, -149.0, -150.0, -1000.0,
    float("inf"), float("-inf"), float("nan"),
]

TANH_SPECIALS = [
    0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 5.0, -5.0, 5.2, 9.0, 9.011, 20.0, -20.0,
    float("inf"), float("-inf"), float("nan"),
]

SIGMOID_SPECIALS = [
    0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 10.0, -10.0, 20.0, -20.0, 88.0, -104.0,
    float("inf"), float("-inf"), float("nan"),
]


# ---------------------------------------------------------------------------
# self-check invariants
# ---------------------------------------------------------------------------

def self_check() -> int:
    checks = 0

    # ex2: identity anchors and saturation
    assert ex2_exact(0.0) == 1.0 and ex2_exact(1.0) == 2.0
    assert ex2_exact(-1.0) == 0.5 and ex2_exact(10.0) == 1024.0
    assert ex2_exact(128.0) == math.inf
    assert ex2_exact(-150.0) == 0.0
    assert is_nan(ex2_exact(float("nan")))
    checks += 1

    # tanh: odd symmetry, limits, |tanh| <= 1
    for x in (0.25, 1.7, 4.9, 12.0):
        assert tanh_exact(x) == -tanh_exact(-x)
    assert tanh_exact(float("inf")) == 1.0
    assert tanh_exact(float("-inf")) == -1.0
    assert all(abs(tanh_exact(v)) <= 1.0 for v in TANH_SPECIALS if not is_nan(v))
    checks += 1

    # sigmoid: symmetry sigmoid(x) + sigmoid(-x) == 1, limits
    for x in (0.25, 1.7, 9.5, 40.0):
        s = sigmoid_exact(x) + sigmoid_exact(-x)
        assert abs(s - 1.0) < 1e-12, s
    assert sigmoid_exact(float("inf")) == 1.0
    assert sigmoid_exact(float("-inf")) == 0.0
    checks += 1

    # f32 view: rounding must not break the f32-representable anchors
    assert ex2_golden(0.5) == f32(math.sqrt(2.0))
    assert tanh_golden(0.0) == 0.0
    assert sigmoid_golden(0.0) == 0.5
    checks += 1

    return checks


if __name__ == "__main__":
    n = self_check()
    print(f"golden_model self-check: {n} invariant groups OK")
    sys.exit(0)
