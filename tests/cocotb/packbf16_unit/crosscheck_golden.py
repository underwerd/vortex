"""Cross-validation of the integer-exact golden model.

Independent second implementation: float32-representable operands,
float64 intermediate, single-step RNE to bf16 (theorem p1 >= p2+2
makes the f32-then-bf16 path equal to one correct rounding step).
The two implementations agreeing bit-exactly over specials^2 and a
large random sweep is the evidence that the golden model itself is
correct (S3 self-check, consumed by the V3 report).

Run:  python3 crosscheck_golden.py [num_random_pairs]
"""
import math
import random
import struct
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from golden_model import QNAN, bf16_add, bf16_mul, directed_specials


def _f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def _to_f32(a):
    return struct.unpack('<f', struct.pack('<I', (a & 0xFFFF) << 16))[0]


def f64_ref(op, a, b):
    fa, fb = _to_f32(a), _to_f32(b)
    r = fa * fb if op == 'mul' else fa + fb
    if math.isnan(r):
        return QNAN
    if math.isinf(r) or abs(r) >= 2.0 ** 128:
        return 0x7F80 if r > 0 else 0xFF80
    b32 = struct.unpack('<I', struct.pack('<f', _f32(r)))[0]
    bf = b32 >> 16
    if ((b32 >> 15) & 1) and ((b32 & 0x7FFF) or (bf & 1)):
        bf = (bf + 1) & 0xFFFF
    return bf


def main() -> int:
    n_random = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    sp = directed_specials()
    bad = []
    for a in sp:
        for b in sp:
            for op, fn in (("mul", bf16_mul), ("add", bf16_add)):
                g, r = fn(a, b), f64_ref(op, a, b)
                if g != r:
                    bad.append((op, hex(a), hex(b), hex(g), hex(r)))
    random.seed(7)
    for _ in range(n_random):
        a, b = random.getrandbits(16), random.getrandbits(16)
        for op, fn in (("mul", bf16_mul), ("add", bf16_add)):
            g, r = fn(a, b), f64_ref(op, a, b)
            if g != r:
                bad.append((op, hex(a), hex(b), hex(g), hex(r)))
    if bad:
        print("CROSS-CHECK MISMATCH: %d" % len(bad))
        for x in bad[:10]:
            print("  ", x)
        return 1
    print("cross-check OK: %d specials^2 + %d random pairs agree "
          "bit-exactly" % (len(sp) ** 2, n_random))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
