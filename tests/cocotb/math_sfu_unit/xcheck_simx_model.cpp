// Cross-check harness: simx math_sfu C++ model vs the Python RTL-model
// oracle (math_rtl_model.py). S4 sanity gate; V4a cocotb coverage keeps
// this contract alive long-term.
//
// Usage:
//   sed -n '/^namespace math_sfu/,/^} \/\/ namespace math_sfu/p' \
//       sim/simx/fpu_unit.cpp > math_sfu_model.inc   (regenerate if model moves)
//   python3 gen_xcheck_vectors.py > /tmp/math_xcheck.txt  (or see below)
//   g++ -O2 -std=c++17 -o /tmp/math_xcheck xcheck_simx_model.cpp && /tmp/math_xcheck
#include <cstdint>
#include <cstdio>

#include "math_sfu_model.inc"

int main() {
  FILE* f = fopen("/tmp/math_xcheck.txt", "r");
  if (!f) { perror("open /tmp/math_xcheck.txt"); return 1; }
  unsigned x, e2, th, sg;
  size_t n = 0, bad = 0;
  while (fscanf(f, "%08x %08x %08x %08x", &x, &e2, &th, &sg) == 4) {
    ++n;
    unsigned ge = math_sfu::ex2(x), gt = math_sfu::tanh(x), gs = math_sfu::sigmoid(x);
    if (ge != e2 || gt != th || gs != sg) {
      ++bad;
      if (bad <= 10)
        printf("MISMATCH x=%08x: ex2 %08x!=%08x tanh %08x!=%08x sigmoid %08x!=%08x\n",
               x, ge, e2, gt, th, gs, sg);
    }
  }
  fclose(f);
  printf("checked=%zu mismatches=%zu %s\n", n, bad, bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
