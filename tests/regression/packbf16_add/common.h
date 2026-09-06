#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

#define NUM_POINTS 64

// 18 directed specials x 18 = 324 operand pairs fill the first 162
// packed words (two bf16 lanes per word); the rest of the buffer is
// random patterns. Covers subnormal, inf, nan, zero and RNE ties.
#define NUM_SPECIALS 18
#define SPECIAL_WORDS (NUM_SPECIALS * NUM_SPECIALS / 2)

struct kernel_arg_t {
    uint64_t src_a_addr;
    uint64_t src_b_addr;
    uint64_t dst_addr;
};

#endif
