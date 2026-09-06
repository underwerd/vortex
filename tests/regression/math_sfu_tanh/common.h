#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

// Points per thread; total vectors = num_cores * num_threads * NUM_POINTS.
#define NUM_POINTS 32

// Directed specials first, random f32 bit patterns after. Covers subnormal,
// inf, nan, zero, odd-symmetry (x, -x) pairs, the 2^-7 bypass edge and the
// 5.25 saturation tail.
#define NUM_SPECIALS 30

struct kernel_arg_t {
    uint64_t src_addr;
    uint64_t dst_addr;
};

#endif
