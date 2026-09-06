#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

// Points per thread; total vectors = num_cores * num_threads * NUM_POINTS.
#define NUM_POINTS 32

// Directed specials first, random f32 bit patterns after. Covers subnormal,
// inf, nan, zero, overflow saturation, underflow saturation (FTZ tail) and
// the domain edges at +/-0.5, x=128 and x=-126.
#define NUM_SPECIALS 20

struct kernel_arg_t {
    uint64_t src_addr;
    uint64_t dst_addr;
};

#endif
