#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

// Points per thread; total vectors = num_cores * num_threads * NUM_POINTS.
#define NUM_POINTS 32

// Directed specials first, random f32 bit patterns after. Covers subnormal,
// inf, nan, zero, the +-10.5 saturation edges and both tails (saturation
// toward 1 on the positive side, FTZ-to-0 flush on the deep negative side).
#define NUM_SPECIALS 28

struct kernel_arg_t {
    uint64_t src_addr;
    uint64_t dst_addr;
};

#endif
