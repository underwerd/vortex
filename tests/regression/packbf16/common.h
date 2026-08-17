#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

#define NUM_POINTS 16

struct kernel_arg_t {
    uint64_t src_a_addr;
    uint64_t src_b_addr;
    uint64_t dst_mul_addr;
    uint64_t dst_add_addr;
};

#endif
