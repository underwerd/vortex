#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

// Rows handled per thread and columns per row; total output vectors =
// num_cores * num_threads * ROWS_PER_THREAD * NUM_COLS.
#define ROWS_PER_THREAD 4
#define NUM_COLS 64

struct kernel_arg_t {
    uint64_t src_addr;
    uint64_t dst_addr;
};

#endif
