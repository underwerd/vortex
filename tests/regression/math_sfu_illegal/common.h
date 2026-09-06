#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

// Words each thread writes to the result buffer:
//   [0] error code (0 = every check passed)
//   [1] trap-observed flags, bit i = illegal case i trapped with mcause=2
//       and mepc = faulting PC
//   [2] ex2(0.75)   result bits with frm=000 (all legal frm agree)
//   [3] tanh(0.75)  result bits with frm=000
//   [4] sigmoid(0.75) result bits with frm=000
//   [5..7] reserved (zero)
#define WORDS_PER_THREAD 8

// 9 illegal cases: rs2!=0 / fmt!=00 (f64 form) / frm in {101,110}, each on
// all three instructions.
#define NUM_ILLEGAL_CASES 9

struct kernel_arg_t {
    uint64_t dst_addr;
};

#endif
