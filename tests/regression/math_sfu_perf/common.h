#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

// SFU perf microbenchmark (math-sfu-002 follow-up):
//   segment 0  empty calibration (vx_rdcycle_sync overhead)
//   segment 1  independent stream  f_k = ex2(src)  (throughput)
//   segment 2  dependency chain    d   = ex2(d)    (single-issue latency)
//
// The independent stream must read a CONSTANT source: f_k = ex2(f_k) would
// form interleaved dependency chains, not an issue-ready stream.
//
// n_blocks must be a multiple of UNROLL (32): each loop iteration issues
// 32 independent fex2. The chain loop is RAW-bound; its integer loop
// overhead executes under the 35-cycle stall and does not pollute b_dep.

#define PERF_UNROLL 32

// out slot layout per thread (uint64_t each)
#define OUT_EMPTY 0   // cycles: rdcycle_sync -> rdcycle_sync, no work
#define OUT_INDEP 1   // cycles: independent stream segment
#define OUT_CHAIN 2   // cycles: dependency chain segment
#define OUT_I0    3   // absolute core-cycle stamp, indep segment start
#define OUT_I1    4   // absolute core-cycle stamp, indep segment end
#define OUT_TAIL  5   // DCE guard: ex2(d) bits << 32 | indep checksum
#define OUT_SLOTS 6

struct kernel_arg_t {
    uint64_t out_addr;
    uint32_t n_blocks;  // independent-stream fex2 count (multiple of 32)
    uint32_t n_chain;   // dependency-chain length
};

#endif
