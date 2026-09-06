#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// volatile alone does NOT fix destination coalescing: __asm__ volatile
// prevents deletion/reordering, but the register allocator is still free to
// assign every DEAD output to the same scratch register (observed in
// disassembly: 25 of 32 unrolled vx_ex2(src) became "fex2.s f14,f15",
// turning the "independent" stream into a WAW chain). LOCAL explicit
// register variables pin the 8 destinations so the emitted stream rotates
// f16-f23 regardless of liveness.
#define VFEX2(fd, fs)                                                    \
    asm volatile(".insn r %2, %1, %3, %0, %4, x0"                       \
                 : "=f"(fd)                                              \
                 : "i"(0), "i"(0x53), "i"(0x30), "f"(fs))  // fex2.s

// One warp measures itself against the core cycle counter (mcycle is
// core-wide: every thread of the warp reads the same value at the same
// warp-synchronous instruction boundary).
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    auto out = reinterpret_cast<uint64_t*>(arg->out_addr);

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 0.5 keeps every result finite (2^0.5); the chain saturates to +Inf
    // after a few iterations -- fixed-latency pipeline, timing unaffected.
    const float src = 0.5f;

    // pinned destinations (local register variables)
    register float f0 __asm__("f16");
    register float f1 __asm__("f17");
    register float f2 __asm__("f18");
    register float f3 __asm__("f19");
    register float f4 __asm__("f20");
    register float f5 __asm__("f21");
    register float f6 __asm__("f22");
    register float f7 __asm__("f23");

    // --- segment 0: empty calibration --------------------------------
    uint64_t t0 = vx_rdcycle_sync();
    uint64_t t1 = vx_rdcycle_sync();
    uint64_t c_empty = t1 - t0;

    // --- segment 1: independent stream (peak issue rate) -------------
    // 8 PINNED rotating destinations f16-f23, constant source: no RAW;
    // dst reuse period (8 instrs) exceeds the writeback latency once the
    // stream runs at queue rate, so WAW does not bind either.
    f0 = src; f1 = src; f2 = src; f3 = src;
    f4 = src; f5 = src; f6 = src; f7 = src;
    uint64_t i0 = vx_rdcycle_sync();
    for (uint32_t r = 0; r < arg->n_blocks; r += PERF_UNROLL) {
        VFEX2(f0, src); VFEX2(f1, src); VFEX2(f2, src); VFEX2(f3, src);
        VFEX2(f4, src); VFEX2(f5, src); VFEX2(f6, src); VFEX2(f7, src);
        VFEX2(f0, src); VFEX2(f1, src); VFEX2(f2, src); VFEX2(f3, src);
        VFEX2(f4, src); VFEX2(f5, src); VFEX2(f6, src); VFEX2(f7, src);
        VFEX2(f0, src); VFEX2(f1, src); VFEX2(f2, src); VFEX2(f3, src);
        VFEX2(f4, src); VFEX2(f5, src); VFEX2(f6, src); VFEX2(f7, src);
        VFEX2(f0, src); VFEX2(f1, src); VFEX2(f2, src); VFEX2(f3, src);
        VFEX2(f4, src); VFEX2(f5, src); VFEX2(f6, src); VFEX2(f7, src);
    }
    uint64_t i1 = vx_rdcycle_sync();
    uint64_t c_indep = i1 - i0;

    // --- segment 2: dependency chain (issue->writeback latency) ------
    float d = src;
    uint64_t j0 = vx_rdcycle_sync();
    for (uint32_t r = 0; r < arg->n_chain; ++r) {
        float d_n;
        VFEX2(d_n, d);   // RAW: next iteration reads this result
        d = d_n;
    }
    uint64_t j1 = vx_rdcycle_sync();
    uint64_t c_chain = j1 - j0;

    // --- consume (DCE guard) + report ---------------------------------
    uint32_t a0, a1, a2, a3, a4, a5, a6, a7, ad;
    __builtin_memcpy(&a0, &f0, 4); __builtin_memcpy(&a1, &f1, 4);
    __builtin_memcpy(&a2, &f2, 4); __builtin_memcpy(&a3, &f3, 4);
    __builtin_memcpy(&a4, &f4, 4); __builtin_memcpy(&a5, &f5, 4);
    __builtin_memcpy(&a6, &f6, 4); __builtin_memcpy(&a7, &f7, 4);
    __builtin_memcpy(&ad, &d, 4);
    uint32_t checksum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;

    uint64_t* row = out + (uint64_t)tid * OUT_SLOTS;
    row[OUT_EMPTY] = c_empty;
    row[OUT_INDEP] = c_indep;
    row[OUT_CHAIN] = c_chain;
    row[OUT_I0]    = i0;
    row[OUT_I1]    = i1;
    row[OUT_TAIL]  = ((uint64_t)ad << 32) | checksum;
}
