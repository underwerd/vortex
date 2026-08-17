#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    auto src_a    = reinterpret_cast<const uint32_t*>(arg->src_a_addr);
    auto src_b    = reinterpret_cast<const uint32_t*>(arg->src_b_addr);
    auto dst_mul  = reinterpret_cast<uint32_t*>(arg->dst_mul_addr);
    auto dst_add  = reinterpret_cast<uint32_t*>(arg->dst_add_addr);

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t p = 0; p < NUM_POINTS; ++p) {
        uint32_t idx = tid * NUM_POINTS + p;
        uint32_t a = src_a[idx];
        uint32_t b = src_b[idx];
        dst_mul[idx] = vx_packbf16_mul(a, b);
        dst_add[idx] = vx_packbf16_add(a, b);
    }
}
