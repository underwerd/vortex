#include <iostream>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vortex2.h>
#include "common.h"

#define RT_CHECK(_expr)                                                   \
  do {                                                                    \
    int _ret = (_expr);                                                   \
    if (0 == _ret) break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, _ret);                  \
    cleanup();                                                            \
    exit(-1);                                                             \
  } while (false)

static const char* kernel_file = "kernel.vxbin";
static vx_device_h device = nullptr;
static vx_buffer_h src_a_buf = nullptr;
static vx_buffer_h src_b_buf = nullptr;
static vx_buffer_h dst_mul_buf = nullptr;
static vx_buffer_h dst_add_buf = nullptr;
static vx_queue_h queue = nullptr;
static vx_module_h module_ = nullptr;
static vx_kernel_h kernel = nullptr;

static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t round_bit = (bits >> 15) & 1;
    uint32_t sticky = bits & 0x7FFF;
    uint16_t bf16 = (bits >> 16) & 0xFFFF;
    if (round_bit && (sticky || lsb)) bf16 += 1;
    return bf16;
}

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

static uint16_t bf16_mul(uint16_t a, uint16_t b) {
    float result = bf16_to_f32(a) * bf16_to_f32(b);
    if (std::isnan(result)) return 0x7FC0;
    if (std::isinf(result)) return result > 0 ? 0x7F80 : 0xFF80;
    return f32_to_bf16(result);
}

static uint16_t bf16_add(uint16_t a, uint16_t b) {
    float result = bf16_to_f32(a) + bf16_to_f32(b);
    if (std::isnan(result)) return 0x7FC0;
    if (std::isinf(result)) return result > 0 ? 0x7F80 : 0xFF80;
    return f32_to_bf16(result);
}

static void cleanup() {
    if (device) {
        if (src_a_buf) vx_buffer_release(src_a_buf);
        if (src_b_buf) vx_buffer_release(src_b_buf);
        if (dst_mul_buf) vx_buffer_release(dst_mul_buf);
        if (dst_add_buf) vx_buffer_release(dst_add_buf);
        if (kernel) vx_kernel_release(kernel);
        if (module_) vx_module_release(module_);
        if (queue) vx_queue_release(queue);
        vx_device_dump_perf(device, stdout);
        vx_device_release(device);
    }
}

int main(int argc, char* argv[]) {
    if (argc > 2 && strcmp(argv[1], "-k") == 0)
        kernel_file = argv[2];

    uint32_t num_threads = 4;
    uint32_t total = num_threads * NUM_POINTS;

    // Generate test data
    std::vector<uint32_t> h_src_a(total), h_src_b(total);
    std::vector<uint32_t> h_dst_mul(total), h_dst_add(total);
    std::vector<uint32_t> ref_mul(total), ref_add(total);

    srand(42);
    for (uint32_t i = 0; i < total; ++i) {
        uint16_t a_lo = rand() & 0xFFFF;
        uint16_t a_hi = rand() & 0xFFFF;
        uint16_t b_lo = rand() & 0xFFFF;
        uint16_t b_hi = rand() & 0xFFFF;
        h_src_a[i] = ((uint32_t)a_hi << 16) | a_lo;
        h_src_b[i] = ((uint32_t)b_hi << 16) | b_lo;
        uint16_t r_mul_lo = bf16_mul(a_lo, b_lo);
        uint16_t r_mul_hi = bf16_mul(a_hi, b_hi);
        uint16_t r_add_lo = bf16_add(a_lo, b_lo);
        uint16_t r_add_hi = bf16_add(a_hi, b_hi);
        ref_mul[i] = ((uint32_t)r_mul_hi << 16) | r_mul_lo;
        ref_add[i] = ((uint32_t)r_add_hi << 16) | r_add_lo;
    }

    RT_CHECK(vx_device_init(&device));
    RT_CHECK(vx_queue_create(device, &queue));

    RT_CHECK(vx_buffer_allocate(device, total * sizeof(uint32_t), &src_a_buf));
    RT_CHECK(vx_buffer_allocate(device, total * sizeof(uint32_t), &src_b_buf));
    RT_CHECK(vx_buffer_allocate(device, total * sizeof(uint32_t), &dst_mul_buf));
    RT_CHECK(vx_buffer_allocate(device, total * sizeof(uint32_t), &dst_add_buf));

    RT_CHECK(vx_buffer_write(src_a_buf, h_src_a.data(), 0, total * sizeof(uint32_t)));
    RT_CHECK(vx_buffer_write(src_b_buf, h_src_b.data(), 0, total * sizeof(uint32_t)));

    RT_CHECK(vx_module_load(device, kernel_file, &module_));
    RT_CHECK(vx_kernel_create(device, module_, "kernel_main", &kernel));

    kernel_arg_t karg;
    karg.src_a_addr  = vx_buffer_addr(src_a_buf);
    karg.src_b_addr  = vx_buffer_addr(src_b_buf);
    karg.dst_mul_addr = vx_buffer_addr(dst_mul_buf);
    karg.dst_add_addr = vx_buffer_addr(dst_add_buf);

    RT_CHECK(vx_kernel_set_arg(kernel, 0, &karg, sizeof(kernel_arg_t)));
    RT_CHECK(vx_kernel_set_work_size(kernel, num_threads, 1, 1));
    RT_CHECK(vx_kernel_enqueue(kernel, queue));
    RT_CHECK(vx_queue_sync(queue));

    RT_CHECK(vx_buffer_read(dst_mul_buf, h_dst_mul.data(), 0, total * sizeof(uint32_t)));
    RT_CHECK(vx_buffer_read(dst_add_buf, h_dst_add.data(), 0, total * sizeof(uint32_t)));

    // Verify results
    int errors = 0;
    for (uint32_t i = 0; i < total; ++i) {
        if (h_dst_mul[i] != ref_mul[i]) {
            if (errors < 5)
                printf("MUL mismatch [%u]: got=0x%08x exp=0x%08x\n", i, h_dst_mul[i], ref_mul[i]);
            ++errors;
        }
        if (h_dst_add[i] != ref_add[i]) {
            if (errors < 5)
                printf("ADD mismatch [%u]: got=0x%08x exp=0x%08x\n", i, h_dst_add[i], ref_add[i]);
            ++errors;
        }
    }

    cleanup();

    if (errors) {
        printf("FAILED: %d mismatches\n", errors);
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
