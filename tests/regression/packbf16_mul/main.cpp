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
static vx_buffer_h dst_buf = nullptr;
static vx_queue_h queue = nullptr;
static vx_module_h module_ = nullptr;
static vx_kernel_h kernel = nullptr;

// Directed specials, same set as the cocotb unit testbench
// (tests/cocotb/packbf16_unit/golden_model.py::directed_specials).
static const uint16_t SPECIALS[NUM_SPECIALS] = {
    0x0000,  // +0
    0x8000,  // -0
    0x7F80,  // +inf
    0xFF80,  // -inf
    0x7FC0,  // qNaN
    0x7F81,  // sNaN payload
    0x0001,  // smallest denormal
    0x007F,  // largest denormal
    0x0080,  // smallest normal (2^-126)
    0x3F80,  // 1.0
    0xBF80,  // -1.0
    0x4000,  // 2.0
    0x3F81,  // 1 + 2^-7 (tie-relevant mantissa)
    0x3F7F,  // just below 1.0
    0x7F7F,  // largest normal
    0xFF7F,  // -largest normal
    0x4300,  // 2^7-ish boundary for add shift clamp
    0x3B00,  // small normal (alignment boundary, d=8 clamp region)
};

// Host reference: bf16 operands widened to f32, exact f32 product, then one
// RNE step to bf16. Since the f32 mantissa (p1=24) exceeds the bf16 mantissa
// (p2=8) by more than 2 bits, double rounding cannot differ from the exact
// product rounded once (Figueroa's theorem), so this matches the RTL's
// single-step RNE bit-for-bit on all finite inputs.
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

void cleanup() {
    if (device) {
        if (src_a_buf) vx_buffer_release(src_a_buf);
        if (src_b_buf) vx_buffer_release(src_b_buf);
        if (dst_buf) vx_buffer_release(dst_buf);
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

    srand(42);

    // open device connection
    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_device_open(0, &device));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    RT_CHECK(vx_queue_create(device, &qi, &queue));

    uint64_t num_cores, num_threads;
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

    uint32_t total = (uint32_t)(num_cores * num_threads) * NUM_POINTS;
    uint32_t buf_size = total * sizeof(uint32_t);
    uint32_t special_words = SPECIAL_WORDS < total ? SPECIAL_WORDS : total;

    std::cout << "num_cores=" << num_cores << " num_threads=" << num_threads << std::endl;
    std::cout << "total elements: " << total << std::endl;
    std::cout << "buffer size: " << buf_size << " bytes" << std::endl;
    std::cout << "directed specials words: " << special_words << std::endl;

    // generate test data: the first SPECIAL_WORDS words carry the full
    // 18x18 directed-special cross product (two bf16 pairs per packed
    // word), the rest are random bf16 patterns.
    std::vector<uint32_t> h_src_a(total), h_src_b(total);
    std::vector<uint32_t> h_dst(total), ref(total);

    for (uint32_t p = 0; (p < NUM_SPECIALS * NUM_SPECIALS) && (p / 2 < total); ++p) {
        uint32_t w = p / 2;
        uint16_t a = SPECIALS[p / NUM_SPECIALS];
        uint16_t b = SPECIALS[p % NUM_SPECIALS];
        if (p & 1) {
            h_src_a[w] |= (uint32_t)a << 16;
            h_src_b[w] |= (uint32_t)b << 16;
        } else {
            h_src_a[w] = a;
            h_src_b[w] = b;
        }
    }
    for (uint32_t i = special_words; i < total; ++i) {
        uint16_t a_lo = rand() & 0xFFFF;
        uint16_t a_hi = rand() & 0xFFFF;
        uint16_t b_lo = rand() & 0xFFFF;
        uint16_t b_hi = rand() & 0xFFFF;
        h_src_a[i] = ((uint32_t)a_hi << 16) | a_lo;
        h_src_b[i] = ((uint32_t)b_hi << 16) | b_lo;
    }

    for (uint32_t i = 0; i < total; ++i) {
        uint16_t a_lo = h_src_a[i] & 0xFFFF;
        uint16_t a_hi = (h_src_a[i] >> 16) & 0xFFFF;
        uint16_t b_lo = h_src_b[i] & 0xFFFF;
        uint16_t b_hi = (h_src_b[i] >> 16) & 0xFFFF;
        uint16_t r_lo = bf16_mul(a_lo, b_lo);
        uint16_t r_hi = bf16_mul(a_hi, b_hi);
        ref[i] = ((uint32_t)r_hi << 16) | r_lo;
    }

    // allocate device memory
    std::cout << "allocate device memory" << std::endl;
    RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src_a_buf));
    RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src_b_buf));
    RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buf));

    kernel_arg_t karg;
    RT_CHECK(vx_buffer_address(src_a_buf, &karg.src_a_addr));
    RT_CHECK(vx_buffer_address(src_b_buf, &karg.src_b_addr));
    RT_CHECK(vx_buffer_address(dst_buf, &karg.dst_addr));

    // upload source buffers
    std::cout << "upload source buffers" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, src_a_buf, 0, h_src_a.data(), buf_size, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(queue, src_b_buf, 0, h_src_b.data(), buf_size, 0, nullptr, nullptr));

    // load kernel module
    std::cout << "load kernel module" << std::endl;
    RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
    RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

    // launch kernel
    std::cout << "launch kernel" << std::endl;
    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    {
        vx_launch_info_t li = {};
        li.struct_size  = sizeof(li);
        li.kernel       = kernel;
        li.args_host    = &karg;
        li.args_size    = sizeof(karg);
        li.ndim         = 1;
        li.grid_dim[0]  = (uint32_t)num_cores;
        li.block_dim[0] = (uint32_t)num_threads;
        RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    }

    // download destination buffer
    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buf, 0, buf_size, 1, &launch_ev, &read_ev));

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    // verify results
    std::cout << "verify results" << std::endl;
    int errors = 0;
    for (uint32_t i = 0; i < total; ++i) {
        if (h_dst[i] != ref[i]) {
            if (errors < 5)
                printf("MUL mismatch [%u]: got=0x%08x exp=0x%08x (a=0x%08x b=0x%08x)\n",
                       i, h_dst[i], ref[i], h_src_a[i], h_src_b[i]);
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
