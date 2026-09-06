#include <iostream>
#include <unistd.h>
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

// System-level 1/(1+e^-x) contract (math-sfu spec): relative error <= 2^-13
// vs a double-precision reference, same adjudication domain as the unit
// oracle (tests/cocotb/math_sfu_unit/math_rtl_model.py): NaN in -> qNaN out,
// sigmoid(+inf) = 1, sigmoid(-inf) = 0, x >= 10.5 saturates to 1 (within the
// bound), deep-negative tail flushes to 0 through the ex2-core FTZ at
// x*log2e <= -126.
static const double REL_BOUND = 1.220703125e-4;  // 2^-13
static const double FTZ_LIMIT = 1.1754943508222875e-38;  // 2^-126
// One-f32-ulp-wide knife zone above 2^-126 where the fixed-point truncation
// of x*log2e may flush a barely-normal true value to zero; both a flush and
// a computed value are legal there.
static const double KNIFE_LIMIT = FTZ_LIMIT * (1.0 + 1.0 / 8388608.0);

static const char* kernel_file = "kernel.vxbin";
static vx_device_h device = nullptr;
static vx_buffer_h src_buf = nullptr;
static vx_buffer_h dst_buf = nullptr;
static vx_queue_h queue = nullptr;
static vx_module_h module_ = nullptr;
static vx_kernel_h kernel = nullptr;

static const uint32_t SPECIALS[NUM_SPECIALS] = {
    0x00000000,  // +0 -> 0.5          (zero)
    0x80000000,  // -0 -> 0.5          (zero)
    0x00000001,  // smallest subnormal (subnormal)
    0x80000001,  // negative smallest subnormal
    0x007FFFFF,  // largest subnormal  (subnormal)
    0x7F800000,  // +inf -> 1          (inf)
    0xFF800000,  // -inf -> 0          (inf)
    0x7FC00000,  // qNaN               (nan)
    0xFFC00000,  // -qNaN              (nan)
    0x7F800001,  // NaN payload        (nan)
    0x3F000000,  // 0.5
    0xBF000000,  // -0.5
    0x3F800000,  // 1.0
    0xBF800000,  // -1.0
    0x40000000,  // 2.0
    0xC0000000,  // -2.0
    0x40480000,  // 3.125
    0xC0480000,  // -3.125
    0x40A00000,  // 5.0
    0xC0A00000,  // -5.0
    0x41200000,  // 10.0
    0xC1200000,  // -10.0
    0x41280000,  // 10.5 saturation boundary (domain_edge)
    0xC1280000,  // -10.5
    0x4127FFFF,  // just below 10.5    (domain_edge)
    0x3F317218,  // ln2 ~= 0.6931472
    0xC2AA0000,  // -85.0  negative tail, still normal-domain output
    0xC2B00000,  // -88.0  deep tail, FTZ flush -> 0 (saturation_tail)
};

static float bits_to_f32(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

static bool check_sigmoid(uint32_t x_bits, uint32_t got_bits, uint32_t idx, int errors) {
    float xf = bits_to_f32(x_bits);
    float got = bits_to_f32(got_bits);
    if (std::isnan(xf)) {
        if (!std::isnan(got)) {
            if (errors < 5)
                printf("SIGMOID nan [%u]: x=0x%08x got=0x%08x expected NaN\n", idx, x_bits, got_bits);
            return false;
        }
        return true;
    }
    bool ok;
    if (xf == std::numeric_limits<float>::infinity()) {
        ok = (got == 1.0f);
    } else if (xf == -std::numeric_limits<float>::infinity()) {
        ok = (got == 0.0f);
    } else {
        double ref = 1.0 / (1.0 + std::exp(-(double)xf));
        if (std::fabs(ref) < FTZ_LIMIT) {
            ok = (got == 0.0f);
        } else if (std::fabs(ref) < KNIFE_LIMIT) {
            double err = std::fabs((double)got - ref) / std::fabs(ref);
            ok = (got == 0.0f) || (err <= REL_BOUND);
        } else {
            double err = std::fabs((double)got - ref) / std::fabs(ref);
            ok = err <= REL_BOUND;
        }
        if (!ok && errors < 5)
            printf("SIGMOID mismatch [%u]: x=0x%08x got=0x%08x ref=%.9e\n", idx, x_bits, got_bits, ref);
        return ok;
    }
    if (!ok && errors < 5)
        printf("SIGMOID mismatch [%u]: x=0x%08x got=0x%08x\n", idx, x_bits, got_bits);
    return ok;
}

void cleanup() {
    if (device) {
        if (src_buf) vx_buffer_release(src_buf);
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

    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_device_open(0, &device));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    RT_CHECK(vx_queue_create(device, &qi, &queue));

    uint64_t num_cores, num_threads;
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

    uint32_t total = (uint32_t)(num_cores * num_threads) * NUM_POINTS;
    uint32_t buf_size = total * sizeof(uint32_t);
    uint32_t special_words = NUM_SPECIALS < total ? NUM_SPECIALS : total;

    std::cout << "num_cores=" << num_cores << " num_threads=" << num_threads << std::endl;
    std::cout << "total vectors: " << total << std::endl;
    std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

    std::vector<uint32_t> h_src(total), h_dst(total);
    for (uint32_t i = 0; i < special_words; ++i)
        h_src[i] = SPECIALS[i];
    for (uint32_t i = special_words; i < total; ++i)
        h_src[i] = ((uint32_t)rand() << 17) ^ ((uint32_t)rand() << 2) ^ ((uint32_t)rand() & 3);

    std::cout << "allocate device memory" << std::endl;
    RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src_buf));
    RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buf));

    kernel_arg_t karg;
    RT_CHECK(vx_buffer_address(src_buf, &karg.src_addr));
    RT_CHECK(vx_buffer_address(dst_buf, &karg.dst_addr));

    std::cout << "upload source buffer" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, src_buf, 0, h_src.data(), buf_size, 0, nullptr, nullptr));

    std::cout << "load kernel module" << std::endl;
    RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
    RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

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

    std::cout << "download destination buffer" << std::endl;
    RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buf, 0, buf_size, 1, &launch_ev, &read_ev));

    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    std::cout << "verify results" << std::endl;
    int errors = 0;
    for (uint32_t i = 0; i < total; ++i) {
        if (!check_sigmoid(h_src[i], h_dst[i], i, errors))
            ++errors;
    }

    cleanup();

    if (errors) {
        printf("FAILED: %d mismatches out of %u vectors\n", errors, total);
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
