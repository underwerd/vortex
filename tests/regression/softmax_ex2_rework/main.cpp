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

// Softmax reworked onto the math-SFU ex2 instruction: the kernel composes
// exp(v) as ex2(v * log2e) after subtracting the row max. This host check
// is the output-equivalence leg against the double-precision softmax gold.
//
// Error budget: each ex2 output carries relative error <= 2^-13 (spec
// contract); the normalized output y_i = E_i / sum(E_j) then carries at
// most ~2 * 2^-13 (the sum's error is a softmax-weighted average of the
// per-exp errors, not a sum of magnitudes), so the 2^-11 bound below has
// >2x margin while still catching real regressions.
static const double REL_BOUND = 2.44140625e-4;   // 2^-11
static const double ABS_FLOOR = 1e-9;

static const char* kernel_file = "kernel.vxbin";
static vx_device_h device = nullptr;
static vx_buffer_h src_buf = nullptr;
static vx_buffer_h dst_buf = nullptr;
static vx_queue_h queue = nullptr;
static vx_module_h module_ = nullptr;
static vx_kernel_h kernel = nullptr;

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

    srand(20260820);

    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_device_open(0, &device));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    RT_CHECK(vx_queue_create(device, &qi, &queue));

    uint64_t num_cores, num_threads;
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

    uint32_t nthreads = (uint32_t)(num_cores * num_threads);
    uint32_t num_rows = nthreads * ROWS_PER_THREAD;
    uint32_t total = num_rows * NUM_COLS;
    uint32_t buf_size = total * sizeof(float);

    std::cout << "num_cores=" << num_cores << " num_threads=" << num_threads << std::endl;
    std::cout << "rows=" << num_rows << " cols=" << NUM_COLS << std::endl;
    std::cout << "total vectors: " << total << std::endl;
    std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

    // Directed rows first (rowmax-subtraction stress), random rows after.
    std::vector<float> h_src(total);
    for (uint32_t row = 0; row < num_rows; ++row) {
        float* p = &h_src[row * NUM_COLS];
        switch (row & 3) {
        case 0:  // all-equal row: exact uniform softmax
            for (uint32_t i = 0; i < NUM_COLS; ++i) p[i] = 0.25f;
            break;
        case 1:  // ramp row
            for (uint32_t i = 0; i < NUM_COLS; ++i) p[i] = (float)i * 0.05f - 1.5f;
            break;
        case 2:  // large positive offset: without rowmax subtraction the
                 // exp composition overflows to +inf and the row turns NaN
            for (uint32_t i = 0; i < NUM_COLS; ++i)
                p[i] = 90.0f + (float)(i % 8) * 0.5f;
            break;
        default:  // mixed-sign row
            for (uint32_t i = 0; i < NUM_COLS; ++i)
                p[i] = ((float)rand() / RAND_MAX - 0.5f) * 12.0f;
            break;
        }
    }

    std::vector<float> h_dst(total);

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
    for (uint32_t row = 0; row < num_rows; ++row) {
        const float* s = &h_src[row * NUM_COLS];
        const float* d = &h_dst[row * NUM_COLS];

        double max = s[0];
        for (uint32_t i = 1; i < NUM_COLS; ++i)
            if (s[i] > max) max = s[i];

        double sum = 0.0;
        for (uint32_t i = 0; i < NUM_COLS; ++i)
            sum += std::exp((double)s[i] - max);

        double row_sum = 0.0;
        for (uint32_t i = 0; i < NUM_COLS; ++i) {
            double ref = std::exp((double)s[i] - max) / sum;
            double got = d[i];
            row_sum += got;
            double abs_err = std::fabs(got - ref);
            bool ok = abs_err <= ABS_FLOOR ||
                      abs_err / std::fabs(ref) <= REL_BOUND;
            if (!ok) {
                if (errors < 5)
                    printf("SOFTMAX_EX2 mismatch row %u col %u: got=%.9e ref=%.9e\n",
                           row, i, got, ref);
                ++errors;
            }
        }
        // a softmax row must sum to 1 within the same contract
        if (std::fabs(row_sum - 1.0) > NUM_COLS * REL_BOUND) {
            if (errors < 5)
                printf("SOFTMAX_EX2 row %u sum=%.9f != 1.0\n", row, row_sum);
            ++errors;
        }
    }

    cleanup();

    if (errors) {
        printf("FAILED: %d mismatches out of %u vectors\n", errors, total);
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
