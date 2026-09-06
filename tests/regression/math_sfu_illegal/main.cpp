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

// Illegal-encoding contract (spec q3 ruling): rs2!=00000, fmt!=00 (f64
// form), and frm in {101,110} each raise an illegal-instruction trap
// (mcause=2, mepc = faulting PC) on ex2/tanh/sigmoid, on both simulators;
// legal frm values 000-100/111 execute identically (rounding ignored).
// The kernel verifies mcause/mepc in-place and reports per-thread status;
// the host asserts every thread saw every trap and no legal-frm anomaly.

static const char* kernel_file = "kernel.vxbin";
static vx_device_h device = nullptr;
static vx_buffer_h dst_buf = nullptr;
static vx_queue_h queue = nullptr;
static vx_module_h module_ = nullptr;
static vx_kernel_h kernel = nullptr;

static const char* illegal_case_name(int bit) {
    switch (bit) {
        case 0: return "ex2 rs2!=0";
        case 1: return "tanh rs2!=0";
        case 2: return "sigmoid rs2!=0";
        case 3: return "ex2 fmt=f64";
        case 4: return "tanh fmt=f64";
        case 5: return "sigmoid fmt=f64";
        case 6: return "ex2 frm=101";
        case 7: return "tanh frm=110";
        case 8: return "sigmoid frm=101";
        default: return "?";
    }
}

void cleanup() {
    if (device) {
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

    std::cout << "open device connection" << std::endl;
    RT_CHECK(vx_device_open(0, &device));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    RT_CHECK(vx_queue_create(device, &qi, &queue));

    uint64_t num_cores, num_threads;
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

    uint32_t nthreads = (uint32_t)(num_cores * num_threads);
    uint32_t total = nthreads * WORDS_PER_THREAD;
    uint32_t buf_size = total * sizeof(uint32_t);

    std::cout << "num_cores=" << num_cores << " num_threads=" << num_threads << std::endl;
    std::cout << "total vectors: " << total << std::endl;
    std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

    std::vector<uint32_t> h_dst(total, 0);

    std::cout << "allocate device memory" << std::endl;
    RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buf));

    kernel_arg_t karg;
    RT_CHECK(vx_buffer_address(dst_buf, &karg.dst_addr));

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
    const uint32_t ALL_TRAPS = (1u << NUM_ILLEGAL_CASES) - 1;
    for (uint32_t t = 0; t < nthreads; ++t) {
        uint32_t err = h_dst[t * WORDS_PER_THREAD + 0];
        uint32_t flags = h_dst[t * WORDS_PER_THREAD + 1];
        if (err == 0 && flags == ALL_TRAPS)
            continue;
        if (errors < 8) {
            printf("thread %u: err=0x%x trap_flags=0x%x (expected 0x%x)\n",
                   t, err, flags, ALL_TRAPS);
            for (int b = 0; b < NUM_ILLEGAL_CASES; ++b)
                if (!(flags & (1u << b)))
                    printf("  missing trap: %s\n", illegal_case_name(b));
            if (err & (1u << 9))  printf("  ex2 legal frm trapped\n");
            if (err & (1u << 10)) printf("  ex2 legal frm results disagree\n");
            if (err & (1u << 11)) printf("  tanh legal frm trapped\n");
            if (err & (1u << 12)) printf("  tanh legal frm results disagree\n");
            if (err & (1u << 13)) printf("  sigmoid legal frm trapped\n");
            if (err & (1u << 14)) printf("  sigmoid legal frm results disagree\n");
        }
        ++errors;
    }

    // Cross-thread agreement: every thread executed the same probe stream.
    for (uint32_t w = 2; w < 5; ++w) {
        uint32_t ref = h_dst[w];
        for (uint32_t t = 1; t < nthreads; ++t) {
            if (h_dst[t * WORDS_PER_THREAD + w] != ref) {
                if (errors < 8)
                    printf("word %u disagrees across threads: thread0=0x%08x thread%u=0x%08x\n",
                           w, ref, t, h_dst[t * WORDS_PER_THREAD + w]);
                ++errors;
                break;
            }
        }
    }

    cleanup();

    if (errors) {
        printf("FAILED: %d thread-level failures\n", errors);
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
