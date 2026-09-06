#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vortex2.h>
#include "common.h"

// SFU perf sweep driver. For each (warps-per-core W, stream length N)
// configuration it launches one grid of W blocks (one warp each) on the
// single-core build, reads back per-thread measurements and prints CSV.
//
// All warps read the SAME core-wide mcycle counter, so multi-warp
// aggregate throughput = (W * n_blocks) / (max(i1) - min(i0)) across the
// warps of the grid.

#define RT_CHECK(_expr)                                                   \
  do {                                                                    \
    int _ret = (_expr);                                                   \
    if (0 == _ret) break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, _ret);                   \
    cleanup();                                                            \
    exit(-1);                                                             \
  } while (false)

static const char* kernel_file = "kernel.vxbin";
static vx_device_h device = nullptr;
static vx_buffer_h out_buf = nullptr;
static vx_queue_h queue = nullptr;
static vx_module_h module_ = nullptr;
static vx_kernel_h kernel = nullptr;

static const uint32_t W_SWEEP[]   = {1, 2, 4};      // warps (blocks) per core
static const uint32_t NB_SWEEP[]  = {64, 256, 1024}; // independent-stream length
static const uint32_t NC_SWEEP[]  = {16, 64, 256};   // dependency-chain length

void cleanup() {
    if (device) {
        if (out_buf) vx_buffer_release(out_buf);
        if (kernel) vx_kernel_release(kernel);
        if (module_) vx_module_release(module_);
        if (queue) vx_queue_release(queue);
        vx_device_dump_perf(device, stdout);
        vx_device_release(device);
    }
}

static void run_config(uint32_t num_warps, uint32_t n_blocks,
                       uint32_t n_chain, uint32_t num_threads) {
    uint32_t total_threads = num_warps * num_threads;
    std::vector<uint64_t> h_out(total_threads * OUT_SLOTS, 0);
    uint64_t out_addr = 0;
    RT_CHECK(vx_buffer_address(out_buf, &out_addr));

    kernel_arg_t karg;
    karg.out_addr = out_addr;
    karg.n_blocks = n_blocks;
    karg.n_chain  = n_chain;

    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    {
        vx_launch_info_t li = {};
        li.struct_size  = sizeof(li);
        li.kernel       = kernel;
        li.args_host    = &karg;
        li.args_size    = sizeof(karg);
        li.ndim         = 1;
        li.grid_dim[0]  = num_warps;      // one block = one warp
        li.block_dim[0] = num_threads;
        RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    }
    RT_CHECK(vx_enqueue_read(queue, h_out.data(), out_buf, 0,
                             h_out.size() * sizeof(uint64_t), 1,
                             &launch_ev, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    // CSV: W, n_blocks, n_chain, tid, empty, indep, chain, i0, i1
    for (uint32_t t = 0; t < total_threads; ++t) {
        const uint64_t* row = &h_out[t * OUT_SLOTS];
        printf("csv,%u,%u,%u,%u,%llu,%llu,%llu,%llu,%llu\n",
               num_warps, n_blocks, n_chain, t,
               (unsigned long long)row[OUT_EMPTY],
               (unsigned long long)row[OUT_INDEP],
               (unsigned long long)row[OUT_CHAIN],
               (unsigned long long)row[OUT_I0],
               (unsigned long long)row[OUT_I1]);
    }

    // multi-warp aggregate over the independent-segment window
    if (num_warps > 1) {
        uint64_t i0_min = ~0ull, i1_max = 0;
        for (uint32_t w = 0; w < num_warps; ++w) {
            // one sample per warp (thread 0 of each block)
            const uint64_t* row = &h_out[w * num_threads * OUT_SLOTS];
            if (row[OUT_I0] < i0_min) i0_min = row[OUT_I0];
            if (row[OUT_I1] > i1_max) i1_max = row[OUT_I1];
        }
        double window = (double)(i1_max - i0_min);
        if (window > 0) {
            printf("agg,%u,%u,%u,%llu,%.3f\n",
                   num_warps, n_blocks, n_chain,
                   (unsigned long long)(i1_max - i0_min),
                   (double)(num_warps * n_blocks) / window);
        }
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
    std::cout << "num_cores=" << num_cores
              << " num_threads=" << num_threads << std::endl;

    uint32_t max_threads = 4 * (uint32_t)num_threads;  // W sweep max = 4
    RT_CHECK(vx_buffer_create(device, max_threads * OUT_SLOTS * sizeof(uint64_t),
                              VX_MEM_WRITE, &out_buf));

    std::cout << "load kernel module" << std::endl;
    RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
    RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

    // throughput sweep: W x N (independent stream), chain fixed small
    for (uint32_t wi = 0; wi < sizeof(W_SWEEP) / sizeof(W_SWEEP[0]); ++wi)
        for (uint32_t ni = 0; ni < sizeof(NB_SWEEP) / sizeof(NB_SWEEP[0]); ++ni)
            run_config(W_SWEEP[wi], NB_SWEEP[ni], NC_SWEEP[0],
                       (uint32_t)num_threads);

    // latency sweep: single warp, chain length scan
    for (uint32_t ni = 0; ni < sizeof(NC_SWEEP) / sizeof(NC_SWEEP[0]); ++ni)
        run_config(1, NB_SWEEP[0], NC_SWEEP[ni], (uint32_t)num_threads);

    std::cout << "PASSED" << std::endl;
    cleanup();
    return 0;
}
