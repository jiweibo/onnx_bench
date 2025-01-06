
#include <cuda_runtime_api.h>

#include "thread_pool.h"
#include "utils/nvtx.h"

#include <future>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <string>

DEFINE_int32(streams, 2, "stream number");
DEFINE_int32(repeats, 1, "repeats");

__global__ void DelayKernel(unsigned long long target_clock = 1000000) {
  unsigned long long start_clock = clock64();
  while (clock64() - start_clock < target_clock) {
    asm volatile(""); // Avoid compiler optimization.
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<cudaStream_t> streams(FLAGS_streams);
  for (size_t i = 0; i < FLAGS_streams; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  ThreadPool tp(FLAGS_streams);

  dim3 blocks(128);
  dim3 grid(8);

  std::vector<std::future<void>> futs;
  for (size_t sid = 0; sid < FLAGS_streams; ++sid) {
    futs.push_back(tp.enqueue(
        [&](int stream_id) {
          for (size_t i = 0; i < FLAGS_repeats; ++i) {
            NvtxRange nvtx(std::to_string(i));
            nvtx.Begin();
            DelayKernel<<<grid, blocks, 0, streams[stream_id]>>>();
            nvtx.End();
          }
        },
        sid));
  }

  for (auto& f : futs) {
    f.get();
  }

  for (size_t i = 0; i < FLAGS_streams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
  return 0;
}