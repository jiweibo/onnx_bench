#include <cuda_runtime_api.h>

#include <future>
#include <iostream>
#include <thread>

#include "thread_pool.h"
#include "utils/barrier.h"
#include "utils/util.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_int32(thread_num, 4, "thread number");
DEFINE_int32(size, 32 * 1024 * 1024, "memory size");
DEFINE_int32(warmup, 0, "warmup");
DEFINE_int32(repeats, 1, "repeats");
DEFINE_bool(pin, false, "use pin memory, default is false");
DEFINE_string(modes, "h2d d2h d2d run", "h2d d2h d2d run");

namespace {

void Copy(void* dst, const void* src, size_t size, cudaMemcpyKind kind,
          cudaStream_t stream, Barrier* barrier, int repeats = 1) {
  for (size_t i = 0; i < repeats; ++i) {
    if (barrier) {
      barrier->Wait();
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, kind, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

__global__ void func(float* device_data, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    device_data[idx] = device_data[idx] * 0.9f + 0.1f;
  }
}

void RunKernel(float* device_data, size_t size, cudaStream_t stream,
               Barrier* barrier, int repetas = 1) {
  dim3 threads_per_block = 512;
  dim3 blocks_per_grid = (size + threads_per_block.x - 1) / threads_per_block.x;
  for (size_t i = 0; i < repetas; ++i) {
    if (barrier) {
      barrier->Wait();
    }
    func<<<blocks_per_grid, threads_per_block, 0, stream>>>(device_data, size);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Barrier barrier(FLAGS_thread_num);
  ThreadPool tp(FLAGS_thread_num);
  std::vector<cudaStream_t> streams(FLAGS_thread_num);
  std::vector<float*> host_datas(FLAGS_thread_num);
  std::vector<float*> device_datas(FLAGS_thread_num);
  std::vector<float*> device_datas2(FLAGS_thread_num);

  for (size_t i = 0; i < streams.size(); ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
    if (FLAGS_pin) {
      CUDA_CHECK(cudaMallocHost(&host_datas[i], FLAGS_size));
    } else {
      host_datas[i] = (float*)malloc(FLAGS_size);
    }
    memset(host_datas[i], 0, FLAGS_size);
    CUDA_CHECK(cudaMalloc((void**)&device_datas[i], FLAGS_size));
    CUDA_CHECK(cudaMalloc((void**)&device_datas2[i], FLAGS_size));
  }

  auto modes = SplitToStringVec(FLAGS_modes, ' ');
  if (modes.empty() || modes.size() == 1) {
    bool empty = modes.empty();
    modes.resize(FLAGS_thread_num);
    for (size_t i = 1; i < FLAGS_thread_num; ++i) {
      modes[i] = empty ? "h2d" : modes[0];
    }
  } else {
    CHECK_EQ(modes.size(), FLAGS_thread_num);
  }

  // warmup
  tp.enqueue([&]() {
      Copy(device_datas[0], host_datas[0], FLAGS_size, cudaMemcpyHostToDevice,
           streams[0], nullptr, FLAGS_warmup);
      Copy(host_datas[0], device_datas[0], FLAGS_size, cudaMemcpyDeviceToHost,
           streams[0], nullptr, FLAGS_warmup);
      Copy(device_datas2[0], device_datas[0], FLAGS_size,
           cudaMemcpyDeviceToDevice, streams[0], nullptr, FLAGS_warmup);
      RunKernel(device_datas[0], FLAGS_size / sizeof(float), streams[0],
                nullptr, FLAGS_warmup);
    }).get();
  LOG(INFO) << "Warmup done.";

  std::vector<std::future<void>> rets(FLAGS_thread_num);
  for (size_t i = 0; i < FLAGS_thread_num; ++i) {
    rets[i] = tp.enqueue(
        [&](int idx) {
          void *dst{nullptr}, *src{nullptr};
          cudaMemcpyKind kind;
          if (modes[idx] == "h2d") {
            dst = device_datas[idx];
            src = host_datas[idx];
            kind = cudaMemcpyHostToDevice;
          } else if (modes[idx] == "d2h") {
            dst = host_datas[idx];
            src = device_datas[idx];
            kind = cudaMemcpyDeviceToHost;
          } else if (modes[idx] == "d2d") {
            dst = device_datas[idx];
            src = device_datas[idx];
            kind = cudaMemcpyDeviceToDevice;
          } else if (modes[idx] == "run") {
            RunKernel(device_datas[idx], FLAGS_size / sizeof(float),
                      streams[idx], &barrier, FLAGS_repeats);
            return;
          } else {
            LOG(FATAL) << "Not supported mode: " << modes[idx];
          }
          Copy(dst, src, FLAGS_size, kind, streams[idx], &barrier,
               FLAGS_repeats);
        },
        i);
  }
  for (size_t i = 0; i < FLAGS_thread_num; ++i) {
    rets[i].get();
  }
  LOG(INFO) << "Run done.";

  for (size_t i = 0; i < FLAGS_thread_num; ++i) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFree(device_datas[i]));
    CUDA_CHECK(cudaFree(device_datas2[i]));
    if (FLAGS_pin) {
      CUDA_CHECK(cudaFreeHost(host_datas[i]));
    } else {
      free(host_datas[i]);
    }
  }
  return 0;
}