#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <ratio>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ifx.h"
#include "utils/barrier.h"
#include "utils/memuse.h"
#include "utils/nvtx.h"
#include "utils/timer.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "cnpy.h"
#include "ifx_sess.h"

// #include "cutlass/gemm/device/gemm.h"
// #include "cutlass/arch/mma.h"
// #include "cutlass/gemm_coord.h"
// #include "cutlass/half.h"
// #include "cutlass/layout/matrix.h"
// #include "cutlass/util/host_tensor.h"
// #include "cutlass/util/reference/host/tensor_copy.h"
// #include "cutlass/util/reference/host/tensor_fill.h"

#include <cublas_v2.h>

using namespace ifx_sess;

DEFINE_string(ifx, "", "ifx model file");
// DEFINE_string(camera_fusion_all_ifx, "", "ifx model file");
// DEFINE_string(lane_detection_ifx, "", "ifx model file");
// DEFINE_string(occupancy_model_ifx, "", "ifx model file");
// DEFINE_string(point_pillar, "", "ifx model file");
DEFINE_string(ifxs, "", "a.ifxmodel b.ifxmodel c.ifxmodel ....");

DEFINE_int32(device_id, 0, "device id");
DEFINE_int32(batch, 1, "batch");
DEFINE_int32(warmup, 0, "warmup");
DEFINE_int32(repeats, 1, "repeats");
DEFINE_string(precision, "fp32", "fp32, fp16, int8");
DEFINE_string(cacheDir, "", "the cache dir");

DEFINE_int32(n, 128, "mnk");

// DEFINE_string(provider, "cpu", "cpu, openvino, cuda, trt");

// DEFINE_string(
//     dumpOutput, "",
//     "Save the output tensor(s) of the last inference iteration in a npz file"
//     "(default = disabled).");

// TODO:
// DEFINE_string(
//     loadInputs, "",
//     "Load input values from files (default = generate random inputs). Input "
//     "names can be wrapped with single quotes (ex: 'Input:0.in')");
// DEFINE_string(inputType, "json", "txt, bin, json etc.");

// DEFINE_string(dataDir, "", "a dir which stores lots of json file");

std::default_random_engine e(1998);

const char* SEP = "-SEP-";

namespace {

void* GenerateData(const std::vector<int64_t>& dims, ifx::DataType type) {
  int64_t num =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
  if (type == ifx::DATA_TYPE_FP32) {
    float* ptr = static_cast<float*>(malloc(num * sizeof(float)));
    std::uniform_real_distribution<float> u(-1, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = 0; // u(e);
    }
    return ptr;
  } else if (type == ifx::DATA_TYPE_FP16) {
    half* ptr = static_cast<half*>(malloc(num * sizeof(half)));
    std::uniform_real_distribution<float> u(-1, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = __half2float(u(e));
    }
    return ptr;
  } else if (type == ifx::DATA_TYPE_INT32) {
    int* ptr = static_cast<int*>(malloc(num * sizeof(int)));
    std::uniform_int_distribution<int> u(0, 127);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = 0; // u(e);
    }
    return ptr;
  } else if (type == ifx::DATA_TYPE_INT64) {
    auto* ptr = static_cast<int64_t*>(malloc(num * sizeof(int64_t)));
    std::uniform_int_distribution<int> u(-128, 127);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ifx::DATA_TYPE_BOOL) {
    bool* ptr = static_cast<bool*>(malloc(num * sizeof(bool)));
    std::uniform_int_distribution<int> u(0, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ifx::DATA_TYPE_UINT8) {
    auto* ptr = static_cast<uint8_t*>(malloc(num * sizeof(uint8_t)));
    std::uniform_int_distribution<uint8_t> u(0, 255);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else {
    LOG(FATAL) << "Not supported data type " << type;
  }

  return nullptr;
}

// template <typename T> std::string PrintShape(const std::vector<T>& v) {
//   std::stringstream ss;
//   for (size_t i = 0; i < v.size() - 1; ++i) {
//     ss << v[i] << "x";
//   }
//   ss << v.back();
//   return ss.str();
// }

std::vector<SessConfig> ParseFlags(const std::vector<std::string>& ifxs) {
  std::vector<SessConfig> configs(ifxs.size());
  for (size_t i = 0; i < ifxs.size(); ++i) {
    SessConfig config;
    config.device_id = FLAGS_device_id;
    config.ifx_file = ifxs[i];
    config.cache_dir = FLAGS_cacheDir;
    config.use_gpu = true;
    config.enable_fp16 = FLAGS_precision == "fp16";
    configs[i] = config;
  }
  return configs;
}

// a.ifxmodel;b.ifxmodel
std::vector<std::string> GetModels(const std::string& ifxs) {
  std::vector<std::string> res;
  int pos = 0;
  const char* sep = " ";

  size_t found = ifxs.find(sep, pos);
  while (found != std::string::npos) {
    auto s = ifxs.substr(pos, found - pos);
    res.push_back(s);
    pos = found + 1;
    found = ifxs.find(sep, pos);
  }
  auto s = ifxs.substr(pos, found - pos);
  res.push_back(s);
  return res;
}

void Run(Ifx_Sess& session, Barrier* barrier = nullptr) {
  std::map<std::string, Tensor> in_tensors;
  std::vector<void*> to_free(session.InputDtypes().size());

  for (size_t i = 0; i < session.InputNames().size(); ++i) {
    auto& name = session.InputNames()[i];
    auto in_dims = session.InputDims()[i];
    auto* data = GenerateData(in_dims, session.InputDtypes()[i]);
    to_free.push_back(data);
    std::vector<int32_t> in_dims_32(in_dims.begin(), in_dims.end());
    auto ifx_tensor = Tensor(name, data, in_dims_32, session.InputDtypes()[i],
                             session.InputFormats()[i], false);
    in_tensors.emplace(name, std::move(ifx_tensor));
  }

  if (barrier)
    barrier->Wait();
  StopWatchTimer timer;
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    timer.Start();
    auto out_tensors = session.Run(in_tensors);
    cudaDeviceSynchronize();
    timer.Stop();
  }
  LOG(INFO) << std::this_thread::get_id() << " " << session.Config().ifx_file
            << " time is " << timer.GetAverageTime() << " ms";

  for (auto* p : to_free) {
    free(p);
  }
}

void gemm_mnk(cublasHandle_t handle, float* a, float* b, float* c, int n) {
  float alpha = 1.0;
  float beta = 0;
  for (size_t i = 0; i < 10000; ++i) {
    auto s = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a,
                          CUDA_R_32F, n, b, CUDA_R_32F, n, &beta, c, CUDA_R_32F,
                          n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK_EQ(s, CUBLAS_STATUS_SUCCESS);
    cudaDeviceSynchronize();
  }
  // cublasHgemm(cublas_handle, cublasOperation_t::CUBLAS_OP_N,
  // cublasOperation_t transb, int m, int n, int k, const __half *alpha, const
  // __half *A, int lda, const __half *B, int ldb, const __half *beta, __half
  // *C, int ldc)
}

class BlasGemmEx {
public:
  BlasGemmEx(float alpha, float beta) {
    status_ = cublasCreate(&handle_);
    CHECK_EQ(status_, CUBLAS_STATUS_SUCCESS);
    cudaError_t status = cudaStreamCreate(&stream_);
    CHECK_EQ(status, cudaSuccess);
    cublasSetStream(handle_, stream_);
  }

  void GemmEx(bool transa, bool transb, int m, int n, int k, const void* alpha,
              const void* A, cudaDataType Atype, int lda, const void* B,
              cudaDataType Btype, int ldb, const void* beta, void* C,
              cudaDataType Ctype, int ldc, cudaDataType computeType,
              cublasGemmAlgo_t algo) {
    status_ = cublasGemmEx(handle_, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                           transb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, alpha,
                           A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc,
                           computeType, algo);
    CHECK_EQ(status_, CUBLAS_STATUS_SUCCESS);
  }

private:
  cublasHandle_t handle_;
  cublasStatus_t status_;
  cudaStream_t stream_;
};

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_ifxs == "") {
    LOG(FATAL) << "Please set --ifxs flag.";
  }

  cublasHandle_t handle = NULL;
  cublasStatus_t status;
  status = cublasCreate(&handle);
  CHECK_EQ(status, CUBLAS_STATUS_SUCCESS);
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  cudaStream_t stream = NULL;
  cudaStreamCreate(&stream);
  cublasSetStream(handle, stream);
  int n = FLAGS_n;
  float* a;
  float* b;
  float* c;
  cudaMalloc((void**)&a, n * n * sizeof(float));
  cudaMalloc((void**)&b, n * n * sizeof(float));
  cudaMalloc((void**)&c, n * n * sizeof(float));
  // gemm_mnk(handle, a, b, c, n);

  auto ifx_paths = GetModels(FLAGS_ifxs);
  auto configs = ParseFlags(ifx_paths);

  std::vector<std::thread> threads(ifx_paths.size() + 2);
  Barrier barrier(ifx_paths.size() + 2);
  std::vector<Ifx_Sess> sessions;
  for (auto& config : configs) {
    sessions.emplace_back(config);
  }
  for (size_t i = 0; i < sessions.size(); ++i) {
    Run(sessions[i]);
  }

  // SessConfig cfa_config = config;
  // cfa_config.ifx_file = "./camera_fusion_all.ifxmodel";
  // Ifx_Sess cfa_session(cfa_config);

  // std::map<std::string, Tensor> in_tensors;

  // StopWatchTimer timer_h2d, timer_run_d2h;
  // NvtxRange nvtx_h2d("h2d"), nvtx_run_d2h("run_d2h");
  // session.RegisterBeforeH2DHook([&]() { timer_h2d.Start(); });
  // session.RegisterAfterH2DHook([&]() { timer_h2d.Stop(); });
  // session.RegisterBeforeRunD2HHook([&]() { timer_run_d2h.Start(); });
  // session.RegisterAfterRunD2HHook([&]() { timer_run_d2h.Stop(); });
  // session.RegisterBeforeRunD2HHook([&]() { nvtx_run_d2h.Begin(); });
  // session.RegisterAfterRunD2HHook([&]() { nvtx_run_d2h.End(); });

  // for (int i = 0; i < FLAGS_warmup; ++i) {
  // Run(session);
  // Run(cfa_session);
  // }
  LOG(INFO) << "--------- Warmup done ---------";
  // MemoryUse checker(configs[0].device_id);
  // checker.Start();
  auto gemm_t = std::thread(gemm_mnk, handle, a, b, c, n);
  for (size_t i = 0; i < sessions.size(); ++i) {
    threads[i] = std::thread(Run, std::ref(sessions[i]), &barrier);
  }
  threads[threads.size() - 2] =
      std::thread(Run, std::ref(sessions[0]), &barrier);
  threads[threads.size() - 1] =
      std::thread(Run, std::ref(sessions[0]), &barrier);
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }

  gemm_t.join();
  // auto [vsz, rss, gpu] = checker.GetMemInfo();
  // checker.Stop();

  // // LOG(INFO) << "------------------------------";
  // // LOG(INFO) << "H2D Average time " << timer_h2d.GetAverageTime()
  // //           << ", variance: " << timer_h2d.ComputeVariance()
  // //           << ", tp50: " << timer_h2d.ComputePercentile(0.5)
  // //           << ", tp99: " << timer_h2d.ComputePercentile(0.99);
  // // LOG(INFO) << "Run+D2H Average time " << timer_run_d2h.GetAverageTime()
  // //           << ", variance: " << timer_run_d2h.ComputeVariance()
  // //           << ", tp50: " << timer_run_d2h.ComputePercentile(0.5)
  // //           << ", tp99: " << timer_run_d2h.ComputePercentile(0.99);
  // // LOG(INFO) << "H2D+RUN+D2H Average time "
  // //           << timer_h2d.GetAverageTime() +
  // timer_run_d2h.GetAverageTime();

  // std::cout << "vsz: " << vsz / 1024.0 << std::endl;
  // std::cout << "rss: " << rss / 1024.0 << std::endl;
  // std::cout << "gpu: " << gpu / (1024.0 * 1024.0) << std::endl;
  return 0;
}
