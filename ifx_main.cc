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

using namespace ifx_sess;

// DEFINE_string(ifx, "", "ifx model file");
DEFINE_string(ifxs, "", "a.ifxmodel b.ifxmodel c.ifxmodel ....");
DEFINE_string(ifx_threads, "", "1 1 1 ...");
DEFINE_string(bs, "", "1 12 1 ...");

DEFINE_int32(device_id, 0, "device id");
DEFINE_int32(batch, -1, "batch");
DEFINE_int32(warmup, 0, "warmup");
DEFINE_int32(repeats, 1, "repeats");
DEFINE_string(precision, "fp32", "fp32, fp16, int8");
DEFINE_string(cacheDir, "", "the cache dir");

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

// "x y z" -> {x, y, z}
std::vector<int> ParseStrToVec(const std::string& str) {
  std::vector<int> res;
  int pos = 0;
  const char* sep = " ";

  size_t found = str.find(sep, pos);
  while (found != std::string::npos) {
    auto s = str.substr(pos, found - pos);
    res.push_back(std::stoi(s));
    pos = found + 1;
    found = str.find(sep, pos);
  }
  auto s = str.substr(pos, found - pos);
  res.push_back(std::stoi(s));
  return res;
}

void Run(Ifx_Sess& session, Barrier* barrier = nullptr, int repeats = 1, int batch=-1) {
  std::map<std::string, Tensor> in_tensors;
  std::vector<void*> to_free(session.InputDtypes().size());

  for (size_t i = 0; i < session.InputNames().size(); ++i) {
    auto& name = session.InputNames()[i];
    auto in_dims = session.InputDims()[i];
    if (batch != -1) in_dims[0] = batch;
    auto* data = GenerateData(in_dims, session.InputDtypes()[i]);
    to_free.push_back(data);
    std::vector<int32_t> in_dims_32(in_dims.begin(), in_dims.end());
    auto ifx_tensor = Tensor(name, data, in_dims_32, session.InputDtypes()[i],
                             session.InputFormats()[i], false);
    in_tensors.emplace(name, std::move(ifx_tensor));
  }

  StopWatchTimer timer;
  NvtxRange nvtx(session.Config().ifx_file);
  for (size_t i = 0; i < repeats; ++i) {
    nvtx.Begin();
    if (barrier)
      barrier->Wait();
    std::uniform_int_distribution<> dis(0, 300);
    std::this_thread::sleep_for(std::chrono::microseconds(dis(e)));
    timer.Start();
    auto out_tensors = session.Run(in_tensors);
    timer.Stop();
    nvtx.End();
  }
  LOG(INFO) << std::this_thread::get_id() << " " << session.Config().ifx_file
            << " time is " << timer.GetAverageTime() << " ms"
            << ", tp50: " << timer.ComputePercentile(0.5)
            << ", tp90: " << timer.ComputePercentile(0.9)
            << ", tp99: " << timer.ComputePercentile(0.99);

  for (auto* p : to_free) {
    free(p);
  }
}
} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_ifxs == "") {
    LOG(FATAL) << "Please set --ifxs flag.";
  }

  auto ifx_paths = GetModels(FLAGS_ifxs);
  std::vector<int> ifx_threads;
  if (FLAGS_ifx_threads.empty()) {
    ifx_threads.resize(ifx_paths.size());
      for (auto& t : ifx_threads)
        t = 1;
  } else {
    ifx_threads = ParseStrToVec(FLAGS_ifx_threads);
    CHECK_EQ(ifx_paths.size(), ifx_threads.size());
  }

  std::vector<int> bs;
  if (FLAGS_bs.empty()) {
    bs.resize(ifx_paths.size());
    for (auto& t : bs) t = -1;
  } else {
    bs = ParseStrToVec(FLAGS_bs);
    CHECK_EQ(ifx_paths.size(), bs.size());
  }

  auto configs = ParseFlags(ifx_paths);
  int total_threads = 0;
  for (auto& t : ifx_threads) {
    total_threads += t;
  }

  std::vector<std::thread> threads(total_threads);
  Barrier barrier(total_threads);
  std::vector<Ifx_Sess> sessions;
  // std::vector<NvtxRange> nvtxs;
  for (auto& config : configs) {
    sessions.emplace_back(config);
    // nvtxs.emplace_back(config.ifx_file);
  }
  for (size_t i = 0; i < sessions.size(); ++i) {
    Run(sessions[i], nullptr, FLAGS_warmup);
  }

  LOG(INFO) << "--------- Warmup done ---------";
  MemoryUse checker(configs[0].device_id);
  checker.Start();

  int thread_num = 0;
  for (size_t i = 0; i < ifx_threads.size(); ++i) {
    for (size_t j = 0; j < ifx_threads[i]; ++j) {
      threads[thread_num++] = std::thread(Run, std::ref(sessions[i]), &barrier, FLAGS_repeats, bs[i]);
    }
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }

  auto mem = checker.GetMemInfo();
  auto vsz = std::get<0>(mem);
  auto rss = std::get<1>(mem);
  auto gpu = std::get<2>(mem);
  checker.Stop();

  std::cout << "vsz: " << vsz / 1024.0 << std::endl;
  std::cout << "rss: " << rss / 1024.0 << std::endl;
  std::cout << "gpu: " << gpu / (1024.0 * 1024.0) << std::endl;
  return 0;
}

