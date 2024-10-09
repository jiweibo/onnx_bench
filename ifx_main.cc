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

#include "core/core.h"
#include "ifx.h"
#include "utils/barrier.h"
#include "utils/cupti_inc.h"
#include "utils/memuse.h"
#include "utils/nvml.h"
#include "utils/nvtx.h"
#include "utils/random_value.h"
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
DEFINE_string(ifxs, "", "a.ifxmodel b.ifxmodel c.ifxmodel <d.ifxmodel e.ifxmodel> ....");
DEFINE_string(bs, "", "1 12 1 <1 3> ...");
DEFINE_string(ifx_threads, "", "1 1 1 1 ...");
DEFINE_string(priority, "", "-5 -5 -4 0");

DEFINE_int32(device_id, 0, "device id");
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

namespace {
void RandomFillTensor(core::TensorRef& tensor) {
  auto num = tensor->Numel();
  auto dtype = tensor->GetDataType();
  switch (dtype) {
  case core::DataType::kBOOL:
    FillBuffer<bool>(tensor->HostData(), num, 0, 1);
    break;
  case core::DataType::kUINT8:
    FillBuffer<uint8_t>(tensor->HostData(), num, 0, 255);
    break;
  case core::DataType::kINT8:
  case core::DataType::kINT32:
  case core::DataType::kINT64:
    FillBuffer<int32_t>(tensor->HostData(), num, -128, 127);
    break;
  case core::DataType::kHALF:
  case core::DataType::kBF16:
  case core::DataType::kFP8:
  case core::DataType::kINT4:
  default:
    LOG(FATAL) << "Not supported dtype " << static_cast<int>(dtype);
  }
}

template <typename T>
std::string PrintShape(const std::vector<T>& v) {
  std::stringstream ss;
  for (size_t i = 0; i < v.size() - 1; ++i) {
    ss << v[i] << "x";
  }
  ss << v.back();
  return ss.str();
}

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

std::vector<std::vector<SessConfig>> ParseFlags(const std::vector<std::vector<std::string>>& ifx_vecs) {
  std::vector<std::vector<SessConfig>> configs(ifx_vecs.size());
  for (size_t i = 0; i < ifx_vecs.size(); ++i) {
    for (size_t j = 0; j < ifx_vecs[i].size(); ++j) {
      SessConfig config;
      config.device_id = FLAGS_device_id;
      config.ifx_file = ifx_vecs[i][j];
      config.cache_dir = FLAGS_cacheDir;
      config.use_gpu = true;
      config.enable_fp16 = FLAGS_precision == "fp16";
      configs[i].emplace_back(config);
    }
  }
  return configs;
}

// Helper function to split a string by a delimiter and return a vector of
// tokens
std::vector<std::string> SplitString(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

// Function to trim whitespace from both ends of a string
std::string TrimString(const std::string& str) {
  size_t first = str.find_first_not_of(' ');
  if (first == std::string::npos)
    return "";
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, last - first + 1);
}

///
/// in:  "a.ifxmodel b.ifxmodel <c.ifxmodel d.ifxmodel> e.ifxmodel"
/// out: {{"a.ifxmodel"}, {"b.ifxmodel"}, {"c.ifxmodel", "d.ifxmodel"},
///      {"e.ifxmodel"}}
///
std::vector<std::vector<std::string>> GetIfxModels(const std::string& ifxs) {
  std::vector<std::vector<std::string>> result;
  std::string remaining = ifxs;
  size_t start = 0;
  size_t end = 0;

  // Process each pair of angle brackets
  while ((start = remaining.find('<')) != std::string::npos && (end = remaining.find('>')) != std::string::npos) {
    // Extract the part before the first '<'
    std::string beforeBrackets = TrimString(remaining.substr(0, start));
    if (!beforeBrackets.empty()) {
      std::vector<std::string> tokens = SplitString(beforeBrackets, ' ');
      for (const std::string& token : tokens) {
        result.push_back({token});
      }
    }

    // Extract the part inside the brackets
    std::string insideBrackets = remaining.substr(start + 1, end - start - 1);
    if (!insideBrackets.empty()) {
      std::vector<std::string> tokens = SplitString(insideBrackets, ' ');
      result.push_back(tokens);
    }

    // Update remaining string to the part after the '>'
    remaining = remaining.substr(end + 1);
  }

  // Process the remaining part after the last '>'
  remaining = TrimString(remaining);
  if (!remaining.empty()) {
    std::vector<std::string> tokens = SplitString(remaining, ' ');
    for (const std::string& token : tokens) {
      result.push_back({token});
    }
  }

  return result;
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

// "-1 1 <16 32> 2" -> {{-1}, {1}, {16, 32}, {2}}
std::vector<std::vector<int>> ParseBatches(const std::string& str) {
  std::vector<std::vector<int>> result;
  std::string remaining = str;
  size_t start = 0;
  size_t end = 0;

  // Process each pair of angle brackets
  while ((start = remaining.find('<')) != std::string::npos && (end = remaining.find('>')) != std::string::npos) {
    // Extract the part before the first '<'
    std::string beforeBrackets = TrimString(remaining.substr(0, start));
    if (!beforeBrackets.empty()) {
      std::vector<std::string> tokens = SplitString(beforeBrackets, ' ');
      for (const std::string& token : tokens) {
        result.push_back({std::stoi(token)});
      }
    }

    // Extract the part inside the brackets
    std::string insideBrackets = remaining.substr(start + 1, end - start - 1);
    if (!insideBrackets.empty()) {
      std::vector<std::string> tokens = SplitString(insideBrackets, ' ');
      std::vector<int> tmp;
      for (const std::string& token : tokens) {
        tmp.push_back({std::stoi(token)});
      }
      result.push_back(tmp);
    }

    // Update remaining string to the part after the '>'
    remaining = remaining.substr(end + 1);
  }

  // Process the remaining part after the last '>'
  remaining = TrimString(remaining);
  if (!remaining.empty()) {
    std::vector<std::string> tokens = SplitString(remaining, ' ');
    for (const std::string& token : tokens) {
      result.push_back({std::stoi(token)});
    }
  }

  return result;
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

void RunCascade(std::vector<std::unique_ptr<Ifx_Sess>>& sessions, std::vector<NvtxRange>& nvtxs,
                const std::vector<int>& batches, Barrier* barrier = nullptr, cudaStream_t stream = nullptr,
                int repeats = 1) {
  int in_tensor_num = 0;
  for (auto& sess : sessions) {
    in_tensor_num += sess->InputDtypes().size();
  }
  std::vector<std::map<std::string, std::shared_ptr<core::Tensor>>> in_tensors(in_tensor_num);

  for (size_t i = 0; i < sessions.size(); ++i) {
    for (size_t j = 0; j < sessions[i]->InputNames().size(); ++j) {
      auto& name = sessions[i]->InputNames()[j];
      auto in_dims = sessions[i]->InputDims()[j];
      if (!batches.empty())
        in_dims[0] = batches[i];

      auto ifx_tensor = std::make_shared<core::Tensor>(
          core::Dims(in_dims), ifx_sess::ToDataType(sessions[i]->InputDtypes()[j]), core::Location::kHOST);
      RandomFillTensor(ifx_tensor);
      in_tensors[i].emplace(name, ifx_tensor);
    }
  }

  std::vector<StopWatchTimer> timers(sessions.size());
  for (size_t repeat = 0; repeat < repeats; ++repeat) {
    if (barrier)
      barrier->Wait();
    // std::uniform_int_distribution<> dis(0, 300);
    // std::this_thread::sleep_for(std::chrono::microseconds(dis(e)));
    for (size_t i = 0; i < sessions.size(); ++i) {
      nvtxs[i].Begin();
      timers[i].Start();
      auto out_tensors = sessions[i]->Run(in_tensors[i], stream);
      timers[i].Stop();
      nvtxs[i].End();
    }
  }

  for (size_t i = 0; i < sessions.size(); ++i) {
    LOG(INFO) << std::this_thread::get_id() << " " << sessions[i]->Config().ifx_file << " time is "
              << timers[i].GetAverageTime() << " ms"
              << ", tp50: " << timers[i].ComputePercentile(0.5) << ", tp90: " << timers[i].ComputePercentile(0.9)
              << ", tp99: " << timers[i].ComputePercentile(0.99);
  }
}
} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  NVMLWrapper nvml(FLAGS_device_id);
  LOG(INFO) << "CUDA Driver: " << nvml.GetCudaDriverVersion();
  LOG(INFO) << "Driver: " << nvml.GetDriverVersion();
  LOG(INFO) << "NVML: " << nvml.GetNvmlVersion();
  LOG(INFO) << "sm_clock: " << nvml.GetNvmlStats().sm_clock;
  LOG(INFO) << "mem_clock: " << nvml.GetNvmlStats().memory_clock;
  LOG(INFO) << "Temperature: " << nvml.GetNvmlStats().temperature;
  LOG(INFO) << "Performance Stat: " << nvml.GetNvmlStats().performance_stat;
  LOG(INFO) << "Power Usage: " << nvml.GetNvmlStats().power_usage;
  LOG(INFO) << "PCIE GEN: " << nvml.GetNvmlStats().cur_pcie_link_gen;
  LOG(INFO) << "PCIE Width: " << nvml.GetNvmlStats().cur_pcie_link_width;
  LOG(INFO) << "PCIE Speed: " << nvml.GetNvmlStats().pcie_speed;
  LOG(INFO) << "Mem Bus Width: " << nvml.GetNvmlStats().mem_bus_width;

  if (FLAGS_ifxs == "") {
    LOG(FATAL) << "Please set --ifxs flag.";
  }

  auto ifx_paths = GetIfxModels(FLAGS_ifxs);
  auto batches = ParseBatches(FLAGS_bs);
  if (!batches.empty()) {
    CHECK_EQ(ifx_paths.size(), batches.size());
    for (size_t i = 0; i < ifx_paths.size(); ++i) {
      CHECK_EQ(ifx_paths[i].size(), batches[i].size());
    }
  }
  std::vector<int> ifx_threads;
  if (FLAGS_ifx_threads.empty()) {
    ifx_threads.resize(ifx_paths.size());
    for (auto& t : ifx_threads)
      t = 1;
  } else {
    ifx_threads = ParseStrToVec(FLAGS_ifx_threads);
    CHECK_EQ(ifx_paths.size(), ifx_threads.size());
  }

  std::vector<int> priorities;
  if (!FLAGS_priority.empty()) {
    priorities = ParseStrToVec(FLAGS_priority);
    CHECK_EQ(priorities.size(), ifx_paths.size());
  }

  auto configs = ParseFlags(ifx_paths);
  int total_threads = 0;
  for (auto& t : ifx_threads) {
    total_threads += t;
  }

  std::vector<std::thread> threads(total_threads);
  Barrier barrier(total_threads);
  std::vector<std::vector<std::unique_ptr<Ifx_Sess>>> sessions;
  std::vector<std::vector<NvtxRange>> nvtxs;
  std::vector<cudaStream_t> streams(configs.size());
  for (size_t i = 0; i < configs.size(); ++i) {
    std::vector<std::unique_ptr<Ifx_Sess>> tmp;
    std::vector<NvtxRange> tmp_nvtx;
    for (size_t j = 0; j < configs[i].size(); ++j) {
      tmp.emplace_back(std::make_unique<Ifx_Sess>(configs[i][j]));
      tmp_nvtx.emplace_back(configs[i][j].ifx_file);
    }
    sessions.emplace_back(std::move(tmp));
    nvtxs.emplace_back(std::move(tmp_nvtx));

    if (priorities.empty()) {
      CHECK_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);
    } else {
      int low_priority, high_priority;
      CHECK_EQ(cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority), cudaSuccess);
      CHECK_EQ(cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, priorities[i]), cudaSuccess);
    }
  }
  LOG(INFO) << "Init session done";

  for (size_t i = 0; i < sessions.size(); ++i) {
    RunCascade(sessions[i], nvtxs[i], batches.empty() ? std::vector<int>{} : batches[i], nullptr,
               streams.empty() ? nullptr : streams[i], FLAGS_warmup);
  }

  // LOG(INFO) << "Util gpu: " << nvml.GetNvmlStats().utilization.gpu;
  // LOG(INFO) << "Util mem: " << nvml.GetNvmlStats().utilization.memory;

  LOG(INFO) << "--------- Warmup done ---------";
  MemoryUse checker(configs[0][0].device_id);
  checker.Start();

  int thread_num = 0;
  for (size_t i = 0; i < ifx_threads.size(); ++i) {
    for (size_t j = 0; j < ifx_threads[i]; ++j) {
      threads[thread_num++] = std::thread(RunCascade, std::ref(sessions[i]), std::ref(nvtxs[i]),
                                          batches.empty() ? std::vector<int>{} : batches[i], &barrier,
                                          streams.empty() ? nullptr : streams[i], FLAGS_repeats);
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
