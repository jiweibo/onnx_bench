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
#include <unordered_map>
#include <utility>
#include <vector>

#include "cnpy.h"
#include "core/core.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "utils/barrier.h"
#include "utils/memuse.h"
#include "utils/timer.h"

#include "ort.h"

DEFINE_string(onnxs, "", "a.onnx,b.onnx,c.onnx");
DEFINE_string(bs, "", "1,4,8");

DEFINE_int32(device_id, 0, "device id");
DEFINE_int32(warmup, 0, "warmup");
DEFINE_int32(repeats, 1, "repeats");
DEFINE_string(log_level, "warning", "verbose, info, warning, error, fatal");
DEFINE_bool(int8, false, "enable int8.");
DEFINE_bool(fp16, false, "enable fp16.");
DEFINE_string(provider, "cpu", "cpu, openvino, cuda, trt");
DEFINE_string(cache_dir, "", "the cache dir");
DEFINE_string(calibration_name, "", "int8 calibration table");
DEFINE_int32(min_subgraph_size, 1, "trt min subgraph size");
DEFINE_uint64(max_workspace_size, 1UL << 31, "trt max workspace size");
DEFINE_string(trt_profile_min_shapes, "", "in1:1x8,in2:1x3x224x224");
DEFINE_string(trt_profile_max_shapes, "", "in1:8x8,in2:8x3x224x224");
DEFINE_string(trt_profile_opt_shapes, "", "in1:8x8,in2:8x3x224x224");

DEFINE_string(trt_filter_nodes, "", "node_a,node_b,node_c");

DEFINE_string(input_data, "", "Load npz input file (default = generate random inputs).");
DEFINE_string(dump_output, "", "dump output to npz.");

namespace {
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

std::vector<int> ParseStrToVec(const std::string& str, char delimiter) {
  auto vec = SplitString(str, delimiter);
  std::vector<int> res(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    res[i] = std::stoi(vec[i]);
  }
  return res;
}

std::vector<SessConfig> ParseFlags(const std::vector<std::string>& onnx_models) {
  std::vector<SessConfig> configs;
  for (size_t i = 0; i < onnx_models.size(); ++i) {
    SessConfig config;
    config.device_id = FLAGS_device_id;
    config.opt_level = GraphOptimizationLevel::ORT_DISABLE_ALL;
    config.onnx_file = onnx_models[i];
    config.intra_op_num_threads = 1;
    if (FLAGS_log_level == "verbose") {
      config.log_level = ORT_LOGGING_LEVEL_VERBOSE;
    } else if (FLAGS_log_level == "info") {
      config.log_level = ORT_LOGGING_LEVEL_INFO;
    } else if (FLAGS_log_level == "warning") {
      config.log_level = ORT_LOGGING_LEVEL_WARNING;
    } else if (FLAGS_log_level == "error") {
      config.log_level = ORT_LOGGING_LEVEL_ERROR;
    } else if (FLAGS_log_level == "fatal") {
      config.log_level = ORT_LOGGING_LEVEL_FATAL;
    } else {
      LOG(FATAL) << "Not supported log level(verbose, info, warning, error, fatal) " << FLAGS_log_level;
    }
    if (FLAGS_provider == "trt") {
      config.use_trt = true;
      config.use_cuda = true;
      config.trt_config = SessConfig::TrtConfig();
      config.trt_config.cache_dir = FLAGS_cache_dir;
      config.trt_config.calibration_table_name = FLAGS_calibration_name;
      config.trt_config.enable_int8 = FLAGS_int8;
      config.trt_config.enable_fp16 = FLAGS_fp16;
      config.trt_config.min_subgraph_size = FLAGS_min_subgraph_size;
      config.trt_config.max_workspace_size = FLAGS_max_workspace_size;
      config.trt_config.trt_profile_min_shapes = FLAGS_trt_profile_min_shapes;
      config.trt_config.trt_profile_max_shapes = FLAGS_trt_profile_max_shapes;
      config.trt_config.trt_profile_opt_shapes = FLAGS_trt_profile_opt_shapes;
      config.trt_config.trt_filter_nodes = FLAGS_trt_filter_nodes;
    } else if (FLAGS_provider == "cuda") {
      config.use_cuda = true;
    } else if (FLAGS_provider == "openvino") {
      config.use_openvino = true;
    }
    configs.emplace_back(std::move(config));
  }
  return configs;
}

cnpy::npz_t LoadNpzFile(const std::string& str) {
  cnpy::npz_t res = cnpy::npz_load(str);
  return res;
}

std::map<std::string, std::shared_ptr<core::Tensor>> ConvertNpzToTensor(const cnpy::npz_t& npz, Sess* sess) {
  std::map<std::string, std::shared_ptr<core::Tensor>> tensors;
  auto input_names = sess->InputNames();
  auto input_dtypes = sess->InputDtypes();
  for (size_t i = 0; i < input_names.size(); ++i) {
    // LOG(INFO) << input_names[i];
    // for (size_t j = 0; j < npz.at(input_names[i]).shape.size(); ++j) {
    //   LOG(INFO) << npz.at(input_names[i]).shape[j];
    // }
    void* data = &(*npz.at(input_names[i]).data_holder)[0];
    auto tensor = std::make_shared<core::Tensor>(data, npz.at(input_names[i]).num_bytes(),
                                                 core::Dims(npz.at(input_names[i]).shape), ToDataType(input_dtypes[i]));
    tensors.emplace(std::make_pair(input_names[i], std::move(tensor)));
  }

  return tensors;
}

void DumpOutputTensors(const std::map<std::string, std::shared_ptr<core::Tensor>>& tensors) {
  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    auto& name = it->first;
    auto tensor = it->second;
    auto dtype = tensor->GetDataType();
    auto shape = tensor->GetDims().ToStdVec<size_t>();
    if (dtype == core::DataType::kFLOAT) {
      auto* data = tensor->HostData<float>();
      cnpy::npz_save(FLAGS_dump_output, name, data, shape, "a");
    } else if (dtype == core::DataType::kHALF) {
      auto* data = tensor->HostData<__half>();
      cnpy::npz_save(FLAGS_dump_output, name, data, shape, "a");
    } else if (dtype == core::DataType::kINT32) {
      auto* data = tensor->HostData<int>();
      cnpy::npz_save(FLAGS_dump_output, name, data, shape, "a");
    } else if (dtype == core::DataType::kINT64) {
      auto* data = tensor->HostData<int64_t>();
      cnpy::npz_save(FLAGS_dump_output, name, data, shape, "a");
    } else if (dtype == core::DataType::kBOOL) {
      auto* data = tensor->HostData<bool>();
      cnpy::npz_save(FLAGS_dump_output, name, data, shape, "a");
    } else if (dtype == core::DataType::kUINT8) {
      auto* data = tensor->HostData<uint8_t>();
      cnpy::npz_save(FLAGS_dump_output, name, data, shape, "a");
    } else {
      LOG(FATAL) << "Not supported data type " << static_cast<int>(dtype);
    }
  }
}

void Run(std::unique_ptr<Sess>& session, int batch = -1, Barrier* barrier = nullptr, int repeats = 1) {
  auto in_tensor_num = session->InputDtypes().size();
  std::map<std::string, std::shared_ptr<core::Tensor>> in_tensors;
  cnpy::npz_t npz;
  if (FLAGS_input_data != "") {
    npz = LoadNpzFile(FLAGS_input_data);
    in_tensors = ConvertNpzToTensor(npz, session.get());
  } else {
    for (size_t j = 0; j < session->InputNames().size(); ++j) {
      auto& name = session->InputNames()[j];
      auto in_dims = session->InputDims()[j];
      if (batch != -1)
        in_dims[0] = batch;
      if (in_dims[0] == -1)
        in_dims[0] = 1;
      auto tensor = std::make_shared<core::Tensor>(core::Dims(in_dims), ToDataType(session->InputDtypes()[j]),
                                                   core::Location::kHOST);
      RandomFillTensor(tensor);
      in_tensors.emplace(name, tensor);
    }
  }

  StopWatchTimer timer;
  for (size_t repeat = 0; repeat < repeats; ++repeat) {
    if (repeat % 50 == 0) {
      LOG(INFO) << repeat;
    }
    if (barrier)
      barrier->Wait();
    timer.Start();
    auto out_tensors = session->RunWithBind(in_tensors);
    timer.Stop();

    if (repeat == repeats - 1 && FLAGS_dump_output != "") {
      DumpOutputTensors(out_tensors);
    }
  }

  LOG(INFO) << std::this_thread::get_id() << " " << session->Config().onnx_file << " time is " << timer.GetAverageTime()
            << " ms"
            << ", tp50: " << timer.ComputePercentile(0.5) << ", tp90: " << timer.ComputePercentile(0.9)
            << ", tp99: " << timer.ComputePercentile(0.99);
}
} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto onnx_paths = SplitString(FLAGS_onnxs, ',');
  auto batches = ParseStrToVec(FLAGS_bs, ',');
  if (!batches.empty()) {
    CHECK_EQ(onnx_paths.size(), batches.size());
  }
  auto configs = ParseFlags(onnx_paths);
  std::vector<std::thread> threads(onnx_paths.size());
  Barrier barrier(onnx_paths.size());
  std::vector<std::unique_ptr<Sess>> sessions;
  for (size_t i = 0; i < configs.size(); ++i) {
    auto sess = std::make_unique<Sess>(configs[i]);
    sessions.emplace_back(std::move(sess));
  }
  LOG(INFO) << "Init session done";

  for (size_t i = 0; i < sessions.size(); ++i) {
    Run(sessions[i], batches.empty() ? -1 : batches[i], nullptr, FLAGS_warmup);
  }
  LOG(INFO) << "--------- Warmup done ---------";
  MemoryUse mem_use(configs.front().device_id, true);

  int thread_num = 0;
  for (size_t i = 0; i < sessions.size(); ++i) {
    threads[thread_num++] =
        std::thread(Run, std::ref(sessions[i]), batches.empty() ? -1 : batches[i], &barrier, FLAGS_repeats);
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }

  return 0;
}