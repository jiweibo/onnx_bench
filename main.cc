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
#include <vector>

#include "dataset.h"

#include "onnxruntime/core/providers/tensorrt/tensorrt_provider_options.h"
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "cnpy.h"

DEFINE_string(onnx, "", "onnx model file");
DEFINE_int32(batch, 1, "batch");
DEFINE_int32(warmup, 0, "warmup");
DEFINE_int32(repeats, 1, "repeats");
DEFINE_string(precision, "fp32", "fp32, fp16");
DEFINE_bool(precisionInt8, false, "enable int8");
DEFINE_string(provider, "cpu", "cpu, openvino, cuda, trt");
DEFINE_string(cacheDir, "", "the cache dir");
DEFINE_string(calibrationName, "", "int8 calibration table");
DEFINE_int32(minSubgraphSize, 1, "trt min subgraph size");
DEFINE_string(
    dumpOutput, "",
    "Save the output tensor(s) of the last inference iteration in a npz file"
    "(default = disabled).");
DEFINE_string(trtFilterOps, "",
              "defaule empty, e.g. 'Flatten_125 Flatten_126'");
DEFINE_string(trtPreferPrecisionOps, "", "prefer fp32 ops");
DEFINE_string(trtPreferPrecisionNodes, "", "prefer fp32 nodes");
DEFINE_string(trtForcePrecisionOps, "", "force ops");
DEFINE_string(trtForcePrecisionNodes, "", "force nodes");

// TODO:
// DEFINE_string(
//     loadInputs, "",
//     "Load input values from files (default = generate random inputs). Input "
//     "names can be wrapped with single quotes (ex: 'Input:0.in')");
// DEFINE_string(inputType, "json", "txt, bin, json etc.");

DEFINE_string(dataDir, "", "a dir which stores lots of json file");

std::default_random_engine e(1998);

namespace {

template <typename T> std::string PrintShape(const std::vector<T>& v) {
  std::stringstream ss;
  for (size_t i = 0; i < v.size() - 1; ++i) {
    ss << v[i] << "x";
  }
  ss << v.back();
  return ss.str();
}

class NvtxRange {
public:
  NvtxRange(const std::string& message) : message_(message) {}

  void Begin() {
    nvtxEventAttributes_t eventAttrib;
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0x00ccffcc;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message_.c_str();

    range_id_ = nvtxRangeStartEx(&eventAttrib);
  }

  void End() { nvtxRangeEnd(range_id_); }

private:
  uint64_t range_id_;
  const std::string message_;
};

void* GenerateData(const std::vector<int64_t>& dims,
                   ONNXTensorElementDataType type) {
  int64_t num =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    float* ptr = static_cast<float*>(malloc(num * sizeof(float)));
    std::uniform_real_distribution<float> u(-1, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    half* ptr = static_cast<half*>(malloc(num * sizeof(half)));
    std::uniform_real_distribution<float> u(-1, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = __half2float(u(e));
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    int* ptr = static_cast<int*>(malloc(num * sizeof(int)));
    std::uniform_int_distribution<int> u(-128, 127);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    auto* ptr = static_cast<int64_t*>(malloc(num * sizeof(int64_t)));
    std::uniform_int_distribution<int> u(-128, 127);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    bool* ptr = static_cast<bool*>(malloc(num * sizeof(bool)));
    std::uniform_int_distribution<int> u(0, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
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

size_t SizeOf(ONNXTensorElementDataType dtype) {
  switch (dtype) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  // maps to c type float
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  // maps to c type int32_t
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: // maps to c type uint32_t
    return 4;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return 1;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: // maps to c type uint16_t
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:  // maps to c type int16_t
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: // Non-IEEE floating-point format
                                               // based on IEEE754
                                               // single-precision
    return 2;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:     // maps to c type int64_t
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:    // maps to c type double
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: // complex with float32 real and
                                                // imaginary components
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:    // maps to c type uint64_t
    return 8;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: // complex with float64 real
                                                 // and imaginary components
    return 16;
  default:
    LOG(FATAL) << "Not supported dtype " << dtype;
  }
}

Ort::Value InitTensorFromData(void* data, const std::vector<int64_t>& dims,
                              ONNXTensorElementDataType type) {
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
  // Ort::MemoryInfo cuda_mem_info{"Cuda", OrtDeviceAllocator, 0,
  //                                 OrtMemTypeDefault};
  size_t num =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  Ort::Value tensor = Ort::Value::CreateTensor(
      mem_info, data, num * SizeOf(type), dims.data(), dims.size(), type);
  return tensor;
}

template <typename T> float mean(T* data, size_t n) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += data[i];
  }
  return sum * 1. / n;
}

const void* GetOrtDataPtr(const Ort::Value& tensor) {
  auto type = tensor.GetTensorTypeAndShapeInfo().GetElementType();
  const void* data{nullptr};
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    data = tensor.GetTensorData<float>();
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    data = tensor.GetTensorData<__half>();
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    data = tensor.GetTensorData<int>();
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    data = tensor.GetTensorData<int64_t>();
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    data = tensor.GetTensorData<bool>();
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    data = tensor.GetTensorData<uint8_t>();
  } else {
    LOG(FATAL) << "Not supported data type " << type;
  }
  return data;
}

float MeanValue(const Ort::Value& tensor) {
  auto type_info = tensor.GetTensorTypeAndShapeInfo();
  auto type = type_info.GetElementType();
  auto dims = type_info.GetShape();
  size_t num =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    auto* data = tensor.GetTensorData<float>();
    return mean(data, num);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    auto* data = tensor.GetTensorData<int>();
    return mean(data, num);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    auto* data = tensor.GetTensorData<int64_t>();
    return mean(data, num);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    auto* data = tensor.GetTensorData<bool>();
    return mean(data, num);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    auto* data = tensor.GetTensorData<uint8_t>();
    return mean(data, num);
  } else {
    LOG(FATAL) << "Not supported data type " << type;
  }
}

void DumpTensors(const std::vector<Ort::Value>& tensors,
                 const std::vector<std::string>& names,
                 const std::string& filename) {
  CHECK_EQ(tensors.size(), names.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& tensor = tensors[i];
    auto& name = names[i];
    auto type_info = tensor.GetTensorTypeAndShapeInfo();
    auto type = type_info.GetElementType();
    auto dims = type_info.GetShape();
    size_t num =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    if (num == 0) {
      continue;
    }
    std::vector<size_t> shape(dims.begin(), dims.end());

    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      auto* data = tensor.GetTensorData<float>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      auto* data = tensor.GetTensorData<__half>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      auto* data = tensor.GetTensorData<int>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      auto* data = tensor.GetTensorData<int64_t>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      auto* data = tensor.GetTensorData<bool>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      auto* data = tensor.GetTensorData<uint8_t>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else {
      LOG(FATAL) << "Not supported data type " << type;
    }
  }
}

void SetCudaProviders(Ort::SessionOptions& session_options) {
  OrtCUDAProviderOptions cuda_opt;
  cuda_opt.device_id = 0;
  cuda_opt.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  cuda_opt.gpu_mem_limit = SIZE_MAX;
  cuda_opt.do_copy_in_default_stream = false;
  cuda_opt.has_user_compute_stream = false;
  cuda_opt.user_compute_stream = nullptr;
  session_options.AppendExecutionProvider_CUDA(cuda_opt);
}

void SetTrtProviders(Ort::SessionOptions& session_options) {
  OrtTensorRTProviderOptions trt_opt{};
  trt_opt.device_id = 0;
  trt_opt.has_user_compute_stream = false;
  trt_opt.user_compute_stream = nullptr;
  trt_opt.trt_max_partition_iterations = 1000;
  trt_opt.trt_min_subgraph_size = FLAGS_minSubgraphSize;
  trt_opt.trt_max_workspace_size = 1UL << 31;
  trt_opt.trt_fp16_enable = FLAGS_precision == "fp16";
  trt_opt.trt_int8_enable = FLAGS_precisionInt8;
  trt_opt.trt_engine_cache_enable = FLAGS_cacheDir != "";
  trt_opt.trt_engine_cache_path = FLAGS_cacheDir.c_str();
  trt_opt.trt_int8_calibration_table_name =
      trt_opt.trt_int8_enable ? FLAGS_calibrationName.c_str() : "";
  trt_opt.trt_dump_subgraphs = false;

  trt_opt.trt_filter_ops = FLAGS_trtFilterOps.c_str();
  // trt_opt.trt_prefer_precision_ops = FLAGS_trtPreferPrecisionOps.c_str();
  // trt_opt.trt_prefer_precision_nodes = FLAGS_trtPreferPrecisionNodes.c_str();
  // trt_opt.trt_force_precision_ops = FLAGS_trtForcePrecisionOps.c_str();
  // trt_opt.trt_force_precision_nodes = FLAGS_trtForcePrecisionNodes.c_str();

  // if (int8_enable) {
  //     trt_opt.trt_int8_calibration_table_name =
  //     int8_calibration_table_file.filename().c_str();
  // }

  session_options.AppendExecutionProvider_TensorRT(trt_opt);
}

void SetCpuProviders(Ort::SessionOptions& session_options) {}

void SetOpenVINOProviders(Ort::SessionOptions& session_options) {
  OrtOpenVINOProviderOptions options;
  options.device_type = "CPU_FP32";
  options.device_id = "";
  options.num_of_threads = 8;
  // options.cache_dir = "";
  // options.context = 0x123456ff;
  // options.enable_opencl_throttling = false;
  session_options.AppendExecutionProvider_OpenVINO(options);

  // https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#onnxruntime-graph-level-optimization
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_DISABLE_ALL);
}

class StopWatchTimer {
public:
  StopWatchTimer()
      : running_(false), clock_sessions_(0), diff_time_(0), total_time_(0) {}
  virtual ~StopWatchTimer() {}

public:
  // Start time measurement
  void Start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    running_ = true;
  }

  // Stop time measurement
  void Stop() {
    diff_time_ = GetDiffTime();
    total_time_ += diff_time_;
    durations_.push_back(diff_time_);
    running_ = false;
    ++clock_sessions_;
  }

  // Reset time counters to zero. Does not change the timer running state but
  // does recapture this point in time as the current start time if it is
  // running.
  void Reset() {
    diff_time_ = 0;
    total_time_ = 0;
    clock_sessions_ = 0;
    durations_.clear();

    if (running_) {
      start_time_ = std::chrono::high_resolution_clock::now();
    }
  }

  // Time in msec. After start if the stop watch is still running (i.e. there
  // was no call to stop()) then the elapsed time is returned, otherwise the
  // time between the last start() and stop call is returned.
  double GetTime() {
    double retval = total_time_;

    if (running_) {
      retval += GetDiffTime();
    }

    return retval;
  }

  // Mean time to date based on the number of times the stopwatch has been
  // stopped and the current total time
  double GetAverageTime() {
    return (clock_sessions_ > 0) ? (total_time_ / clock_sessions_) : 0.0;
  }

  double ComputeVariance() {
    double mean = std::accumulate(durations_.begin(), durations_.end(), 0.0) /
                  durations_.size();
    double sqDiffSum = 0.0;
    for (auto duration : durations_) {
      sqDiffSum += (duration - mean) * (duration - mean);
    }
    return sqDiffSum / (durations_.size() - 1);
  }

  double ComputePercentile(double top) {
    std::sort(durations_.begin(), durations_.end());
    return durations_[(int)(durations_.size() * top)];
  }

  std::vector<double> GetDurations() { return durations_; }

private:
  inline double GetDiffTime() {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time_)
        .count();
  }

private:
  bool running_;

  int clock_sessions_;

  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;

  double diff_time_;

  double total_time_;

  std::vector<double> durations_;
};
} // namespace

Ort::Session InitSession(Ort::Env& env) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  if (FLAGS_provider == "cpu") {
    SetCpuProviders(session_options);
  } else if (FLAGS_provider == "openvino") {
    SetOpenVINOProviders(session_options);
  } else if (FLAGS_provider == "cuda") {
    SetCudaProviders(session_options);
    SetCpuProviders(session_options);
  } else if (FLAGS_provider == "trt") {
    SetTrtProviders(session_options);
    SetCudaProviders(session_options);
    SetCpuProviders(session_options);
  }

  auto session_start = std::chrono::high_resolution_clock::now();
  Ort::Session session(env, FLAGS_onnx.c_str(), session_options);
  auto session_end = std::chrono::high_resolution_clock::now();
  auto dur =
      std::chrono::duration<double, std::milli>(session_end - session_start)
          .count();
  LOG(INFO) << "Init session time is " << dur << ", ms";

  return session;
}

void Run(Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  NvtxRange nvtx_run("run");
  NvtxRange nvtx_h2d("h2d");
  NvtxRange nvtx_d2h("d2h");

  Ort::MemoryInfo cuda_mem_info{"Cuda", OrtDeviceAllocator, 0,
                                OrtMemTypeDefault};
  auto mem_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);

  // Print number of model input nodes
  const size_t num_input_nodes = session.GetInputCount();
  const size_t num_output_nodes = session.GetOutputCount();
  std::vector<std::string> input_names(num_input_nodes);
  std::vector<std::string> output_names(num_output_nodes);
  std::vector<std::vector<int64_t>> input_node_dims(num_input_nodes);
  std::vector<std::vector<int64_t>> output_node_dims(num_output_nodes);
  for (size_t i = 0; i < num_input_nodes; ++i) {
    input_names[i] = session.GetInputNameAllocated(i, allocator).get();
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_node_dims[i] = tensor_info.GetShape();

    if (input_node_dims[i][0] == -1) {
      input_node_dims[i][0] = FLAGS_batch;
    }

    LOG(INFO) << "Input " << i << " : name = " << input_names[i]
              << ", type = " << tensor_info.GetElementType()
              << ", num_dims = " << input_node_dims[i].size()
              << ", dims = " << PrintShape(input_node_dims[i]);
  }
  for (size_t i = 0; i < num_output_nodes; ++i) {
    output_names[i] = session.GetOutputNameAllocated(i, allocator).get();

    auto type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    output_node_dims[i] = type_info.GetTensorTypeAndShapeInfo().GetShape();

    if (output_node_dims[i][0] == -1) {
      output_node_dims[i][0] = FLAGS_batch;
    }

    LOG(INFO) << "Output " << i << " : name = " << output_names[i]
              << ", type = " << tensor_info.GetElementType()
              << ", num_dims = " << output_node_dims[i].size()
              << ", dims = " << PrintShape(output_node_dims[i]);
  }

  Ort::IoBinding bind(session);
  std::vector<void*> ptr_to_free;
  StopWatchTimer timer_run;
  StopWatchTimer timer_h2d;

  auto RunPerBatch = [&]() {
    bind.ClearBoundInputs();
    bind.ClearBoundOutputs();

    for (size_t i = 0; i < num_input_nodes; ++i) {
      auto type = session.GetInputTypeInfo(i)
                      .GetTensorTypeAndShapeInfo()
                      .GetElementType();
      auto* data = GenerateData(input_node_dims[i], type);
      ptr_to_free.push_back(data);
      auto tensor = InitTensorFromData(data, input_node_dims[i], type);
      nvtx_h2d.Begin();
      timer_h2d.Start();
      bind.BindInput(input_names[i].c_str(), tensor);
      bind.SynchronizeInputs();
      timer_h2d.Stop();
      nvtx_h2d.End();
    }

    for (size_t i = 0; i < num_output_nodes; ++i) {
      bind.BindOutput(output_names[i].c_str(), mem_info);
    }

    nvtx_run.Begin();
    timer_run.Start();
    session.Run(Ort::RunOptions{}, bind);
    timer_run.Stop();
    nvtx_run.End();

    bind.SynchronizeOutputs();

    for (auto v : ptr_to_free) {
      free(v);
    }
    ptr_to_free.clear();

    return bind.GetOutputValues();
  };

  for (size_t i = 0; i < FLAGS_warmup; ++i) {
    RunPerBatch();
  }
  timer_run.Reset();
  // timer_h2d.Reset();

  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    auto output_tensors = RunPerBatch();

    if (i == FLAGS_repeats - 1) {
      for (size_t j = 0; j < output_tensors.size(); ++j) {
        auto& out = output_tensors[j];
        if (out.IsTensor()) {
          LOG(INFO) << "Mean value " << j << " : "
                    << MeanValue(output_tensors[j]);
        }
      }
      if (FLAGS_dumpOutput != "") {
        DumpTensors(output_tensors, output_names, FLAGS_dumpOutput);
      }
    }
  }

  LOG(INFO) << "H2D Average time " << timer_h2d.GetAverageTime()
            << ", variance: " << timer_h2d.ComputeVariance()
            << ", tp99: " << timer_h2d.ComputePercentile(0.99);
  LOG(INFO) << "Run+D2H Average time " << timer_run.GetAverageTime()
            << ", variance: " << timer_run.ComputeVariance()
            << ", tp99: " << timer_run.ComputePercentile(0.99);
}

void RunDataSet(Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t num_input_nodes = session.GetInputCount();

  std::vector<std::string> input_names;
  std::vector<const char*> input_names_char;
  std::vector<Ort::Value> input_tensors;

  input_names.reserve(num_input_nodes);
  input_tensors.reserve(num_input_nodes);
  for (int i = 0; i < num_input_nodes; ++i) {
    input_tensors.push_back(Ort::Value{nullptr});
  }

  // Iterator over all input nodes
  for (size_t i = 0; i < num_input_nodes; ++i) {
    input_names.push_back(session.GetInputNameAllocated(i, allocator).get());
    input_names_char.push_back(input_names[i].c_str());
  }

  const size_t num_output_nodes = session.GetOutputCount();
  std::vector<std::string> output_names;
  std::vector<const char*> output_names_char;

  StopWatchTimer timer;
  std::vector<float> max_abs_diff(num_output_nodes, 0);
  std::vector<float> max_abs_diff_base(num_output_nodes, 0);
  std::vector<float> max_abs_diff_ref(num_output_nodes, 0);

  output_names.reserve(num_output_nodes);
  output_names_char.reserve(num_output_nodes);

  for (size_t i = 0; i < num_output_nodes; ++i) {
    output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
    output_names_char.push_back(output_names.back().c_str());
  }

  JsonDataSet ds(FLAGS_dataDir, 5);
  auto names = input_names;
  names.insert(names.end(), output_names.begin(), output_names.end());
  while (1) {
    auto map = ds.GetData(names, FLAGS_batch, true);
    if (map.empty())
      break;
    // LOG(INFO) << "Process for batch " << FLAGS_batch;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
    for (size_t i = 0; i < input_names.size(); ++i) {
      auto name = input_names[i];
      auto dtype = std::get<2>(map[name]);
      auto shape = std::get<1>(map[name]);
      void* data = std::get<0>(map[name]);
      shape[0] = FLAGS_batch;
      size_t num = std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<int>());

      Ort::Value tensor{nullptr};
      if (dtype == Dtype::FLOAT32) {
        tensor =
            Ort::Value::CreateTensor<float>(mem_info, static_cast<float*>(data),
                                            num, shape.data(), shape.size());
      } else if (dtype == Dtype::BOOL) {
        tensor =
            Ort::Value::CreateTensor<bool>(mem_info, static_cast<bool*>(data),
                                           num, shape.data(), shape.size());
      } else {
        LOG(FATAL) << "not support dtype " << static_cast<int>(dtype);
      }
      input_tensors[i] = std::move(tensor);
    }

    timer.Start();
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
        num_input_nodes, output_names_char.data(), num_output_nodes);
    cudaDeviceSynchronize();
    timer.Stop();

    for (size_t i = 0; i < output_names.size(); ++i) {
      auto name = output_names[i];
      auto out_shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
      size_t num = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                   std::multiplies<int>());
      auto dtype = std::get<2>(map[name]);
      // auto shape = std::get<1>(map[name]);
      void* data = std::get<0>(map[name]);

      // CHECK_EQ(out_shape.size(), shape.size());

      // not check precion for int.
      if (dtype == Dtype::FLOAT32) {
        auto* out_data = output_tensors[i].GetTensorData<float>();
        for (size_t j = 0; j < num; ++j) {
          auto diff = abs(out_data[j] - (static_cast<float*>(data))[j]);
          if (diff > max_abs_diff[i]) {
            max_abs_diff[i] = diff;
            max_abs_diff_base[i] = static_cast<float*>(data)[j];
            max_abs_diff_ref[i] = out_data[j];
          }
        }
      } else if (dtype == Dtype::BOOL) {
        auto* out_data = output_tensors[i].GetTensorData<bool>();
        for (size_t j = 0; j < num; ++j) {
          auto diff = abs(out_data[j] - static_cast<bool*>(data)[j]);
          if (diff > max_abs_diff[i]) {
            max_abs_diff[i] = diff;
            max_abs_diff_base[i] = static_cast<bool*>(data)[j];
            max_abs_diff_ref[i] = out_data[j];
          }
        }
      } else if (dtype == Dtype::INT32) {
        // LOG(INFO) << "ignore check for int type.";
      } else if (dtype == Dtype::INT64) {
        // LOG(INFO) << "ignore check for int64 type.";
      } else {
        LOG(FATAL) << "not supported dtype " << static_cast<int>(dtype);
      }
    }
  }

  for (size_t i = 0; i < max_abs_diff.size(); ++i) {
    LOG(INFO) << "max_abs_diff: " << max_abs_diff[i] << ", base is "
              << max_abs_diff_base[i] << ", ref is " << max_abs_diff_ref[i];
  }
  LOG(INFO) << "Average time " << timer.GetAverageTime()
            << ", variance: " << timer.ComputeVariance()
            << ", tp99: " << timer.ComputePercentile(0.99);

  // for (auto dur : timer.GetDurations()) {
  //   LOG(INFO) << dur;
  // }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_onnx == "") {
    LOG(FATAL) << "Please set --onnx flag.";
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer_demo");
  auto session = InitSession(env);

  Run(session);

  if (FLAGS_dataDir != "") {
    RunDataSet(session);
  }
}
