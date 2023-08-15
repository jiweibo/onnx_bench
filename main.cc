#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>

#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(onnx_file, "", "onnx model file");
DEFINE_int32(batch, 1, "batch");
DEFINE_int32(warmup, 0, "warmup");
DEFINE_int32(repeats, 1, "repeats");
DEFINE_string(provider, "cpu", "cpu, cuda, trt");

namespace {
void *GenerateData(const std::vector<int64_t> &dims,
                   ONNXTensorElementDataType type) {
  size_t num =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  std::default_random_engine e;
  e.seed(1998);

  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    float *ptr = static_cast<float *>(malloc(num * sizeof(float)));
    std::uniform_real_distribution<float> u(-1, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    int *ptr = static_cast<int *>(malloc(num * sizeof(int)));
    std::uniform_int_distribution<int> u(-128, 127);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    bool *ptr = static_cast<bool *>(malloc(num * sizeof(bool)));
    std::uniform_int_distribution<int> u(0, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else {
    LOG(FATAL) << "Not supported data type " << type;
  }

  return nullptr;
}

Ort::Value InitTensorFromData(void *data, const std::vector<int64_t> &dims,
                              ONNXTensorElementDataType type) {
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);

  size_t num =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  Ort::Value tensor{nullptr};
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    tensor = Ort::Value::CreateTensor<float>(
        mem_info, static_cast<float *>(data), num, dims.data(), dims.size());
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    tensor = Ort::Value::CreateTensor<int32_t>(
        mem_info, static_cast<int32_t *>(data), num, dims.data(), dims.size());
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    tensor = Ort::Value::CreateTensor<bool>(mem_info, static_cast<bool *>(data),
                                            num, dims.data(), dims.size());
  } else {
    LOG(FATAL) << "Not supported data type " << type;
  }
  return tensor;
}

template <typename T> std::string PrintShape(const std::vector<T> &v) {
  std::stringstream ss;
  for (size_t i = 0; i < v.size() - 1; ++i) {
    ss << v[i] << "x";
  }
  ss << v.back();
  return ss.str();
}

template <typename T> float mean(T *data, size_t n) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += data[i];
  }
  return sum * 1. / n;
}

float MeanValue(const Ort::Value &tensor) {
  auto type_info = tensor.GetTensorTypeAndShapeInfo();
  auto type = type_info.GetElementType();
  auto dims = type_info.GetShape();
  size_t num =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    auto *data = tensor.GetTensorData<float>();
    return mean(data, num);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    auto *data = tensor.GetTensorData<int>();
    return mean(data, num);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    auto *data = tensor.GetTensorData<int64_t>();
    return mean(data, num);
  } else {
    LOG(FATAL) << "Not supported data type " << type;
  }
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
};
} // namespace

void Run() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer_demo");

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  const auto &api = Ort::GetApi();

  if (FLAGS_provider == "cpu") {

  } else if (FLAGS_provider == "cuda") {
    OrtCUDAProviderOptions o;
    memset(&o, 0, sizeof(o));
    o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    o.gpu_mem_limit = SIZE_MAX;
    OrtStatus *onnx_status =
        api.SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);

    // OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    // api.CreateCUDAProviderOptions(&cuda_options);
    // std::vector<const char*> keys{"cudnn_conv1d_pad_to_nc1d"};
    // std::vector<const char*> values{"1"};
    // api.UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(),
    // 1); api.SessionOptionsAppendExecutionProvider_CUDA_V2(session_options,
    // cuda_options);
  } else if (FLAGS_provider == "trt") {
    OrtTensorRTProviderOptionsV2 *tensorrt_options;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
    std::unique_ptr<OrtTensorRTProviderOptionsV2,
                    decltype(api.ReleaseTensorRTProviderOptions)>
        rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
        static_cast<OrtSessionOptions *>(session_options),
        rel_trt_options.get()));
  }

  Ort::Session session(env, FLAGS_onnx_file.c_str(), session_options);

  Ort::AllocatorWithDefaultOptions allocator;

  // Print number of model input nodes
  const size_t num_input_nodes = session.GetInputCount();

  std::vector<std::string> input_names;
  std::vector<const char *> input_names_char;
  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<Ort::Value> input_tensors;
  std::vector<void *> ptr_to_free;

  input_names.reserve(num_input_nodes);
  input_node_dims.reserve(num_input_nodes);
  input_tensors.reserve(num_input_nodes);
  ptr_to_free.reserve(num_input_nodes);

  LOG(INFO) << "Number of inputs " << num_input_nodes;

  // Iterator over all input nodes
  for (size_t i = 0; i < num_input_nodes; ++i) {
    // print input node names
    input_names.push_back(session.GetInputNameAllocated(i, allocator).get());
    input_names_char.push_back(input_names[i].c_str());

    // print input node types
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();

    // Print input shapes/dims
    input_node_dims.push_back(tensor_info.GetShape());

    // TODO(wilber): process for `batch`
    if (input_node_dims[i][0] == -1) {
      input_node_dims[i][0] = FLAGS_batch;
    }

    auto *data = GenerateData(input_node_dims[i], type);
    ptr_to_free.push_back(data);
    input_tensors.emplace_back(
        InitTensorFromData(data, input_node_dims[i], type));

    LOG(INFO) << "Input " << i << " : name = " << input_names[i]
              << ", type = " << type
              << ", num_dims = " << input_node_dims[i].size()
              << ", dims = " << PrintShape(input_node_dims[i]);
  }

  std::vector<std::string> output_names;
  std::vector<const char *> output_names_char;
  std::vector<std::vector<int64_t>> output_node_dims;

  const size_t num_output_nodes = session.GetOutputCount();

  output_names.reserve(num_output_nodes);
  output_names_char.reserve(num_output_nodes);
  output_node_dims.reserve(num_output_nodes);

  LOG(INFO) << "Number of outputs " << num_output_nodes;

  for (size_t i = 0; i < num_output_nodes; ++i) {
    output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
    output_names_char.push_back(output_names.back().c_str());

    auto type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    output_node_dims.push_back(
        type_info.GetTensorTypeAndShapeInfo().GetShape());

    LOG(INFO) << "Output " << i << " : name = " << output_names[i]
              << ", type = " << tensor_info.GetElementType()
              << ", num_dims = " << output_node_dims[i].size()
              << ", dims = " << PrintShape(output_node_dims[i]);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < FLAGS_warmup; ++i) {
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
        num_input_nodes, output_names_char.data(), num_output_nodes);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration<double, std::milli>(end - start).count();
  LOG(INFO) << "warmup done, time is " << time << " ms.";

  StopWatchTimer timer;

  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    timer.Start();
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
        num_input_nodes, output_names_char.data(), num_output_nodes);
    timer.Stop();

    if (i == FLAGS_repeats - 1) {
      for (size_t j = 0; j < output_tensors.size(); ++j) {
        LOG(INFO) << "Mean value " << j << " : "
                  << MeanValue(output_tensors[j]);
      }
    }
  }
  LOG(INFO) << "Average cost time: " << timer.GetAverageTime() << " ms.";

  for (auto v : ptr_to_free) {
    free(v);
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_onnx_file == "") {
    LOG(FATAL) << "Please set --onnx_file flag.";
  }
  Run();
}
