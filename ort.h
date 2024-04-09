#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

#include "onnxruntime/core/providers/tensorrt/tensorrt_provider_options.h"
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "utils/timer.h"

struct SessConfig {
  std::string onnx_file;
  int device_id;
  int intra_op_num_threads;
  OrtLoggingLevel log_level;
  GraphOptimizationLevel opt_level;

  bool use_cuda;
  bool use_trt;
  struct TrtConfig {
    size_t min_subgraph_size;
    size_t max_workspace_size;
    std::string precision; // fp16, fp32.
    std::string cache_dir;
    bool enable_int8;
    std::string calibration_table_name;
    std::string filter_ops;
  } trt_config;
  bool use_openvino;
};

inline size_t SizeOf(ONNXTensorElementDataType dtype) {
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

class Tensor {
public:
  explicit Tensor(const std::string& name, void* data,
                  std::vector<int64_t> dims, ONNXTensorElementDataType dtype,
                  bool on_gpu)
      : name(name), data(data), dims(dims), dtype(dtype), on_gpu(on_gpu) {}

  explicit Tensor(const std::string& name, Ort::Value&& v)
      : name(name), ort_val_(std::move(v)) {
    if (!ort_val_.IsTensor()) {
      LOG(FATAL) << name << " is not Tensor";
    }
    auto info = ort_val_.GetTensorTypeAndShapeInfo();
    dims = info.GetShape();
    dtype = info.GetElementType();
    on_gpu = ort_val_.GetLocation().GetAllocatorName() == "Cuda";
    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) { // maps to c type float
      data = ort_val_.GetTensorMutableData<float>();
    } else if (dtype ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) { // maps to c type uint8_t
      data = ort_val_.GetTensorMutableData<uint8_t>();
    } else if (dtype ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) { // maps to c type int8_t
      data = ort_val_.GetTensorMutableData<int8_t>();
    } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) { // maps to c
                                                                // type uint16_t
      data = ort_val_.GetTensorMutableData<uint16_t>();
    } else if (dtype ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) { // maps to c type int16_t
      data = ort_val_.GetTensorMutableData<int16_t>();
    } else if (dtype ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) { // maps to c type int32_t
      data = ort_val_.GetTensorMutableData<int32_t>();
    } else if (dtype ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) { // maps to c type int64_t
      data = ort_val_.GetTensorMutableData<int64_t>();
    } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      data = ort_val_.GetTensorMutableData<bool>();
    } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      data = ort_val_.GetTensorMutableData<__half>();
    } else if (dtype ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) { // maps to c type double
      data = ort_val_.GetTensorMutableData<double>();
    } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) { // maps to c
                                                                // type uint32_t
      data = ort_val_.GetTensorMutableData<uint32_t>();
    } else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) { // maps to c
                                                                // type uint64_t
      data = ort_val_.GetTensorMutableData<uint64_t>();
    } else if (dtype ==
               ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) { // Non-IEEE
                                                         // floating-point
                                                         // format based on
                                                         // IEEE754
                                                         // single-precision
      data = ort_val_.GetTensorMutableData<uint16_t>();
    } else {
      LOG(FATAL) << "Not supported dtype " << static_cast<int>(dtype);
    }
  }

  Tensor(Tensor&& other) {
    if (ort_val_) {
      this->ort_val_ = std::move(ort_val_);
    }

    this->name = other.name;
    this->data = other.data;
    this->dims = other.dims;
    this->dtype = other.dtype;
    this->on_gpu = other.on_gpu;
  }

  Tensor& operator=(Tensor&& other) {
    if (ort_val_) {
      this->ort_val_ = std::move(ort_val_);
    }

    this->name = other.name;
    this->data = other.data;
    this->dims = other.dims;
    this->dtype = other.dtype;
    this->on_gpu = other.on_gpu;
    return *this;
  }

  std::string name;
  void* data;
  std::vector<int64_t> dims;
  ONNXTensorElementDataType dtype;
  bool on_gpu;

private:
  Ort::Value ort_val_{nullptr};
};

class Sess {
public:
  explicit Sess(SessConfig config)
      : config_(config), env_(config.log_level, config_.onnx_file.c_str()),
        cpu_mem_info_(Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU)),
        cuda_mem_info_(Ort::MemoryInfo("Cuda", OrtDeviceAllocator,
                                       config_.device_id, OrtMemTypeDefault)) {
    InitSession();
  }

  void RegisterBeforeH2DHook(std::function<void(void)> func) {
    before_h2d_hook_.emplace_back(func);
  }
  void RegisterAfterH2DHook(std::function<void(void)> func) {
    after_h2d_hook_.emplace_back(func);
  }
  void RegisterBeforeRunD2HHook(std::function<void(void)> func) {
    before_run_d2h_hook_.emplace_back(func);
  }
  void RegisterAfterRunD2HHook(std::function<void(void)> func) {
    after_run_d2h_hook_.emplace_back(func);
  }

  std::map<std::string, Tensor>
  RunWithBind(const std::map<std::string, Tensor>& inputs,
              bool out_is_gpu = false) {
    CHECK_EQ(inputs.size(), static_cast<size_t>(in_cnt_));
    Ort::IoBinding bind(*session_);

    for (auto& it : inputs) {
      auto& name = it.first;
      auto tensor = InitTensorFromData(it.second.data, it.second.dims,
                                       it.second.dtype, it.second.on_gpu);
      for (auto& hook : before_h2d_hook_)
        hook();
      bind.BindInput(name.c_str(), tensor);
      bind.SynchronizeInputs();
      for (auto& hook : after_h2d_hook_)
        hook();
    }

    for (size_t i = 0; i < out_cnt_; ++i) {
      bind.BindOutput(output_names_[i].c_str(),
                      out_is_gpu ? cuda_mem_info_ : cpu_mem_info_);
    }

    for (auto& hook : before_run_d2h_hook_)
      hook();
    try {
      session_->Run(Ort::RunOptions{}, bind);
      bind.SynchronizeOutputs();
    } catch (const Ort::Exception& ex) {
      LOG(FATAL) << config_.onnx_file << " failed. Onnxruntime Error: " << ex.what();
    } catch (const std::exception& ex) {
      LOG(FATAL) << config_.onnx_file << " failed. Unexpected error occured: " << ex.what();
    } catch (...) {
      LOG(FATAL) << config_.onnx_file << " failed. Unexpected error occured";
    }
    for (auto& hook : after_run_d2h_hook_)
      hook();

    auto outs = bind.GetOutputValues();
    auto out_names = bind.GetOutputNames();
    std::map<std::string, Tensor> res;
    for (size_t i = 0; i < out_names.size(); ++i) {
      auto name = out_names[i];
      auto& val = outs[i];
      // res[name] = Tensor{name, std::move(val)};
      res.emplace(name, Tensor(name, std::move(val)));
    }
    return res;
  }

  std::map<std::string, Tensor>
  RunNoBind(const std::map<std::string, Tensor>& inputs,
            bool out_is_gpu = false) {
    CHECK_EQ(inputs.size(), static_cast<size_t>(in_cnt_));
    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(in_cnt_);
    for (int i = 0; i < in_cnt_; ++i) {
      input_tensors.push_back(Ort::Value{nullptr});
    }

    for (size_t i = 0; i < in_cnt_; ++i) {
      auto name = input_names_[i];
      const auto& in_tensor = inputs.at(name);

      for (auto& hook : before_h2d_hook_)
        hook();
      auto tensor = InitTensorFromData(in_tensor.data, in_tensor.dims,
                                       in_tensor.dtype, in_tensor.on_gpu);
      for (auto& hook : after_h2d_hook_)
        hook();
      input_tensors[i] = std::move(tensor);
    }
    for (auto& hook : before_run_d2h_hook_)
      hook();
    std::vector<Ort::Value> outs;
    try {
      outs = session_->Run(Ort::RunOptions{nullptr},
                                input_names_char_.data(), input_tensors.data(),
                                in_cnt_, output_names_char_.data(), out_cnt_);
      cudaDeviceSynchronize();
    } catch (const Ort::Exception& ex) {
      LOG(FATAL) << config_.onnx_file << " failed. Onnxruntime Error: " << ex.what();
    } catch (const std::exception& ex) {
      LOG(FATAL) << config_.onnx_file << " failed. Unexpected error occured: " << ex.what();
    } catch (...) {
      LOG(FATAL) << config_.onnx_file << " failed. Unexpected error occured";
    }
    for (auto& hook : after_run_d2h_hook_)
      hook();

    std::map<std::string, Tensor> res;
    for (size_t i = 0; i < output_names_.size(); ++i) {
      auto name = output_names_[i];
      auto& val = outs[i];
      // res[name] = Tensor(name, std::move(val));
      res.emplace(name, Tensor(name, std::move(val)));
    }
    return res;
  }

  const std::vector<std::string>& InputNames() { return input_names_; }
  const std::vector<std::string>& OutputNames() { return output_names_; }
  const std::vector<std::vector<int64_t>>& InputDims() {
    return input_node_dims_;
  }
  const std::vector<std::vector<int64_t>>& OutputDims() {
    return output_node_dims_;
  }
  const std::vector<ONNXTensorElementDataType>& InputDtypes() {
    return input_types_;
  }
  const std::vector<ONNXTensorElementDataType>& OutputDtypes() {
    return output_types_;
  }

private:
  void SetCudaProviders(int dev_id = 0) {
    OrtCUDAProviderOptions cuda_opt;
    cuda_opt.device_id = dev_id;
    cuda_opt.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_opt.gpu_mem_limit = SIZE_MAX;
    cuda_opt.do_copy_in_default_stream = false;
    cuda_opt.has_user_compute_stream = false;
    cuda_opt.user_compute_stream = nullptr;
    session_options_.AppendExecutionProvider_CUDA(cuda_opt);
  }

  void SetTrtProviders() {
    OrtTensorRTProviderOptions trt_opt{};
    trt_opt.device_id = config_.device_id;
    trt_opt.has_user_compute_stream = false;
    trt_opt.user_compute_stream = nullptr;
    trt_opt.trt_max_partition_iterations = 1000;
    trt_opt.trt_min_subgraph_size = config_.trt_config.min_subgraph_size;
    trt_opt.trt_max_workspace_size = config_.trt_config.max_workspace_size;
    trt_opt.trt_fp16_enable = config_.trt_config.precision == "fp16";
    trt_opt.trt_int8_enable = config_.trt_config.enable_int8;
    trt_opt.trt_engine_cache_enable = config_.trt_config.cache_dir != "";
    trt_opt.trt_engine_cache_path = config_.trt_config.cache_dir.c_str();
    trt_opt.trt_int8_calibration_table_name =
        trt_opt.trt_int8_enable
            ? config_.trt_config.calibration_table_name.c_str()
            : "";
    trt_opt.trt_dump_subgraphs = false;

    trt_opt.trt_filter_ops = config_.trt_config.filter_ops.c_str();
    // trt_opt.trt_prefer_precision_ops = FLAGS_trtPreferPrecisionOps.c_str();
    // trt_opt.trt_prefer_precision_nodes =
    // FLAGS_trtPreferPrecisionNodes.c_str(); trt_opt.trt_force_precision_ops =
    // FLAGS_trtForcePrecisionOps.c_str(); trt_opt.trt_force_precision_nodes =
    // FLAGS_trtForcePrecisionNodes.c_str();

    session_options_.AppendExecutionProvider_TensorRT(trt_opt);
  }

  void SetCpuProviders() {}

  void SetOpenVINOProviders() {
    OrtOpenVINOProviderOptions options;
    options.device_type = "CPU_FP32";
    options.device_id = "";
    options.num_of_threads = 8;
    // options.cache_dir = "";
    // options.context = 0x123456ff;
    // options.enable_opencl_throttling = false;
    session_options_.AppendExecutionProvider_OpenVINO(options);

    // https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#onnxruntime-graph-level-optimization
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);
  }

  void InitSession() {
    session_options_.SetIntraOpNumThreads(config_.intra_op_num_threads);
    session_options_.SetGraphOptimizationLevel(config_.opt_level);

    if (config_.use_trt) {
      SetTrtProviders();
    }
    if (config_.use_cuda) {
      SetCudaProviders();
    }
    if (config_.use_openvino) {
      // SetOpenVINOProviders();
    }
    SetCpuProviders();

    auto session_start = std::chrono::high_resolution_clock::now();
    session_.reset(
        new Ort::Session(env_, config_.onnx_file.c_str(), session_options_));
    auto session_end = std::chrono::high_resolution_clock::now();
    auto dur =
        std::chrono::duration<double, std::milli>(session_end - session_start)
            .count();
    warmup_time_ms_ += dur;
    // LOG(INFO) << "Init session time is " << dur << ", ms";

    Ort::AllocatorWithDefaultOptions allocator;
    in_cnt_ = session_->GetInputCount();
    input_names_.resize(in_cnt_);
    input_names_char_.resize(in_cnt_);
    input_node_dims_.resize(in_cnt_);
    input_types_.resize(in_cnt_);
    for (int i = 0; i < in_cnt_; ++i) {
      input_names_[i] = session_->GetInputNameAllocated(i, allocator).get();
      input_names_char_[i] = input_names_[i].c_str();
      auto type_info = session_->GetInputTypeInfo(i);
      auto info = type_info.GetTensorTypeAndShapeInfo();
      input_types_[i] = info.GetElementType();
      input_node_dims_[i] = info.GetShape();
    }
    out_cnt_ = session_->GetOutputCount();
    output_names_.resize(out_cnt_);
    output_names_char_.resize(out_cnt_);
    output_node_dims_.resize(out_cnt_);
    output_types_.resize(out_cnt_);
    for (int i = 0; i < out_cnt_; ++i) {
      output_names_[i] = session_->GetOutputNameAllocated(i, allocator).get();
      output_names_char_[i] = output_names_[i].c_str();
      auto type_info = session_->GetOutputTypeInfo(i);
      auto info = type_info.GetTensorTypeAndShapeInfo();
      output_types_[i] = info.GetElementType();
      output_node_dims_[i] = info.GetShape();
    }
  }

  Ort::Value InitTensorFromData(void* data, const std::vector<int64_t>& dims,
                                ONNXTensorElementDataType type, bool on_gpu) {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
    Ort::MemoryInfo cuda_mem_info{"Cuda", OrtDeviceAllocator, config_.device_id,
                                  OrtMemTypeDefault};
    size_t num =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    Ort::Value tensor = Ort::Value::CreateTensor(
        on_gpu ? cuda_mem_info : mem_info, data, num * SizeOf(type),
        dims.data(), dims.size(), type);
    return tensor;
  }

private:
  SessConfig config_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_{nullptr};

  Ort::MemoryInfo cpu_mem_info_;
  Ort::MemoryInfo cuda_mem_info_;

  float warmup_time_ms_{0};

  int in_cnt_{0};
  int out_cnt_{0};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_names_char_;
  std::vector<const char*> output_names_char_;
  std::vector<std::vector<int64_t>> input_node_dims_;
  std::vector<std::vector<int64_t>> output_node_dims_;
  std::vector<ONNXTensorElementDataType> input_types_;
  std::vector<ONNXTensorElementDataType> output_types_;

  std::vector<std::function<void(void)>> before_h2d_hook_;
  std::vector<std::function<void(void)>> after_h2d_hook_;
  std::vector<std::function<void(void)>> before_run_d2h_hook_;
  std::vector<std::function<void(void)>> after_run_d2h_hook_;
};
