#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

#include "core/core.h"

#include "onnxruntime/onnxruntime_c_api.h"
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "onnxruntime/tensorrt_provider_options.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "utils/random_value.h"
#include "utils/timer.h"

#define SUPPORT_FILTER_NODES 0

struct SessConfig {
  std::string onnx_file;
  int device_id;
  int intra_op_num_threads;
  OrtLoggingLevel log_level;
  GraphOptimizationLevel opt_level{ORT_DISABLE_ALL};

  bool use_cuda{false};
  bool use_trt{false};
  struct TrtConfig {
    size_t min_subgraph_size;
    size_t max_workspace_size;
    bool enable_fp16{false};
    bool enable_int8{false};
    std::string cache_dir;
    std::string calibration_table_name;
    std::string trt_profile_min_shapes;
    std::string trt_profile_max_shapes;
    std::string trt_profile_opt_shapes;
    std::string trt_filter_nodes;
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

inline core::DataType ToDataType(ONNXTensorElementDataType dtype) {
  switch (dtype) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return core::DataType::kBOOL;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return core::DataType::kINT8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return core::DataType::kUINT8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return core::DataType::kHALF;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return core::DataType::kBF16;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return core::DataType::kINT32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return core::DataType::kINT64;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return core::DataType::kFLOAT;

  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
  default:
    LOG(FATAL) << "Not supported dtype " << static_cast<int32_t>(dtype);
    return core::DataType::kFLOAT;
  };
}

inline ONNXTensorElementDataType ToDataType(core::DataType dtype) {
  switch (dtype) {
  case core::DataType::kBOOL:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  case core::DataType::kINT8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  case core::DataType::kUINT8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  case core::DataType::kHALF:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  case core::DataType::kBF16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  case core::DataType::kINT32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  case core::DataType::kINT64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case core::DataType::kFLOAT:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  default:
    LOG(FATAL) << "Not supported dtype " << static_cast<int32_t>(dtype);
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
}

class OrtTensor : public core::Tensor {
public:
  explicit OrtTensor(Ort::Value&& val) : ort_val_(std::move(val)) {
    if (!ort_val_.IsTensor()) {
      LOG(FATAL) << "ort_val is not Tensor";
    }
    auto info = ort_val_.GetTensorTypeAndShapeInfo();
    dims_ = info.GetShape();
    dtype_ = ToDataType(info.GetElementType());
    location_ = ort_val_.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU ? core::Location::kDEVICE
                                                                                              : core::Location::kHOST;
    external_size_ = dims_.Numel() * core::DataTypeSize(dtype_);
    if (dtype_ == core::DataType::kFLOAT) { // maps to c type float
      if (location_ == core::Location::kDEVICE) {
        external_device_data_ = ort_val_.GetTensorMutableData<float>();
      } else {
        external_host_data_ = ort_val_.GetTensorMutableData<float>();
      }
    } else if (dtype_ == core::DataType::kUINT8) { // maps to c type uint8_t
      if (location_ == core::Location::kDEVICE) {
        external_device_data_ = ort_val_.GetTensorMutableData<uint8_t>();
      } else {
        external_host_data_ = ort_val_.GetTensorMutableData<uint8_t>();
      }
    } else if (dtype_ == core::DataType::kINT8) { // maps to c type int8_t
      if (location_ == core::Location::kDEVICE) {
        external_device_data_ = ort_val_.GetTensorMutableData<int8_t>();
      } else {
        external_host_data_ = ort_val_.GetTensorMutableData<int8_t>();
      }
    } else if (dtype_ == core::DataType::kINT32) { // maps to c type int32_t
      if (location_ == core::Location::kDEVICE) {
        external_device_data_ = ort_val_.GetTensorMutableData<int32_t>();
      } else {
        external_host_data_ = ort_val_.GetTensorMutableData<int32_t>();
      }
    } else if (dtype_ == core::DataType::kINT64) { // maps to c type int64_t
      if (location_ == core::Location::kDEVICE) {
        external_device_data_ = ort_val_.GetTensorMutableData<int64_t>();
      } else {
        external_host_data_ = ort_val_.GetTensorMutableData<int64_t>();
      }
    } else if (dtype_ == core::DataType::kBOOL) { // maps to c type bool
      if (location_ == core::Location::kDEVICE) {
        external_device_data_ = ort_val_.GetTensorMutableData<bool>();
      } else {
        external_host_data_ = ort_val_.GetTensorMutableData<bool>();
      }
    } else if (dtype_ == core::DataType::kHALF) { // maps to c type bool
      if (location_ == core::Location::kDEVICE) {
        external_device_data_ = ort_val_.GetTensorMutableData<__half>();
      } else {
        external_host_data_ = ort_val_.GetTensorMutableData<__half>();
      }
    } else if (dtype_ == core::DataType::kBF16) { // Non-IEEE
                                                  // floating-point
                                                  // format based on
                                                  // IEEE754
                                                  // single-precision
      if (location_ == core::Location::kDEVICE) {
        external_device_data_ = ort_val_.GetTensorMutableData<uint16_t>();
      } else {
        external_host_data_ = ort_val_.GetTensorMutableData<uint16_t>();
      }
    } else {
      LOG(FATAL) << "Not supported dtype " << static_cast<int>(dtype_);
    }
  }

  ~OrtTensor() override = default;

private:
  Ort::Value ort_val_{nullptr};
};

class Sess {
public:
  explicit Sess(SessConfig config)
      : config_(config), env_(config.log_level, config_.onnx_file.c_str()),
        cpu_mem_info_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU)),
        cuda_mem_info_(Ort::MemoryInfo("Cuda", OrtDeviceAllocator, config_.device_id, OrtMemTypeDefault)) {
    InitSession();
  }

  void RegisterBeforeH2DHook(std::function<void(void)> func) { before_h2d_hook_.emplace_back(func); }
  void RegisterAfterH2DHook(std::function<void(void)> func) { after_h2d_hook_.emplace_back(func); }
  void RegisterBeforeRunD2HHook(std::function<void(void)> func) { before_run_d2h_hook_.emplace_back(func); }
  void RegisterAfterRunD2HHook(std::function<void(void)> func) { after_run_d2h_hook_.emplace_back(func); }

  std::map<std::string, core::TensorRef> RunWithBind(const std::map<std::string, core::TensorRef>& inputs,
                                                     bool out_is_gpu = false) {
    CHECK_EQ(inputs.size(), static_cast<size_t>(in_cnt_));
    Ort::IoBinding bind(*session_);

    for (auto& it : inputs) {
      auto& name = it.first;
      auto ort_val = InitOrtValFromTensor(it.second);
      for (auto& hook : before_h2d_hook_)
        hook();
      bind.BindInput(name.c_str(), ort_val);
      bind.SynchronizeInputs();
      for (auto& hook : after_h2d_hook_)
        hook();
    }

    for (size_t i = 0; i < out_cnt_; ++i) {
      bind.BindOutput(output_names_[i].c_str(), out_is_gpu ? cuda_mem_info_ : cpu_mem_info_);
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
    std::map<std::string, core::TensorRef> res;
    for (size_t i = 0; i < out_names.size(); ++i) {
      res.emplace(out_names[i], std::make_shared<OrtTensor>(std::move(outs[i])));
    }
    return res;
  }

  std::map<std::string, core::TensorRef> RunNoBind(const std::map<std::string, core::TensorRef>& inputs,
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
      auto ort_val = InitOrtValFromTensor(in_tensor);
      for (auto& hook : after_h2d_hook_)
        hook();
      input_tensors[i] = std::move(ort_val);
    }
    for (auto& hook : before_run_d2h_hook_)
      hook();
    std::vector<Ort::Value> outs;
    try {
      outs = session_->Run(Ort::RunOptions{nullptr}, input_names_char_.data(), input_tensors.data(), in_cnt_,
                           output_names_char_.data(), out_cnt_);
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

    std::map<std::string, core::TensorRef> res;
    for (size_t i = 0; i < output_names_.size(); ++i) {
      res.emplace(output_names_[i], std::make_shared<OrtTensor>(std::move(outs[i])));
    }
    return res;
  }

  const std::vector<std::string>& InputNames() { return input_names_; }
  const std::vector<std::string>& OutputNames() { return output_names_; }
  const std::vector<std::vector<int64_t>>& InputDims() { return input_node_dims_; }
  const std::vector<std::vector<int64_t>>& OutputDims() { return output_node_dims_; }
  const std::vector<ONNXTensorElementDataType>& InputDtypes() { return input_types_; }
  const std::vector<ONNXTensorElementDataType>& OutputDtypes() { return output_types_; }

  SessConfig Config() { return config_; }

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
    const auto& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* tensorrt_options;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
    std::vector<const char*> option_keys = {
        "device_id",                    //
        "trt_max_workspace_size",       //
        "trt_max_partition_iterations", //
        "trt_min_subgraph_size",        //
        "trt_fp16_enable",              //
        "trt_int8_enable",              //
        "trt_dump_subgraphs",           //
        "trt_engine_cache_enable",      //
        "trt_detailed_build_log",       //
        // "trt_layer_norm_fp32_fallback",
        // "trt_builder_optimization_level",
    };
    std::string device_id = std::to_string(config_.device_id);
    std::string max_workspace_size = std::to_string(config_.trt_config.max_workspace_size);
    std::string min_subgraph_size = std::to_string(config_.trt_config.min_subgraph_size);
    std::vector<const char*> option_values = {
        device_id.c_str(),
        max_workspace_size.c_str(),
        "1000",
        min_subgraph_size.c_str(),
        config_.trt_config.enable_fp16 ? "1" : "0",
        config_.trt_config.enable_int8 ? "1" : "0",
        "0",
        config_.trt_config.cache_dir.empty() ? "0" : "1",
        config_.log_level == ORT_LOGGING_LEVEL_VERBOSE ? "1" : "0",
        // "1",
        // "3",
    };
#ifdef SUPPORT_FILTER_NODES
    if (!config_.trt_config.trt_filter_nodes.empty()) {
      option_keys.push_back("trt_filter_nodes");
      option_values.push_back(config_.trt_config.trt_filter_nodes.c_str());
    }
#endif
    if (!config_.trt_config.cache_dir.empty()) {
      option_keys.push_back("trt_engine_cache_path");
      option_values.push_back(config_.trt_config.cache_dir.c_str());
      option_keys.push_back("trt_timing_cache_enable");
      option_values.push_back("1");
      option_keys.push_back("trt_timing_cache_path");
      option_values.push_back(config_.trt_config.cache_dir.c_str());
    }
    if (!config_.trt_config.trt_profile_min_shapes.empty()) {
      CHECK_EQ(config_.trt_config.trt_profile_max_shapes.empty(), false);
      CHECK_EQ(config_.trt_config.trt_profile_opt_shapes.empty(), false);
      option_keys.push_back("trt_profile_min_shapes");
      option_values.push_back(config_.trt_config.trt_profile_min_shapes.c_str());
      option_keys.push_back("trt_profile_max_shapes");
      option_values.push_back(config_.trt_config.trt_profile_max_shapes.c_str());
      option_keys.push_back("trt_profile_opt_shapes");
      option_values.push_back(config_.trt_config.trt_profile_opt_shapes.c_str());
    }
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(tensorrt_options, option_keys.data(), option_values.data(),
                                                        option_keys.size()));
    session_options_.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);
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
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
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
    session_.reset(new Ort::Session(env_, config_.onnx_file.c_str(), session_options_));
    auto session_end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration<double, std::milli>(session_end - session_start).count();
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

  Ort::Value InitOrtValFromData(void* data, const std::vector<int64_t>& dims, ONNXTensorElementDataType type,
                                bool on_gpu) {
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
    Ort::MemoryInfo cuda_mem_info{"Cuda", OrtDeviceAllocator, config_.device_id, OrtMemTypeDefault};
    size_t num = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    Ort::Value tensor = Ort::Value::CreateTensor(on_gpu ? cuda_mem_info : mem_info, data, num * SizeOf(type),
                                                 dims.data(), dims.size(), type);
    return tensor;
  }

  Ort::Value InitOrtValFromTensor(const core::TensorRef& t) {
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPU);
    Ort::MemoryInfo cuda_mem_info{"Cuda", OrtDeviceAllocator, config_.device_id, OrtMemTypeDefault};
    auto location = t->GetLocation();
    return Ort::Value::CreateTensor(location == core::Location::kDEVICE ? cuda_mem_info : mem_info,
                                    location == core::Location::kDEVICE ? t->DeviceData() : t->HostData(), t->Bytes(),
                                    t->GetDims().d, t->GetDims().num_dims, ToDataType(t->GetDataType()));
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

inline void RandomFillTensor(core::TensorRef& tensor) {
  auto num = tensor->Numel();
  auto dtype = tensor->GetDataType();
  switch (dtype) {
  case core::DataType::kFLOAT:
    FillBuffer<float>(tensor->HostData(), num, -1., 1.f);
    break;
  case core::DataType::kBOOL:
    FillBuffer<bool>(tensor->HostData(), num, 0, 1);
    break;
  case core::DataType::kUINT8:
    FillBuffer<uint8_t>(tensor->HostData(), num, 0, 255);
    break;
  case core::DataType::kINT8:
  case core::DataType::kINT32:
  case core::DataType::kINT64:
    FillBuffer<int32_t>(tensor->HostData(), num, 0, 127);
    break;
  case core::DataType::kHALF:
  case core::DataType::kBF16:
  case core::DataType::kFP8:
  case core::DataType::kINT4:
  default:
    LOG(FATAL) << "Not supported dtype " << static_cast<int>(dtype);
  }
}
