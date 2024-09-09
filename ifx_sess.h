#pragma once
#include "ifx/ifx.h"
#include "utils/cuda_graph.h"
#include "utils/util.h"

#include <chrono>
#include <cstddef>
#include <functional>

#include <cuda_runtime_api.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

namespace ifx_sess {

struct SessConfig {
  std::string ifx_file;
  int device_id;
  std::string cache_dir;
  bool use_gpu;
  bool enable_fp16;
};

class IfxAllocator : public ifx::IExternalAllocator {
public:
  void* allocate(size_t size) override {
    void* data_ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&data_ptr, size));
    return data_ptr;
  }
  void deallocate(void* ptr, size_t) override { CUDA_CHECK(cudaFreeHost(ptr)); }
};

inline std::vector<int64_t> GetShape(const ifx::IONode& node) {
  std::vector<int64_t> shape;
  shape.reserve(5);
  // NOTE: some types are decrepted, so not support
  switch (node.sDataFormat) {
  case ifx::TensorFormat::TENSOR_FORMAT_NCHW:
    shape = {node.sN, node.sC, node.sH, node.sW};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_NHWC:
    shape = {node.sN, node.sH, node.sW, node.sC};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_W:
    shape = {node.sW};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_NW:
    shape = {node.sN, node.sW};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_NHW:
    shape = {node.sN, node.sH, node.sW};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_NCWH:
    shape = {node.sN, node.sC, node.sW, node.sH};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_NHCW:
    shape = {node.sN, node.sH, node.sC, node.sW};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_NWCH:
    shape = {node.sN, node.sW, node.sC, node.sH};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_NWHC:
    shape = {node.sN, node.sW, node.sH, node.sC};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_NCHWD:
    shape = {node.sN, node.sC, node.sW, node.sH, node.sD};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_HW:
    shape = {node.sH, node.sW};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_CHW:
    shape = {node.sC, node.sH, node.sW};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_NC:
    shape = {node.sN, node.sC};
    break;
  case ifx::TensorFormat::TENSOR_FORMAT_OI:
    shape = {node.sO, node.sI};
    break;
  default:
    LOG(ERROR) << "The tensor format not supported";
  }
  return shape;
}

class Tensor {
public:
  explicit Tensor(const std::string& name, void* data, std::vector<int32_t> dims, ifx::DataType dtype,
                  ifx::TensorFormat data_format, bool on_gpu)
      : name(name), data(data), dims(dims), dtype(dtype), data_format(data_format), on_gpu(on_gpu) {
    if (on_gpu) {
      ifx_tensor_ = ifx::Tensor::create(ifx::DeviceType::DEVICE_TYPE_CUDA, dims, data_format, dtype, data);
    } else {
      ifx_tensor_ = ifx::Tensor::create(ifx::DeviceType::DEVICE_TYPE_NATIVE, dims, data_format, dtype, data);
    }
  }

  ~Tensor() {
    if (ifx_tensor_) {
      ifx_tensor_->destroy();
      ifx_tensor_ = nullptr;
    }
  }

  explicit Tensor(const std::string& name, ifx::Tensor* ifx_tensor) : name(name), ifx_tensor_(ifx_tensor) {
    dims = ifx_tensor->getDims();
    dtype = ifx_tensor->getDataType();
    data_format = ifx_tensor->getFormat();
    // TODO(wilber): now force on host.
    data = ifx_tensor->host();
    on_gpu = false;
  }

  Tensor(Tensor&& other) {
    if (other.ifx_tensor_) {
      this->ifx_tensor_ = std::move(other.ifx_tensor_);
      other.ifx_tensor_ = nullptr;
    }

    this->name = other.name;
    this->data = other.data;
    this->dims = other.dims;
    this->dtype = other.dtype;
    this->on_gpu = other.on_gpu;
  }

  Tensor& operator=(Tensor&& other) {
    if (other.ifx_tensor_) {
      this->ifx_tensor_ = std::move(other.ifx_tensor_);
      other.ifx_tensor_ = nullptr;
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
  std::vector<int32_t> dims;
  ifx::DataType dtype;
  ifx::TensorFormat data_format;
  bool on_gpu;
  ifx::Tensor* ifx_tensor_{nullptr};

private:
};

class Ifx_Sess {
public:
  Ifx_Sess(const SessConfig& cfg) : config_(cfg) {
    LOG(INFO) << "Ifx_Sess this is " << this;
    InitSession();
  }

  std::map<std::string, Tensor> Run(const std::map<std::string, Tensor>& in_tensors, cudaStream_t stream = nullptr) {
    std::map<std::string, ifx::Tensor*> input_ifx_tensors;
    std::map<std::string, ifx::Tensor*> output_ifx_tensors;

    for (size_t i = 0; i < in_cnt_; ++i) {
      auto& name = input_names_[i];
      for (auto& hook : before_h2d_hook_)
        hook();
      input_ifx_tensors[name] = in_tensors.at(name).ifx_tensor_;
      for (auto& hook : after_h2d_hook_)
        hook();
    }

    for (size_t i = 0; i < out_cnt_; ++i) {
      output_ifx_tensors.emplace(std::make_pair(output_names_[i], ifx::Tensor::create(&allocator_)));
    }

    for (auto& hook : before_run_d2h_hook_)
      hook();
    int err;
    try {
      if (stream != nullptr) {
        err = sess_->doInference(input_ifx_tensors, output_ifx_tensors, &stream);
        CHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      } else {
        err = sess_->doInference(input_ifx_tensors, output_ifx_tensors);
      }
    } catch (const ifx::IfxException& e) {
      LOG(FATAL) << "IFX Exception Occured: " << e.what();
    } catch (const std::exception& e) {
      LOG(FATAL) << "STD Exception Occured: " << e.what();
    } catch (...) {
      LOG(FATAL) << "Unknown Exception Occured.";
    }

    for (auto& hook : after_run_d2h_hook_)
      hook();

    std::map<std::string, Tensor> res;
    for (size_t i = 0; i < output_names_.size(); ++i) {
      auto& name = output_names_[i];
      res.emplace(name, Tensor(name, output_ifx_tensors[name]));
    }
    return res;
  }

  const std::vector<std::string>& InputNames() { return input_names_; }
  const std::vector<std::string>& OutputNames() { return output_names_; }
  const std::vector<std::vector<int64_t>>& InputDims() { return input_node_dims_; }
  const std::vector<std::vector<int64_t>>& OutputDims() { return output_node_dims_; }
  const std::vector<ifx::DataType>& InputDtypes() { return input_types_; }
  const std::vector<ifx::DataType>& OutputDtypes() { return output_types_; }
  const std::vector<ifx::TensorFormat>& InputFormats() { return input_formats_; }
  const std::vector<ifx::TensorFormat>& OutputFormats() { return output_formats_; }

  void RegisterBeforeH2DHook(std::function<void(void)> func) { before_h2d_hook_.emplace_back(func); }
  void RegisterAfterH2DHook(std::function<void(void)> func) { after_h2d_hook_.emplace_back(func); }
  void RegisterBeforeRunD2HHook(std::function<void(void)> func) { before_run_d2h_hook_.emplace_back(func); }
  void RegisterAfterRunD2HHook(std::function<void(void)> func) { after_run_d2h_hook_.emplace_back(func); }

  SessConfig Config() { return config_; }

private:
  void InitSession() {
    ifx::GraphOptions ifx_options;
    if (config_.use_gpu) {
      ifx_options.sDeviceType = ifx::DeviceType::DEVICE_TYPE_CUDA;
    } else {
      ifx_options.sDeviceType = ifx::DeviceType::DEVICE_TYPE_X86;
    }
    ifx_options.sEnableMemOpt = false;
    ifx_options.sDeviceId = config_.device_id;
    ifx_options.sCachePath = config_.cache_dir;
    ifx_options.sEnableHalf = config_.enable_fp16;

    graph_ = ifx::Graph::create(ifx_options);

    CHECK_EQ(graph_->loadIfxEncryptModel(config_.ifx_file.c_str(), inputs_, outputs_), 0)
        << "IFX load model failed " << config_.ifx_file;
    sess_ = graph_->createSession();

    in_cnt_ = inputs_.size();
    input_names_.resize(in_cnt_);
    input_node_dims_.resize(in_cnt_);
    input_types_.resize(in_cnt_);
    input_formats_.resize(in_cnt_);

    out_cnt_ = outputs_.size();
    output_names_.resize(out_cnt_);
    output_node_dims_.resize(out_cnt_);
    output_types_.resize(out_cnt_);
    output_formats_.resize(out_cnt_);

    for (size_t i = 0; i < in_cnt_; ++i) {
      input_names_[i] = inputs_[i].sBlobName;
      input_node_dims_[i] = GetShape(inputs_[i]);
      input_types_[i] = inputs_[i].sDataType;
      input_formats_[i] = inputs_[i].sDataFormat;
    }
    for (size_t i = 0; i < out_cnt_; ++i) {
      output_names_[i] = outputs_[i].sBlobName;
      output_node_dims_[i] = GetShape(outputs_[i]);
      output_types_[i] = outputs_[i].sDataType;
      output_formats_[i] = outputs_[i].sDataFormat;
    }
  }

private:
  Ifx_Sess(const Ifx_Sess&) = delete;
  Ifx_Sess& operator=(const Ifx_Sess&) = delete;

  IfxAllocator allocator_;

  SessConfig config_;
  std::vector<ifx::IONode> inputs_;
  std::vector<ifx::IONode> outputs_;

  int in_cnt_{0};
  int out_cnt_{0};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> input_node_dims_;
  std::vector<std::vector<int64_t>> output_node_dims_;
  std::vector<ifx::DataType> input_types_;
  std::vector<ifx::DataType> output_types_;
  std::vector<ifx::TensorFormat> input_formats_;
  std::vector<ifx::TensorFormat> output_formats_;

  ifx::Session* sess_{nullptr};
  ifx::Graph* graph_{nullptr};

  std::vector<std::function<void(void)>> before_h2d_hook_;
  std::vector<std::function<void(void)>> after_h2d_hook_;
  std::vector<std::function<void(void)>> before_run_d2h_hook_;
  std::vector<std::function<void(void)>> after_run_d2h_hook_;
};

} // namespace ifx_sess