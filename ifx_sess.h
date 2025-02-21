#pragma once
#include "core/core.h"
#include "ifx/ifx.h"
#include "utils/util.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include <cuda_runtime_api.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

namespace ifx_sess {
inline core::Dims ToDims(const std::vector<int32_t>& ifx_dims) {
  core::Dims res(ifx_dims);
  return res;
}

inline core::DataType ToDataType(ifx::DataType dtype) {
  switch (dtype) {
  case ifx::DataType::DATA_TYPE_BOOL:
    return core::DataType::kBOOL;
  case ifx::DataType::DATA_TYPE_FP16:
    return core::DataType::kHALF;
  case ifx::DataType::DATA_TYPE_FP32:
    return core::DataType::kFLOAT;
  case ifx::DataType::DATA_TYPE_INT32:
    return core::DataType::kINT32;
  case ifx::DataType::DATA_TYPE_INT64:
    return core::DataType::kINT64;
  case ifx::DataType::DATA_TYPE_INT8:
    return core::DataType::kINT8;
  case ifx::DataType::DATA_TYPE_UINT8:
    return core::DataType::kUINT8;
  case ifx::DataType::DATA_TYPE_BF16:
    return core::DataType::kBF16;
  default:
    return core::DataType::kFLOAT;
  };
}

inline ifx::DataType ToDataType(core::DataType dtype) {
  switch (dtype) {
  case core::DataType::kBOOL:
    return ifx::DataType::DATA_TYPE_BOOL;
  case core::DataType::kHALF:
    return ifx::DataType::DATA_TYPE_FP16;
  case core::DataType::kFLOAT:
    return ifx::DataType::DATA_TYPE_FP32;
  case core::DataType::kINT32:
    return ifx::DataType::DATA_TYPE_INT32;
  case core::DataType::kINT64:
    return ifx::DataType::DATA_TYPE_INT64;
  case core::DataType::kINT8:
    return ifx::DataType::DATA_TYPE_INT8;
  case core::DataType::kUINT8:
    return ifx::DataType::DATA_TYPE_UINT8;
  case core::DataType::kBF16:
    return ifx::DataType::DATA_TYPE_BF16;
  default:
    return ifx::DataType::DATA_TYPE_FP32;
  };
}

inline ifx::Tensor* ToIfxTensor(const core::TensorRef& t, ifx::TensorFormat format) {
  if (t->GetLocation() == core::Location::kDEVICE) {
    return ifx::Tensor::create(ifx::DeviceType::DEVICE_TYPE_CUDA, t->GetDims().ToStdVec<int32_t>(), format,
                               ToDataType(t->GetDataType()), const_cast<void*>(t->DeviceData()));

  } else {
    return ifx::Tensor::create(ifx::DeviceType::DEVICE_TYPE_NATIVE, t->GetDims().ToStdVec<int32_t>(), format,
                               ToDataType(t->GetDataType()), const_cast<void*>(t->HostData()));
  }
}

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
  case ifx::TensorFormat::TENSOR_FORMAT_NCDHW:
    shape = {node.sN, node.sC, node.sD, node.sH, node.sW};
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

class IfxTensor : public core::Tensor {
public:
  explicit IfxTensor(ifx::Tensor* ifx_tensor)
      : core::Tensor(ifx_tensor->host(), ifx_tensor->getByteSize(), ToDims(ifx_tensor->getDims()),
                     ToDataType(ifx_tensor->getDataType())),
        ifx_tensor_(ifx_tensor) {}

  ~IfxTensor() override {
    if (ifx_tensor_) {
      ifx_tensor_->destroy();
    }
  }

private:
  ifx::Tensor* ifx_tensor_{nullptr};
};

class Ifx_Sess {
public:
  Ifx_Sess(const SessConfig& cfg) : config_(cfg) {
    InitSession();
  }

  std::map<std::string, core::TensorRef> Run(const std::map<std::string, core::TensorRef>& in_tensors,
                                             cudaStream_t stream = nullptr) {
    std::map<std::string, ifx::Tensor*> input_ifx_tensors;
    std::map<std::string, ifx::Tensor*> output_ifx_tensors;

    for (size_t i = 0; i < in_cnt_; ++i) {
      auto& name = input_names_[i];
      for (auto& hook : before_h2d_hook_)
        hook();
      input_ifx_tensors[name] = ToIfxTensor(in_tensors.at(name), input_formats_[i]);
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

    std::map<std::string, std::shared_ptr<core::Tensor>> res;
    for (size_t i = 0; i < output_names_.size(); ++i) {
      auto& name = output_names_[i];
      res.emplace(name, std::make_shared<IfxTensor>(output_ifx_tensors[name]));
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
