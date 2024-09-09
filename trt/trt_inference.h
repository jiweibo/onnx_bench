#pragma once

#include "core/core.h"
#include "trt/trt_device.h"
#include "trt/trt_engine.h"
#include "utils/util.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <glog/logging.h>

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>

using namespace nvinfer1;

struct Binding {
  bool isInput{false};
  std::unique_ptr<IMirroredBuffer> buffer;
  std::unique_ptr<OutputAllocator> outputAllocator;
  int64_t volume{0};
  DataType dataType{DataType::kFLOAT};

  void fill();

  void FillTensor(const core::Tensor& tensor, cudaStream_t stream) {
    CHECK_EQ(isInput, true);
    size_t bytes = tensor.Bytes();
    if (bytes > buffer->getSize()) {
      buffer->allocate(bytes);
    }
    if (tensor.GetLocation() == core::Location::kDEVICE) {
      CHECK_NOTNULL(tensor.DeviceData());
      CUDA_CHECK(
          cudaMemcpyAsync(buffer->getDeviceBuffer(), tensor.DeviceData(), bytes, cudaMemcpyDeviceToDevice, stream));
    } else {

      CHECK_NOTNULL(tensor.HostData());
      CUDA_CHECK(cudaMemcpyAsync(buffer->getDeviceBuffer(), tensor.HostData(), bytes, cudaMemcpyHostToDevice, stream));
    }
  }
};

struct TensorInfo {
  int32_t bindingIndex{-1};
  const char* name{nullptr};
  Dims dims{};
  Dims strides{};
  bool isDynamic{};
  int32_t comps{-1};
  int32_t vectorDimIndex{-1};
  bool isInput{};
  DataType dataType{};
  int64_t vol{-1};

  void updateVolume(int32_t batch) { vol = volume(dims, strides, vectorDimIndex, comps, batch); }
};

class Bindings {
public:
  Bindings() = delete;
  explicit Bindings(bool useManaged) : mUseManaged(useManaged) {}

  void addBinding(TensorInfo const& tensorInfo, std::string const& fileName = "");

  void** getDeviceBuffers();

  void transferInputToDevice(cudaStream_t stream);

  void transferOutputToHost(cudaStream_t stream);

  bool setTensorAddresses(nvinfer1::IExecutionContext& context) const;

  // Binding* getBinding(size_t idx) {
  //   CHECK_LT(idx, mBindings.size());
  //   return &mBindings[idx];
  // }

  void FillTensors(const std::unordered_map<std::string, core::Tensor>& inputs, cudaStream_t stream);

private:
  std::unordered_map<std::string, int32_t> mNames;
  std::vector<Binding> mBindings;
  std::vector<void*> mDevicePointers;
  bool mUseManaged{false};
};

struct InferenceEnvironment {
  DISABLE_COPY_MOVE_ASSIGN(InferenceEnvironment);
  InferenceEnvironment() = delete;

  InferenceEnvironment(BuildEnvironment& bEnv) : engine(std::move(bEnv.engine)) {}

  LazilyDeserializedEngine engine;
  // std::unique_ptr<Profiler> profiler;
  std::vector<std::unique_ptr<IExecutionContext>> contexts;
  // Device memory used for inference when the allocation strategy is not static.
  std::vector<TrtDeviceBuffer> deviceMemory;
  std::vector<std::unique_ptr<Bindings>> bindings;
  bool error{false};

  template <class ContextType>
  inline ContextType* getContext(int32_t streamIdx);

  //! Storage for input shape tensors.
  //!
  //! It's important that the addresses of the data do not change between the calls to
  //! setTensorAddress/setInputShape (which tells TensorRT where the input shape tensor is)
  //! and enqueueV3 (when TensorRT might use the input shape tensor).
  //!
  //! The input shape tensors could alternatively be handled via member bindings,
  //! but it simplifies control-flow to store the data here since it's shared across
  //! the bindings.
  std::list<std::vector<int32_t>> inputShapeTensorValues;
};

template <>
inline IExecutionContext* InferenceEnvironment::getContext(int32_t streamIdx) {
  return contexts[streamIdx].get();
}

//!
//! \brief Set up contexts and bindings for inference
//!
bool setUpInference(InferenceEnvironment& iEnv, InferenceOptions const& inference, SystemOptions const& system);
