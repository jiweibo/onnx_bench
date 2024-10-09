#include "trt/trt_inference.h"
#include "NvInferImpl.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeBase.h"
#include "NvInferRuntimePlugin.h"
#include "glog/logging.h"
#include "trt/trt_device.h"
#include "trt/trt_options.h"
#include "trt/trt_utils.h"
#include "utils/util.h"
#include <algorithm>
#include <cstdint>
#include <memory>

namespace {

bool allocateContextMemory(InferenceEnvironment& iEnv, const InferenceOptions& inference) {
  auto* engine = iEnv.engine.get();
  iEnv.deviceMemory.resize(inference.infStream);
  // Delay context memory allocation until input shapes are specified because runtime allocation would require actual
  // input shapes.
  for (int32_t i = 0; i < inference.infStream; ++i) {
    const auto& ec = iEnv.contexts.at(i);
    if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kSTATIC) {
      LOG(INFO) << "Create execution context with device memory size: " << engine->getDeviceMemorySizeV2() / 1.0_MiB
                << " MiB";
    } else {
      size_t sizeToAlloc{0};
      const char* allocReason{nullptr};
      if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kPROFILE) {
        const auto p = inference.optProfileIndex;
        sizeToAlloc = engine->getDeviceMemorySizeForProfileV2(p);
        allocReason = "current profile";
      } else if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kRUNTIME) {
        sizeToAlloc = ec->updateDeviceMemorySizeForShapes();
        allocReason = "current input shapes";
      } else {
        LOG(ERROR) << "Unrecognizable memory allocation strategy.";
        return false;
      }
      iEnv.deviceMemory.at(i) = TrtDeviceBuffer(sizeToAlloc);
      ec->setDeviceMemoryV2(iEnv.deviceMemory.at(i).get(), iEnv.deviceMemory.at(i).getSize());
      LOG(INFO) << "Maximum device memory size across all profiles: " << (engine->getDeviceMemorySizeV2() / 1.0_MiB)
                << " MiB";
      LOG(INFO) << "Only allocated device memory enough for " << allocReason << ": " << (sizeToAlloc / 1.0_MiB)
                << " MiB";
    }
  }
  return true;
}

class FillBindingClosure {
public:
  using BindingsVector = std::vector<std::unique_ptr<Bindings>>;
  FillBindingClosure(const ICudaEngine* engine, const IExecutionContext* context, BindingsVector& bindings,
                     int32_t batch, int32_t endBindingIndex, int32_t profileIndex)
      : engine(engine), context(context), bindingsVec(bindings), batch(batch), endBindingIndex(endBindingIndex),
        profileIndex(profileIndex) {}

  bool operator()() { return fillAllBindings(batch, endBindingIndex); }

private:
  void getTensorInfo(TensorInfo& tensorInfo) {
    const auto b = tensorInfo.bindingIndex;
    const auto name = engine->getIOTensorName(b);
    tensorInfo.name = name;
    tensorInfo.dims = context->getTensorShape(name);
    tensorInfo.isDynamic = std::any_of(tensorInfo.dims.d, tensorInfo.dims.d + tensorInfo.dims.nbDims,
                                       [](int32_t dim) { return dim == -1; });
    tensorInfo.comps = engine->getTensorComponentsPerElement(name, profileIndex);
    tensorInfo.strides = context->getTensorStrides(name);
    tensorInfo.vectorDimIndex = engine->getTensorVectorizedDim(name, profileIndex);
    tensorInfo.isInput = engine->getTensorIOMode(name) == TensorIOMode::kINPUT;
    tensorInfo.dataType = engine->getTensorDataType(name);
  }

  void fillOneBinding(const TensorInfo& tensorInfo) {
    const auto name = tensorInfo.name;
    const auto* bindingInOutStr = tensorInfo.isInput ? "Input" : "Output";
    for (auto& bindings : bindingsVec) {
      if (tensorInfo.isInput) {
        LOG(INFO) << "Using random values for input " << name;
      }
      if (tensorInfo.isDynamic) {
        LOG(INFO) << bindingInOutStr << " binding for " << name
                  << " is dynamic and will be created during execution using OutputAllocator.";
      } else {
        LOG(INFO) << bindingInOutStr << " binding for " << name << " with dimensions " << getDimsStr(tensorInfo.dims);
      }
      bindings->addBinding(tensorInfo);
    }
  }

  bool fillAllBindings(int32_t batch, int32_t endBindingIndex) {
    for (int32_t b = 0; b < endBindingIndex; ++b) {
      TensorInfo tensorInfo;
      tensorInfo.bindingIndex = b;
      getTensorInfo(tensorInfo);
      tensorInfo.updateVolume(batch);
      fillOneBinding(tensorInfo);
    }
    return true;
  }

  const ICudaEngine* engine;
  const IExecutionContext* context;
  BindingsVector& bindingsVec;
  int32_t batch;
  int32_t endBindingIndex;
  int32_t profileIndex;
};
} // namespace

void Binding::fill() {
  switch (dataType) {
  case nvinfer1::DataType::kBOOL: {
    fillBuffer<bool>(buffer->getHostBuffer(), volume, 0, 1);
    break;
  }
  case nvinfer1::DataType::kINT32: {
    fillBuffer<int32_t>(buffer->getHostBuffer(), volume, -128, 127);
    break;
  }
  case nvinfer1::DataType::kINT64: {
    fillBuffer<int64_t>(buffer->getHostBuffer(), volume, -128, 127);
    break;
  }
  case nvinfer1::DataType::kINT8: {
    fillBuffer<int8_t>(buffer->getHostBuffer(), volume, -128, 127);
    break;
  }
  case nvinfer1::DataType::kUINT8: {
    fillBuffer<uint8_t>(buffer->getHostBuffer(), volume, 0, 255);
    break;
  }
  case nvinfer1::DataType::kFLOAT: {
    fillBuffer<float>(buffer->getHostBuffer(), volume, -1.0f, 1.0f);
    break;
  }
  default:
    LOG(FATAL) << "Not supported dtype: " << static_cast<int32_t>(dataType);
  }
}

void Bindings::addBinding(TensorInfo const& tensorInfo, std::string const& fileName) {
  const auto b = tensorInfo.bindingIndex;
  while (mBindings.size() <= static_cast<size_t>(b)) {
    mBindings.emplace_back();
    mDevicePointers.emplace_back();
  }
  mNames[tensorInfo.name] = b;
  mBindings[b].isInput = tensorInfo.isInput;
  mBindings[b].volume = tensorInfo.vol;
  mBindings[b].dataType = tensorInfo.dataType;
  if (tensorInfo.isDynamic) {
    CHECK_EQ(tensorInfo.isInput, false);
    mBindings[b].outputAllocator.reset(new OutputAllocator(new DiscreteMirroredBuffer));
  } else {
    if (mBindings[b].buffer == nullptr) {
      mBindings[b].buffer.reset(new DiscreteMirroredBuffer);
    }
    if (tensorInfo.vol == 0) {
      mBindings[b].buffer->allocate(1);
    } else {
      mBindings[b].buffer->allocate(static_cast<size_t>(tensorInfo.vol) *
                                    static_cast<size_t>(dataTypeSize(tensorInfo.dataType)));
    }
    mDevicePointers[b] = mBindings[b].buffer->getDeviceBuffer();
  }

  if (tensorInfo.isInput) {
    if (!fileName.empty()) {
      LOG(FATAL) << "Not Supported";
    } else {
      mBindings[b].fill();
    }
  }
}

void Bindings::transferInputToDevice(cudaStream_t stream) {
  for (auto& it : mNames) {
    if (mBindings[it.second].isInput) {
      mBindings[it.second].buffer->hostToDevice(stream);
    }
  }
}

void Bindings::transferOutputToHost(cudaStream_t stream) {
  for (auto& it : mNames) {
    if (!mBindings[it.second].isInput) {
      if (mBindings[it.second].outputAllocator != nullptr) {
        mBindings[it.second].outputAllocator->getBuffer()->deviceToHost(stream);
      } else {
        mBindings[it.second].buffer->deviceToHost(stream);
      }
    }
  }
}

bool Bindings::setTensorAddresses(nvinfer1::IExecutionContext& context) const {
  for (const auto& it : mNames) {
    const auto name = it.first.c_str();
    const auto location = context.getEngine().getTensorLocation(name);
    if (location == TensorLocation::kDEVICE) {
      if (mBindings[it.second].outputAllocator != nullptr) {
        if (!context.setOutputAllocator(name, mBindings[it.second].outputAllocator.get())) {
          return false;
        }
      } else {
        if (!context.setTensorAddress(name, mDevicePointers[it.second])) {
          return false;
        }
      }
    }
  }
  return true;
}

void Bindings::FillTensors(const std::unordered_map<std::string, core::Tensor>& inputs, cudaStream_t stream) {
  for (const auto& it : inputs) {
    const auto name = it.first;
    const auto& tensor = it.second;
    const auto idx = mNames[name];
    mBindings[idx].FillTensor(tensor, stream);
    mDevicePointers[idx] = mBindings[idx].buffer->getDeviceBuffer();
  }
}

bool setUpInference(InferenceEnvironment& iEnv, InferenceOptions const& inference, SystemOptions const& system) {
  int32_t device{};
  CUDA_CHECK(cudaGetDevice(&device));

  bool useManagedMemory{false};

  auto* engine = iEnv.engine.get();
  CHECK_NOTNULL(engine);
  iEnv.engine.releaseBlob();

  const int32_t nbOptProfiles = engine->getNbOptimizationProfiles();
  if (inference.optProfileIndex >= nbOptProfiles) {
    LOG(ERROR) << "Selected profile index " << inference.optProfileIndex
               << " exceeds the number of profiles that the engine holds";
    return false;
  }

  if (nbOptProfiles > 1 && !inference.setOptProfile) {
    LOG(WARNING) << nbOptProfiles << " profiles detected but not set. Running with profile 0.";
  }

  for (int32_t s = 0; s < inference.infStream; ++s) {
    IExecutionContext* ec{nullptr};
    if (inference.memoryAllocationStrategy == MemoryAllocationStrategy::kSTATIC) {
      // Let TRT pre-allocate and manage the memory.
      ec = engine->createExecutionContext();
    } else {
      // Allocate based on the current profile or runtime shapes.
      ec = engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED);
    }
    if (ec == nullptr) {
      LOG(ERROR) << "Unable to create execution context of stream " << s;
      return false;
    }

    iEnv.contexts.emplace_back(ec);
    iEnv.bindings.emplace_back(new Bindings(useManagedMemory));
  }

  // TODO(wilber): add debug tensor support.

  if (!allocateContextMemory(iEnv, inference)) {
    return false;
  }

  // return true;

  const int32_t endBindingIndex = engine->getNbIOTensors();

  for (int32_t b = 0; b < endBindingIndex; ++b) {
    const auto& name = engine->getIOTensorName(b);
    const auto& mode = engine->getTensorIOMode(name);
    if (mode == TensorIOMode::kINPUT) {
      const Dims dims = iEnv.contexts.front()->getTensorShape(name);
      bool isShapeInferenceIO{false};
      const bool hasRuntimeDim = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; });
      isShapeInferenceIO = engine->isShapeInferenceIO(name);
      const auto shape = inference.shapes.find(name);
      if (hasRuntimeDim || isShapeInferenceIO) {
        // Set shapeData to either dimensions of the input (if it has a dynamic shape)
        // or set to values of the input (if it is an input shape tensor)
        std::vector<int32_t> shapeData;

        if (shape == inference.shapes.end()) {
          // No information provided. Use default value for missing data.
          constexpr int32_t kDEFAULT_VALUE = 1;
          if (isShapeInferenceIO) {
            // Set shape tensor to all ones.
            shapeData.assign(volume(dims, 0, dims.nbDims), kDEFAULT_VALUE);
            LOG(WARNING) << "Values missing for input shape tensor: " << name
                         << " Automatically setting values to: " << GetShapeStr(shapeData);
          } else {
            // Use default value for unspecified runtime dimensions.
            shapeData.resize(dims.nbDims);
            std::transform(dims.d, dims.d + dims.nbDims, shapeData.begin(),
                           [](int32_t dimension) { return dimension >= 0 ? dimension : kDEFAULT_VALUE; });
            LOG(WARNING) << "Shape missing for input with dynamic shape: " << name
                         << " Automatically setting shape to: " << GetShapeStr(shapeData);
          }
        } else {
          shapeData = shape->second;
        }

        int32_t* shapeTensorData{nullptr};
        if (isShapeInferenceIO) {
          // Save the data in iEnv, in a way that it's address does not change
          // before enqueueV3 is called.
          iEnv.inputShapeTensorValues.emplace_back(shapeData);
          shapeTensorData = iEnv.inputShapeTensorValues.back().data();
        }

        for (auto& c : iEnv.contexts) {
          if (isShapeInferenceIO) {
            LOG(INFO) << "Set input shape tensor " << name << " to " << GetShapeStr(shapeData);
            if (!c->setTensorAddress(name, shapeTensorData)) {
              return false;
            }
          } else {
            LOG(INFO) << "Set shape of input tensor " << name << " to " << GetShapeStr(shapeData);
            if (!c->setInputShape(name, toDims(shapeData))) {
              return false;
            }
          }
        }
      } else if (nbOptProfiles && shape != inference.shapes.end()) {
        // Check if the provided shape matches the static dimensions in the engine.
        for (auto& c : iEnv.contexts) {
          if (!c->setInputShape(name, toDims(shape->second))) {
            LOG(ERROR) << "The engine was built with static shapes for input tensor " << name
                       << " but the provided shapes do not match the static shape!";
            return false;
          }
        }
      }
    }
  }

  const auto* context = iEnv.contexts.front().get();
  return FillBindingClosure(engine, context, iEnv.bindings, 1, endBindingIndex, inference.optProfileIndex)();
}