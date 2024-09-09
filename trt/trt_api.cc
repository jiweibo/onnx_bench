#include "trt/trt_api.h"

std::function<void*(void*, int32_t)> pCreateInferRuntimeInternal{};
std::function<void*(void*, int32_t)> pCreateInferBuilderInternal{};
std::function<void*(void*, void*, int32_t)> pCreateNvOnnxParserInternal{};

const std::string kNVINFER_LIBNAME = std::string{"libnvinfer.so."} + std::to_string(NV_TENSORRT_MAJOR);
const std::string kNVINFER_PLUGIN_LIBNAME = std::string{"libnvinfer_plugin.so."} + std::to_string(NV_TENSORRT_MAJOR);
const std::string kNVONNXPARSER_LIBNAME = std::string{"libnvonnxparser.so."} + std::to_string(NV_TENSORRT_MAJOR);
const std::string kNVINFER_LEAN_LIBNAME = std::string{"libnvinfer_lean.so."} + std::to_string(NV_TENSORRT_MAJOR);
const std::string kNVINFER_DISPATCH_LIBNAME =
    std::string{"libnvinfer_dispatch.so."} + std::to_string(NV_TENSORRT_MAJOR);

inline const std::string& getRuntimeLibraryName(const RuntimeMode mode) {
  switch (mode) {
  case RuntimeMode::kFULL:
    return kNVINFER_LIBNAME;
  case RuntimeMode::kDISPATCH:
    return kNVINFER_DISPATCH_LIBNAME;
  case RuntimeMode::kLEAN:
    return kNVINFER_LEAN_LIBNAME;
  }
  throw std::runtime_error("Unknown runtime mode");
}

template <typename FetchPtrs>
bool initLibrary(LibraryPtr& libPtr, const std::string& libName, FetchPtrs fetchFunc) {
  if (libPtr != nullptr) {
    return true;
  }
  try {
    libPtr.reset(new DynamicLibrary{libName});
    fetchFunc(libPtr.get());
  } catch (...) {
    libPtr.reset();
    LOG(ERROR) << "Could not load library " << libName;
    return false;
  }
  return true;
}

bool initNvInfer() {
  static LibraryPtr libnvinferPtr{};
  auto fetchPtrs = [](DynamicLibrary* l) {
    pCreateInferRuntimeInternal = l->symbolAddress<void*(void*, int32_t)>("createInferRuntime_INTERNAL");
    pCreateInferBuilderInternal = l->symbolAddress<void*(void*, int32_t)>("createInferBuilder_INTERNAL");
  };
  return initLibrary(libnvinferPtr, getRuntimeLibraryName(RuntimeMode::kFULL), fetchPtrs);
}

bool initNvonnxparser() {
  static LibraryPtr libnvonnxparserPtr{};
  auto fetchPtrs = [](DynamicLibrary* l) {
    pCreateNvOnnxParserInternal = l->symbolAddress<void*(void*, void*, int)>("createNvOnnxParser_INTERNAL");
  };
  return initLibrary(libnvonnxparserPtr, kNVONNXPARSER_LIBNAME, fetchPtrs);
}

nvinfer1::IRuntime* createRuntime() {
  if (!initNvInfer())
    return {};
  ASSERT(pCreateInferRuntimeInternal != nullptr);
  return static_cast<IRuntime*>(pCreateInferRuntimeInternal(Logger::getLogger(), NV_TENSORRT_VERSION));
}

nvinfer1::IBuilder* createBuilder() {
  if (!initNvInfer())
    return {};
  ASSERT(pCreateInferBuilderInternal != nullptr);
  return static_cast<IBuilder*>(pCreateInferBuilderInternal(Logger::getLogger(), NV_TENSORRT_VERSION));
}

nvonnxparser::IParser* createONNXParser(INetworkDefinition& network) {
  if (!initNvonnxparser())
    return {};
  ASSERT(pCreateNvOnnxParserInternal != nullptr);
  return static_cast<nvonnxparser::IParser*>(
      pCreateNvOnnxParserInternal(&network, Logger::getLogger(), NV_ONNX_PARSER_VERSION));
}
