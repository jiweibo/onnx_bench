#pragma once

#include "NvInferRuntime.h"
#include "NvInferRuntimeBase.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntimeCommon.h>
#include <NvInferSafeRuntime.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <cstdint>
#include <functional>
#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <string>

#include "trt/trt_logger.h"
#include "utils/dynamic_library.h"
#include "utils/util.h"

using namespace nvinfer1;

using LibraryPtr = std::unique_ptr<DynamicLibrary>;

template <typename FetchPtrs>
inline bool initLibrary(LibraryPtr& libPtr, const std::string& libName, FetchPtrs fetchFunc);

inline bool initNvInfer();

inline bool initNvonnxparser();

nvinfer1::IRuntime* createRuntime();

nvinfer1::IBuilder* createBuilder();

nvonnxparser::IParser* createONNXParser(INetworkDefinition& network);
