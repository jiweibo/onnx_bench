#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "trt/trt_engine.h"
#include "trt/trt_inference.h"
#include "trt/trt_options.h"
#include "trt/trt_utils.h"

#include "core/core.h"

#include <glog/logging.h>

namespace {
inline nvinfer1::Dims ToNvDims(const core::Dims& dims) {
  nvinfer1::Dims res;
  res.nbDims = dims.num_dims;
  std::copy(dims.d, dims.d + dims.num_dims, res.d);
  return res;
}
} // namespace

class TensorRtNet {
public:
  TensorRtNet(const ModelOptions& model, const BuildOptions& build, const SystemOptions& sys,
              const InferenceOptions& inference)
      : model_(model), build_(build), sys_(sys), inference_(inference), stream_(makeCudaStream()) {
    bEnv_.reset(new BuildEnvironment(build.safe, build.versionCompatible, sys.DLACore, build.tempdir,
                                     build.tempfileControls, build.leadDLLPath));
    CHECK_EQ(getEngineBuildEnv(model, build, sys_, *bEnv_), true) << "Engine setup failed";
    bEnv_->engine.setDynamicPlugins(sys.dynamicPlugins);

    // Start inference phase.
    iEnv_.reset(new InferenceEnvironment(*bEnv_));

    // Delete build environment
    bEnv_.reset();

    CHECK_EQ(setUpInference(*iEnv_, inference, sys), true);
  }

  // TODO(wilber): thread-safe.
  void Run(std::unordered_map<std::string, core::Tensor>& inputs) {
    auto* ctx = iEnv_->contexts.front().get();
    auto* bindings = iEnv_->bindings.front().get();
    const auto& engine = ctx->getEngine();

    int32_t num = engine.getNbIOTensors();
    for (int i = 0; i < num; ++i) {
      const char* name = engine.getIOTensorName(i);
      bool is_input = engine.getTensorIOMode(name) == TensorIOMode::kINPUT;
      if (is_input) {
        auto& t = inputs.at(name);
        auto dims = t.GetDims();
        ctx->setInputShape(name, ToNvDims(dims));
      }
    }

    bindings->FillTensors(inputs, *stream_);
    bindings->setTensorAddresses(*ctx);
    CHECK_EQ(ctx->enqueueV3(*stream_), true);
    CUDA_CHECK(cudaStreamSynchronize(*stream_));
  }

private:
  ModelOptions model_;
  BuildOptions build_;
  SystemOptions sys_;
  InferenceOptions inference_;
  std::unique_ptr<BuildEnvironment> bEnv_;
  std::unique_ptr<InferenceEnvironment> iEnv_;
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream_;
};