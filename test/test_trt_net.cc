#include "core/core.h"
#include "trt/trt_net.h"
#include "trt/trt_options.h"

#include <glog/logging.h>

#include <unordered_map>

inline void FillTensor(core::Tensor& t) {
  int64_t volume = t.Numel();
  CHECK_GT(volume, 0);
  switch (t.GetDataType()) {
  case core::DataType::kBOOL: {
    fillBuffer<bool>(t.HostData(), volume, 0, 1);
    break;
  }
  case core::DataType::kINT32: {
    fillBuffer<int32_t>(t.HostData(), volume, 0, 10);
    break;
  }
  case core::DataType::kINT64: {
    fillBuffer<int64_t>(t.HostData(), volume, 0, 10);
    break;
  }
  case core::DataType::kINT8: {
    fillBuffer<int8_t>(t.HostData(), volume, 0, 10);
    break;
  }
  case core::DataType::kUINT8: {
    fillBuffer<uint8_t>(t.HostData(), volume, 0, 10);
    break;
  }
  case core::DataType::kFLOAT: {
    fillBuffer<float>(t.HostData(), volume, -1.0f, 1.0f);
    break;
  }
  default:
    LOG(FATAL) << "Not supported dtype: " << static_cast<int32_t>(t.GetDataType());
  }
}

int main() {
  ModelOptions model;
  model.model = "lidar_second_stage.onnx";
  model.format = ModelFormat::kONNX;

  BuildOptions build;
  build.fp16 = true;
  BuildOptions::ShapeProfile profile;
  ShapeRange a{std::vector<int>{1, 96, 96, 3}, std::vector<int>{96, 96, 96, 3}, std::vector<int>{96, 96, 96, 3}};
  ShapeRange b{std::vector<int>{1}, std::vector<int>{96}, std::vector<int>{96}};
  profile["camera_input"] = a;
  profile["lidar_input"] = a;
  profile["camera_validation"] = b;
  profile["pred_cls"] = b;
  profile["pred_conf"] = b;
  profile["height"] = b;
  profile["length"] = b;
  profile["width"] = b;
  profile["max_intensity"] = b;
  profile["min_intensity"] = b;
  profile["mean_intensity"] = b;
  profile["std_intensity"] = b;
  build.optProfiles.emplace_back(profile);

  build.save = true;
  build.load = true;
  build.engine = "lidar_test_fp16.engine";

  SystemOptions sys;
  InferenceOptions inference;
  TensorRtNet net(model, build, sys, inference);
  std::unordered_map<std::string, core::Tensor> inputs;
  core::Tensor camera_input(core::Dims{1, 96, 96, 3}, core::DataType::kUINT8, core::Location::kHOST, 0);
  core::Tensor lidar_input(core::Dims{1, 96, 96, 3}, core::DataType::kUINT8, core::Location::kHOST, 0);
  core::Tensor camera_validation(core::Dims{1}, core::DataType::kFLOAT, core::Location::kHOST, 0);
  core::Tensor pred_cls(core::Dims{1}, core::DataType::kINT64, core::Location::kHOST, 0);
  core::Tensor pred_conf(core::Dims{1}, core::DataType::kFLOAT, core::Location::kHOST, 0);
  core::Tensor height(core::Dims{1}, core::DataType::kFLOAT, core::Location::kHOST, 0);
  core::Tensor length(core::Dims{1}, core::DataType::kFLOAT, core::Location::kHOST, 0);
  core::Tensor width(core::Dims{1}, core::DataType::kFLOAT, core::Location::kHOST, 0);
  core::Tensor max_intensity(core::Dims{1}, core::DataType::kFLOAT, core::Location::kHOST, 0);
  core::Tensor min_intensity(core::Dims{1}, core::DataType::kFLOAT, core::Location::kHOST, 0);
  core::Tensor mean_intensity(core::Dims{1}, core::DataType::kFLOAT, core::Location::kHOST, 0);
  core::Tensor std_intensity(core::Dims{1}, core::DataType::kFLOAT, core::Location::kHOST, 0);
  FillTensor(camera_input);
  FillTensor(lidar_input);
  FillTensor(camera_validation);
  FillTensor(pred_cls);
  FillTensor(pred_conf);
  FillTensor(height);
  FillTensor(length);
  FillTensor(width);
  FillTensor(max_intensity);
  FillTensor(min_intensity);
  FillTensor(mean_intensity);
  FillTensor(std_intensity);
  LOG(INFO) << "FillTensor done";

  inputs.emplace("camera_input", camera_input);
  inputs.emplace("lidar_input", lidar_input);
  inputs.emplace("camera_validation", camera_validation);
  inputs.emplace("pred_cls", pred_cls);
  inputs.emplace("pred_conf", pred_conf);
  inputs.emplace("height", height);
  inputs.emplace("length", length);
  inputs.emplace("width", width);
  inputs.emplace("max_intensity", max_intensity);
  inputs.emplace("min_intensity", min_intensity);
  inputs.emplace("mean_intensity", mean_intensity);
  inputs.emplace("std_intensity", std_intensity);

  net.Run(inputs);
  LOG(INFO) << "Net run done";
  return 0;
}