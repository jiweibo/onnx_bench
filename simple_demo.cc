#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <ratio>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "utils/memuse.h"
#include "utils/nvtx.h"
#include "utils/random_value.h"
#include "utils/timer.h"
#include "utils/util.h"

#include "ort.h"

#include "cnpy.h"

DEFINE_string(onnx, "", "onnx model file");
DEFINE_string(log_level, "warning", "verbose, info, warning, error, fatal");
DEFINE_int32(batch, 1, "batch");
DEFINE_int32(warmup, 0, "warmup");
DEFINE_int32(repeats, 1, "repeats");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(precision, "fp32", "fp32, fp16");
DEFINE_bool(precisionInt8, false, "enable int8");
DEFINE_string(provider, "cpu", "cpu, openvino, cuda, trt");
DEFINE_string(cacheDir, "", "the cache dir");
DEFINE_string(calibrationName, "", "int8 calibration table");
DEFINE_int32(minSubgraphSize, 1, "trt min subgraph size");
DEFINE_uint64(maxWorkspaceSize, 1UL << 31, "trt max workspace size");
DEFINE_string(dumpOutput, "",
              "Save the output tensor(s) of the last inference iteration in a npz file"
              "(default = disabled).");
DEFINE_string(trtFilterOps, "", "defaule empty, e.g. 'Flatten_125 Flatten_126'");
DEFINE_string(trtPreferPrecisionOps, "", "prefer fp32 ops");
DEFINE_string(trtPreferPrecisionNodes, "", "prefer fp32 nodes");
DEFINE_string(trtForcePrecisionOps, "", "force ops");
DEFINE_string(trtForcePrecisionNodes, "", "force nodes");

DEFINE_string(loadInputs, "",
              "Load input values from files (default = generate random inputs). Input "
              "names can be wrapped with single quotes (ex: "
              "'Input0:a.npy,Input1:b.npy')");
DEFINE_string(inputType, "npz", "npz, npy, txt, bin, json etc.");

DEFINE_string(dataDir, "", "a dir which stores lots of json file");

std::default_random_engine e(1998);

namespace {

void* GenerateData(const std::vector<int64_t>& dims, ONNXTensorElementDataType type) {
  int64_t num = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    float* ptr = static_cast<float*>(malloc(num * sizeof(float)));
    std::uniform_real_distribution<float> u(-1, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    half* ptr = static_cast<half*>(malloc(num * sizeof(half)));
    std::uniform_real_distribution<float> u(-1, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = __half2float(u(e));
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    int* ptr = static_cast<int*>(malloc(num * sizeof(int)));
    std::uniform_int_distribution<int> u(-128, 127);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    auto* ptr = static_cast<int64_t*>(malloc(num * sizeof(int64_t)));
    std::uniform_int_distribution<int> u(-128, 127);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    bool* ptr = static_cast<bool*>(malloc(num * sizeof(bool)));
    std::uniform_int_distribution<int> u(0, 1);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    auto* ptr = static_cast<uint8_t*>(malloc(num * sizeof(uint8_t)));
    std::uniform_int_distribution<uint8_t> u(0, 255);
    for (size_t i = 0; i < num; ++i) {
      ptr[i] = u(e);
    }
    return ptr;
  } else {
    LOG(FATAL) << "Not supported data type " << type;
  }

  return nullptr;
}

const void* GetOrtDataPtr(const Ort::Value& tensor) {
  auto type = tensor.GetTensorTypeAndShapeInfo().GetElementType();
  const void* data{nullptr};
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    data = tensor.GetTensorData<float>();
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    data = tensor.GetTensorData<__half>();
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    data = tensor.GetTensorData<int>();
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    data = tensor.GetTensorData<int64_t>();
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    data = tensor.GetTensorData<bool>();
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    data = tensor.GetTensorData<uint8_t>();
  } else {
    LOG(FATAL) << "Not supported data type " << type;
  }
  return data;
}

void RandomFillTensor(core::TensorRef& tensor) {
  auto num = tensor->Numel();
  auto dtype = tensor->GetDataType();
  switch (dtype) {
  case core::DataType::kBOOL:
    FillBuffer<bool>(tensor->HostData(), num, 0, 1);
    break;
  case core::DataType::kUINT8:
    FillBuffer<uint8_t>(tensor->HostData(), num, 0, 255);
    break;
  case core::DataType::kINT8:
  case core::DataType::kINT32:
  case core::DataType::kINT64:
    FillBuffer<int32_t>(tensor->HostData(), num, -128, 127);
    break;
  case core::DataType::kFLOAT:
    FillBuffer<float>(tensor->HostData(), num, -1.f, 1.f);
    break;
  case core::DataType::kHALF:
  case core::DataType::kBF16:
  case core::DataType::kFP8:
  case core::DataType::kINT4:
  default:
    LOG(FATAL) << "Not supported dtype " << static_cast<int>(dtype);
  }
}

float MeanValue(const Ort::Value& tensor) {
  auto type_info = tensor.GetTensorTypeAndShapeInfo();
  auto type = type_info.GetElementType();
  auto dims = type_info.GetShape();
  size_t num = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    auto* data = tensor.GetTensorData<float>();
    return mean(data, num);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    auto* data = tensor.GetTensorData<int>();
    return mean(data, num);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    auto* data = tensor.GetTensorData<int64_t>();
    return mean(data, num);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    auto* data = tensor.GetTensorData<bool>();
    return mean(data, num);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    auto* data = tensor.GetTensorData<uint8_t>();
    return mean(data, num);
  } else {
    LOG(FATAL) << "Not supported data type " << type;
  }
}

void DumpTensors(const std::vector<Ort::Value>& tensors, const std::vector<std::string>& names,
                 const std::string& filename) {
  CHECK_EQ(tensors.size(), names.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& tensor = tensors[i];
    if (!tensor.IsTensor()) {
      LOG(WARNING) << "Output " << i << " is not Tensor, skip dump.";
      continue;
    }
    auto& name = names[i];
    auto type_info = tensor.GetTensorTypeAndShapeInfo();
    auto type = type_info.GetElementType();
    auto dims = type_info.GetShape();
    size_t num = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    if (num == 0) {
      continue;
    }
    std::vector<size_t> shape(dims.begin(), dims.end());

    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      auto* data = tensor.GetTensorData<float>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      auto* data = tensor.GetTensorData<__half>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      auto* data = tensor.GetTensorData<int>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      auto* data = tensor.GetTensorData<int64_t>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      auto* data = tensor.GetTensorData<bool>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      auto* data = tensor.GetTensorData<uint8_t>();
      cnpy::npz_save(filename, name, data, shape, "a");
    } else {
      LOG(FATAL) << "Not supported data type " << type;
    }
  }
}

SessConfig ParseFlags() {
  SessConfig config;
  config.device_id = FLAGS_device_id;
  config.opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
  config.onnx_file = FLAGS_onnx;
  config.intra_op_num_threads = 1;
  if (FLAGS_log_level == "verbose") {
    config.log_level = ORT_LOGGING_LEVEL_VERBOSE;
  } else if (FLAGS_log_level == "info") {
    config.log_level = ORT_LOGGING_LEVEL_INFO;
  } else if (FLAGS_log_level == "warning") {
    config.log_level = ORT_LOGGING_LEVEL_WARNING;
  } else if (FLAGS_log_level == "error") {
    config.log_level = ORT_LOGGING_LEVEL_ERROR;
  } else if (FLAGS_log_level == "fatal") {
    config.log_level = ORT_LOGGING_LEVEL_FATAL;
  } else {
    LOG(FATAL) << "Not supported log level(verbose, info, warning, error, fatal) " << FLAGS_log_level;
  }
  if (FLAGS_provider == "trt") {
    config.use_trt = true;
    config.use_cuda = true;
    config.trt_config = SessConfig::TrtConfig();
    config.trt_config.cache_dir = FLAGS_cacheDir;
    config.trt_config.calibration_table_name = FLAGS_calibrationName;
    config.trt_config.enable_int8 = FLAGS_precisionInt8;
    config.trt_config.precision = FLAGS_precision;
    config.trt_config.min_subgraph_size = FLAGS_minSubgraphSize;
    config.trt_config.filter_ops = FLAGS_trtFilterOps;
    config.trt_config.max_workspace_size = FLAGS_maxWorkspaceSize;
  } else if (FLAGS_provider == "cuda") {
    config.use_cuda = true;
  } else if (FLAGS_provider == "openvino") {
    config.use_openvino = true;
  }

  return config;
}

} // namespace

// void Run(Sess& session) {
//   NvtxRange nvtx_run_d2h("run_d2h");
//   NvtxRange nvtx_h2d("h2d");

//   // // Print number of model input nodes
//   // const size_t num_input_nodes = session.GetInputCount();
//   // const size_t num_output_nodes = session.GetOutputCount();
//   // std::vector<std::string> input_names(num_input_nodes);
//   // std::vector<std::string> output_names(num_output_nodes);
//   // std::vector<std::vector<int64_t>> input_node_dims(num_input_nodes);
//   // std::vector<std::vector<int64_t>> output_node_dims(num_output_nodes);
//   // std::vector<ONNXTensorElementDataType> input_types(num_input_nodes);
//   // std::vector<ONNXTensorElementDataType> output_types(num_output_nodes);
//   // std::vector<const char*> input_names_char(num_input_nodes);
//   // std::vector<const char*> output_names_char(num_output_nodes);

//   // for (size_t i = 0; i < num_input_nodes; ++i) {
//   //   input_names[i] = session.GetInputNameAllocated(i, allocator).get();
//   //   input_names_char[i] = input_names[i].c_str();
//   //   auto type_info = session.GetInputTypeInfo(i);
//   //   auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
//   //   input_node_dims[i] = tensor_info.GetShape();
//   //   input_types[i] = tensor_info.GetElementType();

//   //   if (input_node_dims[i][0] == -1) {
//   //     input_node_dims[i][0] = FLAGS_batch;
//   //   }

//   //   LOG(INFO) << "Input " << i << " : name = " << input_names[i]
//   //             << ", type = " << input_types[i]
//   //             << ", num_dims = " << input_node_dims[i].size()
//   //             << ", dims = " << PrintShape(input_node_dims[i]);
//   // }
//   // for (size_t i = 0; i < num_output_nodes; ++i) {
//   //   output_names[i] = session.GetOutputNameAllocated(i, allocator).get();
//   //   output_names_char[i] = output_names[i].c_str();

//   //   auto type_info = session.GetOutputTypeInfo(i);
//   //   auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
//   //   output_node_dims[i] =
//   type_info.GetTensorTypeAndShapeInfo().GetShape();
//   //   output_types[i] = tensor_info.GetElementType();

//   //   if (output_node_dims[i][0] == -1) {
//   //     output_node_dims[i][0] = FLAGS_batch;
//   //   }

//   //   LOG(INFO) << "Output " << i << " : name = " << output_names[i]
//   //             << ", type = " << output_types[i]
//   //             << ", num_dims = " << output_node_dims[i].size()
//   //             << ", dims = " << PrintShape(output_node_dims[i]);
//   // }

//   // Ort::IoBinding bind(session);
//   // std::vector<void*> ptr_to_free;
//   // StopWatchTimer timer_run;
//   // StopWatchTimer timer_h2d;

//   // std::vector<Ort::Value> input_tensors;
//   // input_tensors.reserve(num_input_nodes);
//   // for (int i = 0; i < num_input_nodes; ++i) {
//   //   input_tensors.push_back(Ort::Value{nullptr});
//   // }

//   cnpy::npz_t npz_data;
//   if (FLAGS_loadInputs != "") {
//     npz_data = LoadNpzFile(FLAGS_loadInputs);
//   }

//   // auto RunPerBatchNoBind = [&]() {
//   //   for (size_t i = 0; i < num_input_nodes; ++i) {
//   //     auto name = input_names[i];
//   //     auto type = input_types[i];
//   //     auto* data = GenerateData(input_node_dims[i], type);
//   //     ptr_to_free.push_back(data);
//   //     auto& shape = input_node_dims[i];

//   //     size_t num = std::accumulate(shape.begin(), shape.end(),
//   SizeOf(type),
//   //                                  std::multiplies<int>());

//   //     Ort::Value tensor{nullptr};
//   //     timer_h2d.Start();
//   //     tensor = Ort::Value::CreateTensor(mem_info, data, num, shape.data(),
//   //                                       shape.size(), type);
//   //     timer_h2d.Stop();
//   //     input_tensors[i] = std::move(tensor);
//   //   }
//   //   timer_run.Start();
//   //   auto output_tensors = session.Run(
//   //       Ort::RunOptions{nullptr}, input_names_char.data(),
//   //       input_tensors.data(), num_input_nodes, output_names_char.data(),
//   //       num_output_nodes);
//   //   cudaDeviceSynchronize();
//   //   timer_run.Stop();

//   //   for (auto v : ptr_to_free) {
//   //     free(v);
//   //   }
//   //   ptr_to_free.clear();

//   //   return output_tensors;
//   // };

//   auto RunPerBatch = [&]() {
//     // bind.ClearBoundInputs();
//     // bind.ClearBoundOutputs();
//     std::map<std::string, Tensor> inputs;
//     if (FLAGS_loadInputs != "") {
//       data = &(*npz_data.at(input_names[i]).data_holder)[0];
//     } else {
//       data = GenerateData(input_node_dims[i], type);
//       ptr_to_free.push_back(data);
//     }

//     // for (size_t i = 0; i < num_input_nodes; ++i) {
//     //   auto type = input_types[i];
//     //   void* data{nullptr};
//     //   if (FLAGS_loadInputs != "") {
//     //     data = &(*npz_data.at(input_names[i]).data_holder)[0];
//     //   } else {
//     //     data = GenerateData(input_node_dims[i], type);
//     //     ptr_to_free.push_back(data);
//     //   }
//     //   auto tensor = InitTensorFromData(data, input_node_dims[i], type);
//     //   nvtx_h2d.Begin();
//     //   timer_h2d.Start();
//     //   bind.BindInput(input_names[i].c_str(), tensor);
//     //   bind.SynchronizeInputs();
//     //   timer_h2d.Stop();
//     //   nvtx_h2d.End();
//     // }

//     // for (size_t i = 0; i < num_output_nodes; ++i) {
//     //   bind.BindOutput(output_names[i].c_str(), mem_info);
//     // }

//     nvtx_run.Begin();
//     timer_run.Start();
//     session.Run(Ort::RunOptions{}, bind);
//     timer_run.Stop();
//     nvtx_run.End();

//     bind.SynchronizeOutputs();

//     for (auto v : ptr_to_free) {
//       free(v);
//     }
//     ptr_to_free.clear();

//     return bind.GetOutputValues();
//   };

//   for (size_t i = 0; i < FLAGS_warmup; ++i) {
//     RunPerBatch();
//   }
//   timer_run.Reset();
//   timer_h2d.Reset();

//   for (size_t i = 0; i < FLAGS_repeats; ++i) {
//     auto output_tensors = RunPerBatch();

//     if (i == FLAGS_repeats - 1) {
//       for (size_t j = 0; j < output_tensors.size(); ++j) {
//         auto& out = output_tensors[j];
//         if (out.IsTensor()) {
//           LOG(INFO) << "Mean value " << j << " : "
//                     << MeanValue(output_tensors[j]) << ", Shape: "
//                     <<
//                     PrintShape(out.GetTensorTypeAndShapeInfo().GetShape());
//         }
//       }
//       if (FLAGS_dumpOutput != "") {
//         DumpTensors(output_tensors, output_names, FLAGS_dumpOutput);
//       }
//     }
//   }
//   LOG(INFO) << "------------------------------";
//   LOG(INFO) << "H2D Average time " << timer_h2d.GetAverageTime()
//             << ", variance: " << timer_h2d.ComputeVariance()
//             << ", tp99: " << timer_h2d.ComputePercentile(0.99);
//   LOG(INFO) << "Run+D2H Average time " << timer_run.GetAverageTime()
//             << ", variance: " << timer_run.ComputeVariance()
//             << ", tp99: " << timer_run.ComputePercentile(0.99);
//   LOG(INFO) << "H2D+RUN+D2H time "
//             << timer_h2d.GetAverageTime() + timer_run.GetAverageTime();
// }

void PrintSessInfo(Sess& session) {
  const auto& input_names = session.InputNames();
  auto input_dims = session.InputDims();
  const auto& input_types = session.InputDtypes();
  int num_input_nodes = input_names.size();

  const auto& output_names = session.OutputNames();
  auto output_dims = session.OutputDims();
  const auto& output_types = session.OutputDtypes();
  int num_output_nodes = output_names.size();

  for (size_t i = 0; i < num_input_nodes; ++i) {
    LOG(INFO) << "Input " << i << " : name = " << input_names[i] << ", type = " << input_types[i]
              << ", num_dims = " << input_dims[i].size() << ", dims = " << GetShapeStr(input_dims[i]);
  }
  for (size_t i = 0; i < num_output_nodes; ++i) {
    LOG(INFO) << "Output " << i << " : name = " << output_names[i] << ", type = " << output_types[i]
              << ", num_dims = " << output_dims[i].size() << ", dims = " << GetShapeStr(output_dims[i]);
  }
}

void Run(Sess& session) {
  cnpy::npz_t npz_data;
  if (FLAGS_loadInputs != "") {
    npz_data = LoadNpzFile(FLAGS_loadInputs);
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_onnx == "") {
    LOG(FATAL) << "Please set --onnx flag.";
  }

  SessConfig config = ParseFlags();
  Sess session(config);
  PrintSessInfo(session);

  StopWatchTimer timer_h2d, timer_run_d2h;
  NvtxRange nvtx_h2d("h2d"), nvtx_run_d2h("run_d2h");
  session.RegisterBeforeH2DHook([&]() { timer_h2d.Start(); });
  session.RegisterAfterH2DHook([&]() { timer_h2d.Stop(); });
  session.RegisterBeforeRunD2HHook([&]() { timer_run_d2h.Start(); });
  session.RegisterAfterRunD2HHook([&]() { timer_run_d2h.Stop(); });
  session.RegisterBeforeH2DHook([&]() { nvtx_h2d.Begin(); });
  session.RegisterAfterH2DHook([&]() { nvtx_h2d.End(); });
  session.RegisterBeforeRunD2HHook([&]() { nvtx_run_d2h.Begin(); });
  session.RegisterAfterRunD2HHook([&]() { nvtx_run_d2h.End(); });

  auto input_dims = session.InputDims();
  const auto& input_types = session.InputDtypes();
  std::map<std::string, core::TensorRef> inputs;
  std::vector<void*> to_free(input_dims.size());
  for (size_t i = 0; i < input_dims.size(); ++i) {
    auto name = session.InputNames()[i];
    if (input_dims[i][0] == -1) {
      input_dims[i][0] = FLAGS_batch;
    }
    auto ort_tensor = std::make_shared<core::Tensor>(core::Dims(input_dims[i]), ToDataType(session.InputDtypes()[i]),
                                                     core::Location::kHOST);
    RandomFillTensor(ort_tensor);
    inputs.emplace(name, std::move(ort_tensor));
  }
  MemoryUse memuse(FLAGS_device_id);

  for (size_t i = 0; i < 100; ++i)
    auto outs = session.RunWithBind(inputs);

  for (auto* p : to_free)
    free(p);

  LOG(INFO) << "------------------------------";
  LOG(INFO) << "H2D Average time " << timer_h2d.GetAverageTime() << ", variance: " << timer_h2d.ComputeVariance()
            << ", tp99: " << timer_h2d.ComputePercentile(0.99);
  LOG(INFO) << "Run+D2H Average time " << timer_run_d2h.GetAverageTime()
            << ", variance: " << timer_run_d2h.ComputeVariance() << ", tp99: " << timer_run_d2h.ComputePercentile(0.99);
  LOG(INFO) << "H2D+RUN+D2H time " << timer_h2d.GetAverageTime() + timer_run_d2h.GetAverageTime();

}
