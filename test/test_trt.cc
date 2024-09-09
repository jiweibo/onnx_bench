#include <cstdint>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "trt/trt_logger.h"
#include "utils/util.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_bool(save, false, "save");
DEFINE_bool(load, false, "load");
DEFINE_string(onnx, "", "onnx file");
DEFINE_string(engine, "", "engine file");

using namespace nvinfer1;

//======================================================================================================================
/// Optional : Print dimensions as string
std::string printDim(const nvinfer1::Dims& d) {
  using namespace std;
  ostringstream oss;
  for (int j = 0; j < d.nbDims; ++j) {
    oss << d.d[j];
    if (j < d.nbDims - 1)
      oss << "x";
  }
  return oss.str();
}

inline int64_t nvDimsSize(const Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1L, std::multiplies<int64_t>());
}

int32_t nvDataTypeSize(const DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kUINT8:
    return 1;
  case nvinfer1::DataType::kINT64:
    return 8;
  default:
    LOG(FATAL) << "Not Supported dtype " << static_cast<int>(dtype);
    return -1;
  }
}

//======================================================================================================================
/// Optional : Print layers of the network
void printNetwork(const nvinfer1::INetworkDefinition& net) {
  using namespace std;
  using namespace nvinfer1;
  cout << "\n=============\nNetwork info :" << endl;

  cout << "\nInputs : " << endl;
  for (int i = 0; i < net.getNbInputs(); ++i) {
    ITensor* inp = net.getInput(i);
    cout << "input" << i << " , dtype=" << (int)inp->getType() << " , dims=" << printDim(inp->getDimensions()) << endl;
  }

  cout << "\nLayers : " << endl;
  cout << "Number of layers : " << net.getNbLayers() << endl;
  for (int i = 0; i < net.getNbLayers(); ++i) {
    ILayer* l = net.getLayer(i);
    cout << "layer" << i << " , name=" << l->getName() << " , type=" << (int)l->getType() << " , IN ";
    for (int j = 0; j < l->getNbInputs(); ++j)
      cout << printDim(l->getInput(j)->getDimensions()) << " ";
    cout << ", OUT ";
    for (int j = 0; j < l->getNbOutputs(); ++j)
      cout << printDim(l->getOutput(j)->getDimensions()) << " ";
    cout << endl;
  }

  cout << "\nOutputs : " << endl;
  for (int i = 0; i < net.getNbOutputs(); ++i) {
    ITensor* outp = net.getOutput(i);
    cout << "input" << i << " , dtype=" << (int)outp->getType() << " , dims=" << printDim(outp->getDimensions())
         << endl;
  }

  cout << "=============\n" << endl;
}

//======================================================================================================================
/// Parse onnx file and create a TRT engine
nvinfer1::ICudaEngine* createCudaEngine(const std::string& onnxFileName, nvinfer1::ILogger& logger, int batchSize) {

  std::unique_ptr<IBuilder> builder{createInferBuilder(logger)};
  std::unique_ptr<INetworkDefinition> network{
      builder->createNetworkV2(1U << (unsigned)NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
  std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};

  if (!parser->parseFromFile(onnxFileName.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    throw std::runtime_error("ERROR: could not parse ONNX model " + onnxFileName + " !");

  // Optional : print network info
  // printNetwork(*network);

  Dims d;
  d.nbDims = 1;
  d.d[0] = batchSize;

  // Create Optimization profile and set the batch size
  IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("camera_input", OptProfileSelector::kMIN, Dims4{batchSize, 96, 96, 3});
  profile->setDimensions("lidar_input", OptProfileSelector::kMIN, Dims4{batchSize, 96, 96, 3});
  profile->setDimensions("camera_validation", OptProfileSelector::kMIN, d);
  profile->setDimensions("pred_cls", OptProfileSelector::kMIN, d);
  profile->setDimensions("pred_conf", OptProfileSelector::kMIN, d);
  profile->setDimensions("height", OptProfileSelector::kMIN, d);
  profile->setDimensions("length", OptProfileSelector::kMIN, d);
  profile->setDimensions("width", OptProfileSelector::kMIN, d);
  profile->setDimensions("max_intensity", OptProfileSelector::kMIN, d);
  profile->setDimensions("min_intensity", OptProfileSelector::kMIN, d);
  profile->setDimensions("mean_intensity", OptProfileSelector::kMIN, d);
  profile->setDimensions("std_intensity", OptProfileSelector::kMIN, d);
  profile->setDimensions("occ_ratio", OptProfileSelector::kMIN, d);
  profile->setDimensions("velocity", OptProfileSelector::kMIN, d);

  profile->setDimensions("camera_input", OptProfileSelector::kMAX, Dims4{batchSize, 96, 96, 3});
  profile->setDimensions("lidar_input", OptProfileSelector::kMAX, Dims4{batchSize, 96, 96, 3});
  profile->setDimensions("camera_validation", OptProfileSelector::kMAX, d);
  profile->setDimensions("pred_cls", OptProfileSelector::kMAX, d);
  profile->setDimensions("pred_conf", OptProfileSelector::kMAX, d);
  profile->setDimensions("height", OptProfileSelector::kMAX, d);
  profile->setDimensions("length", OptProfileSelector::kMAX, d);
  profile->setDimensions("width", OptProfileSelector::kMAX, d);
  profile->setDimensions("max_intensity", OptProfileSelector::kMAX, d);
  profile->setDimensions("min_intensity", OptProfileSelector::kMAX, d);
  profile->setDimensions("mean_intensity", OptProfileSelector::kMAX, d);
  profile->setDimensions("std_intensity", OptProfileSelector::kMAX, d);
  profile->setDimensions("occ_ratio", OptProfileSelector::kMAX, d);
  profile->setDimensions("velocity", OptProfileSelector::kMAX, d);

  profile->setDimensions("camera_input", OptProfileSelector::kOPT, Dims4{batchSize, 96, 96, 3});
  profile->setDimensions("lidar_input", OptProfileSelector::kOPT, Dims4{batchSize, 96, 96, 3});
  profile->setDimensions("camera_validation", OptProfileSelector::kOPT, d);
  profile->setDimensions("pred_cls", OptProfileSelector::kOPT, d);
  profile->setDimensions("pred_conf", OptProfileSelector::kOPT, d);
  profile->setDimensions("height", OptProfileSelector::kOPT, d);
  profile->setDimensions("length", OptProfileSelector::kOPT, d);
  profile->setDimensions("width", OptProfileSelector::kOPT, d);
  profile->setDimensions("max_intensity", OptProfileSelector::kOPT, d);
  profile->setDimensions("min_intensity", OptProfileSelector::kOPT, d);
  profile->setDimensions("mean_intensity", OptProfileSelector::kOPT, d);
  profile->setDimensions("std_intensity", OptProfileSelector::kOPT, d);
  profile->setDimensions("occ_ratio", OptProfileSelector::kOPT, d);
  profile->setDimensions("velocity", OptProfileSelector::kOPT, d);

  // Build engine
  std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());

  config->addOptimizationProfile(profile);
  config->setFlag(BuilderFlag::kFP16);
  std::unique_ptr<IHostMemory> serializedEngine{builder->buildSerializedNetwork(*network, *config)};
  CHECK_NOTNULL(serializedEngine.get());

  static IRuntime* runtime = createInferRuntime(logger);
  static ICudaEngine* engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());

  std::unique_ptr<IHostMemory> serializeEngine(engine->serialize());
  LOG(INFO) << "nSerialized engine : size = " << serializeEngine->size();
  std::ofstream out("lidar_96.engine", std::ios::binary);
  out.write(reinterpret_cast<char*>(serializeEngine->data()), serializeEngine->size());
  return engine;
}

int main(int argc, char** argv) {
  int batchSize = 96;
  ICudaEngine* engine(createCudaEngine(FLAGS_onnx, *Logger::getLogger()->getTrtLogger(), batchSize));

  // std::vector<char> buffer;
  // {
  //   std::ifstream in("lidar_96.engine", std::ios::binary | std::ios::ate);
  //   if (!in)
  //     throw std::runtime_error("Cannot open trt engine");
  //   std::streamsize ss = in.tellg();
  //   in.seekg(0, std::ios::beg);
  //   LOG(INFO) << "engine size " << ss;
  //   buffer.resize(ss);
  //   if (ss == 0 || !in.read(buffer.data(), ss))
  //     throw std::runtime_error("Cannot read trt engine");
  // }
  // std::unique_ptr<IRuntime> runtime(createInferRuntime(*Logger::getLogger()->getTrtLogger()));
  // std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
  // CHECK_NOTNULL(engine.get());

  LOG(INFO) << "engine done";

  std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Optional : Print all bindings : name + dims + dtype
  std::cout << "=============\nBindings :\n";
  const int32_t endBindingIndex = engine->getNbIOTensors();
  for (int32_t i = 0; i < endBindingIndex; ++i) {
    const char* name = engine->getIOTensorName(i);
    const auto& mode = engine->getTensorIOMode(name);
    if (mode == TensorIOMode::kINPUT) {
      Dims dims = context->getTensorShape(name);
      dims.d[0] = batchSize;
      context->setInputShape(name, dims);
      const nvinfer1::DataType dtype = engine->getTensorDataType(name);
      std::cout << "IN " << name << " " << printDim(dims) << std::endl;
      int64_t num = nvDimsSize(dims);
      CHECK_GE(num, 0);
      void* ptr{nullptr};
      CUDA_CHECK(cudaMalloc(&ptr, num * nvDataTypeSize(dtype)));
      context->setTensorAddress(name, ptr);
    } else if (mode == TensorIOMode::kOUTPUT) {
      const Dims dims = context->getTensorShape(name);
      const nvinfer1::DataType dtype = engine->getTensorDataType(name);
      std::cout << "OUT " << name << " " << printDim(dims) << std::endl;
      int64_t num = nvDimsSize(dims);
      CHECK_GE(num, 0);
      void* ptr{nullptr};
      CUDA_CHECK(cudaMalloc(&ptr, num * nvDataTypeSize(dtype)));
      context->setTensorAddress(name, ptr);
    }
  }
  std::cout << "=============\n\n";

  context->enqueueV3(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaGraph_t graph{};
  cudaGraphExec_t graphExec{};
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  context->enqueueV3(stream);
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  for (size_t i = 0; i < 100; ++i) {
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  return 0;
}