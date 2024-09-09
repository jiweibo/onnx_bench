
#include "trt/trt_engine.h"
#include "trt/trt_api.h"
#include "trt/trt_logger.h"
#include "trt/trt_options.h"
#include "trt/trt_utils.h"
#include "utils/util.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <glog/logging.h>

#include <algorithm>
#include <any>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>

using namespace nvinfer1;

nvinfer1::ICudaEngine* LazilyDeserializedEngine::get() {
  if (mEngine == nullptr) {
    CHECK_EQ(getFileReader().isOpen() || !getBlob().empty(), true) << "Engine is empty. Nothing to deserialize!";

    using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using duration = std::chrono::duration<float>;
    const time_point deserializeStartTime{std::chrono::high_resolution_clock::now()};

    if (mLeanDLLPath.empty()) {
      mRuntime.reset(createRuntime());
    } else {
      LOG(FATAL) << "Not Implemented";
    }

    CHECK_NOTNULL(mRuntime.get());

    if (mVersionCompatible) {
      // Application needs to opt into allowing deserialization of engines with
      // embedded lean runtime.
      mRuntime->setEngineHostCodeAllowed(true);
    }

    if (!mTempdir.empty()) {
      mRuntime->setTemporaryDirectory(mTempdir.c_str());
    }
    mRuntime->setTempfileControlFlags(mTempfileControls);
    CHECK(mRuntime.get() != nullptr) << "runtime creation failed";

    if (mDLACore != -1) {
      LOG(FATAL) << "DLA Not Supported";
    }

    for (const auto& pluginPath : mDynamicPlugins) {
      mRuntime->getPluginRegistry().loadLibrary(pluginPath.c_str());
    }

    if (getFileReader().isOpen()) {
      mEngine.reset(mRuntime->deserializeCudaEngine(getFileReader()));
    } else {
      const auto& engineBlob = getBlob();
      mEngine.reset(mRuntime->deserializeCudaEngine(engineBlob.data, engineBlob.size));
    }
    CHECK(mEngine != nullptr) << "Engine deserialization failed";

    const time_point deserializeEndTime{std::chrono::high_resolution_clock::now()};
    LOG(INFO) << "Engine deserialized in " << duration(deserializeEndTime - deserializeStartTime).count() << " sec.";
  }

  return mEngine.get();
}

nvinfer1::ICudaEngine* LazilyDeserializedEngine::release() { return mEngine.release(); }

//!
//! \brief Generate a network definition for a given model
//!
//! \param[in] model Model options for this network
//! \param[in,out] network Network storing the parsed results
//! \param[in,out] err Error stream
//! \param[out] vcPluginLibrariesUsed If not nullptr, will be populated with paths to VC plugin libraries required by
//! the parsed network.
//!
//! \return Parser The parser used to initialize the network and that holds the weights for the network, or an invalid
//! parser (the returned parser converts to false if tested)
//!
//! Constant input dimensions in the model must not be changed in the corresponding
//! network definition, because its correctness may rely on the constants.
//!
//! \see Parser::operator bool()
//!
Parser modelToNetwork(const ModelOptions& model, const BuildOptions& build, nvinfer1::INetworkDefinition& network,
                      std::vector<std::string>* vcPluginLibrariesUsed) {
  LOG(INFO) << "Start parsing network model.";
  const auto tBgine = std::chrono::high_resolution_clock::now();

  Parser parser;
  switch (model.format) {
  case ModelFormat::kONNX: {
    using namespace nvonnxparser;
    parser.onnxParser.reset(createONNXParser(network));
    CHECK_NOTNULL(parser.onnxParser.get());

    if (!parser.onnxParser->parseFromFile(model.model.c_str(), static_cast<int>(Logger::getLogger()->getSeverity()))) {
      LOG(ERROR) << "Failed to parse onnx file";
      parser.onnxParser.reset();
    }

    if (vcPluginLibrariesUsed && parser.onnxParser.get()) {
      int64_t nbPluginLibs;
      char const* const* pluginLibArray = parser.onnxParser->getUsedVCPluginLibraries(nbPluginLibs);
      if (nbPluginLibs >= 0) {
        vcPluginLibrariesUsed->reserve(nbPluginLibs);
        for (int64_t i = 0; i < nbPluginLibs; ++i) {
          LOG(INFO) << "Using VC plugin library " << pluginLibArray[i];
          vcPluginLibrariesUsed->emplace_back(std::string{pluginLibArray[i]});
        }
      } else {
        LOG(WARNING) << "Failure to query VC plugin libraries required by parsed ONNX network";
      }
    }
    break;
  }
  case ModelFormat::kAny:
    break;
  }
  const auto tEnd = std::chrono::high_resolution_clock::now();
  const float parseTime = std::chrono::duration<float>(tEnd - tBgine).count();
  LOG(INFO) << "Finished parsing network model. Parse time " << parseTime;
  return parser;
}

//!
//! \brief Set up network and config
//!
//! \return boolean Return true if network and config were successfully set
//!
bool setupNetworkAndConfig(const BuildOptions& build, const SystemOptions& sys, nvinfer1::IBuilder& builder,
                           nvinfer1::INetworkDefinition& network, nvinfer1::IBuilderConfig& config,
                           std::vector<std::vector<char>>& sparseWeights) {
  std::vector<IOptimizationProfile*> profiles{};
  profiles.resize(build.optProfiles.size());
  for (auto& profile : profiles) {
    profile = builder.createOptimizationProfile();
  }

  bool hasDynamicShapes{false};
  // bool broadcastInputFormats;

  // Check if the provided input tensor names match the input tensors of the engine.
  // Throw an error if the provided input tensor names cannot be found because it implies a potential typo.
  for (auto const& shapes : build.optProfiles) {
    for (auto const& shape : shapes) {
      bool tensorNameFound{false};
      for (int32_t i = 0; i < network.getNbInputs(); ++i) {
        if (std::string{shape.first} == std::string{network.getInput(i)->getName()}) {
          tensorNameFound = true;
          break;
        }
      }
      if (!tensorNameFound) {
        LOG(ERROR) << "Cannot find input tensor with name \"" << shape.first << "\" in the network "
                   << "inputs! Please make sure the input tensor names are correct.";
        return false;
      }
    }
  }

  for (uint32_t i = 0, n = network.getNbInputs(); i < n; ++i) {
    // Set formats and data types of inputs
    auto* input = network.getInput(i);

    auto const dims = input->getDimensions();
    auto const isScalar = dims.nbDims == 0;
    auto const isDynamicInput =
        std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; }) || input->isShapeTensor();
    if (isDynamicInput) {
      hasDynamicShapes = true;
      for (size_t i = 0; i < build.optProfiles.size(); ++i) {
        const auto& optShapes = build.optProfiles[i];
        auto profile = profiles[i];
        auto const tensorName = input->getName();
        auto shape_it = optShapes.find(tensorName);
        ShapeRange shapes{};

        // If no shape is provided, set dynamic dimensions to 1.
        if (shape_it == optShapes.end()) {
          constexpr int32_t kDEFAULT_DIMENSION{1};
          std::vector<int32_t> staticDims;
          if (input->isShapeTensor()) {
            if (isScalar) {
              staticDims.push_back(1);
            } else {
              staticDims.resize(dims.d[0]);
              std::fill(staticDims.begin(), staticDims.end(), kDEFAULT_DIMENSION);
            }
          } else {
            staticDims.resize(dims.nbDims);
            std::transform(dims.d, dims.d + dims.nbDims, staticDims.begin(),
                           [&](int dimension) { return dimension > 0 ? dimension : kDEFAULT_DIMENSION; });
          }
          auto s = GetShapeStr(staticDims);
          LOG(WARNING) << "Dynamic dimensions required for input: " << tensorName
                       << ", but no shapes were provided. Automatically overriding shape to: ";
          std::fill(shapes.begin(), shapes.end(), staticDims);
        } else {
          shapes = shape_it->second;
        }

        std::vector<int> profileDims{};
        if (input->isShapeTensor()) {

        } else {
          profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMIN)];
          CHECK_EQ(profile->setDimensions(tensorName, OptProfileSelector::kMIN, toDims(profileDims)), true)
              << "Error in set shape values MIN";
          profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
          CHECK_EQ(profile->setDimensions(tensorName, OptProfileSelector::kOPT, toDims(profileDims)), true)
              << "Error in set shape values OPT";
          profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
          CHECK_EQ(profile->setDimensions(tensorName, OptProfileSelector::kMAX, toDims(profileDims)), true)
              << "Error in set shape values MAX";
          LOG(INFO) << "Set shape of input tensor " << tensorName << " for optimization profile " << i << " to:"
                    << " MIN=" << GetShapeStr(shapes[static_cast<size_t>(OptProfileSelector::kMIN)])
                    << " OPT=" << GetShapeStr(shapes[static_cast<size_t>(OptProfileSelector::kOPT)])
                    << " MAX=" << GetShapeStr(shapes[static_cast<size_t>(OptProfileSelector::kMAX)]);
        }
      }
    } // if (isDynamicInput)
  }

  for (uint32_t i = 0, n = network.getNbOutputs(); i < n; ++i) {
    auto* output = network.getOutput(i);
    const auto dims = output->getDimensions();
    // A shape tensor output with known static dimensions may have dynamic shape values inside it.
    const auto isDynamicOutput =
        std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; }) || output->isShapeTensor();
    if (isDynamicOutput) {
      hasDynamicShapes = true;
    }
  }

  if (!hasDynamicShapes && !build.optProfiles[0].empty()) {
    LOG(ERROR) << "Static model does not take explicit shapes since the shape of inference tensors will be determined "
                  "by the model itself";
    return false;
  }

  if (hasDynamicShapes) {
    for (auto profile : profiles) {
      CHECK_EQ(profile->isValid(), true) << "Required optimization profile is invalid";
      CHECK_NE(config.addOptimizationProfile(profile), -1) << "Error in add optimization profile";
    }
  }

  if (!build.tf32) {
    config.clearFlag(BuilderFlag::kTF32);
  }

  if (build.fp16) {
    config.setFlag(BuilderFlag::kFP16);
  }

  if (build.bf16) {
    config.setFlag(BuilderFlag::kBF16);
  }

  return true;
}

//!
//! \brief Create a serialized engine for a network defintion
//!
//! \return Whether the engine creation succeeds or fails.
//!
bool networkToSerializedEngine(const BuildOptions& build, const SystemOptions& sys, IBuilder& builder,
                               BuildEnvironment& env) {
  std::unique_ptr<IBuilderConfig> config(builder.createBuilderConfig());
  CHECK_NOTNULL(config.get());
  std::vector<std::vector<char>> sparseWeights;
  CHECK_EQ(setupNetworkAndConfig(build, sys, builder, *env.network, *config, sparseWeights), true)
      << "Network and config setup failed";

  const auto tBegin = std::chrono::high_resolution_clock::now();
  std::unique_ptr<IHostMemory> serializedEngine(builder.buildSerializedNetwork(*env.network, *config));
  CHECK_NOTNULL(serializedEngine.get());
  const auto tEnd = std::chrono::high_resolution_clock::now();
  float const buildTime = std::chrono::duration<float>(tEnd - tBegin).count();
  LOG(INFO) << "Engine built in " << buildTime << " sec." << std::endl;
  LOG(INFO) << "Created engine with size: " << (serializedEngine->size() / 1.0_MiB) << " MiB" << std::endl;
  env.engine.setBlob(serializedEngine);
  return true;
}

//!
//! \brief Parse a given model, create a network and an engine.
//!
bool modelToBuildEnv(const ModelOptions& model, const BuildOptions& build, SystemOptions& sys, BuildEnvironment& env) {
  env.builder.reset(createBuilder());
  CHECK_NOTNULL(env.builder.get());
  auto networkFlags = 0U;
  for (const auto& pluginPath : sys.plugins) {
    env.builder->getPluginRegistry().loadLibrary(pluginPath.c_str());
  }
  env.network.reset(env.builder->createNetworkV2(networkFlags));
  CHECK_NOTNULL(env.network.get());
  std::vector<std::string> vcPluginLibrariesUsed;
  env.parser = modelToNetwork(model, build, *env.network, build.versionCompatible ? &vcPluginLibrariesUsed : nullptr);
  CHECK_EQ(env.parser.operator bool(), true) << "Parsing model failed";

  if (build.versionCompatible && !vcPluginLibrariesUsed.empty()) {
    LOG(INFO) << "The following plugin libraries were identified by the parser as required for a "
              << "version-compatible engine:";
    for (const auto& lib : vcPluginLibrariesUsed) {
      LOG(INFO) << "  " << lib;
    }

    LOG(INFO) << "These libraries will be added to --dynamicPlugins for use at inference time.";
    std::copy(vcPluginLibrariesUsed.begin(), vcPluginLibrariesUsed.end(), std::back_inserter(sys.dynamicPlugins));

    // Implicitly-added plugins from ONNX parser should be loaded into plugin registry as well.
    for (const auto& pluginPath : vcPluginLibrariesUsed) {
      env.builder->getPluginRegistry().loadLibrary(pluginPath.c_str());
    }
  }

  CHECK_EQ(networkToSerializedEngine(build, sys, *env.builder, env), true);
  return true;
}

bool saveEngine(nvinfer1::ICudaEngine const& engine, std::string const& fileName, std::ostream& err) {
  std::ofstream engineFile(fileName, std::ios::binary);
  if (!engineFile) {
    err << "Cannot open engine file: " << fileName << std::endl;
    return false;
  }

  std::unique_ptr<IHostMemory> serializedEngine{engine.serialize()};
  if (serializedEngine == nullptr) {
    err << "Engine serialization failed" << std::endl;
    return false;
  }

  engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
  return !engineFile.fail();
}

nvinfer1::ICudaEngine* loadEngine(std::string const& engine, int32_t DLACore, std::ostream& err) {
  BuildEnvironment env(false, false, DLACore, "", getTempfileControlDefaults());
  return loadEngineToBuildEnv(engine, false, env) ? env.engine.release() : nullptr;
}

bool getEngineBuildEnv(ModelOptions const& model, BuildOptions const& build, SystemOptions& sys,
                       BuildEnvironment& env) {
  bool createEngineSuccess{false};

  if (build.load) {
    createEngineSuccess = loadStreamingEngineToBuildEnv(build.engine, env);
  } else {
    createEngineSuccess = modelToBuildEnv(model, build, sys, env);
  }
  CHECK_EQ(createEngineSuccess, true) << "Failed to create engine from model or file.";

  if (build.save) {
    std::ofstream engineFile(build.engine, std::ios::binary);
    auto& engineBlob = env.engine.getBlob();
    engineFile.write(static_cast<const char*>(engineBlob.data), engineBlob.size);
    CHECK_EQ(engineFile.fail(), false) << "Saving engine to file failed.";
    engineFile.flush();
    engineFile.close();
    env.engine.releaseBlob();
    CHECK_EQ(loadStreamingEngineToBuildEnv(build.engine, env), true) << "Reading engine file failed.";
  }

  return true;
}

bool loadStreamingEngineToBuildEnv(std::string const& filepath, BuildEnvironment& env) {
  auto& reader = env.engine.getFileReader();
  CHECK_EQ(reader.open(filepath), true) << "Error opening engine file " << filepath;
  return true;
}

bool loadEngineToBuildEnv(const std::string& filepath, bool enableConsistency, BuildEnvironment& env) {
  const auto tBegin = std::chrono::high_resolution_clock::now();
  std::ifstream engineFile(filepath, std::ios::binary);
  CHECK_EQ(engineFile.good(), true) << "Error opening engine file: " << filepath;
  engineFile.seekg(0, std::ifstream::end);
  int64_t fsize = engineFile.tellg();
  engineFile.seekg(0, std::ifstream::beg);

  std::vector<uint8_t> engineBlob(fsize);
  engineFile.read(reinterpret_cast<char*>(engineBlob.data()), fsize);
  CHECK_EQ(engineFile.good(), true) << "Error loading engine file: " << filepath;
  const auto tEnd = std::chrono::high_resolution_clock::now();
  const float loadTime = std::chrono::duration<float>(tEnd - tBegin).count();
  LOG(INFO) << "Engine loaded in " << loadTime << " sec.";
  LOG(INFO) << "Loaded engine with size: " << fsize / 1.0_MiB << " MiB";

  if (enableConsistency) {
    LOG(FATAL) << "Not supported";
  }

  env.engine.setBlob(std::move(engineBlob));
  return true;
}