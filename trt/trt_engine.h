#pragma once

#include <cassert>
#include <fstream>
#include <memory>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <glog/logging.h>

#include "trt/trt_options.h"

using namespace nvinfer1;

struct Parser {
  std::unique_ptr<nvonnxparser::IParser> onnxParser;

  operator bool() const { return onnxParser != nullptr; }
};

//! Implements the TensorRT IStreamReader to allow deserializing an engine
//! directly from the plan file.
class FileStreamReader final : public nvinfer1::IStreamReader {
public:
  bool open(std::string filepath) {
    mFile.open(filepath, std::ios::binary);
    return mFile.is_open();
  }

  void close() {
    if (mFile.is_open()) {
      mFile.close();
    }
  }

  ~FileStreamReader() final { close(); }

  int64_t read(void* dest, int64_t bytes) final {
    if (!mFile.good()) {
      return -1;
    }
    mFile.read(static_cast<char*>(dest), bytes);
    return mFile.gcount();
  }

  void reset() {
    assert(mFile.good());
    mFile.seekg(0);
  }

  bool isOpen() const { return mFile.is_open(); }

private:
  std::ifstream mFile;
};

//!
//! \brief Helper struct to faciliate engine serialization and deserialization.
//! It does not own the underlying memory.
//!
struct EngineBlob {
  EngineBlob(void* engineData, size_t engineSize) : data(engineData), size(engineSize) {}
  void* data{};
  size_t size{};
  bool empty() const { return size == 0; }
};

//!
//! \brief A helper class to hold a serialized engine (std or safe) and only
//! deserialize it when being accessed.
//!
class LazilyDeserializedEngine {
public:
  //!
  //! \brief Delete default constructor to make sure isSafe and DLACore are
  //! always set.
  //!
  LazilyDeserializedEngine() = delete;

  //!
  //! \brief Constructor of LazilyDeserializedEngine.
  //!
  LazilyDeserializedEngine(bool isSafe, bool versionCompatible, int32_t DLACore, std::string const& tempdir,
                           nvinfer1::TempfileControlFlags tempfileControls, std::string const& leanDLLPath)
      : mIsSafe(isSafe), mVersionCompatible(versionCompatible), mDLACore(DLACore), mTempdir(tempdir),
        mTempfileControls(tempfileControls), mLeanDLLPath(leanDLLPath) {
    mFileReader = std::make_unique<FileStreamReader>();
  }

  //!
  //! \brief Move from another LazilyDeserializedEngine.
  //!
  LazilyDeserializedEngine(LazilyDeserializedEngine&& other) = default;

  //!
  //! \brief Delete copy constructor.
  //!
  LazilyDeserializedEngine(LazilyDeserializedEngine const& other) = delete;

  //!
  //! \brief Get the pointer to the ICudaEngine. Triggers deserialization if not
  //! already done so.
  //!
  nvinfer1::ICudaEngine* get();

  //!
  //! \brief Get the pointer to the ICudaEngine and release the ownership.
  //!
  nvinfer1::ICudaEngine* release();

  // //!
  // //! \brief Get the pointer to the safe::ICudaEngine. Triggers
  // deserialization
  // //! if not already done so.
  // //!
  // nvinfer1::safe::ICudaEngine* getSafe();

  //!
  //! \brief Get the underlying blob storing serialized engine.
  //!
  EngineBlob const getBlob() const {
    CHECK((!mFileReader || !mFileReader->isOpen()) &&
          "Attempting to access the glob when there is an open file reader!");
    if (!mEngineBlob.empty()) {
      return EngineBlob{const_cast<void*>(static_cast<void const*>(mEngineBlob.data())), mEngineBlob.size()};
    }
    if (mEngineBlobHostMemory.get() != nullptr && mEngineBlobHostMemory->size() > 0) {
      return EngineBlob{mEngineBlobHostMemory->data(), mEngineBlobHostMemory->size()};
    }
    CHECK(false && "Attempting to access an empty engine!");
    return EngineBlob{nullptr, 0};
  }

  //!
  //! \brief Set the underlying blob storing the serialized engine without
  //! duplicating IHostMemory.
  //!
  void setBlob(std::unique_ptr<nvinfer1::IHostMemory>& data) {
    CHECK(data.get() && data->size() > 0);
    mEngineBlobHostMemory = std::move(data);
    mEngine.reset();
    // mSafeEngine.reset();
  }

  //!
  //! \brief Set the underlying blob storing the serialized engine without
  //! duplicating vector memory.
  //!
  void setBlob(std::vector<uint8_t>&& engineBlob) {
    mEngineBlob = std::move(engineBlob);
    mEngine.reset();
    // mSafeEngine.reset();
  }

  //!
  //! \brief Release the underlying blob without deleting the deserialized
  //! engine.
  //!
  void releaseBlob() {
    mEngineBlob.clear();
    mEngineBlobHostMemory.reset();
  }

  //!
  //! \brief Get the file stream reader used for deserialization
  //!
  FileStreamReader& getFileReader() {
    CHECK(mFileReader);
    return *mFileReader;
  }

  //!
  //! \brief Get if safe mode is enabled.
  //!
  bool isSafe() { return mIsSafe; }

  void setDynamicPlugins(std::vector<std::string> const& dynamicPlugins) { mDynamicPlugins = dynamicPlugins; }

private:
  bool mIsSafe{false};
  bool mVersionCompatible{false};
  int32_t mDLACore{-1};
  std::vector<uint8_t> mEngineBlob;
  std::unique_ptr<FileStreamReader> mFileReader;

  // Directly use the host memory of a serialized engine instead of duplicating
  // the engine in CPU memory.
  std::unique_ptr<nvinfer1::IHostMemory> mEngineBlobHostMemory;

  std::string mTempdir{};
  nvinfer1::TempfileControlFlags mTempfileControls{getTempfileControlDefaults()};
  std::string mLeanDLLPath{};
  std::vector<std::string> mDynamicPlugins;

  //! \name Owned TensorRT objects
  //! Per TensorRT object lifetime requirements as outlined in the developer
  //! guide, the runtime must remain live while any engines created by the
  //! runtime are live. DO NOT ADJUST the declaration order here: runtime ->
  //! (engine|safeEngine). Destruction occurs in reverse declaration order:
  //! (engine|safeEngine) -> runtime.
  //!@{

  //! The runtime used to track parent of mRuntime if one exists.
  //! Needed to load mRuntime if lean.so is supplied through file system path.
  std::unique_ptr<nvinfer1::IRuntime> mParentRuntime{};

  //! The runtime that is used to deserialize the engine.
  std::unique_ptr<nvinfer1::IRuntime> mRuntime{};

  //! If mIsSafe is false, this points to the deserialized std engine
  std::unique_ptr<nvinfer1::ICudaEngine> mEngine{};

  // //! If mIsSafe is true, this points to the deserialized safe engine
  // std::unique_ptr<nvinfer1::safe::ICudaEngine> mSafeEngine{};

  //!@}
};

struct BuildEnvironment {
  BuildEnvironment() = delete;
  BuildEnvironment(BuildEnvironment const& other) = delete;
  BuildEnvironment(BuildEnvironment&& other) = delete;
  BuildEnvironment(bool isSafe, bool versionCompatible, int32_t DLACore, std::string const& tempdir,
                   nvinfer1::TempfileControlFlags tempfileControls, std::string const& leanDLLPath = "")
      : engine(isSafe, versionCompatible, DLACore, tempdir, tempfileControls, leanDLLPath) {}

  //! \name Owned TensorRT objects
  //! Per TensorRT object lifetime requirements as outlined in the developer
  //! guide, factory objects must remain live while the objects created by those
  //! factories are live (with the exception of builder -> engine). DO NOT
  //! ADJUST the declaration order here: builder -> network -> parser.
  //! Destruction occurs in reverse declaration order: parser -> network ->
  //! builder.
  //!@{

  //! The builder used to build the engine.
  std::unique_ptr<nvinfer1::IBuilder> builder;

  //! The network used by the builder.
  std::unique_ptr<nvinfer1::INetworkDefinition> network;

  //! The parser used to specify the network.
  Parser parser;

  //! The engine.
  LazilyDeserializedEngine engine;
  //!@}
};

//!
//! \brief Set up network and config
//!
//! \return boolean Return true if network and config were successfully set
//!
bool setupNetworkAndConfig(const BuildOptions& build, const SystemOptions& sys, nvinfer1::IBuilder& builder,
                           nvinfer1::INetworkDefinition& network, nvinfer1::IBuilderConfig& config, std::ostream& err,
                           std::vector<std::vector<char>>& sparseWeights);

//!
//! \brief Load a serialized engine
//!
//! \return Pointer to the engine loaded or nullptr if the operation failed
//!
nvinfer1::ICudaEngine* loadEngine(std::string const& engine, int32_t DLACore, std::ostream& err);

//!
//! \brief Save an engine into a file
//!
//! \return boolean Return true if the engine was successfully saved
//!
bool saveEngine(nvinfer1::ICudaEngine const& engine, std::string const& fileName, std::ostream& err);

//!
//! \brief Create an engine from model or serialized file, and optionally save engine
//!
//! \return Pointer to the engine created or nullptr if the creation failed
//!
bool getEngineBuildEnv(ModelOptions const& model, BuildOptions const& build, SystemOptions& sys, BuildEnvironment& env);

bool loadStreamingEngineToBuildEnv(std::string const& filepath, BuildEnvironment& env);

bool loadEngineToBuildEnv(const std::string& filepath, bool enableConsistency, BuildEnvironment& env);
