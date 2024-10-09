#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>

using namespace nvinfer1;

// Build default params
constexpr int32_t defaultMaxAuxStreams{-1};

// System default params
constexpr int32_t defaultDevice{0};

// Inference default params
constexpr int32_t defaultStreams{1};
constexpr int32_t defaultOptProfileIndex{0};

enum class RuntimeMode {
  kFULL,
  kDISPATCH,
  kLEAN,
};

enum class MemoryAllocationStrategy {
  kSTATIC,  //< Allocate device memory based on max size across all profiles.
  kPROFILE, //< Allocate device memory based on max size of the current profile.
  kRUNTIME, //< Allocate device memory based on the current input shapes.
};

enum class ModelFormat {
  kAny,
  kONNX,
};

inline std::ostream& operator<<(std::ostream& os, const RuntimeMode mode) {
  switch (mode) {
  case RuntimeMode::kFULL:
    os << "full";
    break;
  case RuntimeMode::kDISPATCH:
    os << "dispatch";
    break;
  case RuntimeMode::kLEAN:
    os << "lean";
    break;
  }
  return os;
}

using Arguments = std::unordered_multimap<std::string, std::pair<std::string, int32_t>>;
using ShapeRange = std::array<std::vector<int32_t>, nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;

class Options {
public:
  virtual ~Options() = default;
  virtual void parse(Arguments& arguments) = 0;
};

class ModelOptions : public Options {
public:
  ModelFormat format{ModelFormat::kAny};
  std::string model;
  std::string prototxt;
  std::vector<std::string> outputs;
  void parse(Arguments& arguments) override {}
};

inline constexpr nvinfer1::TempfileControlFlags getTempfileControlDefaults() {
  using F = nvinfer1::TempfileControlFlag;
  return (1U << static_cast<uint32_t>(F::kALLOW_TEMPORARY_FILES)) |
         (1U << static_cast<uint32_t>(F::kALLOW_IN_MEMORY_FILES));
}

class BuildOptions : public Options {
public:
  // Unit in MB.
  double workspace{-1};
  bool tf32{true};
  bool fp16{false};
  bool bf16{false};
  bool int8{false};

  bool safe{false};
  bool save{false};
  bool load{false};

  bool versionCompatible{false};
  std::string engine;

  std::string tempdir{};
  TempfileControlFlags tempfileControls{getTempfileControlDefaults()};

  RuntimeMode useRuntime{RuntimeMode::kFULL};
  std::string leadDLLPath{};
  int32_t maxAuxStreams{defaultMaxAuxStreams};

  using ShapeProfile = std::unordered_map<std::string, ShapeRange>;
  std::vector<ShapeProfile> optProfiles;

  // TODO: parse gflags.
  void parse(Arguments& arguments) override {}
};

class SystemOptions : public Options {
public:
  int32_t device{defaultDevice};
  int32_t DLACore{-1};
  std::vector<std::string> plugins;
  std::vector<std::string> dynamicPlugins;
  void parse(Arguments& arguments) override {}
};

class InferenceOptions : public Options {
public:
  int32_t infStream{defaultStreams};
  int32_t optProfileIndex{defaultOptProfileIndex};
  bool graph{false};
  bool setOptProfile{false};

  using ShapeProfile = std::unordered_map<std::string, std::vector<int32_t>>;
  ShapeProfile shapes;

  MemoryAllocationStrategy memoryAllocationStrategy{MemoryAllocationStrategy::kSTATIC};

  void parse(Arguments& arguments) override {}
};