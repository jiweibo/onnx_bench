#pragma once

#include "NvInferRuntime.h"
#include "NvInferRuntimeBase.h"
#include "utils/util.h"

#include "trt/trt_options.h"

#include <NvInfer.h>
#include <ostream>

using namespace nvinfer1;

class Logger : ILogger {
public:
  DISABLE_COPY_MOVE_ASSIGN(Logger);

  void log(Severity severity, char const* msg) noexcept override {
    if (severity > severity_)
      return;
    switch (severity) {
    case ILogger::Severity::kINFO:
      LOG(INFO) << "[TRT-INFO]" << msg;
      break;
    case ILogger::Severity::kVERBOSE:
      LOG(INFO) << "[TRT-VERBOSE]" << msg;
      break;
    case ILogger::Severity::kWARNING:
      LOG(WARNING) << "[TRT-WARNING]" << msg;
      break;
    case ILogger::Severity::kERROR:
      LOG(ERROR) << "[TRT-ERROR]" << msg;
      break;
    case ILogger::Severity::kINTERNAL_ERROR:
      LOG(FATAL) << "[TRT-INTERNAL-ERROR]" << msg;
      break;
    default:
      LOG(FATAL) << "Unknown severity";
    }
  }

  static Logger* getLogger() noexcept {
    static Logger logger;
    return &logger;
  }

  ILogger* getTrtLogger() { return static_cast<ILogger*>(getLogger()); }

  ILogger::Severity getSeverity() const { return severity_; }

private:
  explicit Logger(ILogger::Severity severity = ILogger::Severity::kWARNING) : severity_(severity) {}

  ILogger::Severity severity_;
};
