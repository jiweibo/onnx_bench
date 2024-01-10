#pragma once

#include <string>

#include <nvToolsExt.h>

class NvtxRange {
public:
  NvtxRange(const std::string& message) : message_(message) {}

  void Begin() {
    nvtxEventAttributes_t eventAttrib;
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0x00ccffcc;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message_.c_str();

    range_id_ = nvtxRangeStartEx(&eventAttrib);
  }

  void End() { nvtxRangeEnd(range_id_); }

private:
  uint64_t range_id_;
  const std::string message_;
};
