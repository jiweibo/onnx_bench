#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include <cupti.h>
#include <glog/logging.h>

namespace {
#define CUPTI_CALL(call)                                                       \
  do {                                                                         \
    CUptiResult status = call;                                                 \
    if (status != CUPTI_SUCCESS) {                                             \
      const char* errstr;                                                      \
      cuptiGetResultString(status, &errstr);                                   \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",     \
              __FILE__, __LINE__, #call, errstr);                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr int BUF_SIZE = 32 * 1024;
constexpr int ALIGN_SIZE = 8;

struct RuntimeApiTraceStruct {
  const char* function_name;
  uint64_t start_timestamp;
  uint64_t end_timestamp;
  size_t size_in_bytes;
  uint32_t correlation_id;
  const char* memcpy_kind;
  std::unique_ptr<std::mutex> mu;

  RuntimeApiTraceStruct()
      : function_name(nullptr), start_timestamp(0), end_timestamp(0),
        size_in_bytes(0), memcpy_kind(nullptr),
        mu(std::make_unique<std::mutex>()) {}
};

inline const char* CopyKindToString(cudaMemcpyKind kind) {
  switch (kind) {
  case cudaMemcpyDeviceToHost:
    return "D2H";
  case cudaMemcpyDeviceToDevice:
    return "D2D";
  case cudaMemcpyHostToHost:
    return "H2H";
  case cudaMemcpyHostToDevice:
    return "H2D";
  case cudaMemcpyDefault:
    return "DEFAULT";
  default:
    return "UNKNOWN";
  }
}

inline void CUPTIAPI CallbackFunc(void* userdata, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid,
                                  const CUpti_CallbackData* cb_info) {
  auto* traces =
      static_cast<std::map<std::thread::id, RuntimeApiTraceStruct>*>(userdata);
  auto tid = std::this_thread::get_id();

  {
    std::unique_lock<std::mutex> lock;
    if (traces->find(tid) == traces->end()) {
      traces->emplace(tid, RuntimeApiTraceStruct{});
    }
  }
  auto& trace = (*traces)[tid];

  if (cb_info->callbackSite == CUPTI_API_ENTER) {
    cuptiGetTimestamp(&trace.start_timestamp);
    trace.correlation_id = cb_info->correlationId;

    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
      if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunch ||
          cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
          cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx ||
          cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel) {
        LOG(INFO) << "cuLaunch " << cb_info->symbolName << " "
                  << cb_info->functionName;
      } else
        LOG(INFO) << cb_info->functionName;
    }

    if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
        (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
      LOG(ERROR) << "domain " << static_cast<int64_t>(domain);
      // TODO(wilber): cuda11.8 symbolName has bug when encounter cudnn kernel.
      //  trace.function_name = cb_info->symbolName
      LOG(ERROR) << "  cudaLaunch "
                 << reinterpret_cast<const void*>(cb_info->symbolName) << " "
                 << cb_info->correlationId;
      LOG(ERROR) << cb_info->functionName << cb_info->symbolName;

      trace.function_name = cb_info->functionName;
    } else {
      trace.function_name = cb_info->functionName;
    }

    if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
      trace.size_in_bytes =
          static_cast<const cudaMemcpy_v3020_params*>(cb_info->functionParams)
              ->count;
      trace.memcpy_kind = CopyKindToString(
          static_cast<const cudaMemcpy_v3020_params*>(cb_info->functionParams)
              ->kind);
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020) {
      trace.size_in_bytes = static_cast<const cudaMemcpyAsync_v3020_params*>(
                                cb_info->functionParams)
                                ->count;
      trace.memcpy_kind =
          CopyKindToString(static_cast<const cudaMemcpyAsync_v3020_params*>(
                               cb_info->functionParams)
                               ->kind);
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020) {
      trace.size_in_bytes =
          static_cast<const cudaMemset_v3020_params*>(cb_info->functionParams)
              ->count;
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020) {
      trace.size_in_bytes =
          static_cast<const cudaMemsetAsync_ptsz_v7000_params_st*>(
              cb_info->functionParams)
              ->count;
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020) {
      trace.size_in_bytes =
          static_cast<const cudaMalloc_v3020_params*>(cb_info->functionParams)
              ->size;
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020) {
      trace.size_in_bytes = static_cast<const cudaMallocHost_v3020_params*>(
                                cb_info->functionParams)
                                ->size;
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020) {
      trace.size_in_bytes = static_cast<const cudaHostAlloc_v3020_params*>(
                                cb_info->functionParams)
                                ->size;
    }
  }

  if (cb_info->callbackSite == CUPTI_API_EXIT) {
    cuptiGetTimestamp(&trace.end_timestamp);

    if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
        (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
               cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020) {
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020 ||
               cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020 ||
               cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020 ||
               cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020 ||
               cbid == CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020) {
    } else {
    }
  }
}

} // namespace

// class CuptiActivity {
// public:
//   ~CuptiActivity() { CUPTI_CALL(cuptiActivityFlushAll(1)); }
//   void InitTrace();

// private:
//   void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size,
//                                 size_t* max_num_records) {
//     // TODO
//     // uint8_t* bfr = malloc(BUF_SIZE)
//   }

//   void CUPTIAPI BufferCompleted(CUcontext ctx, uint32_t stream_id,
//                                 uint8_t* buffer, size_t size,
//                                 size_t valid_size) {
//     // TODO.
//   }
// };

class CuptiCallbackTracer {
public:
  void EnableCuptiCallback() {
    CUPTI_CALL(cuptiSubscribe(&subscriber_, (CUpti_CallbackFunc)CallbackFunc,
                              &traces_));
    // CUPTI_CALL(cuptiEnableDomain(1, subscriber_,
    // CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API));
  }

  void DisableCuptiCallback() { CUPTI_CALL(cuptiUnsubscribe(subscriber_)); }

private:
  CUpti_SubscriberHandle subscriber_;
  std::map<std::thread::id, RuntimeApiTraceStruct> traces_;
};
