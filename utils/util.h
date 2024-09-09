#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cnpy.h"
#include "glog/logging.h"

template <typename T>
inline std::string GetShapeStr(const std::vector<T>& v) {
  std::stringstream ss;
  for (size_t i = 0; i < v.size() - 1; ++i) {
    ss << v[i] << "x";
  }
  ss << v.back();
  return ss.str();
}

inline std::vector<std::string> SplitToStringVec(std::string const& s, char separator) {
  std::vector<std::string> splitted;

  for (size_t start = 0; start < s.length();) {
    size_t separatorIndex = s.find(separator, start);
    if (separatorIndex == std::string::npos) {
      separatorIndex = s.length();
    }
    splitted.emplace_back(s.substr(start, separatorIndex - start));
    start = separatorIndex + 1;
  }

  return splitted;
}

inline std::map<std::string, std::string> ParseInputs(const std::string& ins) {
  auto items = SplitToStringVec(ins, ',');
  std::map<std::string, std::string> res;
  for (auto& item : items) {
    auto str = SplitToStringVec(item, ':');
    CHECK_EQ(str.size(), 2U);
    res[str[0]] = str[1];
  }
  return res;
}

inline std::map<std::string, cnpy::NpyArray> LoadInputFile(const std::string& str) {
  std::map<std::string, cnpy::NpyArray> res;
  auto input_info = ParseInputs(str);
  for (auto it = input_info.begin(); it != input_info.end(); ++it) {
    auto& name = it->first;
    auto& file = it->second;
    cnpy::NpyArray array = cnpy::npy_load(file);
    res[name] = array;
  }
  return res;
}

inline cnpy::npz_t LoadNpzFile(const std::string& str) {
  cnpy::npz_t res = cnpy::npz_load(str);
  return res;
}

template <typename T>
inline float mean(T* data, size_t n) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += data[i];
  }
  return sum * 1. / n;
}

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", __FILE__, __LINE__, err, cudaGetErrorName(err),     \
              cudaGetErrorString(err));                                                                                \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define DRIVER_API_CALL(apiFuncCall)                                                                                   \
  do {                                                                                                                 \
    CUresult _status = apiFuncCall;                                                                                    \
    if (_status != CUDA_SUCCESS) {                                                                                     \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", __FILE__, __LINE__, #apiFuncCall, _status); \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CUPTI_CALL(call)                                                                                               \
  do {                                                                                                                 \
    CUptiResult status = call;                                                                                         \
    if (status != CUPTI_SUCCESS) {                                                                                     \
      const char* errstr;                                                                                              \
      cuptiGetResultString(status, &errstr);                                                                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr);         \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define NVPW_API_CALL(apiFuncCall)                                                                                     \
  do {                                                                                                                 \
    NVPA_Status _status = apiFuncCall;                                                                                 \
    if (_status != NVPA_STATUS_SUCCESS) {                                                                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", __FILE__, __LINE__, #apiFuncCall, _status); \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define RETURN_IF_NVPW_ERROR(retval, actual)                                                                           \
  do {                                                                                                                 \
    NVPA_Status status = actual;                                                                                       \
    if (NVPA_STATUS_SUCCESS != status) {                                                                               \
      fprintf(stderr, "FAILED: %s with error %s\n", #actual, GetNVPWResultString(status));                             \
      return retval;                                                                                                   \
    }                                                                                                                  \
  } while (0)

#define MEMORY_ALLOCATION_CALL(var)                                                                                    \
  do {                                                                                                                 \
    if (var == NULL) {                                                                                                 \
      fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n", __FILE__, __LINE__);                                \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define DISABLE_COPY_MOVE_ASSIGN(classname)                                                                            \
  classname(const classname&) = delete;                                                                                \
  classname& operator=(const classname&) = delete;                                                                     \
  classname(classname&&) = delete;                                                                                     \
  classname& operator=(classname&&) = delete;

#define ASSERT(condition)                                                                                              \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      std::cerr << "Assertion failure: " << #condition << std::endl;                                                   \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)
