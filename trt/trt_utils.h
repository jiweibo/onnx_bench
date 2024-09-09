#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include <NvInfer.h>

#include <glog/logging.h>

nvinfer1::Dims toDims(const std::vector<int32_t>& vec);

inline std::string getDimsStr(const nvinfer1::Dims& dims) {
  std::stringstream ss;
  for (size_t i = 0; i < dims.nbDims - 1; ++i) {
    ss << dims.d[i] << "x";
  }
  ss << dims.d[dims.nbDims - 1];
  return ss.str();
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void fillBuffer(void* buffer, int64_t volume, T min, T max);

template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
void fillBuffer(void* buffer, int64_t volume, T min, T max);

template <typename T>
inline T roundUp(T m, T n) {
  return ((m + n - 1) / n) * n;
}

template <typename A, typename B>
inline A divUp(A x, B n) {
  return (x + n - 1) / n;
}

//! comps is the number of components in a vector. Ignored if vecDim < 0.
int64_t volume(nvinfer1::Dims const& dims, nvinfer1::Dims const& strides, int32_t vecDim, int32_t comps, int32_t batch);

int64_t volume(const nvinfer1::Dims& dims, int32_t start, int32_t stop);

size_t dataTypeSize(nvinfer1::DataType dataType);

constexpr long double operator"" _MiB(long double val) { return val * (1 << 20); }

static auto StreamDeleter = [](cudaStream_t* stream) {
  if (stream) {
    cudaStreamDestroy(*stream);
    delete stream;
  }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream() {
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
  if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess) {
    pStream.reset(nullptr);
  }
  return pStream;
}