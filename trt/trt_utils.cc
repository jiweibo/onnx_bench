#include "trt/trt_utils.h"

#include <algorithm>
#include <cstdint>
#include <random>

nvinfer1::Dims toDims(const std::vector<int32_t>& vec) {
  int32_t limit = static_cast<int32_t>(nvinfer1::Dims::MAX_DIMS);
  if (static_cast<int32_t>(vec.size()) > limit) {
    LOG(WARNING) << "Vector too long, only first 8 elements are used in dimension.";
  }
  // Pick first nvinfer1::Dims::MAX_DIMS elements
  nvinfer1::Dims dims{std::min(static_cast<int32_t>(vec.size()), limit), {}};
  std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
  return dims;
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type>
void fillBuffer(void* buffer, int64_t volume, T min, T max) {
  T* typedBuffer = static_cast<T*>(buffer);
  std::default_random_engine engine;
  std::uniform_int_distribution<int32_t> distribution(min, max);
  auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
  std::generate(typedBuffer, typedBuffer + volume, generator);
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type>
void fillBuffer(void* buffer, int64_t volume, T min, T max) {
  T* typedBuffer = static_cast<T*>(buffer);
  std::default_random_engine engine;
  std::uniform_real_distribution<float> distribution(min, max);
  auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
  std::generate(typedBuffer, typedBuffer + volume, generator);
}

// Explicit instantiation
template void fillBuffer(void* buffer, int64_t volume, bool min, bool max);
template void fillBuffer(void* buffer, int64_t volume, float min, float max);
template void fillBuffer(void* buffer, int64_t volume, int32_t min, int32_t max);
template void fillBuffer(void* buffer, int64_t volume, int64_t min, int64_t max);
template void fillBuffer(void* buffer, int64_t volume, int8_t min, int8_t max);
template void fillBuffer(void* buffer, int64_t volume, uint8_t min, uint8_t max);

int64_t volume(nvinfer1::Dims const& dims, nvinfer1::Dims const& strides, int32_t vecDim, int32_t comps,
               int32_t batch) {
  int64_t maxNbElems = 1;
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    // Get effective length of axis.
    int64_t d = dims.d[i];
    // Any dimension is 0, it is an empty tensor.
    if (d == 0) {
      return 0;
    }
    if (i == vecDim) {
      d = divUp(d, comps);
    }
    maxNbElems = std::max(maxNbElems, d * strides.d[i]);
  }
  return maxNbElems * batch * (vecDim < 0 ? 1 : comps);
}

int64_t volume(const nvinfer1::Dims& dims, int32_t start, int32_t stop) {
  CHECK_GE(start, 0);
  CHECK_LE(start, stop);
  CHECK_LE(stop, dims.nbDims);
  CHECK(std::all_of(dims.d + start, dims.d + stop, [](int32_t x) { return x >= 0; }));
  return std::accumulate(dims.d + start, dims.d + stop, 1UL, std::multiplies<int64_t>());
}

size_t dataTypeSize(nvinfer1::DataType dataType) {
  switch (dataType) {
  case nvinfer1::DataType::kINT64:
    return 8U;
  case nvinfer1::DataType::kINT32:
  case nvinfer1::DataType::kFLOAT:
    return 4U;
  case nvinfer1::DataType::kBF16:
  case nvinfer1::DataType::kHALF:
    return 2U;
  case nvinfer1::DataType::kBOOL:
  case nvinfer1::DataType::kUINT8:
  case nvinfer1::DataType::kINT8:
  case nvinfer1::DataType::kFP8:
    return 1U;
  case nvinfer1::DataType::kINT4:
    LOG(FATAL) << "Element size is not implemented for sub-byte data-types.";
  }
  return 0;
}
