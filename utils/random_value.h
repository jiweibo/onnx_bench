#pragma once

#include <algorithm>
#include <numeric>
#include <random>

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
inline void FillBuffer(void* buffer, int64_t volume, T min, T max);

template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
inline void FillBuffer(void* buffer, int64_t volume, T min, T max);

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type>
inline void FillBuffer(void* buffer, int64_t volume, T min, T max) {
  T* typedBuffer = static_cast<T*>(buffer);
  std::default_random_engine engine;
  std::uniform_int_distribution<int32_t> distribution(min, max);
  auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
  std::generate(typedBuffer, typedBuffer + volume, generator);
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type>
inline void FillBuffer(void* buffer, int64_t volume, T min, T max) {
  T* typedBuffer = static_cast<T*>(buffer);
  std::default_random_engine engine;
  std::uniform_real_distribution<float> distribution(min, max);
  auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
  std::generate(typedBuffer, typedBuffer + volume, generator);
}
