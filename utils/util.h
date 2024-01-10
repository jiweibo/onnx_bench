#pragma once

#include <map>
#include <string>
#include <vector>

#include "cnpy.h"
#include "glog/logging.h"

template <typename T> inline std::string PrintShape(const std::vector<T>& v) {
  std::stringstream ss;
  for (size_t i = 0; i < v.size() - 1; ++i) {
    ss << v[i] << "x";
  }
  ss << v.back();
  return ss.str();
}

inline std::vector<std::string> SplitToStringVec(std::string const& s,
                                                 char separator) {
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

inline std::map<std::string, cnpy::NpyArray>
LoadInputFile(const std::string& str) {
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

template <typename T> inline float mean(T* data, size_t n) {
  float sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += data[i];
  }
  return sum * 1. / n;
}
