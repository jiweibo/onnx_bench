#pragma once

#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <glog/logging.h>
#include <json/json.h>

enum class Dtype {
  UNKNOWN = 0,
  FLOAT32,
  FLOAT64,
  FLOAT16,
  INT8,
  INT32,
  INT64,
  BOOL,
};

namespace {

inline Dtype ConvertToDtype(const std::string &s) {
  Dtype dtype{Dtype::UNKNOWN};
  if (s == "float32") {
    dtype = Dtype::FLOAT32;
  } else if (s == "float64" || s == "double") {
    dtype = Dtype::FLOAT64;
  } else if (s == "float16" || s == "half") {
    dtype = Dtype::FLOAT16;
  } else if (s == "int8") {
    dtype = Dtype::INT8;
  } else if (s == "int32") {
    dtype = Dtype::INT32;
  } else if (s == "int64") {
    dtype = Dtype::INT64;
  } else if (s == "bool") {
    dtype = Dtype::BOOL;
  } else {
    LOG(FATAL) << "not supported dtype for " << s
               << ", we now support float32, float64/double, float16/half, "
                  "int8, int32, int64, bool";
  }
  return dtype;
}

inline size_t SizeOf(Dtype dtype) {
  switch (dtype) {
  case Dtype::FLOAT32:
    return sizeof(float);
  case Dtype::FLOAT64:
    return sizeof(double);
  case Dtype::FLOAT16:
    return 2;
  case Dtype::INT8:
    return sizeof(int8_t);
  case Dtype::INT32:
    return sizeof(int32_t);
  case Dtype::INT64:
    return sizeof(int64_t);
  case Dtype::BOOL:
    return sizeof(bool);
  default:
    LOG(FATAL) << "not supported dtype ";
  }
}

inline std::map<std::string, std::pair<size_t, size_t>>
CalcSizeForJson(const std::string &json_file,
                const std::vector<std::string> &names) {
  std::ifstream file(json_file);
  Json::Value root;
  Json::Reader reader;
  bool success = reader.parse(file, root);
  file.close();

  if (!success) {
    LOG(FATAL) << "Failed to read JSON data from " << json_file << std::endl;
  }

  std::map<std::string, std::pair<size_t, size_t>> res;

  for (const auto &name : names) {
    auto dtype = ConvertToDtype(root[name]["dtype"].asString());

    auto shape_value = root[name]["shape"];
    CHECK_EQ(shape_value.isArray(), true);

    std::vector<int64_t> shape(shape_value.size());
    for (int i = 0; i < shape.size(); ++i) {
      shape[i] = shape_value[i].asInt64();
    }
    size_t size = std::accumulate(shape.begin(), shape.end(), 1,
                                  std::multiplies<size_t>()) *
                  SizeOf(dtype);
    size_t batch = shape[0];
    res[name] = std::make_pair(size, batch);
  }

  return res;
}

} // namespace

/// Load json data set as model input and output for evaluation.
/// \param path: The dataset dir.
/// \param trunk_size: Load the number of `trunk` files in memory at one time.
/// When the data is used up or the batch is insufficient, the corresponding
/// memory is released and the number of `trunk` files are reloaded. \param
/// list_file: A file which records all dataset.
class JsonDataSet {
public:
  JsonDataSet(const std::string &path, int trunk_size = 32,
              const std::string &list_file = "list.txt")
      : path_(path), trunk_size_(trunk_size), list_file_(list_file) {
    ReadListFile();
  }

  std::map<std::string, std::tuple<void *, std::vector<int64_t>, Dtype>>
  GetData(const std::vector<std::string> &names, size_t batch, bool drop) {
    CHECK_EQ(names.empty(), false);
    CHECK_GE(batch, 1);
    bool success = false;
    std::map<std::string, std::tuple<void *, std::vector<int64_t>, Dtype>> res;

    VLOG(4) << "batch is " << batch << ", cur_batch " << cur_batch_;
    // need to ReadTrunk
    bool read_trunk = false;
    while (cur_batch_ < batch) {
      bool tmp = ReadTrunk(names);
      read_trunk |= tmp;
      if (!tmp)
        break;
    }

    // no data.
    if (cur_batch_ == 0)
      return res;

    size_t actual_batch;
    if (cur_batch_ < batch) {
      if (drop) {
        return res;
      } else {
        LOG(WARNING) << "Not enough for batch " << batch << ", just return "
                     << cur_batch_;
      }
      actual_batch = cur_batch_;
      cur_batch_ = 0;
    } else {
      actual_batch = batch;
      cur_batch_ -= batch;
    }

    for (auto &name : names) {
      std::vector<int64_t> shape = shape_[name];
      shape[0] = actual_batch;
      size_t num = std::accumulate(shape.begin(), shape.end(), 1U,
                                   std::multiplies<size_t>());
      res[name] = std::make_tuple(static_cast<char *>(data_in_mem_[name]) +
                                      data_offset_[name],
                                  shape, dtype_[name]);
      data_offset_[name] += num * SizeOf(dtype_[name]);
      shape_[name][0] -= actual_batch;
    }

    return res;
  }

private:
  int ReadListFile() {
    std::ifstream fp(path_ + "/" + list_file_);
    if (!fp.is_open()) {
      LOG(FATAL) << list_file_ << " not found, please update your dataset.";
      return -1;
    }

    std::string file_name;
    while (fp >> file_name) {
      file_list_.push_back(file_name);
    }
    fp.close();
    VLOG(3) << "find " << file_list_.size() << " files";
    return 0;
  }

  void TryReleaseMemory(const std::vector<std::string> &names) {
    if (cur_batch_ == 0) {
      for (auto it : data_in_mem_) {
        free(it.second);
      }
    }
  }

  bool ReadTrunk(const std::vector<std::string> &names) {
    int end = std::min(idx_ + trunk_size_, file_list_.size());
    if (idx_ >= end) {
      TryReleaseMemory(names);
      return false;
    }

    VLOG(4) << "Load trunk from [" << idx_ << ", " << end << ")";

    // If there is no data left, release it.
    bool use_prev_trunk_data = false;
    std::map<std::string, size_t> prev_reserve_bytes;
    if (cur_batch_ == 0) {
      TryReleaseMemory(names);
    } else {
      // Otherwise copy it to the new trunk memory.
      use_prev_trunk_data = true;
      for (const auto &name : names) {
        auto tmp = shape_[name];
        tmp[0] = cur_batch_;
        size_t num = std::accumulate(tmp.begin(), tmp.end(), 1,
                                     std::multiplies<size_t>());
        size_t bytes = num * SizeOf(dtype_[name]);
        prev_reserve_bytes[name] = bytes;
      }
    }

    // visit trunk_size_ files to calculate the data we need to malloc.
    std::map<std::string, size_t> size_map;
    for (size_t i = idx_; i < end; ++i) {
      std::string filepath = path_ + "/" + file_list_[i];
      auto tmp = CalcSizeForJson(filepath, names);
      cur_batch_ += tmp[names[0]].second;
      for (auto name : names) {
        size_map[name] += tmp[name].first;
        if (use_prev_trunk_data) {
          size_map[name] += prev_reserve_bytes[name];
        }
      }
    }

    // malloc
    for (const auto &name : names) {
      auto *prev_ptr = data_in_mem_[name];
      data_in_mem_[name] = malloc(size_map[name]);
      load_offset_[name] = 0U;

      if (use_prev_trunk_data) {
        memcpy(data_in_mem_[name],
               static_cast<char *>(prev_ptr) + data_offset_[name],
               prev_reserve_bytes[name]);
        load_offset_[name] += prev_reserve_bytes[name];
        free(prev_ptr);
      }

      data_offset_[name] = 0U;
    }

    // load json data
    while (idx_ < end) {
      std::string filepath = path_ + "/" + file_list_[idx_];
      LoadJsonData(filepath);
      ++idx_;
    }

    return true;
  }

private:
  // {
  //   "input_1": {
  //     "data": [
  //       1.0,
  //       2.0,
  //       3.0,
  //       4.0,
  //       5.0,
  //       6.0
  //     ],
  //     "dtype": "float32",
  //     "shape": [
  //       2,
  //       3
  //     ]
  //   }
  // }
  void LoadJsonData(const std::string &json_file) {
    VLOG(5) << "Load json from " << json_file;
    std::ifstream file(json_file);
    Json::Value root;
    Json::Reader reader;
    bool success = reader.parse(file, root);
    file.close();
    if (!success) {
      LOG(FATAL) << "Failed to read JSON data from " << json_file << std::endl;
    }

    for (auto &it : data_in_mem_) {
      auto &name = it.first;
      auto *data =
          (void *)(static_cast<char *>(it.second) + load_offset_[name]);

      dtype_[name] = ConvertToDtype(root[name]["dtype"].asString());

      auto shape_value = root[name]["shape"];
      CHECK_EQ(shape_value.isArray(), true);
      std::vector<int64_t> shape(shape_value.size());
      for (int i = 0; i < shape.size(); ++i) {
        shape[i] = shape_value[i].asInt64();
      }
      shape_[name] = shape;

      auto data_value = root[name]["data"];
      CHECK_EQ(data_value.isArray(), true);
      size_t num = std::accumulate(shape.begin(), shape.end(), 1,
                                   std::multiplies<int>());
      CHECK_EQ(data_value.size(), num);
      load_offset_[name] += num * SizeOf(dtype_[name]);
      switch (dtype_[name]) {
      case Dtype::FLOAT32:
        for (int i = 0; i < num; ++i) {
          static_cast<float *>(data)[i] = data_value[i].asFloat();
        }
        break;
      case Dtype::FLOAT64:
        for (int i = 0; i < num; ++i) {
          static_cast<double *>(data)[i] = data_value[i].asDouble();
        }
        break;
      case Dtype::INT32:
        for (int i = 0; i < num; ++i) {
          static_cast<int32_t *>(data)[i] = data_value[i].asInt();
        }
        break;
      case Dtype::INT64:
        for (int i = 0; i < num; ++i) {
          static_cast<int64_t *>(data)[i] = data_value[i].asInt64();
        }
        break;
      case Dtype::BOOL:
        for (int i = 0; i < num; ++i) {
          static_cast<bool *>(data)[i] = data_value[i].asBool();
        }
        break;
      default:
        LOG(FATAL) << "not implemented for dtype "
                   << static_cast<int>(dtype_[name]);
      }
    }
  }

private:
  std::string path_;
  std::string list_file_;
  size_t trunk_size_;
  size_t idx_{0};
  std::vector<std::string> file_list_;

  // input and output are distinguished by name.
  std::map<std::string, Dtype> dtype_;
  std::map<std::string, std::vector<int64_t>> shape_;
  std::map<std::string, void *> data_in_mem_;
  std::map<std::string, size_t> load_offset_;
  std::map<std::string, size_t> data_offset_;
  size_t cur_batch_{0};
};