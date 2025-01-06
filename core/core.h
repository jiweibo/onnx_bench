#pragma once

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <type_traits>

#include <cuda_runtime_api.h>

#include <glog/logging.h>

#include "NvInferRuntimePlugin.h"
#include "utils/util.h"

namespace core {

enum class Location {
  kDEVICE = 0,
  kHOST = 1,
};

enum class DataType {
  //! 32-bit floating point format.
  kFLOAT = 0,

  //! IEEE 16-bit floating-point format -- has a 5 bit exponent and 11 bit significand.
  kHALF = 1,

  //! Signed 8-bit integer representing a quantized floating-point value.
  kINT8 = 2,

  //! Signed 32-bit integer format.
  kINT32 = 3,

  //! 8-bit boolean. 0 = false, 1 = true, rhs values undefined.
  kBOOL = 4,

  //! Unsigned 8-bit integer format.
  //! Cannot be used to represent quantized floating-point values.
  //! Use the IdentityLayer to convert kUINT8 network-level inputs to {kFLOAT, kHALF} prior
  //! to use with rhs TensorRT layers, or to convert intermediate output
  //! before kUINT8 network-level outputs from {kFLOAT, kHALF} to kUINT8.
  //! kUINT8 conversions are only supported for {kFLOAT, kHALF}.
  //! kUINT8 to {kFLOAT, kHALF} conversion will convert the integer values
  //! to equivalent floating point values.
  //! {kFLOAT, kHALF} to kUINT8 conversion will convert the floating point values
  //! to integer values by truncating towards zero. This conversion has undefined behavior for
  //! floating point values outside the range [0.0F, 256.0F) after truncation.
  //! kUINT8 conversions are not supported for {kINT8, kINT32, kBOOL}.
  kUINT8 = 5,

  //! Signed 8-bit floating point with
  //! 1 sign bit, 4 exponent bits, 3 mantissa bits, and exponent-bias 7.
  kFP8 = 6,

  //! Brain float -- has an 8 bit exponent and 8 bit significand.
  kBF16 = 7,

  //! Signed 64-bit integer type.
  kINT64 = 8,

  //! Signed 4-bit integer type.
  kINT4 = 9,
};

inline size_t DataTypeSize(DataType dtype) {
  switch (dtype) {
  case DataType::kINT64:
    return 8U;
  case DataType::kINT32:
  case DataType::kFLOAT:
    return 4U;
  case DataType::kBF16:
  case DataType::kHALF:
    return 2U;
  case DataType::kBOOL:
  case DataType::kUINT8:
  case DataType::kINT8:
  case DataType::kFP8:
    return 1U;
  case DataType::kINT4:
    LOG(FATAL) << "Element size is not implemented for sub-byte data-types.";
  }
  return 0;
}

//!
//! \class Dims
//! \brief Structure to define the dimensions of a tensor.
//!
//! Dims can also return an "invalid dims" structure. This structure is
//! represented by nbDims == -1 and d[i] == 0 for all i.
//!
//! Dims can also return an "unknown rank" dims structure. This structure is
//! represented by nbDims == -1 and d[i] == -1 for all i.
//!
class Dims {
public:
  //! The maximum rank (number of dimensions) supported for a tensor.
  static constexpr int32_t MAX_DIMS{8};

  //! The rank (number of dimensions).
  int32_t num_dims;

  //! The extent of each dimension.
  int64_t d[MAX_DIMS];

  Dims() = default;

  Dims(std::initializer_list<int> list) {
    num_dims = list.size();
    std::copy(list.begin(), list.end(), d);
  }

  template <typename T>
  Dims(const std::vector<T>& vec) {
    CHECK_LE(vec.size(), MAX_DIMS);
    num_dims = vec.size();
    for (size_t i = 0; i < vec.size(); ++i) {
      this->d[i] = vec[i];
    }
  }

  inline int64_t Numel() const {
    if (num_dims == 0) {
      return 0;
    }
    return std::accumulate(d, d + num_dims, 1L, std::multiplies<int64_t>());
  }

  template <typename T>
  std::vector<T> ToStdVec() const {
    std::vector<T> res;
    res.reserve(num_dims);
    std::copy(d, d + num_dims, std::back_inserter(res));
    return res;
  }
};

inline std::ostream& operator<<(std::ostream& os, const Dims& dims) {
  for (int i = 0; i < dims.num_dims - 1; ++i) {
    os << dims.d[i] << "x";
  }
  os << dims.d[dims.num_dims - 1];
  return os;
}

inline int64_t GetBytes(const Dims& dims, DataType dtype, bool no_zero = true) {
  bool dims_valid = std::all_of(dims.d, dims.d + dims.num_dims, [](int32_t val) { return val >= 0; });
  CHECK_EQ(dims_valid, true);
  int64_t bytes = DataTypeSize(dtype) * dims.Numel();
  if (bytes == 0 && no_zero)
    return 1;
  return bytes;
}

class IAllocator {
public:
  virtual void Allocate(void** ptr, size_t size) = 0;
  virtual void Deallocate(void* ptr) = 0;
  virtual ~IAllocator(){};
};

class DeviceAllocator : public IAllocator {
public:
  void Allocate(void** ptr, size_t size) override { CUDA_CHECK(cudaMalloc(ptr, size)); }
  void Deallocate(void* ptr) override { CUDA_CHECK(cudaFree(ptr)); }
};

class HostAllocator : public IAllocator {
public:
  void Allocate(void** ptr, size_t size) override { CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault)); }
  void Deallocate(void* ptr) override { CUDA_CHECK(cudaFreeHost(ptr)); }
};

//!
//! \class Buffer
//! \brief Managed buffer for host and device
//!
template <typename T, typename = std::enable_if_t<std::is_base_of<IAllocator, T>::value>>
class Buffer {
public:
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& rhs) {
    Reset(rhs.mPtr, rhs.mSize);
    rhs.mPtr = nullptr;
    rhs.mSize = 0;
  }

  Buffer& operator=(Buffer&& rhs) {
    if (this != &rhs) {
      Reset(rhs.mPtr, rhs.mSize);
      rhs.mPtr = nullptr;
      rhs.mSize = 0;
    }
    return *this;
  }

  ~Buffer() { Reset(); }

  Buffer(size_t size) {
    T().Allocate(&ptr_, size);
    size_ = size;
  }

  void Allocate(size_t size) {
    if (size > size_) {
      Reset();
      T().Allocate(&ptr_, size);
      size_ = size;
    }
  }

  void Reset(void* ptr = nullptr, size_t size = 0) {
    if (ptr_) {
      T().Deallocate(ptr_);
    }
    ptr_ = ptr;
    size_ = size;
  }

  void* Data() const { return ptr_; }

  size_t Bytes() const { return size_; }

private:
  void* ptr_{nullptr};
  size_t size_{0};
};

using DeviceBuffer = Buffer<DeviceAllocator>;
using HostBuffer = Buffer<HostAllocator>;

/// There are 4 scenarios:
///   - During initialization, the Tensor object maintains device_buffer, and host_buffer is nullptr. We may call
///     DeviceToHost, so that the Tensor object also maintains host_buffer.
///   - During initialization, the Tensor object maintains host_buffer, and device_buffer is nullptr. We may call
///     HostToDevice, so that the Tensor object also maintains device_buffer.
///   - During initialization, an external device data pointer is used. We may call DeviceToHost, so that the Tensor
///     object maintains host_buffer. Calling interfaces such as Resize is not allowed because we do not have
///     permission to modify external pointers
///   - During initialization, an external host data pointer is used. We may call HostToDevice, so that the Tensor
///     object maintains device_buffer. Calling interfaces such as Resize is not allowed because we do not have
///     permission to modify external pointers
class Tensor {
public:
  explicit Tensor(Dims dims, DataType dtype = DataType::kFLOAT, Location location = Location::kHOST,
                  int32_t device_id = -1)
      : dtype_(dtype), dims_(dims), location_(location), device_id_(device_id) {
    if (location == Location::kDEVICE) {
      CUDA_CHECK(cudaSetDevice(device_id));
      device_buffer_.reset(new DeviceBuffer());
      if (dims.Numel() > 0) {
        device_buffer_->Allocate(GetBytes(dims, dtype));
      }
    } else {
      host_buffer_.reset(new HostBuffer);
      if (dims.Numel() > 0) {
        host_buffer_->Allocate(GetBytes(dims, dtype));
      }
    }
  };

  explicit Tensor(void* data, size_t size, Dims dims, DataType dtype, Location location = Location::kHOST,
                  int32_t devie_id = -1)
      : dtype_(dtype), dims_(dims), location_(location), device_id_(devie_id), external_size_(size) {
    CHECK_GE(external_size_, GetBytes(dims, dtype, false));
    if (location == Location::kDEVICE) {
      external_device_data_ = data;
    } else {
      external_host_data_ = data;
    }
  }

  Tensor(const Tensor& rhs)
      : dtype_(rhs.dtype_), dims_(rhs.dims_), location_(rhs.location_), device_id_(rhs.device_id_),
        device_buffer_(rhs.device_buffer_), host_buffer_(rhs.host_buffer_),
        external_device_data_(rhs.external_device_data_), external_host_data_(rhs.external_host_data_),
        external_size_(rhs.external_size_) {}

  Tensor& operator=(const Tensor& rhs) {
    CHECK_EQ(static_cast<int>(dtype_), static_cast<int>(rhs.dtype_));
    if (this != &rhs) {
      dims_ = rhs.dims_;
      location_ = rhs.location_;
      device_id_ = rhs.device_id_;
      device_buffer_ = rhs.device_buffer_;
      host_buffer_ = rhs.host_buffer_;
      external_device_data_ = rhs.external_device_data_;
      external_host_data_ = rhs.external_host_data_;
      external_size_ = rhs.external_size_;
    }
    return *this;
  }

  Tensor(Tensor&& rhs)
      : dtype_(rhs.dtype_), dims_(rhs.dims_), location_(rhs.location_), device_id_(rhs.device_id_),
        device_buffer_(std::move(rhs.device_buffer_)), host_buffer_(std::move(rhs.host_buffer_)),
        external_device_data_(rhs.external_device_data_), external_host_data_(rhs.external_host_data_),
        external_size_(rhs.external_size_) {
    rhs.device_buffer_ = nullptr;
    rhs.host_buffer_ = nullptr;
  }

  Tensor& operator=(Tensor&& rhs) {
    CHECK_EQ(static_cast<int>(dtype_), static_cast<int>(rhs.dtype_));
    if (this != &rhs) {
      dims_ = rhs.dims_;
      location_ = rhs.location_;
      device_id_ = rhs.device_id_;
      device_buffer_ = std::move(rhs.device_buffer_);
      host_buffer_ = std::move(rhs.host_buffer_);
      external_device_data_ = rhs.external_device_data_;
      external_host_data_ = rhs.external_host_data_;
      external_size_ = rhs.external_size_;
      rhs.device_buffer_ = nullptr;
      rhs.host_buffer_ = nullptr;
    }
    return *this;
  }

  virtual ~Tensor() = default;

  void Resize(const Dims& dims) {
    CHECK_EQ(external_device_data_ == nullptr && external_host_data_ == nullptr, true)
        << "Not allowed call Resize when use external data.";

    if (dims.Numel() > dims_.Numel()) {
      int64_t bytes = GetBytes(dims, dtype_);
      if (device_buffer_) {
        CUDA_CHECK(cudaSetDevice(device_id_));
        device_buffer_->Allocate(bytes);
      }
      if (host_buffer_) {
        host_buffer_->Allocate(bytes);
      }
    }
    dims_ = dims;
  }

  void HostToDevice(cudaStream_t stream) {
    CHECK_EQ(external_device_data_, nullptr);
    CUDA_CHECK(cudaSetDevice(device_id_));
    if (host_buffer_ != nullptr) {
      if (device_buffer_ == nullptr) {
        device_buffer_.reset(new DeviceBuffer(host_buffer_->Bytes()));
      } else {
        device_buffer_->Allocate(host_buffer_->Bytes());
      }
      CUDA_CHECK(
          cudaMemcpyAsync(device_buffer_->Data(), host_buffer_->Data(), host_buffer_->Bytes(), cudaMemcpyHostToDevice));
    } else {
      if (device_buffer_ == nullptr) {
        device_buffer_.reset(new DeviceBuffer(external_size_));
      } else {
        device_buffer_->Allocate(external_size_);
      }
      CUDA_CHECK(cudaMemcpyAsync(device_buffer_->Data(), external_host_data_, external_size_, cudaMemcpyHostToDevice));
    }
  }

  void DeviceToHost(cudaStream_t stream) {
    CHECK_EQ(external_host_data_, nullptr);
    CUDA_CHECK(cudaSetDevice(device_id_));
    if (device_buffer_ != nullptr) {
      if (host_buffer_ == nullptr) {
        host_buffer_.reset(new HostBuffer(device_buffer_->Bytes()));
      } else {
        host_buffer_->Allocate(host_buffer_->Bytes());
      }
      CUDA_CHECK(cudaMemcpyAsync(host_buffer_->Data(), device_buffer_->Data(), device_buffer_->Bytes(),
                                 cudaMemcpyDeviceToHost, stream));
    } else {
      if (host_buffer_ == nullptr) {
        host_buffer_.reset(new HostBuffer(external_size_));
      } else {
        host_buffer_->Allocate(external_size_);
      }
      CUDA_CHECK(
          cudaMemcpyAsync(host_buffer_->Data(), external_device_data_, external_size_, cudaMemcpyDeviceToHost, stream));
    }
  }

  const Dims GetDims() const { return dims_; }

  const DataType GetDataType() const { return dtype_; }

  void* DeviceData() {
    if (external_device_data_) {
      return external_device_data_;
    }

    if (dims_.Numel() > 0 && device_buffer_) {
      return device_buffer_->Data();
    }

    return nullptr;
  }

  template <typename T>
  T* DeviceData() {
    return static_cast<T*>(this->DeviceData());
  }

  const void* DeviceData() const {
    if (external_device_data_) {
      return external_device_data_;
    }

    if (dims_.Numel() > 0 && device_buffer_) {
      return device_buffer_->Data();
    }

    return nullptr;
  }

  template <typename T>
  const T* DeviceData() const {
    return static_cast<T*>(this->DeviceData());
  }

  void* HostData() {
    if (external_host_data_) {
      return external_host_data_;
    }

    if (dims_.Numel() > 0 && host_buffer_) {
      return host_buffer_->Data();
    }

    return nullptr;
  }

  template <typename T>
  T* HostData() {
    return static_cast<T*>(this->HostData());
  }

  const void* HostData() const {
    if (external_host_data_) {
      return external_host_data_;
    }

    if (dims_.Numel() > 0 && host_buffer_) {
      return host_buffer_->Data();
    }

    return nullptr;
  }

  template <typename T>
  const T* HostData() const {
    return static_cast<const T*>(this->HostData());
  }

  int64_t Numel() const { return dims_.Numel(); }

  size_t Bytes() const { return GetBytes(dims_, dtype_); }

  int GetDeviceId() const { return device_id_; }

  Location GetLocation() const { return location_; }

protected:
  // Used to construct sub-class tensor.
  Tensor() = default;

protected:
  DataType dtype_;
  Dims dims_;
  Location location_;
  int device_id_;

  std::shared_ptr<DeviceBuffer> device_buffer_{nullptr};
  std::shared_ptr<HostBuffer> host_buffer_{nullptr};
  void* external_device_data_{nullptr};
  void* external_host_data_{nullptr};
  size_t external_size_{0};
};

using TensorRef = std::shared_ptr<Tensor>;

} // namespace core
