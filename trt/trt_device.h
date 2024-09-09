#pragma once

#include <cstddef>

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>

#include "trt/trt_utils.h"
#include "utils/util.h"

//!
//! \class TrtCudaBuffer
//! \brief Managed buffer for host and device
//!
template <typename Allocator, typename Deallocator>
class TrtCudaBuffer {
public:
  TrtCudaBuffer() = default;

  TrtCudaBuffer(const TrtCudaBuffer&) = delete;

  TrtCudaBuffer& operator=(const TrtCudaBuffer&) = delete;

  TrtCudaBuffer(TrtCudaBuffer&& rhs) {
    reset(rhs.mPtr, rhs.mSize);
    rhs.mPtr = nullptr;
    rhs.mSize = 0;
  }

  TrtCudaBuffer& operator=(TrtCudaBuffer&& rhs) {
    if (this != &rhs) {
      reset(rhs.mPtr, rhs.mSize);
      rhs.mPtr = nullptr;
      rhs.mSize = 0;
    }
    return *this;
  }

  ~TrtCudaBuffer() { reset(); }

  TrtCudaBuffer(size_t size) {
    Allocator()(&mPtr, size);
    mSize = size;
  }

  void allocate(size_t size) {
    reset();
    Allocator()(&mPtr, size);
    mSize = size;
  }

  void reset(void* ptr = nullptr, size_t size = 0) {
    if (mPtr) {
      Deallocator()(mPtr);
    }
    mPtr = ptr;
    mSize = size;
  }

  void* get() const { return mPtr; }

  size_t getSize() const { return mSize; }

private:
  void* mPtr{nullptr};
  size_t mSize{0};
};

struct DeviceAllocator {
  void operator()(void** ptr, size_t size) { CUDA_CHECK(cudaMalloc(ptr, size)); }
};

struct DeviceDeallocator {
  void operator()(void* ptr) { CUDA_CHECK(cudaFree(ptr)); }
};

struct ManagedAllocator {
  void operator()(void** ptr, size_t size) { CUDA_CHECK(cudaMallocManaged(ptr, size)); }
};

struct HostAllocator {
  void operator()(void** ptr, size_t size) { CUDA_CHECK(cudaMallocHost(ptr, size)); }
};

struct HostDeallocator {
  void operator()(void* ptr) { CUDA_CHECK(cudaFreeHost(ptr)); }
};

using TrtDeviceBuffer = TrtCudaBuffer<DeviceAllocator, DeviceDeallocator>;
using TrtManagedBuffer = TrtCudaBuffer<ManagedAllocator, DeviceDeallocator>;
using TrtHostBuffer = TrtCudaBuffer<HostAllocator, HostDeallocator>;

//!
//! \class MirroredBuffer
//! \brief Coupled host and device buffers
//!
class IMirroredBuffer {
public:
  //!
  //! Allocate memory for the mirrored buffer give the size
  //! of the allocation.
  //!
  virtual void allocate(size_t size) = 0;

  //!
  //! Get the pointer to the device side buffer.
  //!
  //! \return pointer to device memory or nullptr if uninitialized.
  //!
  virtual void* getDeviceBuffer() const = 0;

  //!
  //! Get the pointer to the host side buffer.
  //!
  //! \return pointer to host memory or nullptr if uninitialized.
  //!
  virtual void* getHostBuffer() const = 0;

  //!
  //! Copy the memory from host to device.
  //!
  virtual void hostToDevice(cudaStream_t stream) = 0;

  //!
  //! Copy the memory from device to host.
  //!
  virtual void deviceToHost(cudaStream_t stream) = 0;

  //!
  //! Interface to get the size of the memory
  //!
  //! \return the size of memory allocated.
  //!
  virtual size_t getSize() const = 0;

  //!
  //! Virtual destructor declaraion
  //!
  virtual ~IMirroredBuffer() = default;
}; // class IMirroredBuffer

//!
//! Class to have a separate memory buffer for discrete device and host allocations.
//!
class DiscreteMirroredBuffer : public IMirroredBuffer {
public:
  void allocate(size_t size) override {
    mSize = size;
    mHostBuffer.allocate(size);
    mDeviceBuffer.allocate(size);
  }

  void* getDeviceBuffer() const override { return mDeviceBuffer.get(); }

  void* getHostBuffer() const override { return mHostBuffer.get(); }

  void hostToDevice(cudaStream_t stream) override {
    CUDA_CHECK(cudaMemcpyAsync(mDeviceBuffer.get(), mHostBuffer.get(), mSize, cudaMemcpyHostToDevice, stream));
  }

  void deviceToHost(cudaStream_t stream) override {
    CUDA_CHECK(cudaMemcpyAsync(mHostBuffer.get(), mDeviceBuffer.get(), mSize, cudaMemcpyDeviceToHost, stream));
  }

  size_t getSize() const override { return mSize; }

private:
  size_t mSize{0};
  TrtHostBuffer mHostBuffer;
  TrtDeviceBuffer mDeviceBuffer;
}; // class DiscreteMirroredBuffer

//!
//! Class to have a unified memory buffer for embedded devices.
//!
class UnifiedMirroredBuffer : public IMirroredBuffer {
public:
  void allocate(size_t size) override {
    mSize = size;
    mBuffer.allocate(size);
  }

  void* getDeviceBuffer() const override { return mBuffer.get(); }

  void* getHostBuffer() const override { return mBuffer.get(); }

  void hostToDevice(cudaStream_t stream) override {
    // Does nothing since we are using unified memory.
  }

  void deviceToHost(cudaStream_t stream) override {
    // Does nothing since we are using unified memory.
  }

  size_t getSize() const override { return mSize; }

private:
  size_t mSize{0};
  TrtManagedBuffer mBuffer;
}; // class UnifiedMirroredBuffer

//!
//! Class to allocate memory for outputs with data-dependent shapes. The sizes of those are unknown so pre-allocation is
//! not possible.
//!
class OutputAllocator : public nvinfer1::IOutputAllocator {
public:
  OutputAllocator(IMirroredBuffer* buffer) : mBuffer(buffer) {}

  void* reallocateOutput(char const* tensorName, void* currentMemory, uint64_t size,
                         uint64_t alignment) noexcept override {
    // Some memory allocators return nullptr when allocating zero bytes, but TensorRT requires a non-nullptr
    // even for empty tensors, so allocate a dummy byte.
    size = std::max(size, 1UL);
    if (size > mSize) {
      mBuffer->allocate(roundUp(size, alignment));
      mSize = size;
    }
    return mBuffer->getDeviceBuffer();
  }

  //! IMirroredBuffer does not implement Async allocation, hence this is just a wrap around
  void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment,
                              cudaStream_t /*stream*/) noexcept override {
    return reallocateOutput(tensorName, currentMemory, size, alignment);
  }

  void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override {}

  IMirroredBuffer* getBuffer() { return mBuffer.get(); }

  ~OutputAllocator() override {}

private:
  std::unique_ptr<IMirroredBuffer> mBuffer;
  uint64_t mSize{};
};