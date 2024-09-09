#include "core/core.h"
#include "utils/util.h"

#include "gtest/gtest.h"
#include <cstdint>

#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(Core, TestBuffer) {
  core::DeviceBuffer device_buffer;
  device_buffer.Allocate(1);
  EXPECT_EQ(device_buffer.Bytes(), 1);
  EXPECT_NE(device_buffer.Data(), nullptr);
  device_buffer.Reset();
  EXPECT_EQ(device_buffer.Bytes(), 0);
  EXPECT_EQ(device_buffer.Data(), nullptr);
}

TEST(Core, TestDims) {
  core::Dims dims{};
  EXPECT_EQ(dims.num_dims, 0);
  EXPECT_EQ(dims.Numel(), 0);
  EXPECT_EQ(core::GetBytes(dims, core::DataType::kFLOAT, true), 1);
  EXPECT_EQ(core::GetBytes(dims, core::DataType::kFLOAT, false), 0);

  core::Dims dims2{4, 8};
  EXPECT_EQ(dims2.num_dims, 2);
  EXPECT_EQ(dims2.Numel(), 32);
  EXPECT_EQ(core::GetBytes(dims2, core::DataType::kFLOAT), 128);
  auto dims2_vec = dims2.ToStdVec<int32_t>();
  EXPECT_EQ(dims2_vec.size(), 2UL);
  EXPECT_EQ(dims2_vec[0], 4);
  EXPECT_EQ(dims2_vec[1], 8);
  EXPECT_EQ(sizeof(dims2_vec[0]), 4);
  auto dims2_int64_vec = dims2.ToStdVec<int64_t>();
  EXPECT_EQ(dims2_int64_vec.size(), 2UL);
  EXPECT_EQ(dims2_int64_vec[0], 4);
  EXPECT_EQ(dims2_int64_vec[1], 8);
  EXPECT_EQ(sizeof(dims2_int64_vec[0]), 8);
}

TEST(Core, TestTensor) {
  core::Tensor tensor{core::Dims{}};
  EXPECT_EQ(tensor.GetDims().Numel(), 0);
  EXPECT_EQ(tensor.GetDataType(), core::DataType::kFLOAT);
  EXPECT_EQ(tensor.HostData(), nullptr);
  EXPECT_EQ(tensor.DeviceData(), nullptr);

  /// Case1: Maintains host_buffer, and device_buffer is nullptr.
  core::Tensor tensor_host_owned{core::Dims{}, core::DataType::kFLOAT};
  tensor_host_owned.Resize(core::Dims{4, 8});
  CHECK_EQ(tensor_host_owned.DeviceData(), nullptr);
  float* host_ptr = tensor_host_owned.HostData<float>();
  for (size_t i = 0; i < tensor_host_owned.Numel(); ++i) {
    host_ptr[i] = i;
  }
  for (size_t i = 0; i < tensor_host_owned.Numel(); i += 1) {
    EXPECT_NEAR(tensor_host_owned.HostData<float>()[i], i, 1e-5);
  }
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  tensor_host_owned.HostToDevice(stream);
  float* device_data = tensor_host_owned.DeviceData<float>();
  std::vector<float> tmp_data(tensor_host_owned.Numel());
  CUDA_CHECK(cudaMemcpyAsync(tmp_data.data(), device_data, tensor_host_owned.Bytes(), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < tmp_data.size(); i += 1) {
    EXPECT_NEAR(tmp_data[i], i, 1e-5);
  }

  /// Case2: Maintains device_buffer, and host_buffer is nullptr.
  core::Tensor tensor_device_owned{core::Dims{}, core::DataType::kFLOAT, core::Location::kDEVICE, 0};
  tensor_device_owned.Resize(core::Dims{4, 8});
  CHECK_EQ(tensor_device_owned.HostData(), nullptr);
  CUDA_CHECK(cudaMemcpyAsync(tensor_device_owned.DeviceData(), tmp_data.data(), tensor_device_owned.Bytes(),
                             cudaMemcpyHostToDevice, stream));
  tensor_device_owned.DeviceToHost(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int64_t i = 0; i < tensor_device_owned.Numel(); ++i) {
    EXPECT_NEAR(tensor_device_owned.HostData<float>()[i], tmp_data[i], 1e-5);
  }

  /// case3: external host data.
  core::Tensor tensor_ext_host(tensor_host_owned.HostData(), tensor_host_owned.Bytes(), tensor_host_owned.GetDims(),
                               tensor_host_owned.GetDataType());
  EXPECT_EQ(tensor_ext_host.DeviceData(), nullptr);
  CHECK_NOTNULL(tensor_ext_host.HostData());
  tensor_ext_host.HostToDevice(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK_NOTNULL(tensor_ext_host.DeviceData());
  EXPECT_DEATH(tensor_ext_host.Resize(core::Dims{8, 4}), "Not allowed call Resize when use external data");

  /// case4: external device data.
  core::Tensor tensor_ext_device(tensor_device_owned.DeviceData(), tensor_device_owned.Bytes(),
                                 tensor_device_owned.GetDims(), tensor_device_owned.GetDataType(),
                                 core::Location::kDEVICE, 0);
  CHECK_EQ(tensor_ext_device.HostData(), nullptr);
  CHECK_NOTNULL(tensor_ext_device.DeviceData());
  tensor_ext_device.DeviceToHost(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CHECK_NOTNULL(tensor_ext_device.HostData());
  for (int64_t i = 0; i < tensor_ext_device.Numel(); ++i) {
    EXPECT_NEAR(tensor_ext_device.HostData<float>()[i], i, 1e-5);
  }
  EXPECT_DEATH(tensor_ext_device.Resize(core::Dims{2, 16}), "Not allowed call Resize when use external data");
}