#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <sys/types.h>
#include <vector>
#include <iostream>

#include <nvml.h>
#include <unistd.h>

#ifndef NVML_CHECK
#define NVML_CHECK(call)                                                       \
  {                                                                            \
    auto status = static_cast<nvmlReturn_t>(call);                             \
    if (status != NVML_SUCCESS) {                                              \
      fprintf(stderr,                                                          \
              "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "     \
              "with %s (%d).\n",                                               \
              #call, __LINE__, __FILE__, nvmlErrorString(status), status);     \
    }                                                                          \
  }
#endif // NVML_CHECK

struct NvmlStats {
  uint32_t temperature;
  uint32_t power_usage;
  uint32_t sm_clock;
  uint32_t memory_clock;
  nvmlUtilization_t utilization;
  nvmlMemory_t memory;
  nvmlPstates_t performance_stat;

  uint32_t cur_pcie_link_gen;
  uint32_t cur_pcie_link_width;
  uint32_t pcie_speed;
  uint32_t pcie_throughput;

  uint32_t mem_bus_width;

  int cuda_version;
  char driver_version[80];
  char nvml_version[80];
};

class NVMLWrapper {
public:
  explicit NVMLWrapper(int device_id) {
    NVML_CHECK(nvmlInit());
    NVML_CHECK(nvmlDeviceGetHandleByIndex(device_id, &device_));
  }

  NvmlStats GetNvmlStats() {
    NvmlStats stat;

    NVML_CHECK(nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU,
                                        &stat.temperature));
    NVML_CHECK(nvmlDeviceGetPowerUsage(device_, &stat.power_usage));
    NVML_CHECK(nvmlDeviceGetClock(device_, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT,
                                  &stat.sm_clock));
    NVML_CHECK(nvmlDeviceGetClock(device_, NVML_CLOCK_MEM,
                                  NVML_CLOCK_ID_CURRENT, &stat.memory_clock));
    NVML_CHECK(nvmlDeviceGetUtilizationRates(device_, &stat.utilization));
    NVML_CHECK(nvmlDeviceGetMemoryInfo(device_, &stat.memory));
    NVML_CHECK(nvmlDeviceGetPerformanceState(device_, &stat.performance_stat));

    NVML_CHECK(nvmlDeviceGetCurrPcieLinkGeneration(device_, &stat.cur_pcie_link_gen));
    NVML_CHECK(nvmlDeviceGetCurrPcieLinkWidth(device_, &stat.cur_pcie_link_width));
    NVML_CHECK(nvmlDeviceGetPcieSpeed(device_, &stat.pcie_speed));

    NVML_CHECK(nvmlDeviceGetMemoryBusWidth(device_, &stat.mem_bus_width));
    

    NVML_CHECK(nvmlSystemGetCudaDriverVersion_v2(&stat.cuda_version));
    NVML_CHECK(nvmlSystemGetDriverVersion(stat.driver_version, 80));
    NVML_CHECK(nvmlSystemGetNVMLVersion(stat.nvml_version, 80));
    return stat;
  }

  double GetProcessMemory() {
    pid_t pid = getpid();
    // std::cout << "pid is " << pid << std::endl;
    uint32_t process_count = 0;
    auto status = nvmlDeviceGetComputeRunningProcesses_v2(device_, &process_count, nullptr);
    if (status == NVML_ERROR_INSUFFICIENT_SIZE || status == NVML_SUCCESS) {}
    // std::cout << "process_count " << process_count << std::endl;
    std::vector<nvmlProcessInfo_v1_t> processes(process_count);
    NVML_CHECK(nvmlDeviceGetComputeRunningProcesses(device_, &process_count,
                                                    processes.data()));
    for (auto& process : processes) {
      std::cout << process.pid << ", " << process.usedGpuMemory / 1024 / 1024. << std::endl;
      if (process.pid == pid) {
        return process.usedGpuMemory / 1024 / 1024.;
      }
    }
    return -1;
  }

  //
  // System Queries
  //

  int GetCudaDriverVersion() {
    int cuda_driver;
    NVML_CHECK(nvmlSystemGetCudaDriverVersion_v2(&cuda_driver));
    return cuda_driver;
  }

  std::string GetDriverVersion() {
    char version[80];
    NVML_CHECK(nvmlSystemGetDriverVersion(version, 80));
    return std::string(version);
  }

  std::string GetNvmlVersion() {
    char version[80];
    NVML_CHECK(nvmlSystemGetNVMLVersion(version, 80));
    return std::string(version);
  }

  //
  // Device Queries
  //

  uint32_t GetTemperature() {
    uint32_t temp;
    NVML_CHECK(nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &temp));
    return temp;
  }

  ~NVMLWrapper() { NVML_CHECK(nvmlShutdown()); }

private:
  nvmlDevice_t device_;
};