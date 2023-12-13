#pragma once

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <ratio>
#include <thread>
#include <unistd.h>

#include <cuda_runtime_api.h>

class MemoryUse {
public:
  void Start() {
    run_.store(true);
    cpu_mem_thread_ = std::thread(&MemoryUse::GetCpuMemory, this);
    gpu_mem_thread_ = std::thread(&MemoryUse::GetGpuMemory, this);
  }

  std::tuple<size_t, size_t, size_t> GetMemInfo() {
    return {vsz_mem_, rss_mem_, gpu_mem_};
  }

  void Stop() {
    run_.store(false);
    bool cpu_stop = false;
    bool gpu_stop = false;
    while (true) {
      if (cpu_mem_thread_.joinable()) {
        cpu_mem_thread_.join();
        cpu_stop = true;
      }
      if (gpu_mem_thread_.joinable()) {
        gpu_mem_thread_.join();
        gpu_stop = true;
      }
      if (cpu_stop && gpu_stop) {
        break;
      }
    }
  }

private:
  void GetCpuMemory() {
    pid_t pid = getpid();
    while (run_.load()) {
      FILE* fd;
      char line[1024] = {0};
      char virtual_filename[32] = {0};
      char vmrss_name[32] = {0};
      int vmrss_num = 0;
      int vm_size_num = 0;
      sprintf(virtual_filename, "/proc/%d/status", pid);
      fd = fopen(virtual_filename, "r");
      if (fd == NULL) {
        std::cout << "open " << virtual_filename << " failed" << std::endl;
        exit(1);
      }
      for (int i = 0; i < 60; ++i) {
        fgets(line, sizeof(line), fd);
        if (strstr(line, "VmRSS:") != NULL) {
          sscanf(line, "%s %d", vmrss_name, &vmrss_num);
        }
        if (strstr(line, "VmSize:") != NULL) {
          sscanf(line, "%s %d", vmrss_name, &vm_size_num);
        }

        if (vmrss_num != 0 && vm_size_num != 0) {
          break;
        }
      }

      rss_mem_ = rss_mem_ > vmrss_num ? rss_mem_ : vmrss_num;
      vsz_mem_ = vsz_mem_ > vm_size_num ? vsz_mem_ : vm_size_num;
      fclose(fd);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  void GetGpuMemory() {
    while (run_.load()) {
      size_t free_bytes;
      size_t total_bytes;
      auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
      if (status != cudaSuccess) {
        std::cerr << "Error: cudaMemGetInfo fails, "
                  << cudaGetErrorString(status) << std::endl;
        exit(1);
      }

      auto used = total_bytes - free_bytes;
      gpu_mem_ = gpu_mem_ > used ? gpu_mem_ : used;

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

private:
  std::atomic<bool> run_;
  size_t rss_mem_ = 0;
  size_t vsz_mem_ = 0;
  size_t gpu_mem_ = 0;
  std::thread gpu_mem_thread_;
  std::thread cpu_mem_thread_;
};