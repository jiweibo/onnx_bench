#pragma once

#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>

#include <cuda_runtime.h>

class MemoryUse {
public:
  MemoryUse(int device_id = 0, bool to_file = false, const std::string& output_file = "memory_usage.csv",
            int period_ms = 100)
      : run_(true), to_file_(to_file), output_file_(output_file), period_ms_(period_ms), max_rss_mem_(0),
        max_vsz_mem_(0), max_gpu_mem_(0) {

    if (to_file_) {
      output_stream_.open(output_file_, std::ios::out | std::ios::app);
      if (!output_stream_.is_open()) {
        std::cerr << "Error: Unable to open file " << output_file_ << std::endl;
        exit(1);
      }
      output_stream_ << "Timestamp, VmRSS(KB), VmSize(KB), GPU Memory(MB)\n";
    }

    monitor_thread_ = std::thread(&MemoryUse::MonitorMemoryUsage, this);
  }

  ~MemoryUse() {
    Stop();
    PrintMaxMemInfo();
  }

private:
  void MonitorMemoryUsage() {
    pid_t pid = getpid();
    std::string filename = "/proc/" + std::to_string(pid) + "/status";
    size_t prev_rss_mem = 0, prev_vsz_mem = 0, prev_gpu_mem = 0;

    while (run_.load()) {
      std::ifstream file(filename);
      if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        exit(1);
      }

      size_t vmrss = 0, vmsize = 0;
      std::string line;
      while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        size_t value;
        std::string unit;
        iss >> key >> value >> unit;
        if (key == "VmRSS:") {
          vmrss = value;
        } else if (key == "VmSize:") {
          vmsize = value;
        }
        if (vmrss && vmsize) {
          break;
        }
      }

      file.close();

      {
        std::lock_guard<std::mutex> lock(data_mutex_);
        max_rss_mem_ = std::max(max_rss_mem_, vmrss);
        max_vsz_mem_ = std::max(max_vsz_mem_, vmsize);
      }

      size_t free_bytes, total_bytes;
      auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
      if (status != cudaSuccess) {
        std::cerr << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(status) << std::endl;
        exit(1);
      }

      {
        std::lock_guard<std::mutex> lock(data_mutex_);
        max_gpu_mem_ = std::max(max_gpu_mem_, total_bytes - free_bytes);
      }
      RecordMemInfo();

      std::this_thread::sleep_for(std::chrono::milliseconds(period_ms_));
    }
  }

  void Stop() {
    run_.store(false);
    if (monitor_thread_.joinable()) {
      monitor_thread_.join();
    }
  }

  void RecordMemInfo() {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S");

    std::lock_guard<std::mutex> lock(data_mutex_);

    std::stringstream output;
    output << timestamp.str() << ", " << max_rss_mem_ << ", " << max_vsz_mem_ << ", " << max_gpu_mem_ / 1024.0 / 1024.0
           << "\n";

    if (to_file_) {
      output_stream_ << output.str();
    }
  }

  void PrintMaxMemInfo() {
    std::cout << "Max CPU Memory (VmRSS): " << max_rss_mem_ << " KB\n";
    std::cout << "Max Virtual Memory (VmSize): " << max_vsz_mem_ << " KB\n";
    std::cout << "Max GPU Memory: " << max_gpu_mem_ / 1024.0 / 1024.0 << " MB\n";
    if (to_file_) {
      std::cout << "Memory usage data has been written to " << output_file_ << std::endl;
    }
  }

private:
  std::atomic<bool> run_;
  size_t max_rss_mem_ = 0;
  size_t max_vsz_mem_ = 0;
  size_t max_gpu_mem_ = 0;
  std::thread monitor_thread_;
  bool to_file_;
  std::string output_file_;
  int period_ms_;
  std::ofstream output_stream_;
  std::mutex data_mutex_;
};