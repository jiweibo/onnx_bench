#pragma once

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

class StopWatchTimer {
public:
  StopWatchTimer()
      : running_(false), clock_sessions_(0), diff_time_(0), total_time_(0) {}
  virtual ~StopWatchTimer() {}

public:
  // Start time measurement
  void Start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    running_ = true;
  }

  // Stop time measurement
  void Stop() {
    diff_time_ = GetDiffTime();
    total_time_ += diff_time_;
    durations_.push_back(diff_time_);
    running_ = false;
    ++clock_sessions_;
  }

  // Reset time counters to zero. Does not change the timer running state but
  // does recapture this point in time as the current start time if it is
  // running.
  void Reset() {
    diff_time_ = 0;
    total_time_ = 0;
    clock_sessions_ = 0;
    durations_.clear();

    if (running_) {
      start_time_ = std::chrono::high_resolution_clock::now();
    }
  }

  // Time in msec. After start if the stop watch is still running (i.e. there
  // was no call to stop()) then the elapsed time is returned, otherwise the
  // time between the last start() and stop call is returned.
  double GetTime() {
    double retval = total_time_;

    if (running_) {
      retval += GetDiffTime();
    }

    return retval;
  }

  // Mean time to date based on the number of times the stopwatch has been
  // stopped and the current total time
  double GetAverageTime() {
    return (clock_sessions_ > 0) ? (total_time_ / clock_sessions_) : 0.0;
  }

  double ComputeVariance() {
    if (durations_.empty())
      return -1;
    double mean = std::accumulate(durations_.begin(), durations_.end(), 0.0) /
                  durations_.size();
    double sqDiffSum = 0.0;
    for (auto duration : durations_) {
      sqDiffSum += (duration - mean) * (duration - mean);
    }
    return sqDiffSum / (durations_.size() - 1);
  }

  double ComputePercentile(double top) {
    if (durations_.empty())
      return -1;
    std::sort(durations_.begin(), durations_.end());
    if (static_cast<int>(durations_.size() * top) >= durations_.size())
      return durations_.back();
    return durations_[static_cast<int>(durations_.size() * top)];
  }

  std::vector<double> GetDurations() { return durations_; }

private:
  inline double GetDiffTime() {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time_)
        .count();
  }

private:
  bool running_;

  int clock_sessions_;

  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;

  double diff_time_;

  double total_time_;

  std::vector<double> durations_;
};
