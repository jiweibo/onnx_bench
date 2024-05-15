#pragma once

#include <condition_variable>

class Barrier {
public:
  explicit Barrier(size_t cnt) : threshold_(cnt), count_(cnt), generation_(0) {}

  void Wait() {
    std::unique_lock<std::mutex> lock(mu_);
    auto lgen = generation_;
    if (--count_ == 0) {
      generation_++;
      count_ = threshold_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, lgen] { return lgen != generation_; });
    }
  }

private:
  std::mutex mu_;
  std::condition_variable cv_;
  const size_t threshold_;
  size_t count_;
  size_t generation_;
};
