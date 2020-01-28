//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <chrono>
#include <deque>
#include <fstream>
#include <map>
#include <string>
#include <thread>

namespace dlaf {
namespace profiler {
namespace details {

/// Data structure for storing a single profile entry
class task_data_t {
  using clock_t = std::chrono::steady_clock;

  std::string name_;
  std::string group_;

  /// data structure for the time information
  struct {
    std::thread::id id_;
    std::size_t time_;
  } start_data_, end_data_;

public:
  void enter(const std::string& name, const std::string& group) {
    name_ = name;
    group_ = group;
    start_data_ = {std::this_thread::get_id(), get_time()};
  }

  void leave() {
    end_data_ = {std::this_thread::get_id(), get_time()};
  }

  /// @return value is valid just if the same thread has called enter and leave methods
  std::size_t get_time() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t::now().time_since_epoch())
        .count();
  };

  friend std::ostream& operator<<(std::ostream& os, const task_data_t& task_data) {
    os << task_data.name_ << ", ";
    os << task_data.group_ << ", ";
    os << task_data.start_data_.id_ << ", ";
    os << task_data.start_data_.time_ << ", ";
    os << task_data.end_data_.id_ << ", ";
    os << task_data.end_data_.time_;
    return os;
  }
};

}

/// Global profiler manager
///
/// On destruction it dumps on a csv file all tasks
class Manager {
public:
  static Manager& get_global_profiler() {
    static Manager global_;
    return global_;
  }

  ~Manager() {
    std::ofstream profiler_report("report.csv");  // TODO adapt for multiple nodes

    for (const auto& recorder : recorders_)
      for (const auto& task : recorder.second.tasks)
        profiler_report << task << std::endl;
  }

  void add(const details::task_data_t& task_data) {
    thread_local std::thread::id tid = std::this_thread::get_id();
    recorders_[tid].tasks.emplace_back(task_data);
  }

private:
  /// Container for task entries (it is NOT thread-safe)
  struct ThreadLocalRecorder {
    std::deque<details::task_data_t> tasks;
  };

  std::map<std::thread::id, ThreadLocalRecorder> recorders_;
};

/// Helper object to add a profile entry exploiting RAII
struct SectionScoped {
  SectionScoped(const std::string& task_name, const std::string& task_group) {
    data_.enter(task_name, task_group);
  }

  ~SectionScoped() {
    data_.leave();

    Manager::get_global_profiler().add(data_);
  }

private:
  details::task_data_t data_;
};

namespace util {

/// @brief Wraps a callable with a SectionScoped object
///
/// Measure the execution time of the callable
/// @return a proxy callable with the same arguments of the given one
template <class Func>
auto time_it(std::string name, std::string group, Func&& target_function) {
  return [name, group, function = std::forward<Func>(target_function)](auto&&... args) {
    SectionScoped _(name, group);
    return function(std::forward<decltype(args)>(args)...);
  };
}

}

}
}
