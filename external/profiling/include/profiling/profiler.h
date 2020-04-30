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
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <unistd.h>

namespace dlaf {
namespace profiling {

using thread_id_t = std::size_t;

namespace internal {

/// Data structure for storing a single profile entry
class profile_section_data {
  using clock_t = std::chrono::steady_clock;

  std::string name_;
  std::string group_;

  /// Snapshot data
  ///
  /// Data structore for storing thread and time information of a specific point in the timeline
  struct {
    thread_id_t id_;
    std::size_t time_;
  } start_data_, end_data_;

public:
  /// Create a snapshot for keeping track of the entry point in a profile section
  void enter(const thread_id_t tid, const std::string& name, const std::string& group) {
    name_ = name;
    group_ = group;
    start_data_ = {tid, get_time()};
  }

  /// Create a snapshot for keeping track of the exit from a profile section
  void leave(const thread_id_t tid) {
    /// TODO a start_data must exists
    end_data_ = {tid, get_time()};
  }

  friend std::ostream& operator<<(std::ostream& os, const profile_section_data& section_data) {
    os << section_data.name_ << ", ";
    os << section_data.group_ << ", ";
    os << section_data.start_data_.id_ << ", ";
    os << section_data.start_data_.time_ << ", ";
    os << section_data.end_data_.id_ << ", ";
    os << section_data.end_data_.time_;
    return os;
  }

private:
  /// @return current time in nanoseconds on the reference clock
  std::size_t get_time() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(clock_t::now().time_since_epoch()).count();
  };
};

}

/// Profiler Manager
///
/// This is a singleton that manages profiling and at program exit it dumps data to a csv file
/// The report file is a CSV whose name is by default "<hostname>_<pid>.csv" and can be changed or
/// resetted at any time. Each line in the report contains data for a single profile section:
/// 1. task_name
/// 2. task_group
/// 3. thread id on which the profile section started
/// 4. start time of the profile section
/// 5. * thread id on which the profile section ended
/// 6. end time of the profile section
///
/// note: thread id is by default obtained as the std::hash of std::this_thread::get_id(), but it can be
/// changed to use any function with the signature "thread_id_t(*)(void)"
class profiler {
public:
  /// Initializes the profiler or return the existing one
  ///
  /// Nothing will be profiled before the first call to this function
  static profiler& instance() {
    static profiler global_;
    return global_;
  }

  ~profiler() {
    std::ofstream profiler_report(output_filename_);

    for (const auto& recorder : recorders_)
      for (const auto& task : recorder.second.tasks)
        profiler_report << task << std::endl;
  }

  /// Return the current thread id as per configuration
  ///
  /// It returns the identifier for the thread. Once the thread identifier is computed with the currently
  /// configured function this identifier is not changed. So, even if the thread identifier function is
  /// changed, a thread which has already recorded a profile section, it will always have the same identifier.
  const thread_id_t& get_thread_id() const noexcept {
    thread_local const thread_id_t tid = thread_id_getter_();
    return tid;
  }

  /// Add an entry to the profiler database
  ///
  /// This is thread-safe since each thread uses its own recorder
  void add(const internal::profile_section_data& task_data) {
    thread_local auto& recorder = thread_local_recorder();
    recorder.tasks.emplace_back(task_data);
  }

  /// Change the output filename for the report with the given one
  ///
  /// Set the filename of the report that will be stored in the working directory
  /// @param output_filename the name of the file without extensions (csv is automatically appended)
  void set_filename(const std::string& output_filename) {
    output_filename_ = output_filename + ".csv";
  }

  /// Reset the filename used for the report file
  ///
  /// The default is <hostname>_<pid>.csv
  void set_filename() {
    std::ostringstream filename;

    const size_t hostname_max_length = 50;
    char hostname[hostname_max_length];
    gethostname(hostname, hostname_max_length);

    filename << hostname << "_" << getpid();
    set_filename(filename.str());
  }

  /// Set the function used to retrieve the thread id during profiling
  ///
  /// This function will be called by each thread during profiling activities
  /// Required signature is "thread_id_t function_name()"
  void set_thread_id_getter(thread_id_t (*thread_id_getter)(void)) {
    thread_id_getter_ = thread_id_getter;
  }

  /// Reset the function used to retrieve thread id during profiling
  ///
  /// By default std::this_thread:::get_id() is used
  void set_thread_id_getter() {
    set_thread_id_getter(
        []() -> thread_id_t { return std::hash<std::thread::id>{}(std::this_thread::get_id()); });
  }

private:
  /// Container for storing recorded tasks
  struct recorder {
    recorder() = default;

    /// disable copy construction and assignment, it is a singleton
    recorder(const recorder&) = delete;
    recorder& operator=(const recorder&) = delete;

    recorder(recorder&&) = default;

    std::deque<internal::profile_section_data> tasks;
  };

  /// Initialize the profile with default settings
  profiler() {
    set_filename();
    set_thread_id_getter();
  }

  // Disable copy construction and assignment, it is a singleton
  profiler(const profiler&) = delete;
  profiler& operator=(const profiler&) = delete;

  /// Helper member function that gives access to the recorder for the current thread
  recorder& thread_local_recorder() {
    std::lock_guard<std::mutex> _(lock_recorders_);

    bool inserted = false;
    decltype(recorders_)::iterator it;
    std::tie(it, inserted) = recorders_.emplace(std::make_pair(get_thread_id(), recorder{}));

    // TODO do we want an assert instead?
    if (!inserted)
      throw std::runtime_error(
          "recorder for the thread already exists. have you called it more than once?");

    return it->second;
  }

  std::mutex lock_recorders_;
  std::map<thread_id_t, recorder> recorders_;

  std::string output_filename_;
  std::function<thread_id_t(void)> thread_id_getter_;
};

/// Helper object to add a profile entry exploiting RAII
struct profile_scope {
  profile_scope(const std::string& task_name, const std::string& task_group) {
    data_.enter(profiler::instance().get_thread_id(), task_name, task_group);
  }

  ~profile_scope() {
    data_.leave(profiler::instance().get_thread_id());

    profiler::instance().add(data_);
  }

private:
  internal::profile_section_data data_;
};

namespace util {

/// @brief Wraps a callable with a SectionScoped object
///
/// Measure the execution time of the callable
/// @return a proxy callable with the same arguments of the given one
template <class Func>
auto time_it(std::string name, std::string group, Func&& target_function) {
  return [name, group, function = std::forward<Func>(target_function)](auto&&... args) {
    profile_scope _(name, group);
    return function(std::forward<decltype(args)>(args)...);
  };
}

}
}
}
