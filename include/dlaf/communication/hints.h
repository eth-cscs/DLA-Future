//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/execution.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/runtime_local/thread_pool_helpers.hpp>

#include "dlaf/common/assert.h"

namespace dlaf {
namespace comm {

inline std::atomic<bool>* get_hints_mask() {
  using hints_arr_t = std::unique_ptr<std::atomic<bool>[]>;
  static hints_arr_t hints_mask = []() {
    std::size_t nthreads = hpx::resource::get_num_threads();
    hints_arr_t hints(new std::atomic<bool>[nthreads]);
    for (int i = 0; i < nthreads; ++i) {
      hints[i].store(true);
    }
    return hints;
  }();
  return hints_mask.get();
}

inline int get_free_thread_index(const std::string& pool_name) {
  int thread_offset = 0;
  for (int i_pool = 0; i_pool < hpx::resource::get_pool_index(pool_name); ++i_pool) {
    thread_offset += hpx::resource::get_num_threads(i_pool);
  };

  std::atomic<bool>* hints_mask = get_hints_mask();
  for (int i_thd = 0; i_thd < hpx::resource::get_num_threads(pool_name); ++i_thd) {
    int index = i_thd + thread_offset;
    if (hints_mask[index].load()) {
      hints_mask[index].store(false);
      return index;
    }
  }
  return -1;
}

inline bool is_stealing_enabled(const std::string& pool_name) {
  return hpx::resource::get_thread_pool(pool_name).get_scheduler()->has_scheduler_mode(
      hpx::threads::policies::scheduler_mode(
          hpx::threads::policies::scheduler_mode::enable_stealing |
          hpx::threads::policies::scheduler_mode::enable_stealing_numa));
}

class hint_manager {
  int index_;

public:
  hint_manager() : index_(-1) {}
  hint_manager(const std::string& pool_name) {
    using hpx::resource::get_num_threads;
    // Assert that the pool has task stealing disabled
    DLAF_ASSERT(!is_stealing_enabled(pool_name) || get_num_threads(pool_name) == 1, pool_name);
    hpx::util::yield_while([this, &pool_name] {
      index_ = get_free_thread_index(pool_name);
      return index_ == -1;
    });
  }
  hint_manager(const hint_manager& o) = default;
  hint_manager& operator=(const hint_manager& o) = default;
  hint_manager& operator=(hint_manager&& o) {
    index_ = o.index_;
    o.index_ = -1;
    return *this;
  }
  hint_manager(hint_manager&& o) {
    *this = std::move(o);
  }
  ~hint_manager() {
    if (index_ != -1) {
      get_hints_mask()[index_].store(true);
    }
  }

  int get_thread_index() const {
    return index_;
  }
};

}
}
