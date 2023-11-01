//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <vector>

#include <dlaf/common/assert.h>

namespace dlaf {
namespace common {

template <class T>
class RoundRobin {
public:
  RoundRobin() = default;

  template <class... Args>
  RoundRobin(std::size_t n, Args... args) {
    pool_.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
      pool_.emplace_back(args...);
  }

  RoundRobin(RoundRobin&&) = default;
  RoundRobin(const RoundRobin&) = delete;
  RoundRobin& operator=(RoundRobin&&) = default;
  RoundRobin& operator=(const RoundRobin&) = delete;

  T& currentResource() {
    DLAF_ASSERT_HEAVY(curr_index_ < pool_.size(), curr_index_, pool_.size());
    return pool_[curr_index_];
  }

  T& nextResource() {
    DLAF_ASSERT_HEAVY(!pool_.empty(), "");
    curr_index_ = (curr_index_ + 1) % pool_.size();
    DLAF_ASSERT_HEAVY(curr_index_ < pool_.size(), curr_index_, pool_.size());
    return pool_[curr_index_];
  }

  std::size_t size() const noexcept {
    return pool_.size();
  }

private:
  std::size_t curr_index_ = 0;
  std::vector<T> pool_;
};

}
}
