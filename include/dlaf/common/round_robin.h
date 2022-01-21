//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <vector>

namespace dlaf {
namespace common {

template <class T>
struct RoundRobin {
  template <class... Args>
  RoundRobin(std::size_t n, Args... args) : curr_index_(0) {
    pool_.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
      pool_.emplace_back(args...);
  }

  T& nextResource() {
    curr_index_ = (curr_index_ + 1) % pool_.size();
    return pool_[curr_index_];
  }

  std::size_t curr_index_;
  std::vector<T> pool_;
};

}
}
