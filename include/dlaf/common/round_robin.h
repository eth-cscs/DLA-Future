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

#include <vector>

namespace dlaf {
namespace common {

template <class T>
struct RoundRobin {
  template <class... Args>
  RoundRobin(std::size_t n, Args&&... args) : next_index_(0) {
    for (std::size_t i = 0; i < n; ++i)
      pool_.emplace_back(std::forward<Args>(args)...);
  }

  T& nextResource() {
    auto idx = (next_index_ + 1) % pool_.size();
    std::swap(idx, next_index_);
    return pool_[idx];
  }

  std::size_t next_index_;
  std::vector<T> pool_;
};

}
}
