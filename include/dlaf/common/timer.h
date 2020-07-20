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

namespace dlaf {
namespace common {

/// Timer.
///
/// It measures time since its construction on the specified @tparam clock.
template <class clock = std::chrono::high_resolution_clock>
class Timer {
  using time_point = std::chrono::time_point<clock>;

  time_point start_;

  inline time_point now() const {
    return clock::now();
  }

public:
  Timer() : start_(now()) {}

  double elapsed() const {
    using namespace std::chrono;
    return duration_cast<duration<double>>(now() - start_).count();
  }
};

}
}
