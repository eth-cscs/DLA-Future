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

#ifdef DLAF_ASSERT_MODERATE_ENABLE
#include <thread>
#endif

namespace dlaf::common::internal {
class [[nodiscard]] SingleThreadedBlasScope {
public:
  SingleThreadedBlasScope();
  ~SingleThreadedBlasScope();
  SingleThreadedBlasScope(SingleThreadedBlasScope&&) = delete;
  SingleThreadedBlasScope(const SingleThreadedBlasScope&) = delete;
  SingleThreadedBlasScope& operator=(SingleThreadedBlasScope&&) = delete;
  SingleThreadedBlasScope& operator=(const SingleThreadedBlasScope&) = delete;

private:
#ifdef DLAF_ASSERT_MODERATE_ENABLE
  std::thread::id calling_thread;
#endif
#ifdef DLAF_WITH_OPENMP
  int omp_num_threads;
#endif
#ifdef DLAF_WITH_MKL
  int mkl_num_threads;
#endif
};
}
