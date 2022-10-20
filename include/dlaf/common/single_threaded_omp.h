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

// TODO: internal?
namespace dlaf::common::internal {
class [[nodiscard]] SingleThreadedOmpScope {
public:
  SingleThreadedOmpScope();
  ~SingleThreadedOmpScope();
  SingleThreadedOmpScope(SingleThreadedOmpScope &&) = delete;
  SingleThreadedOmpScope(SingleThreadedOmpScope const&) = delete;
  SingleThreadedOmpScope& operator=(SingleThreadedOmpScope&&) = delete;
  SingleThreadedOmpScope& operator=(SingleThreadedOmpScope const&) = delete;

private:
#ifdef DLAF_WITH_OPENMP
  int omp_num_threads;
#endif
#ifdef DLAF_WITH_MKL
  int mkl_num_threads;
#endif
};
}
