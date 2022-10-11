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

#ifdef DLAF_WITH_MKL
#include <mkl_service.h>
#endif
#include <omp.h>

// TODO: internal?
namespace dlaf::common::internal {
class [[nodiscard]] SingleThreadedOmpScope {
public:
  SingleThreadedOmpScope()
      : omp_num_threads(omp_get_max_threads())
#ifdef DLAF_WITH_MKL
        ,
        mkl_num_threads(mkl_set_num_threads_local(1))
#endif
  {
    omp_set_num_threads(1);
  }
  ~SingleThreadedOmpScope() {
    omp_set_num_threads(omp_num_threads);
#ifdef DLAF_WITH_MKL
    mkl_set_num_threads_local(mkl_num_threads);
#endif
  }

  SingleThreadedOmpScope(SingleThreadedOmpScope &&) = delete;
  SingleThreadedOmpScope(SingleThreadedOmpScope const&) = delete;
  SingleThreadedOmpScope& operator=(SingleThreadedOmpScope&&) = delete;
  SingleThreadedOmpScope& operator=(SingleThreadedOmpScope const&) = delete;

private:
  int omp_num_threads;
#ifdef DLAF_WITH_MKL
  int mkl_num_threads;
#endif
};
}
