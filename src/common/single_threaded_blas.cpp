//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#ifdef DLAF_WITH_OPENMP
#include <omp.h>
#endif
#ifdef DLAF_WITH_MKL
#include <mkl_service.h>
#endif

#include <dlaf/common/single_threaded_blas.h>

#ifdef DLAF_ASSERT_MODERATE_ENABLE
#include <dlaf/common/assert.h>
#endif

namespace dlaf::common::internal {
SingleThreadedBlasScope::SingleThreadedBlasScope() {
#ifdef DLAF_ASSERT_MODERATE_ENABLE
  calling_thread = std::this_thread::get_id();
#endif
#ifdef DLAF_WITH_OPENMP
  omp_num_threads = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
#ifdef DLAF_WITH_MKL
  mkl_num_threads = mkl_set_num_threads_local(1);
#endif
}

SingleThreadedBlasScope::~SingleThreadedBlasScope() {
#ifdef DLAF_ASSERT_MODERATE_ENABLE
  auto current_thread = std::this_thread::get_id();
  DLAF_ASSERT_MODERATE(calling_thread == current_thread, calling_thread, current_thread);
#endif
#ifdef DLAF_WITH_OPENMP
  omp_set_num_threads(omp_num_threads);
#endif
#ifdef DLAF_WITH_MKL
  mkl_set_num_threads_local(mkl_num_threads);
#endif
}
}
