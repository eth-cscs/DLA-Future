//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
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

namespace dlaf::common::internal {
SingleThreadedBlasScope::SingleThreadedBlasScope() {
#ifdef DLAF_WITH_OPENMP
  omp_num_threads = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
#ifdef DLAF_WITH_MKL
  mkl_num_threads = mkl_set_num_threads_local(1);
#endif
}

SingleThreadedBlasScope::~SingleThreadedBlasScope() {
#ifdef DLAF_WITH_OPENMP
  omp_set_num_threads(omp_num_threads);
#endif
#ifdef DLAF_WITH_MKL
  mkl_set_num_threads_local(mkl_num_threads);
#endif
}
}
