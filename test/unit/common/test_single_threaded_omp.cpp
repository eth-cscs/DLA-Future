//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/single_threaded_omp.h"

#ifdef DLAF_WITH_OPENMP
#include <omp.h>
#endif
#ifdef DLAF_WITH_MKL
#include <mkl_service.h>
#endif

#include <gtest/gtest.h>

TEST(SingleThreadedOmp, Basic) {
  // Set the number of threads to something bigger than 1 so that we can
  // actually observe if SingleThreadedOmpScope has an effect.
  [[maybe_unused]] constexpr int num_threads = 2;
#ifdef DLAF_WITH_OPENMP
  omp_set_num_threads(num_threads);
#endif
#ifdef DLAF_WITH_MKL
  mkl_set_num_threads(num_threads);
  // If MKL is sequential mkl_set_num_threads will have no effect
  bool mkl_set_num_threads_applied = mkl_get_max_threads() == 2;
#endif
  {
    dlaf::common::internal::SingleThreadedOmpScope omp;
#ifdef DLAF_WITH_OPENMP
    EXPECT_EQ(omp_get_max_threads(), 1);
#endif
#ifdef DLAF_WITH_MKL
    EXPECT_EQ(mkl_get_max_threads(), 1);
#endif
  }

#ifdef DLAF_WITH_OPENMP
  EXPECT_EQ(omp_get_max_threads(), num_threads);
#endif
#ifdef DLAF_WITH_MKL
  if (mkl_set_num_threads_applied) {
    EXPECT_EQ(mkl_get_max_threads(), num_threads);
  }
#endif
}
