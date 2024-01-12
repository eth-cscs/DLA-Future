//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/common/single_threaded_blas.h>

#ifdef DLAF_WITH_OPENMP
#include <omp.h>
#endif
#ifdef DLAF_WITH_MKL
#include <mkl_service.h>
#endif

#include <gtest/gtest.h>

#ifdef DLAF_WITH_MKL
TEST(SingleThreadedBlas, MKL) {
  // Set the number of threads to something bigger than 1 so that we can
  // actually observe if SingleThreadedBlasScope has an effect.
  constexpr int num_threads = 2;

  mkl_set_num_threads(num_threads);

  // If MKL is sequential mkl_set_num_threads will have no effect
  bool mkl_set_num_threads_applied = mkl_get_max_threads() == 2;

  {
    dlaf::common::internal::SingleThreadedBlasScope single;
    EXPECT_EQ(mkl_get_max_threads(), 1);
  }

  if (mkl_set_num_threads_applied) {
    EXPECT_EQ(mkl_get_max_threads(), num_threads);
  }
}
#endif

#ifdef DLAF_WITH_OPENMP
TEST(SingleThreadedBlas, OpenMP) {
  // Set the number of threads to something bigger than 1 so that we can
  // actually observe if SingleThreadedBlasScope has an effect.
  constexpr int num_threads = 2;

  omp_set_num_threads(num_threads);

  {
    dlaf::common::internal::SingleThreadedBlasScope single;
    EXPECT_EQ(omp_get_max_threads(), 1);
  }

  EXPECT_EQ(omp_get_max_threads(), num_threads);
}
#endif
