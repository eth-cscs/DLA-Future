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

#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"
//
#include "hpx/hpx.hpp"
#include "hpx/include/parallel_executors.hpp"
#include "hpx/include/threads.hpp"

/// @file

namespace dlaf {

/// @brief Cholesky implementation on local memory

template <class T>
void cholesky_local(Matrix<T, Device::CPU>& mat) {
  // Setup executors for different task priorities on the default (matrix) pool
  hpx::threads::scheduled_executor matrix_HP_executor =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  hpx::threads::scheduled_executor matrix_normal_executor =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

  // Check if matrix is square
  util_matrix::check_size_square(mat, "Cholesky", "mat");
  // Check if block matrix is square
  util_matrix::check_blocksize_square(mat, "Cholesky", "mat");

  // Define tile index
  LocalTileIndex index;
}

}
