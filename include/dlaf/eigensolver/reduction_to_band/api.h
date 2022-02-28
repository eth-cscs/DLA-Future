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

#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
struct ReductionToBand {
  static common::internal::vector<pika::shared_future<common::internal::vector<T>>> call(
      Matrix<T, D>& mat_a, const SizeType band_size);
  static common::internal::vector<pika::shared_future<common::internal::vector<T>>> call(
      comm::CommunicatorGrid grid, Matrix<T, D>& mat_a);
};

// TODO this is just a placeholder for development purposes
template <class T>
struct ReductionToBand<Backend::GPU, Device::GPU, T> {
  static common::internal::vector<pika::shared_future<common::internal::vector<T>>> call(
      Matrix<T, Device::GPU>&, const SizeType);
};

}
