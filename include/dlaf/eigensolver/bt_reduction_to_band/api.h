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

template <Backend backend, Device device, class T>
struct BackTransformationReductionToBand {
  static void call(Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
                   common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus);

  static void call(comm::CommunicatorGrid grid, Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
                   common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus);
};

}
