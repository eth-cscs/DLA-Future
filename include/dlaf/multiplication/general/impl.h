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

#include "dlaf/multiplication/general/api.h"

#include "dlaf/blas/enum_output.h"
#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/sender/transform.h"
#include "dlaf/sender/when_all_lift.h"

namespace dlaf::multiplication {
namespace internal {

template <Backend B, Device D, class T>
void GeneralSub<B, D, T>::callNN(const SizeType idx_begin, const SizeType idx_end, const blas::Op opA,
                                 const blas::Op opB, const T alpha, Matrix<const T, D>& mat_a,
                                 Matrix<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c) {
  namespace ex = pika::execution::experimental;

  for (SizeType j = idx_begin; j <= idx_end; ++j) {
    for (SizeType i = idx_begin; i <= idx_end; ++i) {
      for (SizeType k = idx_begin; k <= idx_end; ++k) {
        ex::start_detached(dlaf::internal::whenAllLift(opA, opB, alpha,
                                                       mat_a.read_sender(GlobalTileIndex(i, k)),
                                                       mat_b.read_sender(GlobalTileIndex(k, j)),
                                                       k == idx_begin ? beta : T(1),
                                                       mat_c.readwrite_sender(GlobalTileIndex(i, j))) |
                           tile::gemm(dlaf::internal::Policy<B>()));
      }
    }
  }
}
}
}
