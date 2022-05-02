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

template <Device D, class T>
void GeneralSub<D, T>::call(const SizeType a, const SizeType b, const blas::Op opA, const blas::Op opB,
                            const T alpha, Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b,
                            const T beta, Matrix<T, D>& mat_c) {
  namespace ex = pika::execution::experimental;

  if (opA != blas::Op::NoTrans)
    DLAF_UNIMPLEMENTED(opA);
  if (opB != blas::Op::NoTrans)
    DLAF_UNIMPLEMENTED(opB);

  for (SizeType j = a; j <= b; ++j) {
    for (SizeType i = a; i <= b; ++i) {
      for (SizeType k = a; k <= b; ++k) {
        dlaf::internal::whenAllLift(opA, opB, alpha, mat_a.read_sender(GlobalTileIndex(i, k)),
                                    mat_b.read_sender(GlobalTileIndex(k, j)), k == a ? beta : T(1),
                                    mat_c.readwrite_sender(GlobalTileIndex(i, j))) |
            tile::gemm(dlaf::internal::Policy<Backend::MC>()) | ex::start_detached();
      }
    }
  }
}

}
}
