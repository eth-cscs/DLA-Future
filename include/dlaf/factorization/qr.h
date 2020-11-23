//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/factorization/qr/mc.h"
#include "dlaf/factorization/qr/t_factor_mc.h"

namespace dlaf {
namespace factorization {

namespace internal {

// Forms the triangular factor T of a block reflector H of order n, which is defined as a product of k
// elementary reflectors.
//
// A Storage-Efficient WY Representation for Products of Householder Transformations.
// Schreiber, Robert & VanLoan, Charles. (1989)
// SIAM Journal on Scientific and Statistical Computing. 10. 10.1137/0910005.
template <Backend backend, Device device, class T>
void computeTFactor(Matrix<T, device>& t, Matrix<const T, device>& a, const LocalTileIndex ai_start_loc,
                    const GlobalTileIndex ai_start, const SizeType k,
                    common::internal::vector<hpx::shared_future<T>> taus,
                    common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  QR_Tfactor<backend, device, T>::call(t, a, ai_start_loc, ai_start, k, taus, serial_comm);
}

}

}
}
