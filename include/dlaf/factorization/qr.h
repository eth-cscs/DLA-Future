//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
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
void computeTFactor(const SizeType k, Matrix<const T, device>& v, const GlobalTileIndex v_start,
                    hpx::shared_future<common::internal::vector<T>> taus,
                    hpx::future<matrix::Tile<T, device>> t,
                    common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  QR_Tfactor<backend, device, T>::call(k, v, v_start, taus, std::move(t), serial_comm);
}
}

}
}
