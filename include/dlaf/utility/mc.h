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

#include <blas.hh>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"
#include "dlaf/utility/internal.h"

namespace dlaf {

template <>
struct Utility<Backend::MC> {
  /// Compute @p norm_type of the distribtued Matrix @param A
  ///
  /// @param A.size().rows() >= 0, if @p uplo == blas::Uplo::Lower A.size().rows() >= A.size().cols()
  /// @param A.size().cols() >= 0, if @p uplo == blas::Uplo::Upper A.size().cols() >= A.size().rows()
  /// @return 0 if A.size().isEmpty()
  template <class T>
  static dlaf::BaseType<T> norm(comm::CommunicatorGrid grid, lapack::Norm norm_type, blas::Uplo uplo,
                                Matrix<const T, Device::CPU>& A);
};

}

#include "dlaf/utility/norm/mc.tpp"

/// ---- ETI
namespace dlaf {

#define DLAF_NORM_MAX_ETI(KWORD, DATATYPE)                                               \
  KWORD template dlaf::BaseType<DATATYPE>                                                \
  Utility<Backend::MC>::norm<DATATYPE>(comm::CommunicatorGrid, lapack::Norm, blas::Uplo, \
                                       Matrix<const DATATYPE, Device::CPU>&);

DLAF_NORM_MAX_ETI(extern, float)
DLAF_NORM_MAX_ETI(extern, double)
DLAF_NORM_MAX_ETI(extern, std::complex<float>)
DLAF_NORM_MAX_ETI(extern, std::complex<double>)
}
