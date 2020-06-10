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

#include "dlaf/auxiliary/internal.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/types.h"

namespace dlaf {

template <>
struct Auxiliary<Backend::MC> {
  /// Compute the norm @p norm_type of the distributed Matrix @p A (ge/sy/he)
  ///
  /// - With @p norm_type == lapack::Norm::Max:
  ///   - With @p uplo == blas::uplo::Lower
  ///   @pre @p A must be square matrix, A.size().rows() == A.size().cols()
  ///   - With @p uplo == blas::uplo::Upper
  ///   @note not yet implemented
  ///   - With @p uplo == blas::uplo::General
  ///   @note not yet implemented
  /// - With @p norm_type = lapack::Norm::{One, Two, Inf, Fro}
  /// @note not yet implemented
  ///
  /// .
  /// @pre `A.blockSize().rows() == A.blockSize().cols()`
  /// @pre @p A is distributed according to @p grid
  /// @return the norm @p norm_type of the Matrix @p A or 0 if `A.size().isEmpty()` (see LAPACK doc for
  /// additional info)
  template <class T>
  static dlaf::BaseType<T> norm(comm::CommunicatorGrid grid, comm::Index2D rank, lapack::Norm norm_type,
                                blas::Uplo uplo, Matrix<const T, Device::CPU>& A);
};

}

#include "dlaf/auxiliary/norm/mc.tpp"

/// ---- ETI
namespace dlaf {

#define DLAF_NORM_MAX_ETI(KWORD, DATATYPE)                                                    \
  KWORD template dlaf::BaseType<DATATYPE>                                                     \
  Auxiliary<Backend::MC>::norm<DATATYPE>(comm::CommunicatorGrid, comm::Index2D, lapack::Norm, \
                                         blas::Uplo, Matrix<const DATATYPE, Device::CPU>&);

DLAF_NORM_MAX_ETI(extern, float)
DLAF_NORM_MAX_ETI(extern, double)
DLAF_NORM_MAX_ETI(extern, std::complex<float>)
DLAF_NORM_MAX_ETI(extern, std::complex<double>)
}
