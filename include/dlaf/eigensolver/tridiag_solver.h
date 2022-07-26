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

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/tridiag_solver/mc.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// Finds the eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// @param mat_trd  [in/out] (n x 2) local matrix with the diagonal and off-diagonal of the symmetric
/// tridiagonal matrix in the first column and second columns respectively. The last entry of the second
/// column is not used. On exit the eigenvalues are saved in the first column.
//
/// @param d [out] (n x 1) local matrix holding the eigenvalues of the the symmetric tridiagonal mat_trd
//
/// @param mat_ev [out]    (n x n) local matrix holding the eigenvectors of the the symmetric tridiagonal
/// matrix on exit.
///
/// @pre mat_trd and mat_ev are local matrices
/// @pre mat_trd has 2 columns
/// @pre mat_ev is a square matrix
/// @pre mat_ev has a square block size
template <Backend backend, Device device, class T>
void tridiagSolver(Matrix<T, device>& mat_trd, Matrix<T, device>& d, Matrix<T, device>& mat_ev) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "tridagSolver accepts only real values (float, double)!");

  DLAF_ASSERT(matrix::local_matrix(mat_trd), mat_trd);
  DLAF_ASSERT(mat_trd.distribution().size().cols() == 2, mat_trd);

  DLAF_ASSERT(matrix::local_matrix(d), d);
  DLAF_ASSERT(d.distribution().size().cols() == 1, d);

  DLAF_ASSERT(matrix::local_matrix(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_size(mat_ev), mat_ev);
  DLAF_ASSERT(matrix::square_blocksize(mat_ev), mat_ev);

  DLAF_ASSERT(mat_trd.distribution().blockSize().rows() == mat_ev.distribution().blockSize().rows(),
              mat_ev.distribution().blockSize().rows(), mat_trd.distribution().blockSize().rows());
  DLAF_ASSERT(mat_trd.distribution().blockSize().rows() == d.distribution().blockSize().rows(),
              mat_trd.distribution().blockSize().rows(), d.distribution().blockSize().rows());
  DLAF_ASSERT(mat_trd.distribution().size().rows() == mat_ev.distribution().size().rows(),
              mat_ev.distribution().size().rows(), mat_trd.distribution().size().rows());
  DLAF_ASSERT(mat_trd.distribution().size().rows() == d.distribution().size().rows(),
              mat_trd.distribution().size().rows(), d.distribution().size().rows());

  internal::TridiagSolver<backend, device, T>::call(mat_trd, d, mat_ev);
}

// Overload which provides the eigenvector matrix as complex values where the imaginery part is set to zero.
template <Backend backend, Device device, class T>
void tridiagSolver(Matrix<T, device>& mat_trd, Matrix<T, device>& d,
                   Matrix<std::complex<T>, device>& mat_ev) {
  Matrix<T, Device::CPU> mat_real_ev(mat_ev.distribution());
  tridiagSolver<backend, device, T>(mat_trd, d, mat_real_ev);

  // Convert real to complex numbers
  const matrix::Distribution& dist = mat_ev.distribution();
  for (auto tile_wrt_local : iterate_range2d(dist.localNrTiles())) {
    auto convert_to_complex_fn = [](const matrix::Tile<const T, Device::CPU>& in,
                                    const matrix::Tile<std::complex<T>, Device::CPU>& out) {
      for (auto el_idx : iterate_range2d(out.size())) {
        out(el_idx) = std::complex<T>(in(el_idx), 0);
      }
    };

    pika::execution::experimental::when_all(mat_real_ev.read_sender(tile_wrt_local),
                                            mat_ev.readwrite_sender(tile_wrt_local)) |
        dlaf::internal::transformDetach(dlaf::internal::Policy<Backend::MC>(),
                                        std::move(convert_to_complex_fn));
  }
}

}
}
