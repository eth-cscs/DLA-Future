//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#ifdef DLAF_WITH_HDF5
#include <atomic>
#include <sstream>
#endif

#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/eigensolver/gen_eigensolver/api.h>
#include <dlaf/eigensolver/gen_to_std.h>
#include <dlaf/factorization/cholesky.h>
#include <dlaf/matrix/hdf5.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/solver/triangular.h>
#include <dlaf/util_matrix.h>

#include "api.h"

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
void GenEigensolver<B, D, T>::call(blas::Uplo uplo, Matrix<T, D>& mat_a, Matrix<T, D>& mat_b,
                                   Matrix<BaseType<T>, D>& eigenvalues, Matrix<T, D>& eigenvectors,
                                   const Factorization factorization,
                                   const SizeType eigenvalues_index_begin,
                                   const SizeType eigenvalues_index_end) {
  if (factorization == Factorization::do_factorization) {
    cholesky_factorization<B>(uplo, mat_b);
  }
  generalized_to_standard<B>(uplo, mat_a, mat_b);

  hermitian_eigensolver<B>(uplo, mat_a, eigenvalues, eigenvectors, eigenvalues_index_begin,
                           eigenvalues_index_end);

  auto spec = matrix::util::internal::sub_matrix_spec_slice_cols(eigenvectors, eigenvalues_index_begin,
                                                                 eigenvalues_index_end);
  matrix::internal::MatrixRef eigenvectors_ref(eigenvectors, spec);
  solver::internal::triangular_solver<B>(blas::Side::Left, uplo, blas::Op::ConjTrans,
                                         blas::Diag::NonUnit, T(1), mat_b, eigenvectors_ref);
}

template <Backend B, Device D, class T>
void GenEigensolver<B, D, T>::call(comm::CommunicatorGrid& grid, blas::Uplo uplo, Matrix<T, D>& mat_a,
                                   Matrix<T, D>& mat_b, Matrix<BaseType<T>, D>& eigenvalues,
                                   Matrix<T, D>& eigenvectors, const Factorization factorization,
                                   const SizeType eigenvalues_index_begin,
                                   const SizeType eigenvalues_index_end) {
#ifdef DLAF_WITH_HDF5
  static std::atomic<size_t> num_gen_eigensolver_calls = 0;
  std::stringstream fname;
  fname << "generalized-eigensolver-";
  if (factorization == Factorization::already_factorized) {
    fname << "factorized-";
  }
  fname << matrix::internal::TypeToString_v<T> << "-" << std::to_string(num_gen_eigensolver_calls)
        << ".h5";

  std::optional<matrix::internal::FileHDF5> file;

  if (getTuneParameters().debug_dump_generalized_eigensolver_data) {
    file = matrix::internal::FileHDF5(grid.fullCommunicator(), fname.str());
    file->write(mat_a, "/input-a");
    if (factorization == Factorization::do_factorization) {
      file->write(mat_b, "/input-b");
    }
    else {  // Already factorized
      file->write(mat_b, "/input-b-factorized");
    }
  }
#endif

  if (factorization == Factorization::do_factorization) {
    cholesky_factorization<B>(grid, uplo, mat_b);
  }

  generalized_to_standard<B>(grid, uplo, mat_a, mat_b);

  hermitian_eigensolver<B>(grid, uplo, mat_a, eigenvalues, eigenvectors, eigenvalues_index_begin,
                           eigenvalues_index_end);

  auto spec = matrix::util::internal::sub_matrix_spec_slice_cols(eigenvectors, eigenvalues_index_begin,
                                                                 eigenvalues_index_end);
  matrix::internal::MatrixRef eigenvectors_ref(eigenvectors, spec);
  solver::internal::triangular_solver<B>(grid, blas::Side::Left, uplo, blas::Op::ConjTrans,
                                         blas::Diag::NonUnit, T(1), mat_b, eigenvectors_ref);

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_generalized_eigensolver_data) {
    file->write(eigenvalues, "/evals");
    file->write(eigenvectors, "/evecs");
  }

  num_gen_eigensolver_calls++;
#endif
}
}
