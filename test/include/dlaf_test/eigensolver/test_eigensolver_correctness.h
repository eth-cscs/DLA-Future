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

#include <blas.hh>

#include <pika/init.hpp>

#include <dlaf/blas/scal.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/util_types.h>

namespace dlaf::test {

template <class T, Device D, class... GridIfDistributed>
void testEigensolverCorrectness(const blas::Uplo uplo, Matrix<const T, Device::CPU>& reference,
                                Matrix<const BaseType<T>, D>& eigenvalues,
                                Matrix<const T, D>& eigenvectors, SizeType eigenvalue_index_start,
                                SizeType eval_index_end, GridIfDistributed&... grid) {
  using dlaf::matrix::MatrixMirror;
  using dlaf::matrix::test::allGather;
  using dlaf::matrix::test::MatrixLocal;

  DLAF_ASSERT(eigenvalue_index_start == 0l, eigenvalue_index_start);

  // Note:
  // Wait for the algorithm to finish all scheduled tasks, because verification has MPI blocking
  // calls that might lead to deadlocks.
  constexpr bool isDistributed = (sizeof...(grid) == 1);
  if constexpr (isDistributed)
    pika::wait();

  const SizeType m = reference.size().rows();

  auto mat_a_local = allGather<T>(blas::Uplo::General, reference, grid...);
  auto mat_evalues_local = [&]() {
    MatrixMirror<const BaseType<T>, Device::CPU, D> mat_evals(eigenvalues);

    return allGather<BaseType<T>>(blas::Uplo::General, mat_evals.get());
  }();
  auto mat_e_local = [&]() {
    MatrixMirror<const T, Device::CPU, D> mat_e(eigenvectors);
    // Reference to sub-matrix representing only valid (i.e. back-transformed) eigenvectors
    auto spec = matrix::util::internal::sub_matrix_spec_slice_cols(reference, eigenvalue_index_start,
                                                                   eval_index_end);
    matrix::internal::MatrixRef mat_e_ref(mat_e.get(), spec);

    return allGather<T>(blas::Uplo::General, mat_e_ref, grid...);
  }();

  // Number of valid eigenvectors
  const SizeType n = mat_e_local.size().cols();

  // eigenvalues are contiguous in the mat_local buffer
  const BaseType<T>* evals_start = mat_evalues_local.ptr({0, 0});
  EXPECT_TRUE(std::is_sorted(evals_start, evals_start + m));

  MatrixLocal<T> workspace1({n, n}, reference.blockSize());

  dlaf::common::internal::SingleThreadedBlasScope single;

  // Check eigenvectors orthogonality (E^H E == Id)
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, n, n, m, T{1},
             mat_e_local.ptr(), mat_e_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0},
             workspace1.ptr(), workspace1.ld());

  auto id = [](GlobalElementIndex index) {
    if (index.row() == index.col())
      return T{1};
    return T{0};
  };
  CHECK_MATRIX_NEAR(id, workspace1, m * TypeUtilities<T>::error, 10 * m * TypeUtilities<T>::error);

  MatrixLocal<T> workspace2({m, n}, reference.blockSize());

  // Check Ax = lambda x
  // Compute A E
  blas::hemm(blas::Layout::ColMajor, blas::Side::Left, uplo, m, n, T{1}, mat_a_local.ptr(),
             mat_a_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0}, workspace2.ptr(),
             workspace2.ld());

  // Compute Lambda E (in place in mat_e_local)
  for (SizeType j = 0; j < n; ++j) {
    blas::scal(m, mat_evalues_local({j, 0}), mat_e_local.ptr({0, j}), 1);
  }

  // Check A E == Lambda E
  auto res = [&mat_e_local](GlobalElementIndex index) { return mat_e_local(index); };
  CHECK_MATRIX_NEAR(res, workspace2, 2 * m * TypeUtilities<T>::error, 2 * m * TypeUtilities<T>::error);
}

}
