//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <functional>
#include <tuple>

#include <blas.hh>

#include <pika/init.hpp>

#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/types.h>

#include <gtest/gtest.h>

#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/util_types.h>

namespace dlaf::test {

template <class T, Device D, class... GridIfDistributed>
void testGenEigensolverCorrectness(const blas::Uplo uplo, Matrix<const T, Device::CPU>& reference_a,
                                   Matrix<const T, Device::CPU>& reference_b,
                                   dlaf::EigensolverResult<T, D>& ret, GridIfDistributed... grid) {
  using dlaf::matrix::MatrixMirror;
  using dlaf::matrix::test::allGather;
  using dlaf::matrix::test::MatrixLocal;

  // Note:
  // Wait for the algorithm to finish all scheduled tasks, because verification has MPI blocking
  // calls that might lead to deadlocks.
  constexpr bool isDistributed = (sizeof...(grid) == 1);
  if constexpr (isDistributed)
    pika::wait();

  const SizeType m = reference_a.size().rows();

  auto mat_a_local = allGather(blas::Uplo::General, reference_a, grid...);
  auto mat_b_local = allGather(blas::Uplo::General, reference_b, grid...);
  auto mat_evalues_local = [&]() {
    MatrixMirror<const BaseType<T>, Device::CPU, D> mat_evals(ret.eigenvalues);
    return allGather(blas::Uplo::General, mat_evals.get());
  }();
  auto mat_e_local = [&]() {
    MatrixMirror<const T, Device::CPU, D> mat_e(ret.eigenvectors);
    return allGather(blas::Uplo::General, mat_e.get(), grid...);
  }();

  // eigenvalues are contiguous in the mat_local buffer
  BaseType<T>* evals_start = mat_evalues_local.ptr({0, 0});
  BaseType<T>* evals_end = evals_start + m;
  EXPECT_TRUE(std::is_sorted(evals_start, evals_end));

  MatrixLocal<T> mat_be_local({m, m}, reference_a.blockSize());

  dlaf::common::internal::SingleThreadedBlasScope single;

  // Compute B E which is needed for both checks.
  blas::hemm(blas::Layout::ColMajor, blas::Side::Left, uplo, m, m, T{1}, mat_b_local.ptr(),
             mat_b_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0}, mat_be_local.ptr(),
             mat_be_local.ld());

  MatrixLocal<T> workspace({m, m}, reference_a.blockSize());

  // Check eigenvectors orthogonality (E^H B E == Id)
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, m, m, m, T{1},
             mat_e_local.ptr(), mat_e_local.ld(), mat_be_local.ptr(), mat_be_local.ld(), T{0},
             workspace.ptr(), workspace.ld());

  auto id = [](GlobalElementIndex index) {
    if (index.row() == index.col())
      return T{1};
    return T{0};
  };
  CHECK_MATRIX_NEAR(id, workspace, m * TypeUtilities<T>::error, 10 * m * TypeUtilities<T>::error);

  // Check Ax = lambda B x
  // Compute A E
  blas::hemm(blas::Layout::ColMajor, blas::Side::Left, uplo, m, m, T{1}, mat_a_local.ptr(),
             mat_a_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0}, workspace.ptr(),
             workspace.ld());

  // Compute Lambda E (in place in mat_e_local)
  for (SizeType j = 0; j < m; ++j) {
    blas::scal(m, mat_evalues_local({j, 0}), mat_be_local.ptr({0, j}), 1);
  }

  // Check A E == Lambda E
  CHECK_MATRIX_NEAR(mat_be_local, workspace, m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

}
