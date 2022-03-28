//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/gen_eigensolver.h"

#include <functional>
#include <tuple>

#include <gtest/gtest.h>

#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

template <typename Type>
class GenEigensolverTestMC : public ::testing::Test {};
TYPED_TEST_SUITE(GenEigensolverTestMC, MatrixElementTypes);

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType>> sizes = {
    {0, 2},                              // m = 0
    {5, 8}, {34, 34},                    // m <= mb
    {4, 3}, {16, 10}, {34, 13}, {32, 5}  // m > mb
};

template <class T, Backend B, Device D>
void testGenEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb) {
  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<const T, Device::CPU> reference_a = [&]() {
    Matrix<T, Device::CPU> reference(size, block_size);
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<const T, Device::CPU> reference_b = [&]() {
    Matrix<T, Device::CPU> reference(size, block_size);
    matrix::util::set_random_hermitian_positive_definite(reference);
    return reference;
  }();

  Matrix<T, D> mat_a(reference_a.distribution());
  copy(reference_a, mat_a);
  Matrix<T, D> mat_b(reference_b.distribution());
  copy(reference_b, mat_b);

  auto ret = eigensolver::genEigensolver<B>(uplo, mat_a, mat_b);

  if (mat_a.size().isEmpty())
    return;

  auto mat_a_local = allGather(blas::Uplo::General, reference_a);
  auto mat_b_local = allGather(blas::Uplo::General, reference_b);
  auto mat_e_local = allGather(blas::Uplo::General, ret.eigenvectors);

  MatrixLocal<T> mat_be_local({m, m}, block_size);
  // Compute B E which is needed for both checks.
  blas::hemm(blas::Layout::ColMajor, blas::Side::Left, uplo, m, m, T{1}, mat_b_local.ptr(),
             mat_b_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0}, mat_be_local.ptr(),
             mat_be_local.ld());

  MatrixLocal<T> workspace({m, m}, block_size);

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
    blas::scal(m, ret.eigenvalues[j], mat_be_local.ptr({0, j}), 1);
  }

  // Check A E == Lambda E
  CHECK_MATRIX_NEAR(mat_be_local, workspace, m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

TYPED_TEST(GenEigensolverTestMC, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    for (auto sz : sizes) {
      const auto& [m, mb] = sz;
      testGenEigensolver<TypeParam, Backend::MC, Device::CPU>(uplo, m, mb);
    }
  }
}
