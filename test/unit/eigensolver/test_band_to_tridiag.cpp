//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/band_to_tridiag.h"

#include <exception>
#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/traits.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
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
class EigensolverBandToTridiagTest : public ::testing::Test {};

TYPED_TEST_SUITE(EigensolverBandToTridiagTest, MatrixElementTypes);

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, band_size}
    {0, 2, 2},                                                // m = 0
    {1, 2, 2},                                                // m = 1
    {5, 5, 5}, {4, 4, 2},                                     // m = mb
    {4, 6, 3}, {8, 4, 2}, {18, 4, 4}, {34, 6, 6}, {37, 9, 3}  // m != mb
};

template <class T>
void testBandToTridiag(const blas::Uplo uplo, const SizeType band_size, const SizeType m,
                       const SizeType mb) {
  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<T, Device::CPU> mat_a(size, block_size);
  matrix::util::set_random_hermitian(mat_a);

  auto ret = eigensolver::bandToTridiag<Backend::MC>(uplo, band_size, mat_a);
  auto& mat_trid = ret.tridiagonal;
  auto& mat_v = ret.hh_reflectors;

  if (m == 0)
    return;

  auto mat_trid_local = matrix::test::allGather(lapack::MatrixType::General, mat_trid);
  MatrixLocal<T> mat_local(mat_a.size(), mat_a.blockSize());
  const auto ld = mat_local.ld();
  set(mat_local, [](auto) { return T{0}; });

  for (SizeType j = 0; j < m - 1; ++j) {
    mat_local({j, j}) = mat_trid_local({0, j});
    mat_local({j + 1, j}) = mat_trid_local({1, j});
    mat_local({j, j + 1}) = mat_trid_local({1, j});
  }
  mat_local({m - 1, m - 1}) = mat_trid_local({0, m - 1});

  auto mat_v_local = matrix::test::allGather(lapack::MatrixType::General, mat_v);

  auto apply_left_right = [&mat_local, m, ld](SizeType size_hhr, T* v, SizeType first_index) {
    T tau = v[0];
    v[0] = T{1};
    lapack::larf(blas::Side::Left, size_hhr, m, v, 1, tau, mat_local.ptr({first_index, 0}), ld);
    lapack::larf(blas::Side::Right, m, size_hhr, v, 1, dlaf::conj(tau), mat_local.ptr({0, first_index}),
                 ld);
  };

  if (isComplex_v<T> && m > 1) {
    T* v = mat_v_local.ptr({(m - 2) / band_size * band_size, m - 2});
    apply_left_right(1, v, m - 1);
  }

  for (SizeType sweep = m - 3; sweep >= 0; --sweep) {
    for (SizeType step = dlaf::util::ceilDiv(m - sweep - 2, band_size) - 1; step >= 0; --step) {
      SizeType first_index = sweep + 1 + step * band_size;
      SizeType size_hhr = std::min(band_size, m - first_index);

      SizeType i = (sweep / band_size + step) * band_size;
      T* v = mat_v_local.ptr({i, sweep});
      apply_left_right(size_hhr, v, first_index);
    }
  }

  // mat_a is a const input so it has not changed.
  auto res = [uplo, band_size, &mat_a, &mat_local](const GlobalElementIndex& index) {
    auto diag_index = index.row() - index.col();
    if (uplo == blas::Uplo::Upper && -diag_index >= 0 && -diag_index > band_size + 1)
      return mat_local(index);
    if (uplo == blas::Uplo::Lower && diag_index >= 0 && diag_index < band_size + 1)
      return mat_local(index);

    const auto& dist_a = mat_a.distribution();
    return mat_a.read(dist_a.globalTileIndex(index)).get()(dist_a.tileElementIndex(index));
  };

  CHECK_MATRIX_NEAR(res, mat_a, mb * m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

TYPED_TEST(EigensolverBandToTridiagTest, CorrectnessLocal) {
  const blas::Uplo uplo = blas::Uplo::Lower;

  for (const auto& [m, mb, b] : sizes)
    testBandToTridiag<TypeParam>(uplo, b, m, mb);
}
