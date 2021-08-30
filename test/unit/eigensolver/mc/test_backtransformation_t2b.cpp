//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause

#include "dlaf/eigensolver/backtransformation.h"

#include <gtest/gtest.h>

#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::util;

using namespace dlaf::test;
using namespace dlaf::matrix::test;

template <typename Type>
class BacktransformationT2BTest : public ::testing::Test {};

TYPED_TEST_SUITE(BacktransformationT2BTest, MatrixElementTypes);

struct calculateTau {
  template <class T>
  static T call(const T* v, const SizeType size) {
    const auto dotprod = blas::dot(size, v, 1, v, 1);
    return 2 / dotprod;
  }

  template <class T>
  static std::complex<T> call(const std::complex<T>* v, const SizeType size) {
    const T dotprod = std::real(blas::dot(size, v, 1, v, 1));
    const T imag = 1;
    return {(T(1) + std::sqrt(T(1) - dotprod * imag * imag)) / dotprod, imag};
  }
};

template <class T>
void computeTaus(const SizeType k, matrix::Tile<T, Device::CPU> tile) {
  const SizeType b = tile.size().rows();

  for (SizeType j = 0; j < k; ++j) {
    const SizeType i = j;
    const SizeType n = i + b;

    auto tau = calculateTau::call(tile.ptr({0, j}), n);

    *tile.ptr({0, j}) = tau;
  }
}

TYPED_TEST(BacktransformationT2BTest, CorrectnessLocal) {
  const SizeType m = 12;
  const SizeType n = 12;
  const SizeType mb = 4;
  const SizeType nb = 4;

  const SizeType b = mb;

  const LocalElementSize sz_e(m, n);
  const TileElementSize bsz_e(mb, nb);

  const LocalElementSize sz_v(m, m);
  const TileElementSize bsz_v(mb, mb);

  Matrix<TypeParam, Device::CPU> mat_e(sz_e, bsz_e);
  set_random(mat_e);
  auto mat_e_local = allGather(lapack::MatrixType::General, mat_e);

  Matrix<const TypeParam, Device::CPU> mat_v = [sz_v, bsz_v]() {
    Matrix<TypeParam, Device::CPU> mat_v(sz_v, bsz_v);
    set_random(mat_v); // TODO ? same seed ==> mat_v == mat_e

    for (SizeType j = 0; j < mat_v.distribution().localNrTiles().cols(); ++j) {
      for (SizeType i = 0; i < j; ++i) {
        const SizeType k = (i == j) ? b - 2 : b;
        hpx::dataflow(hpx::unwrapping(computeTaus<TypeParam>), k, mat_v(LocalTileIndex(i, j)));
      }
    }

    return mat_v;
  }();

  MatrixLocal<TypeParam> mat_v_local = allGather(lapack::MatrixType::Lower, mat_v);

  eigensolver::backTransformationT2B<Backend::MC>(mat_e, mat_v);

  for (SizeType j = m - 1; j >= 0; --j) {
    SizeType j_t = j / mb;
    for (SizeType i = m - mb + ((j + 1) % mb); i >= j + 1; i -= mb) {
      SizeType i_t = (i - 1) / mb;
      SizeType k = j % mb;

      const SizeType size = std::min(mb, m - i);

      if (size == 1) // TODO check this (also for complex)
        continue;

      auto& tile_v = mat_v_local.tile({i_t, j_t});
      TypeParam& v_head = *tile_v.ptr({0, k});
      const TypeParam tau = v_head;
      v_head = 1;

      lapack::larf(blas::Side::Left, size, n, &v_head, 1, tau, mat_e_local.ptr({i, 0}), mat_e_local.ld());
    }
  }

  auto result = [&dist = mat_e.distribution(), &mat_local = mat_e_local](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  const auto error = 2 * TypeUtilities<TypeParam>::error; // TODO how much error
  CHECK_MATRIX_NEAR(result, mat_e, error, error);
}
