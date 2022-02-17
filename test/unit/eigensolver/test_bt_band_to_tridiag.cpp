//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause

#include "dlaf/eigensolver/bt_band_to_tridiag.h"

#include <gtest/gtest.h>

#include "dlaf/eigensolver/band_to_tridiag.h"  // for nrSweeps/nrStepsForSweep
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::util;

using namespace dlaf::test;
using namespace dlaf::matrix::test;

template <typename Type>
class BacktransformationT2BTest : public ::testing::Test {};

TYPED_TEST_SUITE(BacktransformationT2BTest, MatrixElementTypes);

// Note: Helper functions for computing the tau of a given reflector. Reflector pointer should
// point to its 2nd component, i.e. the 1st component equal to 1 is implicitly considered in the
// computation without the need to have it in-place in the reflector. Moreover, this also means
// that the size given is not equal to the reflector size, instead it is equal to the number of
// components after the first one (i.e. reflector size - 1).
struct calculateTau {
  template <class T>
  static T call(const T* v, const SizeType size) {
    const T dotprod = blas::dot(size, v, 1, v, 1) + 1;
    return 2 / dotprod;
  }

  template <class T>
  static std::complex<T> call(const std::complex<T>* v, const SizeType size) {
    const T dotprod = std::real(blas::dot(size, v, 1, v, 1)) + 1;
    return {T(1) / dotprod, T(1) / dotprod};
  }
};

template <class T>
void computeTaus(const SizeType max_refl_size, const SizeType k, matrix::Tile<T, Device::CPU> tile) {
  for (SizeType j = 0; j < k; ++j) {
    const SizeType size = std::min(max_refl_size, tile.size().rows());
    // Note: calculateTau implicitly considers the first component equal to 1
    DLAF_ASSERT(size > 0, size);
    const auto tau = calculateTau::call(tile.ptr({1, j}), size - 1);
    *tile.ptr({0, j}) = tau;
  }
}

struct config_t {
  const SizeType m, n, mb, nb, b = mb;
};

std::vector<config_t> configs{
    {0, 0, 4, 4}, {12, 12, 4, 4}, {12, 12, 4, 3}, {20, 30, 5, 5}, {20, 30, 5, 6},
    {8, 8, 3, 3}, {10, 10, 3, 3}, {12, 12, 5, 5}, {12, 30, 5, 6},
};

template <class T>
void testBacktransformation(SizeType m, SizeType n, SizeType mb, SizeType nb, const SizeType b) {
  Matrix<T, Device::CPU> mat_e({m, n}, {mb, nb});
  set_random(mat_e);
  auto mat_e_local = allGather(blas::Uplo::General, mat_e);

  Matrix<const T, Device::CPU> mat_hh = [m, mb, b]() {
    Matrix<T, Device::CPU> mat_hh({m, m}, {mb, mb});
    set_random(mat_hh);

    const auto& dist = mat_hh.distribution();

    for (SizeType j = 0; j < mat_hh.size().cols(); j += b) {
      for (SizeType i = j; i < mat_hh.size().rows(); i += b) {
        const GlobalElementIndex ij(i, j);

        const TileElementIndex sub_origin = dist.tileElementIndex(ij);
        const TileElementSize sub_size(std::min(b, mat_hh.size().rows() - ij.row()),
                                       std::min(b, mat_hh.size().cols() - ij.col()));

        const SizeType n = std::min(2 * b - 1, mat_hh.size().rows() - ij.row() - 1);
        const SizeType k = std::min(n - 1, sub_size.cols());

        if (k <= 0)
          continue;

        const GlobalTileIndex ij_tile = dist.globalTileIndex(ij);
        auto tile_v = mat_hh(ij_tile);
        pika::dataflow(pika::unwrapping(computeTaus<T>), b, k,
                       splitTile(tile_v, {sub_origin, sub_size}));
      }
    }

    return mat_hh;
  }();

  MatrixLocal<T> mat_hh_local = allGather(blas::Uplo::Lower, mat_hh);

  eigensolver::backTransformationBandToTridiag<Backend::MC>(mat_e, mat_hh, b);

  if (m == 0 || n == 0)
    return;

  using eigensolver::internal::nrSweeps;
  using eigensolver::internal::nrStepsForSweep;
  for (SizeType sweep = nrSweeps<T>(m) - 1; sweep >= 0; --sweep) {
    for (SizeType step = nrStepsForSweep(sweep, m, b) - 1; step >= 0; --step) {
      const SizeType j = sweep;
      const SizeType i = j + 1 + step * b;

      const SizeType size = std::min(b, m - i);
      const SizeType i_v = (i - 1) / b * b;

      T& v_head = *mat_hh_local.ptr({i_v, j});
      const T tau = v_head;
      v_head = 1;

      using blas::Side;
      lapack::larf(Side::Left, size, n, &v_head, 1, tau, mat_e_local.ptr({i, 0}), mat_e_local.ld());
    }
  }

  auto result = [&dist = mat_e.distribution(),
                 &mat_local = mat_e_local](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  CHECK_MATRIX_NEAR(result, mat_e, m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

TYPED_TEST(BacktransformationT2BTest, CorrectnessLocal) {
  for (const auto& [m, n, mb, nb, b] : configs)
    testBacktransformation<TypeParam>(m, n, mb, nb, b);
}

std::vector<config_t> configs_subband{
    {12, 12, 4, 4, 2},
    {12, 12, 6, 6, 2},
    {11, 11, 6, 6, 3},
};

TYPED_TEST(BacktransformationT2BTest, CorrectnessLocalSubBand) {
  for (const auto& [m, n, mb, nb, b] : configs_subband)
    testBacktransformation<TypeParam>(m, n, mb, nb, b);
}
