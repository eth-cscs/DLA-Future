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

struct calculateTau {
  template <class T>
  static T call(const T* v, const SizeType size) {
    const T dotprod = blas::dot(size, v, 1, v, 1);
    return 2 / dotprod;
  }

  template <class T>
  static std::complex<T> call(const std::complex<T>* v, const SizeType size) {
    const T dotprod = std::real(blas::dot(size, v, 1, v, 1));
    const T imag = T(1) / size;  // TODO check tau vs dotprod
    return {(T(1.0) + std::sqrt(T(1) - dotprod * imag * imag)) / dotprod, imag};
  }

  // TODO random number has to be "resetted" each time
};

template <class T>
void computeTaus(const SizeType n, const SizeType k, matrix::Tile<T, Device::CPU> tile) {
  for (SizeType j = 0; j < k; ++j) {
    const SizeType size = std::min(n - j, tile.size().rows() - 1);
    DLAF_ASSERT(size > 0, size);
    const auto tau = calculateTau::call(tile.ptr({0, j}), size);
    *tile.ptr({0, j}) = tau;
  }
}

struct config_t {
  const SizeType m, n, mb, nb;
};

std::vector<config_t> configs{
    {0, 0, 4, 4}, {12, 12, 4, 4}, {12, 12, 4, 3}, {20, 30, 5, 5}, {20, 30, 5, 6},
    {8, 8, 3, 3}, {10, 10, 3, 3}, {12, 12, 5, 5}, {12, 30, 5, 6},
};

TYPED_TEST(BacktransformationT2BTest, CorrectnessLocal) {
  for (const auto& config : configs) {
    const SizeType m = config.m;
    const SizeType n = config.n;
    const SizeType mb = config.mb;
    const SizeType nb = config.nb;

    Matrix<TypeParam, Device::CPU> mat_e({m, n}, {mb, nb});
    set_random(mat_e);
    auto mat_e_local = allGather(lapack::MatrixType::General, mat_e);

    auto nrSweeps = [m]() {
      const bool is_complex = std::is_same<TypeParam, ComplexType<TypeParam>>::value;
      return std::max<SizeType>(0, is_complex ? m - 1 : m - 2);
    };

    auto nrStepsPerSweep = [m, mb](const SizeType sweep) {
      return std::max<SizeType>(0, sweep == m - 2 ? 1 : dlaf::util::ceilDiv(m - sweep - 2, mb));
    };

    Matrix<const TypeParam, Device::CPU> mat_i = [m, mb]() {
      Matrix<TypeParam, Device::CPU> mat_i({m, m}, {mb, mb});
      set_random(mat_i);  // TODO ? same seed ==> mat_i == mat_e

      const auto m = mat_i.distribution().localNrTiles().cols();
      for (SizeType j = 0; j < m; ++j) {
        for (SizeType i = j; i < m; ++i) {
          const bool affectsTwoRows = i < m - 1;
          const SizeType k =
              affectsTwoRows ? mat_i.tileSize({i, j}).cols() : mat_i.tileSize({i, j}).rows() - 2;
          const SizeType n = mat_i.tileSize({i, j}).rows() - 1 +
                             (affectsTwoRows ? mat_i.tileSize({i + 1, j}).rows() : 0);
          if (k <= 0)
            continue;
          hpx::dataflow(hpx::unwrapping(computeTaus<TypeParam>), n, k, mat_i(LocalTileIndex(i, j)));
        }
      }

      return mat_i;
    }();

    MatrixLocal<TypeParam> mat_i_local = allGather(lapack::MatrixType::Lower, mat_i);

    eigensolver::backTransformationT2B<Backend::MC>(mat_e, mat_i);

    for (SizeType sweep = nrSweeps() - 1; sweep >= 0; --sweep) {
      for (SizeType step = nrStepsPerSweep(sweep) - 1; step >= 0; --step) {
        const SizeType j = sweep;
        const SizeType i = j + 1 + step * mb;

        const SizeType size = std::min(mb, m - i);
        const SizeType i_v = (i - 1) / mb * mb;

        TypeParam& v_head = *mat_i_local.ptr({i_v, j});
        const TypeParam tau = v_head;
        v_head = 1;

        using blas::Side;
        lapack::larf(Side::Left, size, n, &v_head, 1, tau, mat_e_local.ptr({i, 0}), mat_e_local.ld());
      }
    }

    auto result = [& dist = mat_e.distribution(),
                   &mat_local = mat_e_local](const GlobalElementIndex& element) {
      const auto tile_index = dist.globalTileIndex(element);
      const auto tile_element = dist.tileElementIndex(element);
      return mat_local.tile_read(tile_index)(tile_element);
    };

    // TODO how much error
    const auto error = std::max<SizeType>(1, 40 * m * n) * TypeUtilities<TypeParam>::error;
    CHECK_MATRIX_NEAR(result, mat_e, error, error);
  }
}
