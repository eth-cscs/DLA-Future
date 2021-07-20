//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/backtransformation.h"

#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/common/index2d.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::common;
using namespace dlaf::matrix;
using namespace dlaf::matrix::internal;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using dlaf::matrix::test::MatrixLocal;
using namespace testing;

template <typename Type>
class BackTransformationEigenSolverTestMC : public ::testing::Test {};
TYPED_TEST_SUITE(BackTransformationEigenSolverTestMC, MatrixElementTypes);

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes =
    // m, n, mb, nb
    {{3, 0, 1, 1},   {0, 5, 2, 3},  // m, n = 0
     {2, 2, 3, 3},   {3, 4, 6, 7},  // m < mb
     {3, 3, 1, 1},   {4, 4, 2, 2}, {6, 3, 3, 3}, {12, 2, 4, 4}, {12, 24, 3, 3},
     {24, 36, 6, 6}, {5, 8, 3, 2}, {4, 6, 2, 3}, {5, 5, 2, 3},  {8, 27, 3, 4},
     {15, 34, 4, 6}, {7, 6, 2, 3}, {8, 5, 2, 3}, {7, 5, 2, 3}};

template <class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().blockSize()};
}

template <class T>
void set_zero(Matrix<T, Device::CPU>& mat) {
  dlaf::matrix::util::set(mat, [](auto&&) { return static_cast<T>(0.0); });
}

template <class T>
void getTau(T& tau, T dotprod, BaseType<T> /*tau_i*/) {
  tau = static_cast<T>(2.0) / dotprod;
}

template <class T>
void getTau(std::complex<T>& tau, T dotprod, BaseType<T> tau_i) {
  tau = {(static_cast<T>(1.0) + sqrt(static_cast<T>(1.0) - dotprod * tau_i * tau_i)) / dotprod, tau_i};
}

template <class T>
void testBacktransformationEigenv(SizeType m, SizeType n, SizeType mb, SizeType nb) {
  LocalElementSize sizeC(m, n);
  TileElementSize blockSizeC(mb, nb);
  Matrix<T, Device::CPU> mat_c(sizeC, blockSizeC);
  dlaf::matrix::util::set_random(mat_c);

  LocalElementSize sizeV(m, m);
  TileElementSize blockSizeV(mb, mb);
  Matrix<T, Device::CPU> mat_v(sizeV, blockSizeV);
  dlaf::matrix::util::set_random(mat_v);

  SizeType tottaus;
  if (m < mb || m == 0 || n == 0)
    tottaus = 0;
  else
    tottaus = m - mb;

  if (tottaus > 0) {
    // Reset diagonal and upper values of V
    MatrixLocal<T> v({m, m}, blockSizeV);
    for (const auto& ij_tile : iterate_range2d(v.nrTiles())) {
      // copy only the panel
      const auto& source_tile = mat_v.read(ij_tile).get();
      if (ij_tile.row() <= ij_tile.col()) {
        tile::set0<T>(v.tile(ij_tile));
      }
      else if (ij_tile.row() == ij_tile.col() + 1) {
        copy(source_tile, v.tile(ij_tile));
        tile::laset<T>(lapack::MatrixType::Upper, 0.f, 1.f, v.tile(ij_tile));
      }
      else {
        copy(source_tile, v.tile(ij_tile));
      }
    }

    // Create C local
    MatrixLocal<T> c({m, n}, blockSizeC);
    for (const auto& ij_tile : iterate_range2d(c.nrTiles())) {
      // copy only the panel
      const auto& source_tile = mat_c.read(ij_tile).get();
      copy(source_tile, c.tile(ij_tile));
    }

    common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus;

    common::internal::vector<T> tausloc;
    tausloc.reserve(m);

    // Impose orthogonality: Q = I - v tau vH is orthogonal (Q QH = I).
    // Real case: tau = 2 / (vH v)
    // Complex case: real part of tau = [1 + sqrt(1 - vH v taui^2)]/(vH v)
    for (SizeType k = 0; k < tottaus; k += mb) {
      common::internal::vector<T> tau_tile;
      tau_tile.reserve(mb);
      auto seed = 10000 * k / mb + 1;
      dlaf::matrix::util::internal::getter_random<BaseType<T>> random_value(seed);
      for (SizeType j = k; j < std::min(k + mb, tottaus); ++j) {
        const GlobalElementIndex v_offset{j + mb, j};
        auto dotprod = blas::dot(tottaus - j, v.ptr(v_offset), 1, v.ptr(v_offset), 1);
        BaseType<T> tau_i = 0;
        if (std::is_same<T, ComplexType<T>>::value) {
          tau_i = random_value();
        }
        T tau;
        getTau(tau, dotprod, tau_i);
        tausloc.push_back(tau);
        tau_tile.push_back(tau);
      }
      taus.push_back(hpx::make_ready_future(tau_tile));
    }

    for (SizeType j = tottaus - 1; j > -1; --j) {
      const GlobalElementIndex v_offset{j + mb, j};
      auto tau = tausloc[j];
      lapack::larf(lapack::Side::Left, tottaus - j, n, v.ptr(v_offset), 1, tau,
                   c.ptr(GlobalElementIndex{j + mb, 0}), c.ld());
    }

    eigensolver::backTransformation<Backend::MC>(mat_c, mat_v, taus);

    auto result = [& dist = mat_c.distribution(), &mat_local = c](const GlobalElementIndex& element) {
      const auto tile_index = dist.globalTileIndex(element);
      const auto tile_element = dist.tileElementIndex(element);
      return mat_local.tile_read(tile_index)(tile_element);
    };

    const auto error = (mat_c.size().rows() + 1) * dlaf::test::TypeUtilities<T>::error;
    CHECK_MATRIX_NEAR(result, mat_c, error, error);
  }
}

TYPED_TEST(BackTransformationEigenSolverTestMC, CorrectnessLocal) {
  SizeType m, n, mb, nb;

  for (auto sz : sizes) {
    std::tie(m, n, mb, nb) = sz;
    testBacktransformationEigenv<TypeParam>(m, n, mb, nb);
  }
}
