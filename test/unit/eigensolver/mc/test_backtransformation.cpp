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

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    {3, 0, 1, 1}, {0, 5, 2, 3},  // m, n = 0
    {2, 2, 3, 3}, {3, 4, 6, 7},  // m < mb
    {3, 3, 1, 1}, {4, 4, 2, 2}, {6, 3, 3, 3}, {12, 2, 4, 4}, {12, 24, 3, 3}, {24, 36, 6, 6},
    //  {5, 8, 3, 2}, {4, 6, 2, 3}, {5, 5, 2, 3},  {8, 27, 3, 4}, {15, 34, 4, 6},   // These cases lead
    //  to operations between incomplete tiles where sizes are not detected correctly.
};

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

  int tottaus;
  if (m < mb || m == 0 || n == 0)
    tottaus = 0;
  else
    tottaus = (m / mb - 1) * mb + m % mb;

  if (tottaus > 0) {
    // Impose orthogonality: Q = I - v tau v^H is orthogonal (Q Q^H = I)
    // leads to tau = [1 + sqrt(1 - vH v taui^2)]/(vH v) for real
    LocalElementSize sizeTau(m, 1);
    TileElementSize blockSizeTau(1, 1);
    Matrix<T, Device::CPU> mat_tau(sizeTau, blockSizeTau);

    // Reset diagonal and upper values of V
    MatrixLocal<T> v({m, m}, blockSizeV);
    for (const auto& ij_tile : iterate_range2d(v.nrTiles())) {
      // copy only the panel
      const auto& source_tile = mat_v.read(ij_tile).get();
      copy(source_tile, v.tile(ij_tile));
      if (ij_tile.row() <= ij_tile.col()) {
        tile::set0<T>(v.tile(ij_tile));
      }
      else if (ij_tile.row() == ij_tile.col() + 1) {
        tile::laset<T>(lapack::MatrixType::Upper, 0.f, 1.f, v.tile(ij_tile));
      }
    }

    // Create C local
    MatrixLocal<T> c({m, n}, blockSizeC);
    for (const auto& ij_tile : iterate_range2d(c.nrTiles())) {
      // copy only the panel
      const auto& source_tile = mat_c.read(ij_tile).get();
      copy(source_tile, c.tile(ij_tile));
    }

    // TODO: creating a whole matrix, solves issues on the last tile when m%mb != 0 ==> find out how to
    // use only a panel
    LocalElementSize sizeT(tottaus, tottaus);
    TileElementSize blockSizeT(mb, mb);
    Matrix<T, Device::CPU> mat_t(sizeT, blockSizeT);
    set_zero(mat_t);

    common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus;

    MatrixLocal<T> tausloc({tottaus, 1}, {mb, mb});
    auto tau_rows = tausloc.nrTiles().rows();

    auto nt = 0;
    for (SizeType i = 0; i < tau_rows; ++i) {
      common::internal::vector<T> t_tile;
      auto seed = 10000 * i / mb + 1;
      dlaf::matrix::util::internal::getter_random<BaseType<T>> random_value(seed);
      for (SizeType t = 0; t < mb && nt < tottaus; ++t) {
        const GlobalElementIndex v_offset{i * mb + t, i * mb + t};
        auto dotprod = blas::dot(m - t, v.ptr(v_offset), 1, v.ptr(v_offset), 1);
        BaseType<T> tau_i = 0;
        if (std::is_same<T, ComplexType<T>>::value) {
          tau_i = random_value();
        }
        T tau;
        getTau(tau, dotprod, tau_i);
        tausloc({nt, 0}) = static_cast<T>(tau);
        t_tile.push_back(static_cast<T>(tau));
        ++nt;
      }
      taus.push_back(hpx::make_ready_future(t_tile));
    }

    for (SizeType i = tottaus - 1; i > -1; --i) {
      const GlobalElementIndex v_offset{i, i};
      auto tau = tausloc({i, 0});
      lapack::larf(lapack::Side::Left, m - i, n, v.ptr(v_offset), 1, tau,
                   c.ptr(GlobalElementIndex{i, 0}), c.ld());
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
