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
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::common;
using namespace dlaf::matrix;
using namespace dlaf::matrix::internal;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class BackTransformationEigenSolverTestMC : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};
TYPED_TEST_SUITE(BackTransformationEigenSolverTestMC, MatrixElementTypes);

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes =
    // m, n, mb, nb
    {
        {3, 0, 1, 1}, {0, 5, 2, 3},                                  // m, n = 0
        {2, 2, 3, 3}, {3, 4, 6, 7},                                  // m < mb
        {3, 3, 1, 1}, {4, 4, 2, 2},  {12, 2, 4, 4}, {24, 36, 6, 6},  // mb = nb
        {5, 8, 3, 2}, {8, 27, 3, 4}, {15, 34, 4, 6}                  // mb != nb
};

template <class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().blockSize()};
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
  const LocalElementSize sizeC(m, n);
  const TileElementSize blockSizeC(mb, nb);
  Matrix<T, Device::CPU> mat_c(sizeC, blockSizeC);
  dlaf::matrix::util::set_random(mat_c);

  const LocalElementSize sizeV(m, m);
  const TileElementSize blockSizeV(mb, mb);
  Matrix<T, Device::CPU> mat_v(sizeV, blockSizeV);
  dlaf::matrix::util::set_random(mat_v);

  const SizeType nr_reflector = std::max(static_cast<SizeType>(0), m - mb - 1);

  // Reset diagonal and upper values of V
  const MatrixLocal<T> v({m, m}, blockSizeV);
  for (const auto& ij_tile : iterate_range2d(v.nrTiles())) {
    const auto& source_tile = mat_v.read(ij_tile).get();
    copy(source_tile, v.tile(ij_tile));
    if (ij_tile.row() == ij_tile.col() + 1)
      tile::internal::laset<T>(lapack::MatrixType::Upper, 0.f, 1.f, v.tile(ij_tile));
  }

  // Create C local
  const MatrixLocal<T> c({m, n}, blockSizeC);
  for (const auto& ij_tile : iterate_range2d(c.nrTiles())) {
    const auto& source_tile = mat_c.read(ij_tile).get();
    copy(source_tile, c.tile(ij_tile));
  }

  common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus;
  SizeType nr_reflectors_blocks = std::max<SizeType>(0, dlaf::util::ceilDiv(m - mb - 1, mb));
  taus.reserve(nr_reflectors_blocks);

  common::internal::vector<T> tausloc;
  tausloc.reserve(nr_reflector);

  // Impose orthogonality: Q = I - v tau vH is orthogonal (Q QH = I).
  // Real case: tau = 2 / (vH v)
  // Complex case: real part of tau = [1 + sqrt(1 - vH v taui^2)]/(vH v)
  for (SizeType k = 0; k < nr_reflector; k += mb) {
    common::internal::vector<T> tau_tile;
    tau_tile.reserve(mb);
    auto seed = 10000 * k / mb + 1;
    dlaf::matrix::util::internal::getter_random<BaseType<T>> random_value(seed);
    for (SizeType j = k; j < std::min(k + mb, nr_reflector); ++j) {
      const GlobalElementIndex v_offset{j + mb, j};
      auto dotprod = blas::dot(m - mb - j, v.ptr(v_offset), 1, v.ptr(v_offset), 1);
      BaseType<T> tau_i = 0;
      if (std::is_same_v<T, ComplexType<T>>) {
        tau_i = random_value();
      }
      T tau;
      getTau(tau, dotprod, tau_i);
      tausloc.push_back(tau);
      tau_tile.push_back(tau);
    }
    taus.push_back(hpx::make_ready_future(tau_tile));
  }

  if (n != 0) {
    for (SizeType j = nr_reflector - 1; j >= 0; --j) {
      const GlobalElementIndex v_offset{j + mb, j};
      auto tau = tausloc[j];
      lapack::larf(lapack::Side::Left, m - mb - j, n, v.ptr(v_offset), 1, tau,
                   c.ptr(GlobalElementIndex{j + mb, 0}), c.ld());
    }
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

template <class T>
void testBacktransformationEigenv(comm::CommunicatorGrid grid, SizeType m, SizeType n, SizeType mb,
                                  SizeType nb) {
  comm::Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  const LocalElementSize sizeC(m, n);
  const TileElementSize blockSizeC(mb, nb);
  const GlobalElementSize szC = globalTestSize(sizeC);
  const Distribution distrC(szC, blockSizeC, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_c(std::move(distrC));
  dlaf::matrix::util::set_random(mat_c);

  const LocalElementSize sizeV(m, m);
  const TileElementSize blockSizeV(mb, mb);
  const GlobalElementSize szV = globalTestSize(sizeV);
  const Distribution distrV(szV, blockSizeV, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_v(std::move(distrV));
  dlaf::matrix::util::set_random(mat_v);

  const SizeType nr_reflector = std::max(static_cast<SizeType>(0), m - mb - 1);

  // Copy matrices locally
  const auto mat_c_loc = dlaf::matrix::test::allGather<T>(lapack::MatrixType::General, mat_c, grid);
  auto mat_v_loc = dlaf::matrix::test::allGather<T>(lapack::MatrixType::General, mat_v, grid);

  // Reset diagonal and upper values of V
  for (const auto& ij_tile : iterate_range2d(mat_v_loc.nrTiles())) {
    if (ij_tile.row() == ij_tile.col() + 1)
      tile::internal::laset<T>(lapack::MatrixType::Upper, 0.f, 1.f, mat_v_loc.tile(ij_tile));
  }

  common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus;
  SizeType nr_reflectors_blocks = std::max<SizeType>(0, dlaf::util::ceilDiv(m - mb - 1, mb));
  taus.reserve(nr_reflectors_blocks);

  common::internal::vector<T> tausloc;
  tausloc.reserve(nr_reflector);

  // Impose orthogonality: Q = I - v tau vH is orthogonal (Q QH = I).
  // Real case: tau = 2 / (vH v)
  // Complex case: real part of tau = [1 + sqrt(1 - vH v taui^2)]/(vH v)
  for (SizeType k = 0; k < nr_reflector; k += mb) {
    common::internal::vector<T> tau_tile;
    tau_tile.reserve(mb);
    auto seed = 10000 * k / mb + 1;
    dlaf::matrix::util::internal::getter_random<BaseType<T>> random_value(seed);
    for (SizeType j = k; j < std::min(k + mb, nr_reflector); ++j) {
      const GlobalElementIndex v_offset{j + mb, j};
      auto dotprod = blas::dot(m - mb - j, mat_v_loc.ptr(v_offset), 1, mat_v_loc.ptr(v_offset), 1);
      BaseType<T> tau_i = 0;
      if (std::is_same_v<T, ComplexType<T>>) {
        tau_i = random_value();
      }
      T tau;
      getTau(tau, dotprod, tau_i);
      tausloc.push_back(tau);
      tau_tile.push_back(tau);
    }
    if (grid.rank().col() == mat_v.distribution().template rankGlobalTile<Coord::Col>(k / mb))
      taus.push_back(hpx::make_ready_future(tau_tile));
  }

  if (n != 0) {
    for (SizeType j = nr_reflector - 1; j >= 0; --j) {
      const GlobalElementIndex v_offset{j + mb, j};
      auto tau = tausloc[j];
      lapack::larf(lapack::Side::Left, m - mb - j, n, mat_v_loc.ptr(v_offset), 1, tau,
                   mat_c_loc.ptr(GlobalElementIndex{j + mb, 0}), mat_c_loc.ld());
    }
  }

  eigensolver::backTransformation<Backend::MC>(grid, mat_c, mat_v, taus);

  auto result = [& dist = mat_c.distribution(),
                 &mat_local = mat_c_loc](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  const auto error = (mat_c.size().rows() + 1) * dlaf::test::TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(result, mat_c, error, error);
}

TYPED_TEST(BackTransformationEigenSolverTestMC, CorrectnessLocal) {
  for (auto sz : sizes) {
    auto [m, n, mb, nb] = sz;
    testBacktransformationEigenv<TypeParam>(m, n, mb, nb);
  }
}

TYPED_TEST(BackTransformationEigenSolverTestMC, CorrectnessDistributed) {
  for (const auto& comm_grid : {this->commGrids()[0]}) {
    for (auto sz : sizes) {
      auto [m, n, mb, nb] = sz;
      testBacktransformationEigenv<TypeParam>(comm_grid, m, n, mb, nb);
    }
  }
}
