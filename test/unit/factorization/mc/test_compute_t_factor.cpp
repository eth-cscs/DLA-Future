//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/factorization/qr.h"

#include <tuple>

#include <gtest/gtest.h>
#include <blas.hh>
#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/lapack/tile.h"  // workaround for importing lapack.hh
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::test;
using dlaf::comm::CommunicatorGrid;
using dlaf::matrix::test::MatrixLocal;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T, Device D>
struct ComputeTFactorTest : public ::testing::Test {
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

template <class T>
using ComputeTFactorTestMC = ComputeTFactorTest<T, Device::CPU>;

TYPED_TEST_SUITE(ComputeTFactorTestMC, MatrixElementTypes);

template <class T>
T preset_eye(const GlobalElementIndex& index) {
  return index.row() == index.col() ? 1 : 0;
}

template <class T>
void is_orthogonal(const MatrixLocal<const T>& matrix) {
  if (matrix.size().isEmpty())
    return;

  MatrixLocal<T> ortho(matrix.size(), matrix.blockSize());

  // ortho = matrix . matrix*
  // clang-format off
  blas::gemm(blas::Layout::ColMajor,
      blas::Op::NoTrans, blas::Op::ConjTrans,
      matrix.size().rows(), matrix.size().cols(), matrix.size().rows(),
      1,
      matrix.ptr(), matrix.ld(),
      matrix.ptr(), matrix.ld(),
      0,
      ortho.ptr(), ortho.ld());
  // clang-format on

  MatrixLocal<const T> eye = [&]() {
    MatrixLocal<T> m(matrix.size(), matrix.blockSize());
    set(m, preset_eye<T>);
    return m;
  }();

  // Note:
  // Orthogonality requires a strict error test, => n * error
  SCOPED_TRACE("Orthogonality test");
  const auto error = matrix.size().rows() * TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(eye, ortho, 0, error);
}

template <class T>
std::tuple<dlaf::common::internal::vector<T>, MatrixLocal<T>> computeHAndTFactor(
    const SizeType k, const MatrixLocal<const T>& v) {
  const SizeType m = v.size().rows();
  const TileElementSize block_size = v.blockSize();

  // compute taus and H_exp
  common::internal::vector<T> taus;
  taus.reserve(k);

  MatrixLocal<T> h_expected({m, m}, block_size);
  set(h_expected, preset_eye<T>);

  for (auto j = 0; !v.size().isEmpty() && j < k; ++j) {
    const SizeType reflector_size = m - j;

    const T* data_ptr = v.ptr({j, j});
    const auto norm = blas::nrm2(reflector_size, data_ptr, 1);
    const T tau = 2 / std::pow(norm, static_cast<decltype(norm)>(2));

    taus.push_back(tau);

    MatrixLocal<T> h_i({reflector_size, reflector_size}, block_size);
    set(h_i, preset_eye<T>);

    // Hi = (I - tau . v . v*)
    // clang-format off
    blas::ger(blas::Layout::ColMajor,
        reflector_size, reflector_size,
        -tau,
        v.ptr({j, j}), 1,
        v.ptr({j, j}), 1,
        h_i.ptr(), h_i.ld());
    // clang-format on

    // H_exp[:, j:] = H_exp[:, j:] . Hi
    const GlobalElementIndex h_offset{0, j};
    MatrixLocal<T> workspace({h_expected.size().rows(), h_i.size().cols()}, h_i.blockSize());

    // clang-format off
    blas::gemm(blas::Layout::ColMajor,
        blas::Op::NoTrans, blas::Op::NoTrans,
        h_expected.size().rows(), h_i.size().cols(), h_i.size().rows(),
        1,
        h_expected.ptr(h_offset), h_expected.ld(),
        h_i.ptr(), h_i.ld(),
        0,
        workspace.ptr(), workspace.ld());
    // clang-format on
    std::copy(workspace.ptr(), workspace.ptr() + workspace.size().linear_size(),
              h_expected.ptr(h_offset));
  }

  return std::make_tuple(taus, std::move(h_expected));
}

template <class T>
MatrixLocal<T> computeHFromTFactor(const SizeType k, const Tile<const T, Device::CPU>& t,
                                   const MatrixLocal<const T>& v) {
  const SizeType m = v.size().rows();
  const TileElementSize block_size = v.blockSize();

  // TV* = (VT*)* = W
  MatrixLocal<T> w({m, k}, block_size);
  std::copy(v.ptr(), v.ptr() + w.size().linear_size(), w.ptr());

  // clang-format off
  blas::trmm(blas::Layout::ColMajor,
      blas::Side::Right, blas::Uplo::Upper,
      blas::Op::ConjTrans, blas::Diag::NonUnit,
      m, k,
      1,
      t.ptr(), t.ld(),
      w.ptr(), w.ld());
  // clang-format on

  // H_result = I - V W*
  MatrixLocal<T> h_result({m, m}, block_size);
  set(h_result, preset_eye<T>);

  // clang-format off
  blas::gemm(blas::Layout::ColMajor,
      blas::Op::NoTrans, blas::Op::ConjTrans,
      m, m, k,
      -1,
      v.ptr(), v.ld(),
      w.ptr(), w.ld(),
      1,
      h_result.ptr(), h_result.ld());
  // clang-format on

  return h_result;
}

std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType, GlobalTileIndex>> configs{
    // m, n, mb, nb, k, v_start
    {39, 26, 6, 6, 6, {0, 3}},  // all reflectors (complete tile)
    {39, 26, 6, 6, 2, {1, 4}},  // all reflectors (incomplete tile)
    {39, 26, 6, 6, 4, {0, 3}},  // k reflectors (complete tile)
    {39, 26, 6, 6, 1, {1, 4}},  // k reflectors (incomplete tile)

    {26, 39, 6, 6, 6, {0, 5}},  // all reflectors (complete tile)
    {26, 39, 6, 6, 2, {1, 6}},  // all reflectors (incomplete tile)
    {26, 39, 6, 6, 4, {1, 5}},  // k reflectors (complete tile)
    {26, 39, 6, 6, 1, {0, 6}},  // k reflectors (incomplete tile)

    // non-square tiles
    {39, 26, 10, 4, 4, {1, 5}},  // all reflectors (complete tile)
    {39, 26, 10, 4, 2, {0, 6}},  // all reflectors (incomplete tile)
    {39, 26, 10, 4, 3, {1, 5}},  // k reflectors (complete tile)
    {39, 26, 10, 4, 1, {0, 6}},  // k reflectors (incomplete tile)

    {26, 39, 10, 6, 6, {1, 5}},  // all reflectors (complete tile)
    {26, 39, 10, 6, 2, {0, 6}},  // all reflectors (incomplete tile)
    {26, 39, 10, 6, 4, {0, 5}},  // k reflectors (complete tile)
    {26, 39, 10, 6, 1, {1, 6}},  // k reflectors (incomplete tile)

    {26, 39, 10, 6, 0, {0, 6}},  // 0 reflectors

    // empty matrices
    {0, 0, 10, 6, 0, {0, 0}},
    {39, 0, 10, 6, 0, {0, 0}},
    {0, 26, 10, 6, 0, {0, 0}},
};

// Note:
// Testing this function requires next input values:
// - V      the matrix with the elementary reflectors (columns)
// - taus   the set of tau coefficients, one for each reflector
//
// V is generated randomly and the related tau values are computed, and at the same time they are used to
// compute the expected resulting Householder transformation by applying one at a time the reflectors in
// V, by applying on each one the equation
//
// Hi = I - tau . v . v*
//
// and updating the final expected Householder transformation with
//
// H_exp = H_exp * Hi
//
// resulting in
//
// H = H1 . H2 . ... . Hk
//
// On the other side, from the function, by providing V and taus we get back a T factor, that we
// can use to compute the Householder transformation by applying all reflectors in block.
// This Householder transformation is obtained with the equation
//
// H = I - V . T . V*
//
// Which we expect to be the equal to the one computed previously.
TYPED_TEST(ComputeTFactorTestMC, CorrectnessLocal) {
  SizeType a_m, a_n, mb, nb, k;
  GlobalTileIndex v_start;

  for (auto config : configs) {
    std::tie(a_m, a_n, mb, nb, k, v_start) = config;

    ASSERT_LE(k, nb);

    const TileElementSize block_size(mb, nb);

    Matrix<const TypeParam, Device::CPU> v_input = [&]() {
      Matrix<TypeParam, Device::CPU> V({a_m, a_n}, block_size);
      dlaf::matrix::util::set_random(V);
      return V;
    }();

    GlobalElementSize v_size(a_m, a_n);

    if (!v_size.isEmpty()) {
      const auto v_start_el = GlobalElementIndex(v_start.row() * mb, v_start.col() * nb);
      const auto v_end_el = GlobalElementIndex{a_m, std::min((v_start.col() + 1) * nb, a_n)};
      v_size = v_end_el - v_start_el;
    }

    MatrixLocal<TypeParam> v(v_size, block_size);

    const GlobalTileSize v_offset{v_start.row(), v_start.col()};
    for (const auto& ij_tile : iterate_range2d(v.nrTiles())) {
      // copy only the panel
      const auto& source_tile = v_input.read(ij_tile + v_offset).get();
      matrix::internal::copy(source_tile, v.tile(ij_tile));
      if (ij_tile.row() == 0)
        tile::internal::laset<TypeParam>(lapack::MatrixType::Upper, 0.f, 1.f, v.tile(ij_tile));
    }

    auto tmp = computeHAndTFactor(k, v);
    hpx::shared_future<dlaf::common::internal::vector<TypeParam>> taus_input =
        hpx::make_ready_future<dlaf::common::internal::vector<TypeParam>>(std::move(std::get<0>(tmp)));
    auto h_expected = std::move(std::get<1>(tmp));

    is_orthogonal(h_expected);

    const auto max_k = std::min(mb, nb);
    Matrix<TypeParam, Device::CPU> t_output({max_k, max_k}, block_size);
    const LocalTileIndex t_idx(0, 0);

    dlaf::factorization::internal::computeTFactor<Backend::MC>(k, v_input, v_start, taus_input,
                                                               t_output(t_idx));

    // No reflectors, so nothing to check
    if (k == 0)
      continue;

    // Note:
    // In order to check T correctness, the test will compute the H transformation matrix
    // that will results from using it with the V panel.
    //
    // In particular
    //
    // H_res = I - V T V*
    //
    // is computed and compared to the one previously obtained by applying reflectors sequentially
    const auto& t = t_output.read(t_idx).get();
    MatrixLocal<TypeParam> h_result = computeHFromTFactor(k, t, v);

    is_orthogonal(h_result);

    SCOPED_TRACE(::testing::Message() << "Comparison test m = " << a_m << " n=" << a_n << " mb=" << mb
                                      << " nb=" << nb << " k=" << k << " v_start=" << v_start);
    const auto error = h_result.size().rows() * k * TypeUtilities<TypeParam>::error;
    CHECK_MATRIX_NEAR(h_expected, h_result, 0, error);
  }
}

TYPED_TEST(ComputeTFactorTestMC, CorrectnessDistributed) {
  SizeType a_m, a_n, mb, nb, k;
  GlobalTileIndex v_start;

  for (auto comm_grid : this->commGrids()) {
    for (auto config : configs) {
      std::tie(a_m, a_n, mb, nb, k, v_start) = config;

      ASSERT_LE(k, nb);

      const TileElementSize block_size(mb, nb);

      comm::Index2D source_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                      std::min(1, comm_grid.size().cols() - 1));
      const matrix::Distribution dist_v({a_m, a_n}, block_size, comm_grid.size(), comm_grid.rank(),
                                        source_rank_index);

      Matrix<const TypeParam, Device::CPU> v_input = [&]() {
        Matrix<TypeParam, Device::CPU> V(dist_v);
        dlaf::matrix::util::set_random(V);
        return V;
      }();

      const MatrixLocal<const TypeParam> v = [&v_input, &dist_v, &comm_grid, a_m, a_n, nb, v_start] {
        // TODO this can be improved by communicating just the selected column starting at v_start
        // gather the entire A matrix
        auto a = matrix::test::allGather<TypeParam>(lapack::MatrixType::General, v_input, comm_grid);

        // panel shape
        GlobalElementSize v_size = dist_v.size();

        if (!v_size.isEmpty()) {
          const auto v_start_el = dist_v.globalElementIndex(v_start, {0, 0});
          const auto v_end_el = GlobalElementIndex{a_m, std::min(v_start_el.col() + nb, a_n)};
          v_size = v_end_el - v_start_el;
        }

        MatrixLocal<TypeParam> v(v_size, a.blockSize());

        // copy only the panel
        const GlobalTileSize v_offset{v_start.row(), v_start.col()};
        for (const auto& ij : iterate_range2d(v.nrTiles())) {
          matrix::internal::copy(a.tile_read(ij + v_offset), v.tile(ij));
          if (ij.row() == 0)
            tile::internal::laset<TypeParam>(lapack::MatrixType::Upper, 0.f, 1.f, v.tile(ij));
        }

        return v;
      }();

      auto tmp = computeHAndTFactor(k, v);
      hpx::shared_future<dlaf::common::internal::vector<TypeParam>> taus_input =
          hpx::make_ready_future<dlaf::common::internal::vector<TypeParam>>(std::move(std::get<0>(tmp)));
      auto h_expected = std::move(std::get<1>(tmp));

      is_orthogonal(h_expected);

      common::Pipeline<comm::Communicator> serial_comm(comm_grid.colCommunicator());

      const auto max_k = std::min(mb, nb);
      Matrix<TypeParam, Device::CPU> t_output({max_k, max_k}, block_size);
      const LocalTileIndex t_idx(0, 0);

      dlaf::factorization::internal::computeTFactor<Backend::MC>(k, v_input, v_start, taus_input,
                                                                 t_output(t_idx), serial_comm);

      // No reflectors, so nothing to check
      if (k == 0)
        continue;

      const auto column_involved = dist_v.rankGlobalTile(v_start).col();
      if (dist_v.rankIndex().col() != column_involved)
        continue;

      // Note:
      // In order to check T correctness, the test will compute the H transformation matrix
      // that will results from using it with the V panel.
      //
      // In particular
      //
      // H_res = I - V T V*
      //
      // is computed and compared to the one previously obtained by applying reflectors sequentially
      const auto& t = t_output.read(t_idx).get();
      MatrixLocal<TypeParam> h_result = computeHFromTFactor(k, t, v);

      is_orthogonal(h_result);

      // Note:
      // The error threshold has been determined considering that ~2*n*nb arithmetic operations (n*nb
      // multiplications and n*nb addition) are needed to compute each of the element of the matrix `h_result`,
      // and that TypeUtilities<TypeParam>::error indicates maximum error for a multiplication + addition.
      SCOPED_TRACE(::testing::Message() << "Comparison test m = " << a_m << " n=" << a_n << " mb=" << mb
                                        << " nb=" << nb << " k=" << k << " v_start=" << v_start);
      const auto error = h_result.size().rows() * k * TypeUtilities<TypeParam>::error;
      CHECK_MATRIX_NEAR(h_expected, h_result, 0, error);
    }
  }
}
