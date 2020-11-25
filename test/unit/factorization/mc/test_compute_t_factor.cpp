//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/factorization/qr.h"

#include <gtest/gtest.h>
#include <blas.hh>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/lapack_tile.h"  // workaround for importing lapack.hh
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using dlaf::Device;
using dlaf::Matrix;
using dlaf::comm::CommunicatorGrid;
using dlaf::matrix::test::MatrixLocal;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new dlaf::test::CommunicatorGrid6RanksEnvironment);

template <typename Type>
struct ComputeTFactorDistributedTest : public ::testing::Test {
  const std::vector<dlaf::comm::CommunicatorGrid>& commGrids() {
    return dlaf::test::comm_grids;
  }
};

TYPED_TEST_SUITE(ComputeTFactorDistributedTest, dlaf::test::MatrixElementTypes);

template <class T>
T preset_eye(const dlaf::GlobalElementIndex& index) {
  return index.row() == index.col() ? 1 : 0;
}

template <class T>
void is_orthogonal(const MatrixLocal<const T>& matrix) {
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

  constexpr auto error = 2 * dlaf::test::TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(eye, ortho, error, error);
}

std::vector<std::array<dlaf::SizeType, 5>> configs{
    // m, n, mb, nb, k
    {39, 26, 6, 6, 6},  // all reflectors
    {39, 26, 6, 6, 4},  // k reflectors
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
TYPED_TEST(ComputeTFactorDistributedTest, Correctness) {
  using namespace dlaf;

  constexpr auto error = test::TypeUtilities<TypeParam>::error;

  SizeType a_m, a_n, mb, nb, k;

  for (auto comm_grid : this->commGrids()) {
    for (auto config : configs) {
      std::tie(a_m, a_n, mb, nb, k) = config;

      const TileElementSize block_size(mb, nb);

      comm::Index2D source_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                      std::min(1, comm_grid.size().cols() - 1));
      const matrix::Distribution dist_v({a_m, a_n}, block_size, comm_grid.size(), comm_grid.rank(),
                                        source_rank_index);

      const GlobalTileIndex v_start{dist_v.nrTiles().rows() / 2, dist_v.nrTiles().cols() / 2};

      Matrix<const TypeParam, Device::CPU> v_input = [&]() {
        Matrix<TypeParam, Device::CPU> V(dist_v);
        dlaf::matrix::util::set_random(V);
        return V;
      }();

      const MatrixLocal<const TypeParam> v = [&v_input, &dist_v, &comm_grid, a_m, a_n, nb, v_start] {
        // TODO this can be improved by communicating just the interesting part
        // gather the entire A matrix
        auto a = matrix::test::all_gather<TypeParam>(v_input, comm_grid);

        // panel shape
        const auto v_start_el = dist_v.globalElementIndex(v_start, {0, 0});
        const auto v_end_el = GlobalElementIndex{a_m, std::min(v_start_el.col() + nb, a_n)};
        const auto v_size_el = v_end_el - v_start_el;

        MatrixLocal<TypeParam> v(v_size_el, a.blockSize());
        DLAF_ASSERT_HEAVY(v.size().rows() > v.size().cols(), v.size());

        // copy just the panel
        const GlobalTileSize v_offset{v_start.row(), v_start.col()};
        for (const auto& ij : iterate_range2d(v.nrTiles()))
          copy(a.tile_read(ij + v_offset), v.tile(ij));

        // clean reflectors
        lapack::laset(lapack::MatrixType::Upper, v.size().rows(), v.size().cols(), 0, 1, v.ptr(),
                      v.ld());

        return v;
      }();

      const SizeType m = v.size().rows();

      // compute taus and H_exp
      common::internal::vector<hpx::shared_future<TypeParam>> taus_input;
      taus_input.reserve(k);

      MatrixLocal<TypeParam> h_expected({m, m}, block_size);
      set(h_expected, preset_eye<TypeParam>);

      for (auto j = 0; j < k; ++j) {
        const TypeParam* data_ptr = v.ptr({j, j});
        const auto norm = blas::nrm2(m - j, data_ptr, 1);
        const TypeParam tau = 2 / std::pow(norm, 2);

        taus_input.push_back(hpx::make_ready_future<TypeParam>(tau));

        // TODO work just on the submatrix

        MatrixLocal<TypeParam> h_i({m, m}, block_size);
        set(h_i, preset_eye<TypeParam>);

        // Hi = (I - tau . v . v*)
        // clang-format off
        blas::ger(blas::Layout::ColMajor,
            m - j, m - j,
            -tau,
            v.ptr({j, j}), 1,
            v.ptr({j, j}), 1,
            h_i.ptr({j, j}), h_i.ld());
        // clang-format on

        // H_exp = H_exp . Hi
        // clang-format off
        MatrixLocal<TypeParam> workspace(h_expected.size(), h_expected.blockSize());
        blas::gemm(blas::Layout::ColMajor,
            blas::Op::NoTrans, blas::Op::NoTrans,
            m, m, m,
            1,
            h_expected.ptr(), h_expected.ld(),
            h_i.ptr(), h_i.ld(),
            0,
            workspace.ptr(), workspace.ld());
        // clang-format on
        copy(workspace, h_expected);
      }

      is_orthogonal(h_expected);

      common::Pipeline<comm::CommunicatorGrid> serial_comm(comm_grid);
      Matrix<TypeParam, Device::CPU> t_output(LocalElementSize{k, k}, block_size);

      dlaf::factorization::internal::computeTFactor<Backend::MC>(k, v_input, v_start, taus_input,
                                                                 t_output, serial_comm);

      // Note:
      const comm::Index2D owner_t = dist_v.rankGlobalTile(v_start);
      if (dist_v.rankIndex() != owner_t)
        continue;

      // Note:
      // T factor is reduced just on the rank owning V0.
      //
      // In order to check T correcteness, the test will compute the H transformation matrix
      // that will results from using it with the V panel.
      //
      // In particular
      //
      // H_res = I - V T V*
      //
      // is computed and compared to the one previously obtained by applying reflectors sequentially

      const auto& t = t_output.read(LocalTileIndex{0, 0}).get();

      // TV* = (VT*)* = W
      MatrixLocal<TypeParam> w({m, k}, block_size);
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
      MatrixLocal<TypeParam> h_result(h_expected.size(), block_size);
      set(h_result, preset_eye<TypeParam>);

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

      is_orthogonal(h_result);

      CHECK_MATRIX_NEAR(h_expected, h_result, error, error);
    }
  }
}
