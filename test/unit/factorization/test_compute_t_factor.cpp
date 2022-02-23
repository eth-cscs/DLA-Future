//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/factorization/qr.h"

#include <tuple>

#include <gtest/gtest.h>
#include <blas.hh>
#include <pika/future.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/lapack/tile.h"  // workaround for importing lapack.hh
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/matrix/views.h"
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

template <class T>
struct ComputeTFactorTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(ComputeTFactorTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <class T>
struct ComputeTFactorTestGPU : public TestWithCommGrids {};

TYPED_TEST_SUITE(ComputeTFactorTestGPU, MatrixElementTypes);
#endif

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
    const SizeType k, const MatrixLocal<const T>& v, GlobalElementIndex v_start) {
  // PRE: m >= k
  const SizeType m = v.size().rows() - v_start.row();
  const TileElementSize block_size = v.blockSize();

  // compute taus and H_exp
  common::internal::vector<T> taus;
  taus.reserve(k);

  MatrixLocal<T> h_expected({m, m}, block_size);
  set(h_expected, preset_eye<T>);

  for (auto j = 0; j < k; ++j) {
    const SizeType reflector_size = m - j;

    const T* data_ptr = v.ptr({v_start.row() + j, v_start.col() + j});
    const auto norm = blas::nrm2(reflector_size, data_ptr, 1);
    const T tau = 2 / (norm * norm);

    taus.push_back(tau);

    MatrixLocal<T> h_i({reflector_size, reflector_size}, block_size);
    set(h_i, preset_eye<T>);

    // Hi = (I - tau . v . v*)
    // clang-format off
    blas::ger(blas::Layout::ColMajor,
        reflector_size, reflector_size,
        -tau,
        data_ptr, 1,
        data_ptr, 1,
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
                                   const MatrixLocal<const T>& v, GlobalElementIndex v_start) {
  // PRE: m >= k
  const SizeType m = v.size().rows() - v_start.row();
  const TileElementSize block_size = v.blockSize();

  // TV* = (VT*)* = W
  MatrixLocal<T> w({m, k}, block_size);
  lapack::lacpy(blas::Uplo::General, m, k, v.ptr(v_start), v.ld(), w.ptr(), w.ld());

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
      v.ptr(v_start), v.ld(),
      w.ptr(), w.ld(),
      1,
      h_result.ptr(), h_result.ld());
  // clang-format on

  return h_result;
}

std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, GlobalElementIndex>> configs{
    // m, k, mb, nb, v_start
    {39, 6, 6, 6, {0, 0}},  // all reflectors
    {39, 4, 6, 6, {6, 0}},  // k reflectors
    {26, 6, 6, 6, {0, 0}},  // all reflectors
    {26, 4, 6, 6, {6, 0}},  // k reflectors

    // non-square tiles
    {39, 4, 10, 4, {10, 0}},  // all reflectors
    {39, 3, 10, 4, {0, 0}},   // k reflectors
    {26, 6, 3, 6, {6, 0}},    // all reflectors
    {26, 3, 3, 6, {0, 0}},    // k reflectors

    // Same panel, different v_start
    {26, 4, 5, 4, {0, 0}},
    {26, 3, 5, 4, {1, 1}},
    {26, 2, 5, 4, {2, 2}},
    {26, 4, 5, 4, {3, 0}},
    {26, 4, 5, 4, {4, 0}},
    {26, 2, 5, 4, {5, 2}},
    {26, 4, 5, 4, {22, 0}},
    {26, 3, 5, 4, {23, 0}},
    {26, 2, 5, 4, {24, 2}},
    {26, 1, 5, 4, {25, 2}},
};

// Note:
// Testing this function requires the following input values:
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
template <class T, Backend B, Device D>
void testComputeTFactor(const SizeType m, const SizeType k, const SizeType mb, const SizeType nb,
                        const GlobalElementIndex v_start) {
  ASSERT_LE(v_start.row() + k, m);
  ASSERT_LE(v_start.col() + k, nb);

  const TileElementSize block_size(mb, nb);

  Matrix<T, Device::CPU> v_h({m, nb}, block_size);
  dlaf::matrix::util::set_random(v_h);
  auto dist_v = v_h.distribution();

  // set up HHReflectors correctly.
  for (SizeType i = v_start.row(); i < v_start.row() + k && i < m; ++i) {
    SizeType j = i - v_start.row() + v_start.col();
    GlobalElementIndex ij(i, j);
    auto tile = v_h(dist_v.globalTileIndex(ij)).get();
    auto ij_tile = dist_v.tileElementIndex(ij);
    tile(ij_tile) = T{1};
    for (SizeType jj = j + 1; jj < v_start.col() + k; ++jj) {
      tile({ij_tile.row(), jj}) = T{0};
    }
  }

  auto v_local = dlaf::matrix::test::allGather<T>(blas::Uplo::General, v_h);

  auto [taus, h_expected] = computeHAndTFactor(k, v_local, v_start);
  pika::shared_future<dlaf::common::internal::vector<T>> taus_input =
      pika::make_ready_future<dlaf::common::internal::vector<T>>(std::move(taus));

  is_orthogonal(h_expected);

  Matrix<T, Device::CPU> t_output_h({k, k}, {k, k});
  const LocalTileIndex t_idx(0, 0);

  {
    MatrixMirror<T, D, Device::CPU> v(v_h);
    MatrixMirror<T, D, Device::CPU> t_output(t_output_h);
    const matrix::SubPanelView panel_view(dist_v, v_start, k);
    Panel<Coord::Col, T, D> panel_v(dist_v);
    panel_v.setRangeStart(v_start);
    panel_v.setWidth(k);
    for (const auto& i : panel_view.iteratorLocal()) {
      panel_v.setTile(i, splitTile(v.get().read(i), panel_view(i)));
    }

    using dlaf::factorization::internal::computeTFactor;
    computeTFactor<B>(panel_v, taus_input, t_output.get()(t_idx));
  }

  // Note:
  // In order to check T correctness, the test will compute the H transformation matrix
  // that will results from using it with the V panel.
  //
  // In particular
  //
  // H_res = I - V T V*
  //
  // is computed and compared to the one previously obtained by applying reflectors sequentially
  const auto& t = t_output_h.read(t_idx).get();
  MatrixLocal<T> h_result = computeHFromTFactor(k, t, v_local, v_start);

  is_orthogonal(h_result);

  SCOPED_TRACE(::testing::Message() << "Comparison test m=" << m << " k=" << k << " mb=" << mb
                                    << " nb=" << nb << " v_start=" << v_start);
  const auto error = h_result.size().rows() * k * TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(h_expected, h_result, 0, error);
}

template <class T, Backend B, Device D>
void testComputeTFactor(comm::CommunicatorGrid grid, const SizeType m, const SizeType k,
                        const SizeType mb, const SizeType nb, const GlobalElementIndex v_start) {
  ASSERT_LE(v_start.row() + k, m);
  ASSERT_LE(v_start.col() + k, nb);

  const TileElementSize block_size(mb, nb);

  comm::Index2D source_rank_index(std::max(0, grid.size().rows() - 1),
                                  std::min(1, grid.size().cols() - 1));

  const matrix::Distribution dist_v({m, nb}, block_size, grid.size(), grid.rank(), source_rank_index);

  Matrix<T, Device::CPU> v_h(dist_v);
  dlaf::matrix::util::set_random(v_h);

  // set up HHReflectors correctly.
  for (SizeType i = v_start.row(); i < v_start.row() + k && i < m; ++i) {
    SizeType j = i - v_start.row() + v_start.col();
    GlobalElementIndex ij(i, j);
    if (dist_v.rankIndex() == dist_v.rankGlobalTile(dist_v.globalTileIndex(ij))) {
      auto tile = v_h(dist_v.globalTileIndex(ij)).get();
      auto ij_tile = dist_v.tileElementIndex(ij);
      tile(ij_tile) = T{1};
      for (SizeType jj = j + 1; jj < v_start.col() + k; ++jj) {
        tile({ij_tile.row(), jj}) = T{0};
      }
    }
  }

  auto v_local = matrix::test::allGather<T>(blas::Uplo::General, v_h, grid);

  // Ranks without HHReflectors can return.
  if (dist_v.rankIndex().col() != source_rank_index.col())
    return;

  auto [taus, h_expected] = computeHAndTFactor(k, v_local, v_start);
  pika::shared_future<dlaf::common::internal::vector<T>> taus_input =
      pika::make_ready_future<dlaf::common::internal::vector<T>>(std::move(taus));

  is_orthogonal(h_expected);

  common::Pipeline<comm::Communicator> serial_comm(grid.colCommunicator());

  Matrix<T, Device::CPU> t_output_h({k, k}, {k, k});
  const LocalTileIndex t_idx(0, 0);

  {
    MatrixMirror<T, D, Device::CPU> v(v_h);
    MatrixMirror<T, D, Device::CPU> t_output(t_output_h);
    const matrix::SubPanelView panel_view(dist_v, v_start, k);
    Panel<Coord::Col, T, D> panel_v(dist_v);
    panel_v.setRangeStart(v_start);
    panel_v.setWidth(k);
    for (const auto& i : panel_view.iteratorLocal()) {
      panel_v.setTile(i, splitTile(v.get().read(i), panel_view(i)));
    }

    using dlaf::factorization::internal::computeTFactor;
    computeTFactor<B>(panel_v, taus_input, t_output.get()(t_idx), serial_comm);
  }

  // Note:
  // In order to check T correctness, the test will compute the H transformation matrix
  // that will results from using it with the V panel.
  //
  // In particular
  //
  // H_res = I - V T V*
  //
  // is computed and compared to the one previously obtained by applying reflectors sequentially
  const auto& t = t_output_h.read(t_idx).get();
  MatrixLocal<T> h_result = computeHFromTFactor(k, t, v_local, v_start);

  is_orthogonal(h_result);

  // Note:
  // The error threshold has been determined considering that ~2*n*nb arithmetic operations (n*nb
  // multiplications and n*nb addition) are needed to compute each of the element of the matrix `h_result`,
  // and that TypeUtilities<T>::error indicates maximum error for a multiplication + addition.
  SCOPED_TRACE(::testing::Message() << "Comparison test m=" << m << " k=" << k << " mb=" << mb
                                    << " nb=" << nb << " v_start=" << v_start);
  const auto error = h_result.size().rows() * k * TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(h_expected, h_result, 0, error);
}

TYPED_TEST(ComputeTFactorTestMC, CorrectnessLocal) {
  for (const auto& [m, k, mb, nb, v_start] : configs) {
    testComputeTFactor<TypeParam, Backend::MC, Device::CPU>(m, k, mb, nb, v_start);
  }
}

TYPED_TEST(ComputeTFactorTestMC, CorrectnessDistributed) {
  for (auto comm_grid : {this->commGrids()[0]}) {
    for (const auto& [m, k, mb, nb, v_start] : configs) {
      testComputeTFactor<TypeParam, Backend::MC, Device::CPU>(comm_grid, m, k, mb, nb, v_start);
    }
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(ComputeTFactorTestGPU, CorrectnessLocal) {
  for (const auto& [m, k, mb, nb, v_start] : configs) {
    testComputeTFactor<TypeParam, Backend::GPU, Device::GPU>(m, k, mb, nb, v_start);
  }
}
#endif
