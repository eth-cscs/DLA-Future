//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <blas.hh>

#include <dlaf/common/range2d.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/factorization/qr.h>
#include <dlaf/factorization/qr/internal/get_tfactor_num_workers.h>
#include <dlaf/lapack/tile.h>  // workaround for importing lapack.hh
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/matrix/views.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::test;
using dlaf::comm::CommunicatorGrid;
using dlaf::matrix::test::MatrixLocal;

using pika::execution::experimental::any_sender;
using pika::execution::experimental::just;
using pika::this_thread::experimental::sync_wait;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct ComputeTFactorTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(ComputeTFactorTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
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
  {
    dlaf::common::internal::SingleThreadedBlasScope single;

    blas::gemm(blas::Layout::ColMajor,
               blas::Op::NoTrans, blas::Op::ConjTrans,
               matrix.size().rows(), matrix.size().cols(), matrix.size().rows(),
               1,
               matrix.ptr(), matrix.ld(),
               matrix.ptr(), matrix.ld(),
               0,
               ortho.ptr(), ortho.ld());
  }
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
std::tuple<Matrix<T, Device::CPU>, MatrixLocal<T>> computeHAndTFactor(const SizeType k,
                                                                      const MatrixLocal<const T>& v,
                                                                      GlobalElementIndex v_start) {
  // PRE: m >= k
  dlaf::common::internal::SingleThreadedBlasScope single;

  const SizeType m = v.size().rows() - v_start.row();
  const TileElementSize block_size = v.blockSize();

  // compute taus and H_exp
  Matrix<T, Device::CPU> mat_taus(matrix::Distribution(GlobalElementSize(k, 1), TileElementSize(k, 1),
                                                       comm::Size2D(1, 1), comm::Index2D(0, 0),
                                                       comm::Index2D(0, 0)));
  auto taus_tile = sync_wait(mat_taus.readwrite(GlobalTileIndex(0, 0)));

  MatrixLocal<T> h_expected({m, m}, block_size);
  set(h_expected, preset_eye<T>);

  for (auto j = 0; j < k; ++j) {
    const SizeType reflector_size = m - j;

    const T* data_ptr = v.ptr({v_start.row() + j, v_start.col() + j});
    const auto norm = blas::nrm2(reflector_size, data_ptr, 1);
    const T tau = 2 / (norm * norm);

    taus_tile(TileElementIndex(j, 0)) = tau;

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

  return std::make_tuple(std::move(mat_taus), std::move(h_expected));
}

template <class T>
MatrixLocal<T> computeHFromTFactor(const SizeType k, const Tile<const T, Device::CPU>& t,
                                   const MatrixLocal<const T>& v, GlobalElementIndex v_start) {
  // PRE: m >= k
  dlaf::common::internal::SingleThreadedBlasScope single;

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
  using dlaf::factorization::internal::computeTFactor;

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
    auto tile = sync_wait(v_h.readwrite(dist_v.globalTileIndex(ij)));
    auto ij_tile = dist_v.tileElementIndex(ij);
    tile(ij_tile) = T{1};
    for (SizeType jj = j + 1; jj < v_start.col() + k; ++jj) {
      tile({ij_tile.row(), jj}) = T{0};
    }
  }

  auto v_local = dlaf::matrix::test::allGather<const T>(blas::Uplo::General, v_h);

  auto [mat_taus_h, h_expected] = computeHAndTFactor(k, v_local, v_start);

  is_orthogonal(h_expected);

  Matrix<T, Device::CPU> t_output_h({k, k}, {k, k});
  const LocalTileIndex t_idx(0, 0);

  {
    MatrixMirror<const T, D, Device::CPU> v(v_h);
    MatrixMirror<T, D, Device::CPU> t_output(t_output_h);
    MatrixMirror<const T, D, Device::CPU> mat_taus(mat_taus_h);

    const matrix::SubPanelView panel_view(dist_v, v_start, k);
    Panel<Coord::Col, T, D> panel_v(dist_v);
    panel_v.setRangeStart(v_start);
    panel_v.setWidth(k);
    for (const auto& i : panel_view.iteratorLocal()) {
      panel_v.setTile(i, splitTile(v.get().read(i), panel_view(i)));
    }

    auto workspaces = [k]() -> matrix::Panel<Coord::Col, T, D> {
      const SizeType nworkspaces = to_SizeType(
          std::max<std::size_t>(0, factorization::internal::get_tfactor_num_workers<B>() - 1));
      const SizeType nrefls_step = k;
      return matrix::Panel<Coord::Col, T, D>({{nworkspaces * nrefls_step, nrefls_step},
                                              {nrefls_step, nrefls_step}});
    }();

    computeTFactor<B>(panel_v, mat_taus.get().read(GlobalTileIndex(0, 0)),
                      t_output.get().readwrite(t_idx), workspaces);
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
  auto t_holder = sync_wait(t_output_h.read(t_idx));
  const auto& t = t_holder.get();
  MatrixLocal<T> h_result = computeHFromTFactor(k, t, v_local, v_start);

  is_orthogonal(h_result);

  SCOPED_TRACE(::testing::Message() << "Comparison test m=" << m << " k=" << k << " mb=" << mb
                                    << " nb=" << nb << " v_start=" << v_start);
  const auto error = h_result.size().rows() * k * TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(h_expected, h_result, 0, error);
}

template <class T, Backend B, Device D>
void testComputeTFactor(comm::CommunicatorGrid& grid, const SizeType m, const SizeType k,
                        const SizeType mb, const SizeType nb, const GlobalElementIndex v_start) {
  using dlaf::factorization::internal::computeTFactor;

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
      auto tile = sync_wait(v_h.readwrite(dist_v.globalTileIndex(ij)));
      auto ij_tile = dist_v.tileElementIndex(ij);
      tile(ij_tile) = T{1};
      for (SizeType jj = j + 1; jj < v_start.col() + k; ++jj) {
        tile({ij_tile.row(), jj}) = T{0};
      }
    }
  }

  auto v_local = matrix::test::allGather<const T>(blas::Uplo::General, v_h, grid);

  // Ranks without HHReflectors can return.
  if (dist_v.rankIndex().col() != source_rank_index.col())
    return;

  auto [mat_taus_h, h_expected] = computeHAndTFactor(k, v_local, v_start);

  is_orthogonal(h_expected);

  auto serial_comm(grid.col_communicator_pipeline());

  Matrix<T, Device::CPU> t_output_h({k, k}, {k, k});
  const LocalTileIndex t_idx(0, 0);

  {
    MatrixMirror<const T, D, Device::CPU> v(v_h);
    MatrixMirror<T, D, Device::CPU> t_output(t_output_h);
    MatrixMirror<const T, D, Device::CPU> mat_taus(mat_taus_h);
    const matrix::SubPanelView panel_view(dist_v, v_start, k);
    Panel<Coord::Col, T, D> panel_v(dist_v);
    panel_v.setRangeStart(v_start);
    panel_v.setWidth(k);
    for (const auto& i : panel_view.iteratorLocal()) {
      panel_v.setTile(i, splitTile(v.get().read(i), panel_view(i)));
    }

    auto workspaces = [k]() -> matrix::Panel<Coord::Col, T, D> {
      const SizeType nworkspaces = to_SizeType(
          std::max<std::size_t>(0, factorization::internal::get_tfactor_num_workers<B>() - 1));
      const SizeType nrefls_step = k;
      return matrix::Panel<Coord::Col, T, D>({{nworkspaces * nrefls_step, nrefls_step},
                                              {nrefls_step, nrefls_step}});
    }();

    computeTFactor<B>(panel_v, mat_taus.get().read(GlobalTileIndex(0, 0)),
                      t_output.get().readwrite(t_idx), workspaces, serial_comm);
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
  auto t_holder = sync_wait(t_output_h.read(t_idx));
  const auto& t = t_holder.get();
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
  for (auto& comm_grid : this->commGrids()) {
    for (const auto& [m, k, mb, nb, v_start] : configs) {
      testComputeTFactor<TypeParam, Backend::MC, Device::CPU>(comm_grid, m, k, mb, nb, v_start);
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(ComputeTFactorTestGPU, CorrectnessLocal) {
  for (const auto& [m, k, mb, nb, v_start] : configs) {
    testComputeTFactor<TypeParam, Backend::GPU, Device::GPU>(m, k, mb, nb, v_start);
  }
}

TYPED_TEST(ComputeTFactorTestGPU, CorrectnessDistributed) {
  for (auto& comm_grid : this->commGrids()) {
    for (const auto& [m, k, mb, nb, v_start] : configs) {
      testComputeTFactor<TypeParam, Backend::GPU, Device::GPU>(comm_grid, m, k, mb, nb, v_start);
    }
  }
}

#endif
