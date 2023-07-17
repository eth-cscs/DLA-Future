//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <exception>
#include <functional>
#include <sstream>
#include <tuple>

#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/eigensolver/band_to_tridiag.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/traits.h>
#include <dlaf/tune.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_generic_lapack.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;
using pika::this_thread::experimental::sync_wait;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct EigensolverBandToTridiagTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(EigensolverBandToTridiagTest, MatrixElementTypes);

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, mb_1d, band_size}
    {0, 2, 2, 2},                                    // m = 0
    {1, 2, 2, 2},                                    // m = 1
    {5, 5, 5, 5},  {4, 4, 4, 2},                     // m = mb
    {4, 6, 6, 3},  {8, 4, 8, 2},   {16, 12, 12, 6},  // m != mb
    {18, 4, 8, 4}, {34, 6, 18, 6}, {37, 9, 9, 3}     // m != mb
};

template <class T, class... GridIfDistributed>
void testBandToTridiagOutputCorrectness(const blas::Uplo uplo, const SizeType band_size,
                                        const SizeType m, const SizeType mb,
                                        Matrix<const T, Device::CPU>& mat_a_h,
                                        Matrix<BaseType<T>, Device::CPU>& mat_trid,
                                        Matrix<T, Device::CPU>& mat_v, GridIfDistributed... grid) {
  auto mat_trid_local = matrix::test::allGather(blas::Uplo::General, mat_trid);
  MatrixLocal<T> mat_local(mat_a_h.size(), mat_a_h.blockSize());
  const auto ld = mat_local.ld();
  set(mat_local, [](auto) { return T{0}; });

  for (SizeType j = 0; j < m - 1; ++j) {
    mat_local({j, j}) = mat_trid_local({j, 0});
    mat_local({j + 1, j}) = mat_trid_local({j, 1});
    mat_local({j, j + 1}) = mat_trid_local({j, 1});
  }
  mat_local({m - 1, m - 1}) = mat_trid_local({m - 1, 0});

  auto mat_v_local = matrix::test::allGather(blas::Uplo::General, mat_v, grid...);

  auto apply_left_right = [&mat_local, m, ld](SizeType size_hhr, T* v, SizeType first_index) {
    dlaf::common::internal::SingleThreadedBlasScope single;

    T tau = v[0];
    v[0] = T{1};
    lapack::larf(blas::Side::Left, size_hhr, m, v, 1, tau, mat_local.ptr({first_index, 0}), ld);
    lapack::larf(blas::Side::Right, m, size_hhr, v, 1, dlaf::conj(tau), mat_local.ptr({0, first_index}),
                 ld);
  };

  if (isComplex_v<T> && m > 1) {
    T* v = mat_v_local.ptr({(m - 2) / band_size * band_size, m - 2});
    apply_left_right(1, v, m - 1);
  }

  for (SizeType sweep = m - 3; sweep >= 0; --sweep) {
    for (SizeType step = dlaf::util::ceilDiv(m - sweep - 2, band_size) - 1; step >= 0; --step) {
      SizeType first_index = sweep + 1 + step * band_size;
      SizeType size_hhr = std::min(band_size, m - first_index);

      SizeType i = (sweep / band_size + step) * band_size;
      T* v = mat_v_local.ptr({i, sweep});
      apply_left_right(size_hhr, v, first_index);
    }
  }

  // mat_a_h is a const input so it has not changed.
  auto res = [uplo, band_size, &mat_a_h, &mat_local](const GlobalElementIndex& index) {
    auto diag_index = index.row() - index.col();
    if (uplo == blas::Uplo::Upper && -diag_index >= 0 && -diag_index > band_size + 1)
      return mat_local(index);
    if (uplo == blas::Uplo::Lower && diag_index >= 0 && diag_index < band_size + 1)
      return mat_local(index);

    const auto& dist_a = mat_a_h.distribution();
    return sync_wait(mat_a_h.read(dist_a.globalTileIndex(index))).get()(dist_a.tileElementIndex(index));
  };

  CHECK_MATRIX_NEAR(res, mat_a_h, mb * m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

template <Device D, class T>
void testBandToTridiag(const blas::Uplo uplo, const SizeType band_size, const SizeType m,
                       const SizeType mb) {
  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<T, Device::CPU> mat_a_h(size, block_size);
  matrix::util::set_random_hermitian(mat_a_h);

  auto [mat_trid, mat_v] = [&]() {
    MatrixMirror<const T, D, Device::CPU> mat_a(mat_a_h);
    return eigensolver::band_to_tridiag<Backend::MC>(uplo, band_size, mat_a.get());
  }();

  if (m == 0)
    return;

  testBandToTridiagOutputCorrectness(uplo, band_size, m, mb, mat_a_h, mat_trid, mat_v);
}

template <Device D, class T>
void testBandToTridiag(CommunicatorGrid grid, blas::Uplo uplo, const SizeType band_size,
                       const SizeType m, const SizeType mb) {
  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));
  Distribution distr({m, m}, {mb, mb}, grid.size(), grid.rank(), src_rank_index);

  Matrix<T, Device::CPU> mat_a_h(std::move(distr));
  matrix::util::set_random_hermitian(mat_a_h);

  auto [mat_trid, mat_v] = [&]() {
    MatrixMirror<const T, D, Device::CPU> mat_a(mat_a_h);
    return eigensolver::band_to_tridiag<Backend::MC>(grid, uplo, band_size, mat_a.get());
  }();

  if (m == 0)
    return;

  // SCOPED_TRACE cannot yield.
  // As not all the tiles are needed by the algorithm,
  // this wait is needed to ensure that the full matrix is setup to avoid yielding.
  mat_a_h.waitLocalTiles();
  mat_trid.waitLocalTiles();
  mat_v.waitLocalTiles();
  SCOPED_TRACE(::testing::Message() << "size " << m << ", block " << mb << ", band " << band_size
                                    << ", grid " << grid.size());

  testBandToTridiagOutputCorrectness(uplo, band_size, m, mb, mat_a_h, mat_trid, mat_v, grid);
}

TYPED_TEST(EigensolverBandToTridiagTest, CorrectnessLocalFromCPU) {
  const blas::Uplo uplo = blas::Uplo::Lower;

  for (const auto& [m, mb, mb_1d, b] : sizes) {
    getTuneParameters().band_to_tridiag_1d_block_size_base = mb_1d;
    testBandToTridiag<Device::CPU, TypeParam>(uplo, b, m, mb);
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(EigensolverBandToTridiagTest, CorrectnessLocalFromGPU) {
  const blas::Uplo uplo = blas::Uplo::Lower;

  for (const auto& [m, mb, mb_1d, b] : sizes) {
    getTuneParameters().band_to_tridiag_1d_block_size_base = mb_1d;
    testBandToTridiag<Device::GPU, TypeParam>(uplo, b, m, mb);
  }
}
#endif

TYPED_TEST(EigensolverBandToTridiagTest, CorrectnessDistributed) {
  const blas::Uplo uplo = blas::Uplo::Lower;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [m, mb, mb_1d, b] : sizes) {
      getTuneParameters().band_to_tridiag_1d_block_size_base = mb_1d;
      testBandToTridiag<Device::CPU, TypeParam>(comm_grid, uplo, b, m, mb);
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(EigensolverBandToTridiagTest, CorrectnessDistributedFromGPU) {
  const blas::Uplo uplo = blas::Uplo::Lower;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [m, mb, mb_1d, b] : sizes) {
      getTuneParameters().band_to_tridiag_1d_block_size_base = mb_1d;
      testBandToTridiag<Device::GPU, TypeParam>(comm_grid, uplo, b, m, mb);
    }
  }
}
#endif
