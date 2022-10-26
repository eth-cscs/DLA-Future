//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/band_to_tridiag.h"

#include <exception>
#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/traits.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

#include "dlaf/matrix/print_csv.h"
#include "dlaf/matrix/print_numpy.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct EigensolverBandToTridiagTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(EigensolverBandToTridiagTest, MatrixElementTypes);

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, band_size}
    {0, 2, 2},                                                // m = 0
    {1, 2, 2},                                                // m = 1
    {5, 5, 5}, {4, 4, 2},                                     // m = mb
    {4, 6, 3}, {8, 4, 2}, {18, 4, 4}, {34, 6, 6}, {37, 9, 3}  // m != mb
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
    return mat_a_h.read(dist_a.globalTileIndex(index)).get()(dist_a.tileElementIndex(index));
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
    return eigensolver::bandToTridiag<Backend::MC>(uplo, band_size, mat_a.get());
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

  auto el_a = [](auto index) { return T(((double) 100 + 100 * index.row() - 99 * index.col()) / 500.); };
  set(mat_a_h, el_a);

  auto [mat_trid, mat_v] = [&]() {
    MatrixMirror<const T, D, Device::CPU> mat_a(mat_a_h);
    return eigensolver::bandToTridiag<Backend::MC>(grid, uplo, band_size, mat_a.get());
  }();

  if (m == 0)
    return;

  GlobalElementIndex unused = {m - 1, 1};
  mat_trid(mat_trid.distribution().globalTileIndex(unused))
      .get()(mat_trid.distribution().tileElementIndex(unused)) = BaseType<T>{9};
  {
    const LocalElementSize size(m, m);
    const TileElementSize block_size(mb, mb);

    Matrix<T, Device::CPU> mat_a_h(size, block_size);
    set(mat_a_h, el_a);

    auto mat_v_local = matrix::test::allGather(blas::Uplo::General, mat_v, grid);

    auto [mat_trid2, mat_v] = [&]() {
      MatrixMirror<const T, D, Device::CPU> mat_a(mat_a_h);
      return eigensolver::bandToTridiag<Backend::MC>(uplo, band_size, mat_a.get());
    }();

    auto up = [](T& a, const T& b) {
      a -= b;
      if (std::abs(a) < 1e-7)
        a = T{0};
    };
    for (SizeType j = 0; j < mat_v_local.size().cols(); ++j) {
      for (SizeType i = 0; i < mat_v_local.size().rows(); ++i) {
        GlobalElementIndex gei{i, j};
        auto ti = mat_v.distribution().tileElementIndex(gei);
        auto gi = mat_v.distribution().globalTileIndex(gei);
        up(mat_v(gi).get()(ti), mat_v_local(gei));
      }
    }

    auto& ref = mat_trid2;
    auto exp = [unused, &ref](const GlobalElementIndex& i) {
      if (i == unused)
        return BaseType<T>{9};
      return ref.read(ref.distribution().globalTileIndex(i))
          .get()(ref.distribution().tileElementIndex(i));
    };
    CHECK_MATRIX_NEAR(exp, mat_trid, 100 * m * TypeUtilities<T>::error,
                      100 * m * TypeUtilities<T>::error);
  }

  // SCOPED_TRACE cannot yield.
  mat_trid.waitLocalTiles();
  mat_v.waitLocalTiles();
  SCOPED_TRACE(::testing::Message() << "size " << m << ", block " << mb << ", band " << band_size
                                    << ", grid " << grid.size());

  testBandToTridiagOutputCorrectness(uplo, band_size, m, mb, mat_a_h, mat_trid, mat_v, grid);
}

TYPED_TEST(EigensolverBandToTridiagTest, CorrectnessLocalFromCPU) {
  const blas::Uplo uplo = blas::Uplo::Lower;

  for (const auto& [m, mb, b] : sizes)
    testBandToTridiag<Device::CPU, TypeParam>(uplo, b, m, mb);
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(EigensolverBandToTridiagTest, CorrectnessLocalFromGPU) {
  const blas::Uplo uplo = blas::Uplo::Lower;

  for (const auto& [m, mb, b] : sizes)
    testBandToTridiag<Device::GPU, TypeParam>(uplo, b, m, mb);
}
#endif

TYPED_TEST(EigensolverBandToTridiagTest, CorrectnessDistributed) {
  const blas::Uplo uplo = blas::Uplo::Lower;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [m, mb, b] : sizes) {
      //      if (comm_grid.rank() == comm::Index2D{0, 0}) {
      std::cout << "\n----------------------------\n" << std::endl;
      std::cout << m << ", " << mb << ":" << b << " " << comm_grid.size() << std::endl;
      //      }

      testBandToTridiag<Device::CPU, TypeParam>(comm_grid, uplo, b, m, mb);
    }
  }
}
