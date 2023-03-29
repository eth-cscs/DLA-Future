//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/bt_band_to_tridiag.h"

#include <gtest/gtest.h>

#include "dlaf/eigensolver/band_to_tridiag.h"  // for nrSweeps/nrStepsForSweep
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::util;

using namespace dlaf::test;
using namespace dlaf::matrix::test;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class BacktransformationBandToTridiagTest : public TestWithCommGrids {};

template <class T>
using BacktransformationBandToTridiagTestMC = BacktransformationBandToTridiagTest<T>;

TYPED_TEST_SUITE(BacktransformationBandToTridiagTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using BacktransformationBandToTridiagTestGPU = BacktransformationBandToTridiagTest<T>;

TYPED_TEST_SUITE(BacktransformationBandToTridiagTestGPU, MatrixElementTypes);
#endif

// Note: Helper functions for computing the tau of a given reflector. Reflector pointer should
// point to its 2nd component, i.e. the 1st component equal to 1 is implicitly considered in the
// computation without the need to have it in-place in the reflector. Moreover, this also means
// that the size given is not equal to the reflector size, instead it is equal to the number of
// components after the first one (i.e. reflector size - 1).
struct calculateTau {
  template <class T>
  static T call(const T* v, const SizeType size) {
    const T dotprod = blas::dot(size, v, 1, v, 1) + 1;
    return 2 / dotprod;
  }

  template <class T>
  static std::complex<T> call(const std::complex<T>* v, const SizeType size) {
    const T dotprod = std::real(blas::dot(size, v, 1, v, 1)) + 1;
    return {T(1) / dotprod, T(1) / dotprod};
  }
};

template <class T>
void computeTaus(const SizeType max_refl_size, const SizeType k, matrix::Tile<T, Device::CPU> tile) {
  for (SizeType j = 0; j < k; ++j) {
    const SizeType size = std::min(max_refl_size, tile.size().rows());
    // Note: calculateTau implicitly considers the first component equal to 1
    DLAF_ASSERT(size > 0, size);
    const auto tau = calculateTau::call(tile.ptr({1, j}), size - 1);
    *tile.ptr({0, j}) = tau;
  }
}

template <Backend B, Device D, class T>
void testBacktransformation(SizeType m, SizeType n, SizeType mb, SizeType nb, const SizeType b) {
  Matrix<T, Device::CPU> mat_e_h({m, n}, {mb, nb});
  set_random(mat_e_h);
  auto mat_e_local = allGather(blas::Uplo::General, mat_e_h);

  Matrix<const T, Device::CPU> mat_hh = [m, mb, b]() {
    Matrix<T, Device::CPU> mat_hh({m, m}, {mb, mb});
    set_random(mat_hh);

    const auto& dist = mat_hh.distribution();

    for (SizeType j = 0; j < mat_hh.size().cols(); j += b) {
      for (SizeType i = j; i < mat_hh.size().rows(); i += b) {
        const GlobalElementIndex ij(i, j);

        const TileElementIndex sub_origin = dist.tileElementIndex(ij);
        const TileElementSize sub_size(std::min(b, mat_hh.size().rows() - ij.row()),
                                       std::min(b, mat_hh.size().cols() - ij.col()));

        const SizeType n = std::min(2 * b - 1, mat_hh.size().rows() - ij.row() - 1);
        const SizeType k = std::min(n - 1, sub_size.cols());

        if (k <= 0)
          continue;

        const GlobalTileIndex ij_tile = dist.globalTileIndex(ij);
        dlaf::internal::transformLiftDetach(dlaf::internal::Policy<dlaf::Backend::MC>(), computeTaus<T>,
                                            b, k, splitTile(mat_hh(ij_tile), {sub_origin, sub_size}));
      }
    }

    return mat_hh;
  }();

  MatrixLocal<T> mat_hh_local = allGather(blas::Uplo::Lower, mat_hh);

  {
    MatrixMirror<T, D, Device::CPU> mat_e(mat_e_h);
    eigensolver::backTransformationBandToTridiag<B>(b, mat_e.get(), mat_hh);
  }

  if (m == 0 || n == 0)
    return;

  using eigensolver::internal::nrStepsForSweep;
  using eigensolver::internal::nrSweeps;
  for (SizeType sweep = nrSweeps<T>(m) - 1; sweep >= 0; --sweep) {
    for (SizeType step = nrStepsForSweep(sweep, m, b) - 1; step >= 0; --step) {
      const SizeType j = sweep;
      const SizeType i = j + 1 + step * b;

      const SizeType size = std::min(b, m - i);
      const SizeType i_v = (i - 1) / b * b;

      T& v_head = *mat_hh_local.ptr({i_v, j});
      const T tau = v_head;
      v_head = 1;

      using blas::Side;
      lapack::larf(Side::Left, size, n, &v_head, 1, tau, mat_e_local.ptr({i, 0}), mat_e_local.ld());
    }
  }

  auto result = [&dist = mat_e_h.distribution(),
                 &mat_local = mat_e_local](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  CHECK_MATRIX_NEAR(result, mat_e_h, m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

template <Backend B, Device D, class T>
void testBacktransformation(comm::CommunicatorGrid grid, SizeType m, SizeType n, SizeType mb,
                            SizeType nb, const SizeType b) {
  const Distribution dist({m, n}, {mb, nb}, grid.size(), grid.rank(), {0, 0});

  Matrix<T, Device::CPU> mat_e_h(dist);
  set_random(mat_e_h);
  auto mat_e_local = allGather(blas::Uplo::General, mat_e_h, grid);

  Matrix<const T, Device::CPU> mat_hh = [grid, m, mb, b]() {
    const Distribution dist({m, m}, {mb, mb}, grid.size(), grid.rank(), {0, 0});

    Matrix<T, Device::CPU> mat_hh(dist);
    set_random(mat_hh);

    for (SizeType j = 0; j < dist.localNrTiles().cols(); j += b) {
      for (SizeType i = j; i < dist.localNrTiles().rows(); i += b) {
        const TileElementIndex sub_origin(0, 0);
        const TileElementSize sub_size(std::min(b, dist.localSize().rows() - i * b),
                                       std::min(b, dist.localSize().cols() - j * b));

        const SizeType n = std::min(2 * b - 1, dist.localSize().rows() - i * b - 1);
        const SizeType k = std::min(n - 1, sub_size.cols());

        if (k <= 0)
          continue;

        dlaf::internal::transformLiftDetach(dlaf::internal::Policy<dlaf::Backend::MC>(), computeTaus<T>,
                                            b, k,
                                            splitTile(mat_hh(LocalTileIndex{i, j}),
                                                      {sub_origin, sub_size}));
      }
    }

    return mat_hh;
  }();

  MatrixLocal<T> mat_hh_local = allGather(blas::Uplo::Lower, mat_hh, grid);

  {
    MatrixMirror<T, D, Device::CPU> mat_e(mat_e_h);
    eigensolver::backTransformationBandToTridiag<B>(grid, b, mat_e.get(), mat_hh);
  }

  if (m == 0 || n == 0)
    return;

  using eigensolver::internal::nrStepsForSweep;
  using eigensolver::internal::nrSweeps;
  for (SizeType sweep = nrSweeps<T>(m) - 1; sweep >= 0; --sweep) {
    for (SizeType step = nrStepsForSweep(sweep, m, b) - 1; step >= 0; --step) {
      const SizeType j = sweep;
      const SizeType i = j + 1 + step * b;

      const SizeType size = std::min(b, m - i);
      const SizeType i_v = (i - 1) / b * b;

      T& v_head = *mat_hh_local.ptr({i_v, j});
      const T tau = v_head;
      v_head = 1;

      using blas::Side;
      lapack::larf(Side::Left, size, n, &v_head, 1, tau, mat_e_local.ptr({i, 0}), mat_e_local.ld());
    }
  }

  auto result = [&dist = mat_e_h.distribution(),
                 &mat_local = mat_e_local](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  CHECK_MATRIX_NEAR(result, mat_e_h, m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

struct config_t {
  const SizeType m, n, mb, nb, b = mb;
};

std::vector<config_t> configs{
    {0, 0, 4, 4},                                  // empty
    {1, 1, 4, 4},   {2, 2, 4, 4},   {2, 2, 2, 2},  // edge-cases
    {12, 12, 4, 4}, {12, 12, 4, 3}, {20, 30, 5, 5}, {20, 30, 5, 6},
    {8, 8, 3, 3},   {10, 10, 3, 3}, {12, 12, 5, 5}, {12, 30, 5, 6},
};

TYPED_TEST(BacktransformationBandToTridiagTestMC, CorrectnessLocal) {
  for (const auto& [m, n, mb, nb, b] : configs)
    testBacktransformation<Backend::MC, Device::CPU, TypeParam>(m, n, mb, nb, b);
}

TYPED_TEST(BacktransformationBandToTridiagTestMC, CorrectnessDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [m, n, mb, nb, b] : configs) {
      testBacktransformation<Backend::MC, Device::CPU, TypeParam>(comm_grid, m, n, mb, nb, b);
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(BacktransformationBandToTridiagTestGPU, CorrectnessLocal) {
  for (const auto& [m, n, mb, nb, b] : configs)
    testBacktransformation<Backend::GPU, Device::GPU, TypeParam>(m, n, mb, nb, b);
}

TYPED_TEST(BacktransformationBandToTridiagTestGPU, CorrectnessDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [m, n, mb, nb, b] : configs) {
      testBacktransformation<Backend::GPU, Device::GPU, TypeParam>(comm_grid, m, n, mb, nb, b);
    }
  }
}
#endif

std::vector<config_t> configs_subband{
    {0, 12, 4, 4, 2}, {4, 4, 4, 4, 2}, {12, 12, 4, 4, 2}, {12, 25, 6, 3, 2}, {11, 13, 6, 4, 2},
};

TYPED_TEST(BacktransformationBandToTridiagTestMC, CorrectnessLocalSubBand) {
  for (const auto& [m, n, mb, nb, b] : configs_subband)
    testBacktransformation<Backend::MC, Device::CPU, TypeParam>(m, n, mb, nb, b);
}

TYPED_TEST(BacktransformationBandToTridiagTestMC, CorrectnessDistributedSubBand) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [m, n, mb, nb, b] : configs_subband) {
      testBacktransformation<Backend::MC, Device::CPU, TypeParam>(comm_grid, m, n, mb, nb, b);
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(BacktransformationBandToTridiagTestGPU, CorrectnessLocalSubBand) {
  for (const auto& [m, n, mb, nb, b] : configs_subband)
    testBacktransformation<Backend::GPU, Device::GPU, TypeParam>(m, n, mb, nb, b);
}

TYPED_TEST(BacktransformationBandToTridiagTestGPU, CorrectnessDistributedSubBand) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [m, n, mb, nb, b] : configs_subband) {
      testBacktransformation<Backend::GPU, Device::GPU, TypeParam>(comm_grid, m, n, mb, nb, b);
    }
  }
}
#endif
