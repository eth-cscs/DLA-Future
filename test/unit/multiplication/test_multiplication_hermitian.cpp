//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/multiplication/hermitian.h"

#include <functional>
#include <tuple>

#include <gtest/gtest.h>
#include <pika/runtime.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_generic_blas.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::util;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct HermitianMultiplicationTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(HermitianMultiplicationTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
struct HermitianMultiplicationTestGPU : public TestWithCommGrids {};

TYPED_TEST_SUITE(HermitianMultiplicationTestGPU, MatrixElementTypes);
#endif

const std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    {0, 0, 1, 1},                                                // m, n = 0
    {0, 2, 1, 2}, {7, 0, 2, 1},                                  // m = 0 or n = 0
    {2, 2, 5, 5}, {10, 10, 2, 3}, {7, 7, 3, 2},                  // m = n
    {3, 2, 7, 7}, {12, 3, 5, 5},  {7, 6, 3, 2}, {15, 7, 3, 5},   // m > n
    {2, 3, 7, 7}, {4, 13, 5, 5},  {7, 8, 2, 9}, {19, 25, 6, 5},  // m < n
};

template <class T, Backend B, Device D>
void testHermitianMultiplication(const blas::Side side, const blas::Uplo uplo, const SizeType m,
                                 const SizeType n, const SizeType mb, const SizeType nb, const T alpha,
                                 const T beta) {
  const SizeType k = side == blas::Side::Left ? m : n;
  const SizeType kb = side == blas::Side::Left ? mb : nb;

  const LocalElementSize size_a(k, k);
  const TileElementSize block_size_a(kb, kb);
  Matrix<T, Device::CPU> mat_ah(size_a, block_size_a);

  const LocalElementSize size_b(m, n);
  const TileElementSize block_size_b(mb, nb);
  Matrix<T, Device::CPU> mat_bh(size_b, block_size_b);

  const LocalElementSize size_c(m, n);
  const TileElementSize block_size_c(mb, nb);
  Matrix<T, Device::CPU> mat_ch(size_c, block_size_c);

  auto [el_a, el_b, el_c, res_c] =
      getHermitianMatrixMultiplication<GlobalElementIndex, T>(side, uplo, k, alpha, beta);

  set(mat_ah, el_a);
  set(mat_bh, el_b);
  set(mat_ch, el_c);

  {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<T, D, Device::CPU> mat_b(mat_bh);
    MatrixMirror<T, D, Device::CPU> mat_c(mat_ch);

    multiplication::hermitian<B>(side, uplo, alpha, mat_a.get(), mat_b.get(), beta, mat_c.get());
  }

  // SCOPED_TRACE cannot yield.
  mat_ch.waitLocalTiles();
  SCOPED_TRACE(::testing::Message() << "m " << m << "n " << n << ", mb " << mb << ", nb " << nb);
  CHECK_MATRIX_NEAR(res_c, mat_ch, 10 * (m + 1) * TypeUtilities<T>::error,
                    10 * (m + 1) * TypeUtilities<T>::error);
}

template <class T, Backend B, Device D>
void testHermitianMultiplication(comm::CommunicatorGrid grid, const blas::Side side,
                                 const blas::Uplo uplo, const SizeType m, const SizeType n,
                                 const SizeType mb, const SizeType nb, const T alpha, const T beta) {
  const SizeType k = side == blas::Side::Left ? m : n;
  const SizeType kb = side == blas::Side::Left ? mb : nb;
  const Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  const GlobalElementSize size_a(k, k);
  const TileElementSize block_size_a(kb, kb);
  Distribution distr_a(size_a, block_size_a, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_ah(std::move(distr_a));

  const GlobalElementSize size_b(m, n);
  const TileElementSize block_size_b(mb, nb);
  Distribution distr_b(size_b, block_size_b, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_bh(std::move(distr_b));

  const GlobalElementSize size_c(m, n);
  const TileElementSize block_size_c(mb, nb);
  Distribution distr_c(size_c, block_size_c, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_ch(std::move(distr_c));

  auto [el_a, el_b, el_c, res_c] =
      getHermitianMatrixMultiplication<GlobalElementIndex, T>(side, uplo, k, alpha, beta);

  set(mat_ah, el_a);
  set(mat_bh, el_b);
  set(mat_ch, el_c);

  {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<T, D, Device::CPU> mat_b(mat_bh);
    MatrixMirror<T, D, Device::CPU> mat_c(mat_ch);

    multiplication::hermitian<B>(grid, side, uplo, alpha, mat_a.get(), mat_b.get(), beta, mat_c.get());
  }

  SCOPED_TRACE(::testing::Message() << "m " << m << ", n " << n << ", mb " << mb << ", nb " << nb);
  CHECK_MATRIX_NEAR(res_c, mat_ch, 10 * (m + 1) * TypeUtilities<T>::error,
                    10 * (m + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(HermitianMultiplicationTestMC, CorrectnessLocal) {
  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      if (side != blas::Side::Left || uplo != blas::Uplo::Lower)
        continue;
      for (const auto& [m, n, mb, nb] : sizes) {
        TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);
        TypeParam beta = TypeUtilities<TypeParam>::element(1.12, -.1);

        testHermitianMultiplication<TypeParam, Backend::MC, Device::CPU>(side, uplo, m, n, mb, nb, alpha,
                                                                         beta);
      }
    }
  }
}

TYPED_TEST(HermitianMultiplicationTestMC, CorrectnessDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto side : blas_sides) {
      for (const auto uplo : blas_uplos) {
        if (side != blas::Side::Left || uplo != blas::Uplo::Lower)
          continue;

        for (const auto& [m, n, mb, nb] : sizes) {
          TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);
          TypeParam beta = TypeUtilities<TypeParam>::element(1.12, -.1);

          testHermitianMultiplication<TypeParam, Backend::MC, Device::CPU>(comm_grid, side, uplo, m, n,
                                                                           mb, nb, alpha, beta);
          pika::threads::get_thread_manager().wait();
        }
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(HermitianMultiplicationTestGPU, CorrectnessLocal) {
  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      if (side != blas::Side::Left || uplo != blas::Uplo::Lower)
        continue;
      for (const auto& [m, n, mb, nb] : sizes) {
        TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);
        TypeParam beta = TypeUtilities<TypeParam>::element(1.12, -.1);

        testHermitianMultiplication<TypeParam, Backend::GPU, Device::GPU>(side, uplo, m, n, mb, nb,
                                                                          alpha, beta);
      }
    }
  }
}

TYPED_TEST(HermitianMultiplicationTestGPU, CorrectnessDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto side : blas_sides) {
      for (const auto uplo : blas_uplos) {
        if (side != blas::Side::Left || uplo != blas::Uplo::Lower)
          continue;

        for (const auto& [m, n, mb, nb] : sizes) {
          TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);
          TypeParam beta = TypeUtilities<TypeParam>::element(1.12, -.1);

          testHermitianMultiplication<TypeParam, Backend::GPU, Device::GPU>(comm_grid, side, uplo, m, n,
                                                                            mb, nb, alpha, beta);
          pika::threads::get_thread_manager().wait();
        }
      }
    }
  }
}
#endif
