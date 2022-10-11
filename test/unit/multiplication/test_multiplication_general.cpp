//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/multiplication/general.h"

#include <gtest/gtest.h>

#include "dlaf/blas/enum_output.h"
#include "dlaf/common/assert.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_generic_blas.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::test;

template <class T>
struct GeneralMultiplicationTestMC : public ::testing::Test {};

TYPED_TEST_SUITE(GeneralMultiplicationTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
struct GeneralMultiplicationTestGPU : public ::testing::Test {};

TYPED_TEST_SUITE(GeneralMultiplicationTestGPU, MatrixElementTypes);
#endif

const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType, SizeType>> sizes = {
    // m, n, k, mb, a, b
    // full gemm
    {3, 3, 3, 1, 0, 2},
    {3, 3, 3, 3, 0, 0},
    {6, 6, 6, 3, 0, 1},
    {9, 9, 9, 3, 0, 2},
    {21, 21, 21, 3, 0, 6},
    // sub gemm
    {9, 9, 9, 3, 1, 2},
    {21, 21, 21, 3, 3, 6},
};

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

template <class T, Backend B, Device D>
void testGeneralMultiplication(const SizeType a, const SizeType b, const blas::Op opA,
                               const blas::Op opB, const T alpha, const T beta, const SizeType m,
                               const SizeType n, const SizeType k, const SizeType mb) {
  const SizeType a_el = a * mb;
  const SizeType b_el = std::min((b + 1) * mb - 1, m - 1);

  auto [refA, refB, refC, refResult] =
      matrix::test::getSubMatrixMatrixMultiplication(a_el, b_el, m, n, k, alpha, beta, opA, opB);

  auto setMatrix = [&](auto elSetter, const LocalElementSize size, const TileElementSize block_size) {
    Matrix<T, Device::CPU> matrix(size, block_size);
    dlaf::matrix::util::set(matrix, elSetter);
    return matrix;
  };

  Matrix<const T, Device::CPU> mat_ah = setMatrix(refA, {m, k}, {mb, mb});
  Matrix<const T, Device::CPU> mat_bh = setMatrix(refB, {k, n}, {mb, mb});
  Matrix<T, Device::CPU> mat_ch = setMatrix(refC, {m, n}, {mb, mb});

  {
    MatrixMirror<const T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<const T, D, Device::CPU> mat_b(mat_bh);
    MatrixMirror<T, D, Device::CPU> mat_c(mat_ch);

    multiplication::generalSubMatrix<B>(a, b, opA, opB, alpha, mat_a.get(), mat_b.get(), beta,
                                        mat_c.get());
  }

  CHECK_MATRIX_NEAR(refResult, mat_ch, 40 * (mat_ch.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_ch.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(GeneralMultiplicationTestMC, CorrectnessLocal) {
  for (const auto opA : blas_ops) {
    for (const auto opB : blas_ops) {
      // Note: not yet implemented
      if (opA != blas::Op::NoTrans || opB != blas::Op::NoTrans)
        continue;

      for (const auto& [m, n, k, mb, a, b] : sizes) {
        const TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
        const TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);
        testGeneralMultiplication<TypeParam, Backend::MC, Device::CPU>(a, b, opA, opB, alpha, beta, m, n,
                                                                       k, mb);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GeneralMultiplicationTestGPU, CorrectnessLocal) {
  for (const auto opA : blas_ops) {
    for (const auto opB : blas_ops) {
      // Note: not yet implemented
      if (opA != blas::Op::NoTrans || opB != blas::Op::NoTrans)
        continue;

      for (const auto& [m, n, k, mb, a, b] : sizes) {
        const TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
        const TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);
        testGeneralMultiplication<TypeParam, Backend::GPU, Device::GPU>(a, b, opA, opB, alpha, beta, m,
                                                                        n, k, mb);
      }
    }
  }
}
#endif

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct GeneralSubMultiplicationDistTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(GeneralSubMultiplicationDistTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
struct GeneralSubKMultiplicationTestGPU : public TestWithCommGrids {};

TYPED_TEST_SUITE(GeneralSubMultiplicationDistTestGPU, MatrixElementTypes);
#endif

template <class T, Backend B, Device D>
void testGeneralSubMultiplication(comm::CommunicatorGrid grid, const SizeType a, const SizeType b,
                                  const T alpha, const T beta, const SizeType m, const SizeType mb) {
  const comm::Index2D src_rank_index(std::max(0, grid.size().rows() - 1),
                                     std::min(1, grid.size().cols() - 1));
  matrix::Distribution dist({m, m}, {mb, mb}, grid.size(), grid.rank(), src_rank_index);

  const SizeType a_el = a * mb;
  const SizeType b_el = std::min((b + 1) * mb - 1, m - 1);
  const SizeType nrefls = b_el + 1 - a_el;

  auto [refA, refB, refC, refResult] =
      matrix::test::getSubMatrixMatrixMultiplication(a_el, b_el, m, m, nrefls, alpha, beta,
                                                     blas::Op::NoTrans, blas::Op::NoTrans);

  auto distributedMatrixFrom = [&dist](auto elSetter) {
    Matrix<T, Device::CPU> matrix(dist);
    dlaf::matrix::util::set(matrix, elSetter);
    return matrix;
  };

  Matrix<const T, Device::CPU> mat_ah(distributedMatrixFrom(refA));
  Matrix<const T, Device::CPU> mat_bh(distributedMatrixFrom(refB));
  Matrix<T, Device::CPU> mat_ch(distributedMatrixFrom(refC));

  {
    MatrixMirror<const T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<const T, D, Device::CPU> mat_b(mat_bh);
    MatrixMirror<T, D, Device::CPU> mat_c(mat_ch);

    multiplication::generalSubMatrix<B>(grid, a, b, alpha, mat_a.get(), mat_b.get(), beta, mat_c.get());
  }

  CHECK_MATRIX_NEAR(refResult, mat_ch, 40 * (mat_ch.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_ch.size().rows() + 1) * TypeUtilities<T>::error);
}

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> subk_sizes = {
    // m, mb, a, b
    // full tile, all reflectors
    {3, 1, 0, 2},
    {3, 3, 0, 0},
    {6, 3, 0, 1},
    {9, 3, 0, 2},
    {21, 3, 0, 6},
    // partial tile, all reflectors
    {9, 3, 1, 2},
    {21, 3, 3, 6},
    // incomplete tiles, full reflectors
    {8, 3, 1, 2},
};

TYPED_TEST(GeneralSubMultiplicationDistTestMC, CorrectnessDistributed) {
  for (auto comm_grid : this->commGrids()) {
    for (const auto& [m, mb, a, b] : subk_sizes) {
      const TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
      const TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);
      testGeneralSubMultiplication<TypeParam, Backend::MC, Device::CPU>(comm_grid, a, b, alpha, beta, m,
                                                                        mb);
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GeneralSubMultiplicationDistTestGPU, CorrectnessDistributed) {
  for (auto comm_grid : this->commGrids()) {
    for (const auto& [m, mb, a, b, nrefls] : subk_sizes) {
      const TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
      const TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);
      testGeneralSubMultiplication<TypeParam, Backend::GPU, Device::GPU>(comm_grid, a, b, alpha, beta, m,
                                                                         mb);
    }
  }
}
#endif
