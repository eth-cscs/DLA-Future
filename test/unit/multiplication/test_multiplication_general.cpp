//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <pika/init.hpp>

#include <dlaf/blas/enum_output.h>
#include <dlaf/common/assert.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/multiplication/general.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_generic_blas.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::test;

template <class T>
struct GeneralSubMultiplicationTestMC : public ::testing::Test {};

TYPED_TEST_SUITE(GeneralSubMultiplicationTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
struct GeneralSubMultiplicationTestGPU : public ::testing::Test {};

TYPED_TEST_SUITE(GeneralSubMultiplicationTestGPU, MatrixElementTypes);
#endif

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct GeneralSubMultiplicationDistTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(GeneralSubMultiplicationDistTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
struct GeneralSubMultiplicationDistTestGPU : public TestWithCommGrids {};

TYPED_TEST_SUITE(GeneralSubMultiplicationDistTestGPU, MatrixElementTypes);
#endif

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    // m, mb, a, b
    // full gemm
    {3, 1, 0, 3},
    {3, 3, 0, 1},
    {6, 3, 0, 2},
    {9, 3, 0, 3},
    {21, 3, 0, 7},
    // sub gemm empty
    {9, 3, 0, 0},
    {9, 3, 2, 2},
    {9, 3, 3, 3},
    // sub gemm
    {9, 3, 1, 3},
    {21, 3, 3, 7},
    // full gemm, incomplete tiles
    {8, 3, 1, 3},
};

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

template <class T, Backend B, Device D>
void testGeneralSubMultiplication(const SizeType a, const SizeType b, const T alpha, const T beta,
                                  const SizeType m, const SizeType mb) {
  const SizeType a_el = a * mb;
  const SizeType b_el = std::min(b * mb, m);

  auto [refA, refB, refC, refResult] =
      matrix::test::getSubMatrixMatrixMultiplication(a_el, b_el, m, m, m, alpha, beta, blas::Op::NoTrans,
                                                     blas::Op::NoTrans);

  auto setMatrix = [&](auto elSetter, const LocalElementSize size, const TileElementSize block_size) {
    Matrix<T, Device::CPU> matrix(size, block_size);
    dlaf::matrix::util::set(matrix, elSetter);
    return matrix;
  };

  Matrix<const T, Device::CPU> mat_ah = setMatrix(refA, {m, m}, {mb, mb});
  Matrix<const T, Device::CPU> mat_bh = setMatrix(refB, {m, m}, {mb, mb});
  Matrix<T, Device::CPU> mat_ch = setMatrix(refC, {m, m}, {mb, mb});

  {
    MatrixMirror<const T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<const T, D, Device::CPU> mat_b(mat_bh);
    MatrixMirror<T, D, Device::CPU> mat_c(mat_ch);

    multiplication::internal::generalSubMatrix<B>(a, b, blas::Op::NoTrans, blas::Op::NoTrans, alpha,
                                                  mat_a.get(), mat_b.get(), beta, mat_c.get());
  }

  CHECK_MATRIX_NEAR(refResult, mat_ch, 40 * (mat_ch.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_ch.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(GeneralSubMultiplicationTestMC, CorrectnessLocal) {
  for (const auto& [m, mb, a, b] : sizes) {
    const TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
    const TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);
    testGeneralSubMultiplication<TypeParam, Backend::MC, Device::CPU>(a, b, alpha, beta, m, mb);
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GeneralSubMultiplicationTestGPU, CorrectnessLocal) {
  for (const auto& [m, mb, a, b] : sizes) {
    const TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
    const TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);
    testGeneralSubMultiplication<TypeParam, Backend::GPU, Device::GPU>(a, b, alpha, beta, m, mb);
  }
}
#endif

template <class T, Backend B, Device D>
void testGeneralSubMultiplication(comm::CommunicatorGrid grid, const SizeType a, const SizeType b,
                                  const T alpha, const T beta, const SizeType m, const SizeType mb) {
  const comm::Index2D src_rank_index(std::max(0, grid.size().rows() - 1),
                                     std::min(1, grid.size().cols() - 1));
  matrix::Distribution dist({m, m}, {mb, mb}, grid.size(), grid.rank(), src_rank_index);

  const SizeType a_el = a * mb;
  const SizeType b_el = std::min(b * mb, m);

  auto [refA, refB, refC, refResult] =
      matrix::test::getSubMatrixMatrixMultiplication(a_el, b_el, m, m, m, alpha, beta, blas::Op::NoTrans,
                                                     blas::Op::NoTrans);

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

    multiplication::internal::generalSubMatrix<B>(grid, a, b, alpha, mat_a.get(), mat_b.get(), beta,
                                                  mat_c.get());
  }

  CHECK_MATRIX_NEAR(refResult, mat_ch, 40 * (mat_ch.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_ch.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(GeneralSubMultiplicationDistTestMC, CorrectnessDistributed) {
  for (auto comm_grid : this->commGrids()) {
    for (const auto& [m, mb, a, b] : sizes) {
      const TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
      const TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);
      testGeneralSubMultiplication<TypeParam, Backend::MC, Device::CPU>(comm_grid, a, b, alpha, beta, m,
                                                                        mb);
      pika::wait();
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GeneralSubMultiplicationDistTestGPU, CorrectnessDistributed) {
  for (auto comm_grid : this->commGrids()) {
    for (const auto& [m, mb, a, b] : sizes) {
      const TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
      const TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);
      testGeneralSubMultiplication<TypeParam, Backend::GPU, Device::GPU>(comm_grid, a, b, alpha, beta, m,
                                                                         mb);
      pika::wait();
    }
  }
}
#endif

template <class T, Backend B, Device D>
void testGeneralSubMultiplication(dlaf::matrix::internal::SubMatrixSpec sub_spec, const T alpha,
                                  const T beta, const SizeType m, const SizeType mb) {
  using dlaf::matrix::internal::MatrixRef;

  const SizeType a_el = sub_spec.origin.row();
  const SizeType b_el = sub_spec.origin.row() + sub_spec.size.rows();

  auto [refA, refB, refC, refResult] =
      matrix::test::getSubMatrixMatrixMultiplication(a_el, b_el, m, m, m, alpha, beta, blas::Op::NoTrans,
                                                     blas::Op::NoTrans);

  auto setMatrix = [&](auto elSetter, const LocalElementSize size, const TileElementSize block_size) {
    Matrix<T, Device::CPU> matrix(size, block_size);
    dlaf::matrix::util::set(matrix, elSetter);
    return matrix;
  };

  Matrix<const T, Device::CPU> mat_ah = setMatrix(refA, {m, m}, {mb, mb});
  Matrix<const T, Device::CPU> mat_bh = setMatrix(refB, {m, m}, {mb, mb});
  Matrix<T, Device::CPU> mat_ch = setMatrix(refC, {m, m}, {mb, mb});

  {
    MatrixMirror<const T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<const T, D, Device::CPU> mat_b(mat_bh);
    MatrixMirror<T, D, Device::CPU> mat_c(mat_ch);

    MatrixRef<const T, D> mat_sub_a(mat_a.get(), sub_spec);
    MatrixRef<const T, D> mat_sub_b(mat_b.get(), sub_spec);
    MatrixRef<T, D> mat_sub_c(mat_c.get(), sub_spec);

    multiplication::internal::GeneralSub<B, D, T>::callNN(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
                                                          mat_sub_a, mat_sub_b, beta, mat_sub_c);
  }

  CHECK_MATRIX_NEAR(refResult, mat_ch, 40 * (mat_ch.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_ch.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(GeneralSubMultiplicationTestMC, MatrixRefCorrectnessLocal) {
  constexpr TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
  constexpr TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);

  for (const auto& [m, mb, a, b] : sizes) {
    const SizeType a_el = a * mb;
    const SizeType b_el = std::min(b * mb, m);
    dlaf::matrix::internal::SubMatrixSpec spec{GlobalElementIndex{a_el, a_el},
                                               GlobalElementSize{b_el - a_el, b_el - a_el}};
    testGeneralSubMultiplication<TypeParam, Backend::MC, Device::CPU>(spec, alpha, beta, m, mb);
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GeneralSubMultiplicationTestGPU, MatrixRefCorrectnessLocal) {
  constexpr TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
  constexpr TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);

  for (const auto& [m, mb, a, b] : sizes) {
    const SizeType a_el = a * mb;
    const SizeType b_el = std::min(b * mb, m);
    dlaf::matrix::internal::SubMatrixSpec spec{GlobalElementIndex{a_el, a_el},
                                               GlobalElementSize{b_el - a_el, b_el - a_el}};
    testGeneralSubMultiplication<TypeParam, Backend::GPU, Device::GPU>(spec, alpha, beta, m, mb);
  }
}
#endif
