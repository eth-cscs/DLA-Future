//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <tuple>
#include <utility>
#include <vector>

#include <pika/init.hpp>

#include <dlaf/blas/enum_output.h>
#include <dlaf/common/assert.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/communication/communicator.h>
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
using namespace dlaf::matrix::test;
using namespace dlaf::test;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct GeneralMultiplicationTestMC : public ::testing::Test {};
TYPED_TEST_SUITE(GeneralMultiplicationTestMC, MatrixElementTypes);

template <class T>
struct GeneralMultiplicationDistTestMC : public TestWithCommGrids {};
TYPED_TEST_SUITE(GeneralMultiplicationDistTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
struct GeneralMultiplicationTestGPU : public ::testing::Test {};
TYPED_TEST_SUITE(GeneralMultiplicationTestGPU, MatrixElementTypes);

template <class T>
struct GeneralMultiplicationDistTestGPU : public TestWithCommGrids {};
TYPED_TEST_SUITE(GeneralMultiplicationDistTestGPU, MatrixElementTypes);
#endif

struct GemmConfig {
  const blas::Op opA, opB;
  const SizeType m, n, k;
  const SizeType mb, nb, kb;
  const struct {
    const GlobalElementIndex tl = {0, 0};
    const GlobalElementIndex br = {0, 0};
  } margin_a = {}, margin_b = {}, margin_c = {};

  matrix::internal::SubMatrixSpec sub_a() const noexcept {
    return {margin_a.tl, {m, k}};
  }
  matrix::internal::SubMatrixSpec sub_b() const noexcept {
    return {margin_b.tl, {k, n}};
  }
  matrix::internal::SubMatrixSpec sub_c() const noexcept {
    return {margin_c.tl, {m, n}};
  }

  GlobalElementSize full_a() const noexcept {
    return sizeFromOrigin(margin_a.tl) + common::sizeFromOrigin(margin_a.br) + sub_a().size;
  }
  GlobalElementSize full_b() const noexcept {
    return sizeFromOrigin(margin_b.tl) + common::sizeFromOrigin(margin_b.br) + sub_b().size;
  }
  GlobalElementSize full_c() const noexcept {
    return sizeFromOrigin(margin_c.tl) + common::sizeFromOrigin(margin_c.br) + sub_c().size;
  }
};

template <class T, Backend B, Device D>
void testGeneralMultiplication(const T alpha, const T beta, const GemmConfig& config) {
  using dlaf::matrix::internal::MatrixRef;

  auto setMatrix = [&](auto&& elSetter, const GlobalElementSize& size,
                       const TileElementSize& block_size) {
    Matrix<T, Device::CPU> matrix({size.rows(), size.cols()}, block_size);
    dlaf::matrix::util::set(matrix, elSetter);
    return matrix;
  };

  auto [subValuesA, subValuesB, subValuesC, subValuesResult] =
      matrix::test::getMatrixMatrixMultiplication<GlobalElementIndex, T>(config.opA, config.opB,
                                                                         config.k, alpha, beta);

  const auto fullValuesA = mix_values(config.sub_a(), subValuesA, [](auto) { return T(-99); });
  const auto fullValuesB = mix_values(config.sub_b(), subValuesB, [](auto) { return T(-99); });
  const auto fullValuesC = mix_values(config.sub_c(), subValuesC, [](auto) { return T(-99); });

  Matrix<const T, Device::CPU> mat_ah = setMatrix(fullValuesA, config.full_a(), {config.mb, config.kb});
  Matrix<const T, Device::CPU> mat_bh = setMatrix(fullValuesB, config.full_b(), {config.kb, config.nb});
  Matrix<T, Device::CPU> mat_ch = setMatrix(fullValuesC, config.full_c(), {config.mb, config.nb});

  {
    MatrixMirror<const T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<const T, D, Device::CPU> mat_b(mat_bh);
    MatrixMirror<T, D, Device::CPU> mat_c(mat_ch);

    MatrixRef<const T, D> mat_sub_a(mat_a.get(), config.sub_a());
    MatrixRef<const T, D> mat_sub_b(mat_b.get(), config.sub_b());
    MatrixRef<T, D> mat_sub_c(mat_c.get(), config.sub_c());

    // Note: currently it is implemented just the NoTrans/NoTrans case
    ASSERT_EQ(config.opA, blas::Op::NoTrans);
    ASSERT_EQ(config.opB, blas::Op::NoTrans);
    multiplication::internal::generalMatrix<B>(config.opA, config.opB, alpha, mat_sub_a, mat_sub_b, beta,
                                               mat_sub_c);
  }

  const auto fullValuesResult = mix_values(config.sub_c(), subValuesResult, fullValuesC);
  CHECK_MATRIX_NEAR(fullValuesResult, mat_ch, 2 * (mat_ah.size().cols() + 1) * TypeUtilities<T>::error,
                    2 * (mat_ah.size().cols() + 1) * TypeUtilities<T>::error);
}

template <class T, Backend B, Device D>
void testGeneralMultiplication(const T alpha, const T beta, const GemmConfig& config,
                               comm::CommunicatorGrid& grid) {
  using dlaf::matrix::internal::MatrixRef;

  auto mpi_row_chain = grid.row_communicator_pipeline();
  auto mpi_col_chain = grid.col_communicator_pipeline();

  const TileElementSize blocksize_a(config.mb, config.kb);
  const TileElementSize blocksize_b(config.kb, config.nb);
  const TileElementSize blocksize_c(config.mb, config.nb);

  const comm::Index2D src_rank_c(std::max(0, grid.size().rows() - 1),
                                 std::min(1, grid.size().cols() - 1));
  const matrix::Distribution dist_c(config.full_c(), blocksize_c, grid.size(), grid.rank(), src_rank_c);

  const comm::IndexT_MPI rank_aligned_row =
      align_sub_rank_index<Coord::Row>(dist_c, config.sub_c().origin, blocksize_a,
                                       config.sub_a().origin);
  const comm::IndexT_MPI rank_aligned_col =
      align_sub_rank_index<Coord::Col>(dist_c, config.sub_c().origin, blocksize_b,
                                       config.sub_b().origin);

  // Note:
  // GEMM(NoTrans, NoTrans) requires:
  // - a is rank aligned with c for what concerns rows
  // - b is rank aligned with c for what concerns cols
  const comm::Index2D src_rank_a{rank_aligned_row, 0};
  const comm::Index2D src_rank_b{0, rank_aligned_col};

  const matrix::Distribution dist_a(config.full_a(), blocksize_a, grid.size(), grid.rank(), src_rank_a);
  const matrix::Distribution dist_b(config.full_b(), blocksize_b, grid.size(), grid.rank(), src_rank_b);

  auto setMatrix = [&](auto&& elSetter, matrix::Distribution dist) {
    Matrix<T, Device::CPU> matrix(std::move(dist));
    dlaf::matrix::util::set(matrix, elSetter);
    return matrix;
  };

  auto [subValuesA, subValuesB, subValuesC, subValuesResult] =
      matrix::test::getMatrixMatrixMultiplication<GlobalElementIndex, T>(config.opA, config.opB,
                                                                         config.k, alpha, beta);

  const auto fullValuesA = mix_values(config.sub_a(), subValuesA, [](auto) { return T(-99); });
  const auto fullValuesB = mix_values(config.sub_b(), subValuesB, [](auto) { return T(-99); });
  const auto fullValuesC = mix_values(config.sub_c(), subValuesC, [](auto) { return T(-99); });

  Matrix<const T, Device::CPU> mat_ah = setMatrix(fullValuesA, dist_a);
  Matrix<const T, Device::CPU> mat_bh = setMatrix(fullValuesB, dist_b);
  Matrix<T, Device::CPU> mat_ch = setMatrix(fullValuesC, dist_c);

  {
    MatrixMirror<const T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<const T, D, Device::CPU> mat_b(mat_bh);
    MatrixMirror<T, D, Device::CPU> mat_c(mat_ch);

    MatrixRef<const T, D> mat_sub_a(mat_a.get(), config.sub_a());
    MatrixRef<const T, D> mat_sub_b(mat_b.get(), config.sub_b());
    MatrixRef<T, D> mat_sub_c(mat_c.get(), config.sub_c());

    // Note: currently it is implemented just the NoTrans/NoTrans case
    ASSERT_EQ(config.opA, blas::Op::NoTrans);
    ASSERT_EQ(config.opB, blas::Op::NoTrans);
    multiplication::internal::generalMatrix<B>(mpi_row_chain, mpi_col_chain, alpha, mat_sub_a, mat_sub_b,
                                               beta, mat_sub_c);
  }

  const auto fullValuesResult = mix_values(config.sub_c(), subValuesResult, fullValuesC);
  CHECK_MATRIX_NEAR(fullValuesResult, mat_ch, 2 * (mat_ah.size().cols() + 1) * TypeUtilities<T>::error,
                    2 * (mat_ah.size().cols() + 1) * TypeUtilities<T>::error);
}

std::vector<GemmConfig> gemm_configs = {
    // empty matrices
    {blas::Op::NoTrans, blas::Op::NoTrans, 0, 0, 7, 3, 6, 2},
    {blas::Op::NoTrans, blas::Op::NoTrans, 26, 0, 7, 3, 6, 2},
    {blas::Op::NoTrans, blas::Op::NoTrans, 0, 13, 7, 3, 6, 2},
    {blas::Op::NoTrans, blas::Op::NoTrans, 26, 13, 0, 3, 6, 2},

    // full
    {blas::Op::NoTrans, blas::Op::NoTrans, 3, 3, 3, 3, 3, 3},
    {blas::Op::NoTrans, blas::Op::NoTrans, 8, 8, 11, 10, 9, 13},
    {blas::Op::NoTrans, blas::Op::NoTrans, 3, 2, 4, 1, 1, 1},
    {blas::Op::NoTrans, blas::Op::NoTrans, 6, 9, 8, 2, 3, 4},
    {blas::Op::NoTrans, blas::Op::NoTrans, 21, 21, 21, 3, 4, 5},
    {blas::Op::NoTrans, blas::Op::NoTrans, 12, 20, 11, 3, 4, 5},
    {blas::Op::NoTrans, blas::Op::NoTrans, 8, 8, 11, 3, 3, 5},
};

std::vector<GemmConfig> sub_gemm_configs = {
    // empty matrices
    {blas::Op::NoTrans, blas::Op::NoTrans, 0, 0, 7, 3, 6, 2, {{1, 2}}, {{2, 3}}, {{3, 4}}},
    {blas::Op::NoTrans, blas::Op::NoTrans, 26, 0, 7, 3, 6, 2, {{1, 2}}, {{2, 3}}, {{3, 4}}},
    {blas::Op::NoTrans, blas::Op::NoTrans, 0, 13, 7, 3, 6, 2, {{1, 2}}, {{2, 3}}, {{3, 4}}},
    // k = 0
    {blas::Op::NoTrans, blas::Op::NoTrans, 26, 13, 0, 3, 6, 2, {{1, 2}}, {{2, 3}}, {{3, 4}}},
    // single-tile
    {blas::Op::NoTrans, blas::Op::NoTrans, 8, 8, 11, 10, 9, 13, {{2, 1}}, {{1, 1}}, {{0, 0}}},
    // multi-tile
    {blas::Op::NoTrans, blas::Op::NoTrans, 12, 20, 11, 3, 4, 5, {{7, 1}}, {{11, 10}}, {{4, 2}}},
    {blas::Op::NoTrans, blas::Op::NoTrans, 12, 20, 11, 3, 4, 5, {{6, 10}}, {{5, 8}}, {{9, 12}}},
};

TYPED_TEST(GeneralMultiplicationTestMC, CorrectnessLocal) {
  constexpr TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
  constexpr TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);

  for (const GemmConfig& test_config : gemm_configs) {
    testGeneralMultiplication<TypeParam, Backend::MC, Device::CPU>(alpha, beta, test_config);
  }
}

TYPED_TEST(GeneralMultiplicationTestMC, CorrectnessLocalSub) {
  constexpr TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
  constexpr TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);

  for (const GemmConfig& test_config : sub_gemm_configs) {
    testGeneralMultiplication<TypeParam, Backend::MC, Device::CPU>(alpha, beta, test_config);
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GeneralMultiplicationTestGPU, CorrectnessLocal) {
  constexpr TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
  constexpr TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);

  for (const GemmConfig& test_config : gemm_configs) {
    testGeneralMultiplication<TypeParam, Backend::GPU, Device::GPU>(alpha, beta, test_config);
  }
}

TYPED_TEST(GeneralMultiplicationTestGPU, CorrectnessLocalSub) {
  constexpr TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
  constexpr TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);

  for (const GemmConfig& test_config : sub_gemm_configs) {
    testGeneralMultiplication<TypeParam, Backend::GPU, Device::GPU>(alpha, beta, test_config);
  }
}
#endif

TYPED_TEST(GeneralMultiplicationDistTestMC, CorrectnessDistributed) {
  constexpr TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
  constexpr TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);

  for (auto& comm_grid : this->commGrids()) {
    for (const GemmConfig& test_config : gemm_configs) {
      testGeneralMultiplication<TypeParam, Backend::MC, Device::CPU>(alpha, beta, test_config,
                                                                     comm_grid);
      pika::wait();
    }
  }
}

TYPED_TEST(GeneralMultiplicationDistTestMC, CorrectnessDistributedSub) {
  constexpr TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
  constexpr TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);

  for (auto& comm_grid : this->commGrids()) {
    for (const GemmConfig& test_config : sub_gemm_configs) {
      testGeneralMultiplication<TypeParam, Backend::MC, Device::CPU>(alpha, beta, test_config,
                                                                     comm_grid);
      pika::wait();
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GeneralMultiplicationDistTestGPU, CorrectnessDistributed) {
  constexpr TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
  constexpr TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);

  for (auto& comm_grid : this->commGrids()) {
    for (const GemmConfig& test_config : gemm_configs) {
      testGeneralMultiplication<TypeParam, Backend::GPU, Device::GPU>(alpha, beta, test_config,
                                                                      comm_grid);
      pika::wait();
    }
  }
}

TYPED_TEST(GeneralMultiplicationDistTestGPU, CorrectnessDistributedSub) {
  constexpr TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
  constexpr TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);

  for (auto& comm_grid : this->commGrids()) {
    for (const GemmConfig& test_config : sub_gemm_configs) {
      testGeneralMultiplication<TypeParam, Backend::GPU, Device::GPU>(alpha, beta, test_config,
                                                                      comm_grid);
      pika::wait();
    }
  }
}
#endif
