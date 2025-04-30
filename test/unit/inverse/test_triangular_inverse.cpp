//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include <pika/init.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/inverse/triangular.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_generic_lapack.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct TriangularInverseTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(TriangularInverseTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
struct TriangularInverseTestGPU : public TestWithCommGrids {};

TYPED_TEST_SUITE(TriangularInverseTestGPU, MatrixElementTypes);
#endif

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<blas::Diag> blas_diags({blas::Diag::Unit, blas::Diag::NonUnit});

const std::vector<std::tuple<SizeType, SizeType>> sizes = {
    {0, 2},                              // m = 0
    {5, 8}, {34, 34},                    // m <= mb
    {4, 3}, {16, 10}, {34, 13}, {32, 5}  // m > mb
};

template <class T, Backend B, Device D>
void test_triangular_inverse(const blas::Uplo uplo, const blas::Diag diag, const SizeType m,
                             const SizeType mb) {
  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<T, Device::CPU> mat_h(size, block_size);

  auto [el, res] = get_triangular_inverse_setters<GlobalElementIndex, T>(uplo, diag);

  set(mat_h, el);

  {
    MatrixMirror<T, D, Device::CPU> mat(mat_h);
    triangular_inverse<B, D, T>(uplo, diag, mat.get());
  }

  CHECK_MATRIX_NEAR(res, mat_h, 4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error,
                    4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error);
}

template <class T, Backend B, Device D>
void test_triangular_inverse(comm::CommunicatorGrid& grid, const blas::Uplo uplo, const blas::Diag diag,
                             const SizeType m, const SizeType mb) {
  const GlobalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);
  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  Distribution distribution(size, block_size, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_h(std::move(distribution));

  auto [el, res] = get_triangular_inverse_setters<GlobalElementIndex, T>(uplo, diag);

  set(mat_h, el);

  {
    MatrixMirror<T, D, Device::CPU> mat(mat_h);
    triangular_inverse<B, D, T>(grid, uplo, diag, mat.get());
  }

  CHECK_MATRIX_NEAR(res, mat_h, 4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error,
                    4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(TriangularInverseTestMC, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    if (uplo == blas::Uplo::Upper)
      continue;
    for (auto diag : blas_diags) {
      for (const auto& [m, mb] : sizes) {
        test_triangular_inverse<TypeParam, Backend::MC, Device::CPU>(uplo, diag, m, mb);
      }
    }
  }
}

// TYPED_TEST(TriangularInverseTestMC, CorrectnessDistributed) {
//   for (auto& comm_grid : this->commGrids()) {
//     for (auto uplo : blas_uplos) {
//       for (auto diag : blas_diags) {
//         for (const auto& [m, mb] : sizes) {
//           test_triangular_inverse<TypeParam, Backend::MC, Device::CPU>(comm_grid, uplo, m, mb);
//           pika::wait();
//         }
//       }
//     }
//   }
// }

#ifdef DLAF_WITH_GPU
// TYPED_TEST(TriangularInverseTestGPU, CorrectnessLocal) {
//   for (auto uplo : blas_uplos) {
//     for (auto diag : blas_diags) {
//       for (const auto& [m, mb] : sizes) {
//         test_triangular_inverse<TypeParam, Backend::GPU, Device::GPU>(uplo, m, mb);
//       }
//     }
//   }
// }
//
// TYPED_TEST(TriangularInverseTestGPU, CorrectnessDistributed) {
//   for (auto& comm_grid : this->commGrids()) {
//     for (auto uplo : blas_uplos) {
//       for (auto diag : blas_diags) {
//         for (const auto& [m, mb] : sizes) {
//           test_triangular_inverse<TypeParam, Backend::GPU, Device::GPU>(comm_grid, uplo, m, mb);
//           pika::wait();
//         }
//       }
//     }
//   }
// }
#endif
