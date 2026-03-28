//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

#include <blas/util.hh>

#include <pika/init.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/util_matrix.h>
#include <dlaf_c_test/c_api_helpers.h>

#include "test_triangular_c_api_config.h"
#include "test_triangular_c_api_wrapper.h"

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_generic_blas.h>
#include <dlaf_test/matrix/util_matrix.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::util;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksCAPIEnvironment);

template <class T>
struct TriangularMultiplicationTestCapi : public TestWithCommGrids {};

TYPED_TEST_SUITE(TriangularMultiplicationTestCapi, MatrixElementTypes);

const std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<blas::Diag> blas_diags({blas::Diag::NonUnit, blas::Diag::Unit});

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    {0, 0, 1, 1},                                              // m, n = 0
    {0, 2, 1, 2}, {7, 0, 2, 1},                                // m = 0 or n = 0
    {2, 2, 5, 5}, {10, 10, 2, 3},                              // m = n
    {12, 3, 5, 5}, {7, 6, 3, 2}, {15, 7, 3, 5},                // m > n
    {4, 13, 5, 5}, {7, 8, 2, 9}, {19, 25, 6, 5},               // m < n
};

template <class T, API api>
void testTriangularMultiplication(comm::CommunicatorGrid& grid, blas::Side side, blas::Uplo uplo,
                                 blas::Op op, blas::Diag diag, const SizeType m, const SizeType n,
                                 const SizeType mb, const SizeType nb) {
  auto dlaf_context = c_api_test_initialize<api>(pika_argc, pika_argv, dlaf_argc, dlaf_argv, grid);

  // In normal use the runtime is resumed by the C API call
  // The pika runtime is suspended by dlaf_initialize
  // Here we need to resume it manually to build the matrices with DLA-Future
  pika::resume();

  // Setup matrix A (triangular)
  SizeType a_size = (side == blas::Side::Left) ? m : n;
  SizeType a_block = (side == blas::Side::Left) ? mb : nb;

  const GlobalElementSize size_a(a_size, a_size);
  const TileElementSize block_size_a(a_block, a_block);
  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  Distribution distribution_a(size_a, block_size_a, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_a(std::move(distribution_a));

  // Setup matrix B
  const GlobalElementSize size_b(m, n);
  const TileElementSize block_size_b(mb, nb);

  Distribution distribution_b(size_b, block_size_b, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_b(std::move(distribution_b));

  // Set up matrices with test data
  T alpha = TypeUtilities<T>::element(1.5, .5);
  auto [el_op_a, res_b, el_b] =
      getTriangularSystem<GlobalElementIndex, T>(side, uplo, op, diag, static_cast<T>(1.0) / alpha,
                                                 m, n);

  set(mat_a, el_op_a, op);
  set(mat_b, el_b);
  mat_a.waitLocalTiles();
  mat_b.waitLocalTiles();

  // Get pointers to local matrices
  auto [local_a_ptr, lld_a] = top_left_tile(mat_a);
  auto [local_b_ptr, lld_b] = top_left_tile(mat_b);

  char dlaf_side = blas::to_char(side);
  char dlaf_uplo = blas::to_char(uplo);
  char dlaf_op = blas::to_char(op);
  char dlaf_diag = blas::to_char(diag);

  // Suspend pika to ensure it is resumed by the C API
  pika::suspend();

  if constexpr (api == API::dlaf) {
    DLAF_descriptor dlaf_desc_a = {(int) a_size,     (int) a_size,     (int) a_block, (int) a_block,
                                   src_rank_index.row(), src_rank_index.col(), 0,              0,
                                   lld_a};
    DLAF_descriptor dlaf_desc_b = {(int) m,     (int) n,     (int) mb, (int) nb,
                                   src_rank_index.row(), src_rank_index.col(), 0,              0,
                                   lld_b};

    int err = -1;
    if constexpr (std::is_same_v<T, double>) {
      err = C_dlaf_triangular_multiplication_d(dlaf_context, dlaf_side, dlaf_uplo, dlaf_op, dlaf_diag,
                                              alpha, local_a_ptr, dlaf_desc_a, local_b_ptr,
                                              dlaf_desc_b);
    }
    else if constexpr (std::is_same_v<T, float>) {
      err = C_dlaf_triangular_multiplication_s(dlaf_context, dlaf_side, dlaf_uplo, dlaf_op, dlaf_diag,
                                              alpha, local_a_ptr, dlaf_desc_a, local_b_ptr,
                                              dlaf_desc_b);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
      err = C_dlaf_triangular_multiplication_z(dlaf_context, dlaf_side, dlaf_uplo, dlaf_op, dlaf_diag,
                                              alpha, local_a_ptr, dlaf_desc_a, local_b_ptr,
                                              dlaf_desc_b);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
      err = C_dlaf_triangular_multiplication_c(dlaf_context, dlaf_side, dlaf_uplo, dlaf_op, dlaf_diag,
                                              alpha, local_a_ptr, dlaf_desc_a, local_b_ptr,
                                              dlaf_desc_b);
    }
    else {
      DLAF_ASSERT(false, typeid(T).name());
    }
    DLAF_ASSERT(err == 0, err);
  }
  else if constexpr (api == API::scalapack) {
#ifdef DLAF_WITH_SCALAPACK
    int desc_a[] = {1,
                    dlaf_context,
                    (int) a_size,
                    (int) a_size,
                    (int) a_block,
                    (int) a_block,
                    src_rank_index.row(),
                    src_rank_index.col(),
                    lld_a};

    int desc_b[] = {1,
                    dlaf_context,
                    (int) m,
                    (int) n,
                    (int) mb,
                    (int) nb,
                    src_rank_index.row(),
                    src_rank_index.col(),
                    lld_b};

    if constexpr (std::is_same_v<T, double>) {
      C_dlaf_pdtrmm(dlaf_side, dlaf_uplo, dlaf_op, dlaf_diag, (int) m, (int) n, alpha, local_a_ptr, 1,
                   1, desc_a, local_b_ptr, 1, 1, desc_b);
    }
    else if constexpr (std::is_same_v<T, float>) {
      C_dlaf_pstrmm(dlaf_side, dlaf_uplo, dlaf_op, dlaf_diag, (int) m, (int) n, alpha, local_a_ptr, 1,
                   1, desc_a, local_b_ptr, 1, 1, desc_b);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
      C_dlaf_pztrmm(dlaf_side, dlaf_uplo, dlaf_op, dlaf_diag, (int) m, (int) n, alpha, local_a_ptr, 1,
                   1, desc_a, local_b_ptr, 1, 1, desc_b);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
      C_dlaf_pctrmm(dlaf_side, dlaf_uplo, dlaf_op, dlaf_diag, (int) m, (int) n, alpha, local_a_ptr, 1,
                   1, desc_a, local_b_ptr, 1, 1, desc_b);
    }
    else {
      DLAF_ASSERT(false, typeid(T).name());
    }
#else
    static_assert(api != API::scalapack, "DLA-Future compiled without ScaLAPACK support.");
#endif
  }

  // Resume pika for the checks (suspended by the C API)
  pika::resume();

  CHECK_MATRIX_NEAR(res_b, mat_b, 40 * (mat_b.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_b.size().rows() + 1) * TypeUtilities<T>::error);

  // Suspend pika to make sure dlaf_finalize resumes it
  pika::suspend();

  c_api_test_finalize<api>(dlaf_context);
}

TYPED_TEST(TriangularMultiplicationTestCapi, CorrectnessDistributedDLAF) {
  for (auto& grid : this->commGrids()) {
    for (auto side : blas_sides) {
      for (auto uplo : blas_uplos) {
        for (auto op : blas_ops) {
          if (op != blas::Op::NoTrans)
            continue;

          for (auto diag : blas_diags) {
            for (const auto& [m, n, mb, nb] : sizes) {
              testTriangularMultiplication<TypeParam, API::dlaf>(grid, side, uplo, op, diag, m, n, mb,
                                                                 nb);
            }
          }
        }
      }
    }
  }
}

#ifdef DLAF_WITH_SCALAPACK
TYPED_TEST(TriangularMultiplicationTestCapi, CorrectnessDistributedScaLAPACK) {
  for (auto& grid : this->commGrids()) {
    for (auto side : blas_sides) {
      for (auto uplo : blas_uplos) {
        for (auto op : blas_ops) {
          if (op != blas::Op::NoTrans)
            continue;

          for (auto diag : blas_diags) {
            for (const auto& [m, n, mb, nb] : sizes) {
              testTriangularMultiplication<TypeParam, API::scalapack>(grid, side, uplo, op, diag, m, n,
                                                                      mb, nb);
            }
          }
        }
      }
    }
  }
}
#endif
