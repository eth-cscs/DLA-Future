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
#include <dlaf_c_test/c_api_helpers.h>

#include "test_inverse_from_cholesky_factor_c_api_config.h"
#include "test_inverse_from_cholesky_factor_c_api_wrapper.h"

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_generic_lapack.h>
#include <dlaf_test/matrix/util_matrix.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksCAPIEnvironment);

template <class T>
struct InverseFromCholeskyFactorTestCapi : public TestWithCommGrids {};

TYPED_TEST_SUITE(InverseFromCholeskyFactorTestCapi, MatrixElementTypes);

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

const std::vector<std::tuple<SizeType, SizeType>> sizes = {
    {0, 2},                              // m = 0
    {5, 8}, {34, 34},                    // m <= mb
    {4, 3}, {16, 10}, {34, 13}, {32, 5}  // m > mb
};

template <class T, API api>
void test_inverse_from_cholesky_factor(comm::CommunicatorGrid& grid, const blas::Uplo uplo,
                                       const SizeType m, const SizeType mb) {
  auto dlaf_context = c_api_test_initialize<api>(pika_argc, pika_argv, dlaf_argc, dlaf_argv, grid);

  // In normal use the runtime is resumed by the C API call
  // The pika runtime is suspended by dlaf_initialize
  // Here we need to resume it manually to build the matrices with DLA-Future
  pika::resume();

  const GlobalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);
  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  Distribution distribution(size, block_size, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_h(std::move(distribution));

  auto [el, res] = get_inverse_cholesky_factor_setters<GlobalElementIndex, T>(m, uplo);
  set(mat_h, el);
  mat_h.waitLocalTiles();

  char dlaf_uplo = blas::to_char(uplo);

  // Get pointer to first element of local matrix
  auto [local_a_ptr, lld] = top_left_tile(mat_h);

  // Suspend pika to ensure it is resumed by the C API
  pika::suspend();

  if constexpr (api == API::dlaf) {
    DLAF_descriptor dlaf_desc =
        {(int) m, (int) m, (int) mb, (int) mb, src_rank_index.row(), src_rank_index.col(), 0, 0, lld};
    int err = -1;
    if constexpr (std::is_same_v<T, double>) {
      err = C_dlaf_inverse_from_cholesky_factor_d(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc);
    }
    else if constexpr (std::is_same_v<T, float>) {
      err = C_dlaf_inverse_from_cholesky_factor_s(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
      err = C_dlaf_inverse_from_cholesky_factor_z(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
      err = C_dlaf_inverse_from_cholesky_factor_c(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc);
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
                    (int) m,
                    (int) m,
                    (int) mb,
                    (int) mb,
                    src_rank_index.row(),
                    src_rank_index.col(),
                    lld};
    int info = -1;
    if constexpr (std::is_same_v<T, double>) {
      C_dlaf_pdpotri(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, &info);
    }
    else if constexpr (std::is_same_v<T, float>) {
      C_dlaf_pspotri(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, &info);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
      C_dlaf_pzpotri(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, &info);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
      C_dlaf_pcpotri(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, &info);
    }
    else {
      DLAF_ASSERT(false, typeid(T).name());
    }
    DLAF_ASSERT(info == 0, info);
#else
    static_assert(api != API::scalapack, "DLA-Future compiled without ScaLAPACK support.");
#endif
  }

  // Resume pika for the checks (suspended by the C API)
  pika::resume();

  CHECK_MATRIX_NEAR(res, mat_h, 4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error,
                    4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error);

  // Suspend pika to make sure dlaf_finalize resumes it
  pika::suspend();

  c_api_test_finalize<api>(dlaf_context);
}

TYPED_TEST(InverseFromCholeskyFactorTestCapi, CorrectnessDistributedDLAF) {
  for (auto& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (const auto& [m, mb] : sizes) {
        test_inverse_from_cholesky_factor<TypeParam, API::dlaf>(grid, uplo, m, mb);
      }
    }
  }
}

#ifdef DLAF_WITH_SCALAPACK
TYPED_TEST(InverseFromCholeskyFactorTestCapi, CorrectnessDistributedScaLAPACK) {
  for (auto& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (const auto& [m, mb] : sizes) {
        test_inverse_from_cholesky_factor<TypeParam, API::scalapack>(grid, uplo, m, mb);
      }
    }
  }
}
#endif
