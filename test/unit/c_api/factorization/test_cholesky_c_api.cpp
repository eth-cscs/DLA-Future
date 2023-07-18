//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>
#include <functional>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <blas/util.hh>

#include <pika/init.hpp>
#include <pika/runtime.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/factorization/cholesky.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf_c/grid.h>
#include <dlaf_c/init.h>

#include "test_cholesky_c_api_wrapper.h"

#include <gtest/gtest.h>

#ifdef DLAF_WITH_SCALAPACK
#include <dlaf_test/blacs.h>
#endif
#include <dlaf_c_test/c_api_helpers.h>

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
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksCAPIEnvironment);

template <class T>
struct CholeskyTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(CholeskyTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
struct CholeskyTestGPU : public TestWithCommGrids {};

TYPED_TEST_SUITE(CholeskyTestGPU, MatrixElementTypes);
#endif

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

const std::vector<std::tuple<SizeType, SizeType>> sizes = {
    {0, 2},                              // m = 0
    {5, 8}, {34, 34},                    // m <= mb
    {4, 3}, {16, 10}, {34, 13}, {32, 5}  // m > mb
};

template <class T, Backend B, Device D, API api>
void testCholesky(comm::CommunicatorGrid grid, const blas::Uplo uplo, const SizeType m,
                  const SizeType mb, int dlaf_context) {
  // In normal use the runtime is resumed by the C API call
  // The pika runtime is suspended by dlaf_initialize
  // Here we need to resume it manually to build the matrices with DLA-Future
  pika::resume();

  const GlobalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);
  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  Distribution distribution(size, block_size, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_h(std::move(distribution));

  auto [el, res] = getCholeskySetters<GlobalElementIndex, T>(uplo);
  set(mat_h, el);
  mat_h.waitLocalTiles();

  char dlaf_uplo = uplo == blas::uplo2char(uplo);

  // Get pointer to first element of local matrix
  auto [local_a_ptr, lld] = top_left_tile(mat_h);

  // Suspend pika to ensure it is resumed by the C API
  pika::suspend();

  if constexpr (api == API::dlaf) {
    DLAF_descriptor dlaf_desc =
        {(int) m, (int) m, (int) mb, (int) mb, src_rank_index.row(), src_rank_index.col(), 0, 0, lld};
    int err = -1;
    if constexpr (std::is_same_v<T, double>) {
      err = C_dlaf_cholesky_d(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc);
    }
    else if constexpr (std::is_same_v<T, float>) {
      err = C_dlaf_cholesky_s(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
      err = C_dlaf_cholesky_z(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
      err = C_dlaf_cholesky_c(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc);
    }
    else {
      DLAF_ASSERT(false, typeid(T).name());
    }
    EXPECT_EQ(0, err);
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
      C_dlaf_pdpotrf(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, &info);
    }
    else if constexpr (std::is_same_v<T, float>) {
      C_dlaf_pspotrf(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, &info);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
      C_dlaf_pzpotrf(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, &info);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
      C_dlaf_pcpotrf(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, &info);
    }
    else {
      DLAF_ASSERT(false, typeid(T).name());
    }
    EXPECT_EQ(0, info);
#endif
  }

  // Resume pika for the checks (suspended by the C API)
  pika::resume();

  CHECK_MATRIX_NEAR(res, mat_h, 4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error,
                    4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error);

  // Suspend pika to make sure dlaf_finalize resumes it
  pika::suspend();
}

TYPED_TEST(CholeskyTestMC, CorrectnessDistributedDLAF) {
  constexpr API api = API::dlaf;
  for (const auto& grid : this->commGrids()) {
    auto dlaf_context = c_api_test_inititialize<api>(grid);

    for (auto uplo : blas_uplos) {
      for (const auto& [m, mb] : sizes) {
        testCholesky<TypeParam, Backend::MC, Device::CPU, api>(grid, uplo, m, mb, dlaf_context);
      }
    }

    c_api_test_finalize<api>(dlaf_context);
  }
}

#ifdef DLAF_WITH_SCALAPACK
TYPED_TEST(CholeskyTestMC, CorrectnessDistributedScaLAPACK) {
  constexpr API api = API::scalapack;
  for (const auto& grid : this->commGrids()) {
    auto dlaf_context = c_api_test_inititialize<api>(grid);

    for (auto uplo : blas_uplos) {
      for (const auto& [m, mb] : sizes) {
        testCholesky<TypeParam, Backend::MC, Device::CPU, api>(grid, uplo, m, mb, dlaf_context);
      }
    }

    c_api_test_finalize<api>(dlaf_context);
  }
}
#endif

#ifdef DLAF_WITH_GPU
TYPED_TEST(CholeskyTestGPU, CorrectnessDistributedDLAF) {
  constexpr API api = API::dlaf;
  for (const auto& grid : this->commGrids()) {
    auto dlaf_context = c_api_test_inititialize<api>(grid);

    for (auto uplo : blas_uplos) {
      for (const auto& [m, mb] : sizes) {
        testCholesky<TypeParam, Backend::GPU, Device::GPU, api>(grid, uplo, m, mb, dlaf_context);
      }
    }

    c_api_test_finalize<api>(dlaf_context);
  }
}

#ifdef DLAF_WITH_SCALAPACK
TYPED_TEST(CholeskyTestGPU, CorrectnessDistributedScaLapack) {
  constexpr API api = API::scalapack;
  for (const auto& grid : this->commGrids()) {
    auto dlaf_context = c_api_test_inititialize<api>(grid);

    for (auto uplo : blas_uplos) {
      for (const auto& [m, mb] : sizes) {
        testCholesky<TypeParam, Backend::GPU, Device::GPU, api>(grid, uplo, m, mb, dlaf_context);
      }
    }

    c_api_test_finalize<api>(dlaf_context);
  }
}
#endif
#endif
