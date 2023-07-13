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
#include <tuple>

#include <pika/init.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/eigensolver/eigensolver/api.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/tune.h>
#include <dlaf/types.h>
#include <dlaf_c/grid.h>
#include <dlaf_c/init.h>

#include "config.h"
#include "test_eigensolver_c_api_wrapper.h"

#include <gtest/gtest.h>

#include <dlaf_test/blacs.h>
#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/eigensolver/test_eigensolver_correctness.h>
#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksCAPIEnvironment);

template <typename Type>
class EigensolverTest : public TestWithCommGrids {};

template <class T>
using EigensolverTestMC = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using EigensolverTestGPU = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestGPU, MatrixElementTypes);
#endif

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, eigensolver_min_band}
    {0, 2, 100},                                              // m = 0
    {5, 8, 100}, {34, 34, 100},                               // m <= mb
    {4, 3, 100}, {16, 10, 100}, {34, 13, 100}, {32, 5, 100},  // m > mb
    {34, 8, 3},  {32, 6, 3}                                   // m > mb, sub-band
};

enum class API { dlaf, scalapack };

template <class T, Backend B, Device D, API api>
void testEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb, CommunicatorGrid grid) {
  dlaf_initialize(pika_argc, pika_argv, dlaf_argc, dlaf_argv);

  char grid_order = grid_ordering(MPI_COMM_WORLD, grid.size().rows(), grid.size().cols(),
                                  grid.rank().row(), grid.rank().col());

  int dlaf_context = -1;
  if constexpr (api == API::dlaf) {
    // Create DLAF grid directly
    dlaf_context = dlaf_create_grid(MPI_COMM_WORLD, grid.size().rows(), grid.size().cols(), grid_order);
  }
  else if constexpr (api == API::scalapack) {
#ifdef DLAF_WITH_SCALAPACK
    // Create BLACS grid
    Cblacs_get(0, 0, &dlaf_context);  // Default system context
    Cblacs_gridinit(&dlaf_context, &grid_order, grid.size().rows(), grid.size().cols());

    // Create DLAF grid from BLACS context
    dlaf_create_grid_from_blacs(dlaf_context);
#endif
  }

  // In normal use the runtime is resumed by the C API call
  // The pika runtime is suspended by dlaf_initialize
  // Here we need to resume it manually to build the matrices with DLA-Future
  pika::resume();

  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<const T, Device::CPU> reference = [&]() {
    auto reference = [&]() -> auto{
      return Matrix<T, Device::CPU>(GlobalElementSize(m, m), block_size, grid);
    }
    ();
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(reference.distribution());
  copy(reference, mat_a_h);
  mat_a_h.waitLocalTiles();

  eigensolver::EigensolverResult<T, D> ret = [&]() {
    const SizeType size = mat_a_h.size().rows();
    Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                       TileElementSize(mat_a_h.blockSize().rows(), 1));
    Matrix<T, D> eigenvectors(GlobalElementSize(size, size), mat_a_h.blockSize(), grid);

    eigenvalues.waitLocalTiles();
    eigenvectors.waitLocalTiles();

    char dlaf_uplo = uplo == blas::Uplo::Upper ? 'U' : 'L';

    // Get top left local tiles
    int lld_a, lld_eigenvectors;
    T* local_a_ptr;
    T* local_eigenvectors_ptr;
    dlaf::BaseType<T>* eigenvalues_ptr;
    {
      if (LocalTileIndex(0, 0).isIn(mat_a_h.distribution().localNrTiles())) {
        auto toplefttile_a =
            pika::this_thread::experimental::sync_wait(mat_a_h.readwrite(LocalTileIndex(0, 0)));

        lld_a = static_cast<int>(toplefttile_a.ld());
        local_a_ptr = toplefttile_a.ptr();
      }
      else {
        lld_a = 1;
        local_a_ptr = nullptr;
      }

      if (LocalTileIndex(0, 0).isIn(eigenvectors.distribution().localNrTiles())) {
        auto toplefttile_eigenvectors =
            pika::this_thread::experimental::sync_wait(eigenvectors.readwrite(LocalTileIndex(0, 0)));

        lld_eigenvectors = static_cast<int>(toplefttile_eigenvectors.ld());
        local_eigenvectors_ptr = toplefttile_eigenvectors.ptr();
      }
      else {
        lld_eigenvectors = 1;
        local_eigenvectors_ptr = nullptr;
      }

      if (LocalTileIndex(0, 0).isIn(eigenvalues.distribution().localNrTiles())) {
        auto toplefttile_eigenvalues =
            pika::this_thread::experimental::sync_wait(eigenvalues.readwrite(LocalTileIndex(0, 0)));

        eigenvalues_ptr = toplefttile_eigenvalues.ptr();
      }
      else {
        eigenvalues_ptr = nullptr;
      }
    }  // Destroy tiles (avoid spurious dependencies)

    // Suspend pika to ensure it is resumed by the C API
    pika::suspend();

    if constexpr (api == API::dlaf) {
      DLAF_descriptor dlaf_desc_a = {(int) m, (int) m, (int) mb, (int) mb, 0, 0, 0, 0, lld_a};
      DLAF_descriptor dlaf_desc_eigenvectors = {(int) m, (int) m, (int) mb, (int) mb,        0,
                                                0,       0,       0,        lld_eigenvectors};

      int err = -1;
      if constexpr (std::is_same_v<T, double>) {
        err = C_dlaf_eigensolver_d(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a, eigenvalues_ptr,
                                   local_eigenvectors_ptr, dlaf_desc_eigenvectors);
      }
      else if constexpr (std::is_same_v<T, float>) {
        err = C_dlaf_eigensolver_s(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a, eigenvalues_ptr,
                                   local_eigenvectors_ptr, dlaf_desc_eigenvectors);
      }
      else if constexpr (std::is_same_v<T, std::complex<double>>) {
        err = C_dlaf_eigensolver_z(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a, eigenvalues_ptr,
                                   local_eigenvectors_ptr, dlaf_desc_eigenvectors);
      }
      else if constexpr (std::is_same_v<T, std::complex<float>>) {
        err = C_dlaf_eigensolver_c(dlaf_context, dlaf_uplo, local_a_ptr, dlaf_desc_a, eigenvalues_ptr,
                                   local_eigenvectors_ptr, dlaf_desc_eigenvectors);
      }
      else {
        DLAF_ASSERT(false, typeid(T).name());
      }
      EXPECT_EQ(0, err);
    }
    else if constexpr (api == API::scalapack) {
#ifdef DLAF_WITH_SCALAPACK
      int desc_a[] = {1, dlaf_context, (int) m, (int) m, (int) mb, (int) mb, 0, 0, lld_a};
      int desc_z[] = {1, dlaf_context, (int) m, (int) m, (int) mb, (int) mb, 0, 0, lld_eigenvectors};
      int info = -1;
      if constexpr (std::is_same_v<T, double>) {
        C_dlaf_pdsyevd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                       local_eigenvectors_ptr, 1, 1, desc_z, &info);
      }
      else if constexpr (std::is_same_v<T, float>) {
        C_dlaf_pssyevd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                       local_eigenvectors_ptr, 1, 1, desc_z, &info);
      }
      else if constexpr (std::is_same_v<T, std::complex<double>>) {
        C_dlaf_pzheevd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                       local_eigenvectors_ptr, 1, 1, desc_z, &info);
      }
      else if constexpr (std::is_same_v<T, std::complex<float>>) {
        C_dlaf_pcheevd(dlaf_uplo, (int) m, local_a_ptr, 1, 1, desc_a, eigenvalues_ptr,
                       local_eigenvectors_ptr, 1, 1, desc_z, &info);
      }
      else {
        DLAF_ASSERT(false, typeid(T).name());
      }
      EXPECT_EQ(0, info);
#endif
    }

    return eigensolver::EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
  }();

  // Resume pika runtime suspended by C API for correctness checks
  pika::resume();

  if (!mat_a_h.size().isEmpty())
    testEigensolverCorrectness(uplo, reference, ret.eigenvalues, ret.eigenvectors, grid);

  // Suspend pika to make sure dlaf_finalize resumes it
  pika::suspend();

  dlaf_free_grid(dlaf_context);
  dlaf_finalize();

#ifdef DLAF_WITH_SCALAPACK
  if constexpr (api == API::scalapack) {
    Cblacs_gridexit(dlaf_context);
  }
#endif
}

TYPED_TEST(EigensolverTestMC, CorrectnessDistributedDLAF) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        testEigensolver<TypeParam, Backend::MC, Device::CPU, API::dlaf>(uplo, m, mb, grid);
      }
    }
  }
}

#ifdef DLAF_WITH_SCALAPACK
TYPED_TEST(EigensolverTestMC, CorrectnessDistributedScalapack) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        testEigensolver<TypeParam, Backend::MC, Device::CPU, API::scalapack>(uplo, m, mb, grid);
      }
    }
  }
}
#endif

#ifdef DLAF_WITH_GPU
TYPED_TEST(EigensolverTestGPU, CorrectnessDistributedDLAF) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        testEigensolver<TypeParam, Backend::GPU, Device::GPU, API::dlaf>(uplo, m, mb, grid);
      }
    }
  }
}

#ifdef DLAF_WITH_SCALAPACK
TYPED_TEST(EigensolverTestGPU, CorrectnessDistributedScalapack) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        testEigensolver<TypeParam, Backend::GPU, Device::GPU, API::scalapack>(uplo, m, mb, grid);
      }
    }
  }
}
#endif
#endif
