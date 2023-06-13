//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <functional>
#include <tuple>

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

#include "test_eigensolver_c_api_wrapper.h"

#include <gtest/gtest.h>

#include <pika/init.hpp>

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
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class EigensolverTest : public TestWithCommGrids {};

template <class T>
using EigensolverTestMC = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestMC, RealMatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using EigensolverTestGPU = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestGPU, RealMatrixElementTypes);
#endif

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, eigensolver_min_band}
    // {0, 2, 100},                                              // m = 0
    // {5, 8, 100}, {34, 34, 100},                               // m <= mb
    //{4, 3, 100}, {16, 10, 100}, {34, 13, 100}, {32, 5, 100},  // m > mb
    {32, 5, 100},
    //{34, 8, 3},  {32, 6, 3}                                   // m > mb, sub-band
};

template <class T, Backend B, Device D, class... GridIfDistributed>
void testEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb, CommunicatorGrid grid) {
  const char* argv[] = {"test_c_api_", nullptr};
  dlaf_initialize(1, argv);

  // The pika runtime is suspended by dlaf_initialize
  // In normal use the runtime is resumed by the C API call
  // Here we need to resume it manually to build the matrices with DLA-Future
  pika::resume();

  int dlaf_context =
      dlaf_create_grid(grid.fullCommunicator(), grid.size().rows(), grid.size().cols(), 'R');

  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<const T, Device::CPU> reference = [&]() {
    auto reference = [&]() -> auto {
      return Matrix<T, Device::CPU>(GlobalElementSize(m, m), block_size, grid);
    }();
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
    auto toplefttile_a =
        pika::this_thread::experimental::sync_wait(mat_a_h.readwrite(LocalTileIndex(0, 0)));
    auto toplefttile_eigenvalues =
        pika::this_thread::experimental::sync_wait(eigenvalues.readwrite(LocalTileIndex(0, 0)));
    auto toplefttile_eigenvectors =
        pika::this_thread::experimental::sync_wait(eigenvectors.readwrite(LocalTileIndex(0, 0)));

    // Get local leading dimension
    int m_local = mat_a_h.distribution().localSize().rows();
    DLAF_descriptor desc = {(int) m, (int) m, (int) mb, (int) mb, 0, 0, 1, 1, m_local};

    // Suspend pika to ensure it is resumed by the C API
    pika::suspend();

    if constexpr (std::is_same_v<T, double>) {
      C_dlaf_eigensolver_d(dlaf_context, dlaf_uplo, toplefttile_a.ptr(), desc,
                           toplefttile_eigenvalues.ptr(), toplefttile_eigenvectors.ptr(), desc);
    }
    else {
      C_dlaf_eigensolver_s(dlaf_context, dlaf_uplo, toplefttile_a.ptr(), desc,
                           toplefttile_eigenvalues.ptr(), toplefttile_eigenvectors.ptr(), desc);
    }

    // eigensolver::eigensolver<B>(grid..., uplo, mat_a.get(), eigenvalues, eigenvectors);
    return eigensolver::EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
  }();

  // if (mat_a_h.size().isEmpty())
  //   return;

  testEigensolverCorrectness(uplo, reference, ret.eigenvalues, ret.eigenvectors, grid);

  dlaf_free_grid(dlaf_context);
  dlaf_finalize();
}

TYPED_TEST(EigensolverTestMC, CorrectnessDistributed) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        //getTuneParameters().eigensolver_min_band = b_min;
        testEigensolver<TypeParam, Backend::MC, Device::CPU>(uplo, m, mb, grid);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(EigensolverTestGPU, CorrectnessDistributed) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        //getTuneParameters().eigensolver_min_band = b_min;
        testEigensolver<TypeParam, Backend::GPU, Device::GPU>(uplo, m, mb, grid);
      }
    }
  }
}
#endif
