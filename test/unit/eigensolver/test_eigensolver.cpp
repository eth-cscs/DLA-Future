//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/eigensolver.h"

#include <functional>
#include <tuple>

#include <gtest/gtest.h>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/eigensolver/api.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/tune.h"
#include "dlaf/types.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/eigensolver/test_eigensolver_correctness.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

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

TYPED_TEST_SUITE(EigensolverTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using EigensolverTestGPU = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestGPU, MatrixElementTypes);
#endif

enum class Allocation { use_preallocated, do_allocation };

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, eigensolver_min_band}
    {0, 2, 100},                                              // m = 0
    {5, 8, 100}, {34, 34, 100},                               // m <= mb
    {4, 3, 100}, {16, 10, 100}, {34, 13, 100}, {32, 5, 100},  // m > mb
    {34, 8, 3},  {32, 6, 3}                                   // m > mb, sub-band
};

template <class T, Backend B, Device D, Allocation allocation>
void testEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb) {
  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<const T, Device::CPU> reference = [&]() {
    Matrix<T, Device::CPU> reference(size, block_size);
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(reference.distribution());
  copy(reference, mat_a_h);

  eigensolver::EigensolverResult<T, D> ret = [&]() {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);

    if constexpr (allocation == Allocation::do_allocation) {
      return eigensolver::eigensolver<B>(uplo, mat_a.get());
    }
    else if constexpr (allocation == Allocation::use_preallocated) {
      const SizeType size = mat_a_h.size().rows();
      Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                         TileElementSize(mat_a_h.blockSize().rows(), 1));
      Matrix<T, D> eigenvectors(LocalElementSize(size, size), mat_a_h.blockSize());

      eigensolver::eigensolver<B>(uplo, mat_a.get(), eigenvalues, eigenvectors);

      return eigensolver::EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
    }
  }();

  if (mat_a_h.size().isEmpty())
    return;

  testEigensolverCorrectness(uplo, reference, ret.eigenvalues, ret.eigenvectors);
}

template <class T, Backend B, Device D, Allocation allocation>
void testEigensolver(comm::CommunicatorGrid grid, const blas::Uplo uplo, const SizeType m,
                     const SizeType mb) {
  const GlobalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<const T, Device::CPU> reference = [&]() {
    Matrix<T, Device::CPU> reference(size, block_size, grid);
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(reference.distribution());
  copy(reference, mat_a_h);

  eigensolver::EigensolverResult<T, D> ret = [&]() {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);

    if constexpr (allocation == Allocation::do_allocation) {
      return eigensolver::eigensolver<B>(grid, uplo, mat_a.get());
    }
    else if constexpr (allocation == Allocation::use_preallocated) {
      const SizeType size = mat_a_h.size().rows();
      matrix::Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                                 TileElementSize(mat_a_h.blockSize().rows(), 1));
      matrix::Matrix<T, D> eigenvectors(GlobalElementSize(size, size), mat_a_h.blockSize(), grid);

      eigensolver::eigensolver<B>(grid, uplo, mat_a.get(), eigenvalues, eigenvectors);

      return eigensolver::EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
    }
  }();

  if (mat_a_h.size().isEmpty())
    return;

  testEigensolverCorrectness(uplo, reference, ret.eigenvalues, ret.eigenvectors, grid);
}

TYPED_TEST(EigensolverTestMC, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    for (auto [m, mb, b_min] : sizes) {
      getTuneParameters().eigensolver_min_band = b_min;
      testEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation>(uplo, m, mb);
      testEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::use_preallocated>(uplo, m, mb);
    }
  }
}

TYPED_TEST(EigensolverTestMC, CorrectnessDistributed) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        getTuneParameters().eigensolver_min_band = b_min;
        testEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation>(grid, uplo, m,
                                                                                        mb);
        testEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::use_preallocated>(grid, uplo, m,
                                                                                           mb);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(EigensolverTestGPU, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    for (auto [m, mb, b_min] : sizes) {
      getTuneParameters().eigensolver_min_band = b_min;
      testEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation>(uplo, m, mb);
      testEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::use_preallocated>(uplo, m, mb);
    }
  }
}

TYPED_TEST(EigensolverTestGPU, CorrectnessDistributed) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        getTuneParameters().eigensolver_min_band = b_min;
        testEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation>(grid, uplo, m,
                                                                                         mb);
        testEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::use_preallocated>(grid, uplo,
                                                                                            m, mb);
      }
    }
  }
}
#endif
