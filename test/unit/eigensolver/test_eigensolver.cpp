//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <functional>
#include <tuple>
#include <utility>
#include <vector>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/eigensolver/eigensolver/api.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/tune.h>
#include <dlaf/types.h>

#include <gtest/gtest.h>

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

TYPED_TEST_SUITE(EigensolverTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using EigensolverTestGPU = EigensolverTest<T>;

TYPED_TEST_SUITE(EigensolverTestGPU, MatrixElementTypes);
#endif

enum class Allocation { use_preallocated, do_allocation };
enum class MatrixType { random, identity };

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, eigensolver_min_band}
    {0, 2, 100},                                              // m = 0
    {5, 8, 100}, {34, 34, 100},                               // m <= mb
    {4, 3, 100}, {16, 10, 100}, {34, 13, 100}, {32, 5, 100},  // m > mb
    {34, 8, 3},  {32, 6, 3}                                   // m > mb, sub-band
};

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes_id = {
    // {m, mb, eigensolver_min_band}
    {8, 4, 4},
    {34, 8, 4},
};

template <class T, Backend B, Device D, Allocation allocation, class... GridIfDistributed>
void testEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb, const MatrixType type,
                     GridIfDistributed&... grid) {
  constexpr bool isDistributed = (sizeof...(grid) == 1);
  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<const T, Device::CPU> reference = [&]() {
    auto reference = [&]() -> auto {
      if constexpr (isDistributed)
        return Matrix<T, Device::CPU>(GlobalElementSize(m, m), block_size, grid...);
      else
        return Matrix<T, Device::CPU>(LocalElementSize(m, m), block_size);
    }();
    switch (type) {
      case MatrixType::identity:
        matrix::util::set_identity(reference);
        break;
      case MatrixType::random:
        matrix::util::set_random_hermitian(reference);
        break;
    }
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(reference.distribution());
  copy(reference, mat_a_h);

  EigensolverResult<T, D> ret = [&]() {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);

    if constexpr (allocation == Allocation::do_allocation) {
      if constexpr (isDistributed) {
        return hermitian_eigensolver<B>(grid..., uplo, mat_a.get());
      }
      else {
        return hermitian_eigensolver<B>(uplo, mat_a.get());
      }
    }
    else if constexpr (allocation == Allocation::use_preallocated) {
      const SizeType size = mat_a_h.size().rows();
      Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                         TileElementSize(mat_a_h.blockSize().rows(), 1));
      if constexpr (isDistributed) {
        Matrix<T, D> eigenvectors(GlobalElementSize(size, size), mat_a_h.blockSize(), grid...);
        hermitian_eigensolver<B>(grid..., uplo, mat_a.get(), eigenvalues, eigenvectors);
        return EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
      }
      else {
        Matrix<T, D> eigenvectors(LocalElementSize(size, size), mat_a_h.blockSize());
        hermitian_eigensolver<B>(uplo, mat_a.get(), eigenvalues, eigenvectors);
        return EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
      }
    }
  }();

  if (mat_a_h.size().isEmpty())
    return;

  testEigensolverCorrectness(uplo, reference, ret.eigenvalues, ret.eigenvectors, grid...);
}

TYPED_TEST(EigensolverTestMC, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    for (auto [m, mb, b_min] : sizes) {
      getTuneParameters().eigensolver_min_band = b_min;
      testEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation>(
          uplo, m, mb, MatrixType::random);
      testEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::use_preallocated>(
          uplo, m, mb, MatrixType::random);
    }
    for (auto [m, mb, b_min] : sizes_id) {
      getTuneParameters().eigensolver_min_band = b_min;
      testEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation>(
          uplo, m, mb, MatrixType::identity);
    }
  }
}

TYPED_TEST(EigensolverTestMC, CorrectnessDistributed) {
  for (comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        getTuneParameters().eigensolver_min_band = b_min;
        testEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation>(
            uplo, m, mb, MatrixType::random, grid);
        testEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::use_preallocated>(
            uplo, m, mb, MatrixType::random, grid);
      }
      for (auto [m, mb, b_min] : sizes_id) {
        getTuneParameters().eigensolver_min_band = b_min;
        testEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation>(
            uplo, m, mb, MatrixType::identity, grid);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(EigensolverTestGPU, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    for (auto [m, mb, b_min] : sizes) {
      getTuneParameters().eigensolver_min_band = b_min;
      testEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation>(
          uplo, m, mb, MatrixType::random);
      testEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::use_preallocated>(
          uplo, m, mb, MatrixType::random);
    }
    for (auto [m, mb, b_min] : sizes_id) {
      getTuneParameters().eigensolver_min_band = b_min;
      testEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation>(
          uplo, m, mb, MatrixType::identity);
    }
  }
}

TYPED_TEST(EigensolverTestGPU, CorrectnessDistributed) {
  for (comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        getTuneParameters().eigensolver_min_band = b_min;
        testEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation>(
            uplo, m, mb, MatrixType::random, grid);
        testEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::use_preallocated>(
            uplo, m, mb, MatrixType::random, grid);
      }
      for (auto [m, mb, b_min] : sizes_id) {
        getTuneParameters().eigensolver_min_band = b_min;
        testEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation>(
            uplo, m, mb, MatrixType::identity, grid);
      }
    }
  }
}
#endif
