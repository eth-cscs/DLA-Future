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

#include <pika/init.hpp>

#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/eigensolver/gen_eigensolver.h>
#include <dlaf/factorization/cholesky.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/tune.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/eigensolver/test_gen_eigensolver_correctness.h>
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
class GenEigensolverTest : public TestWithCommGrids {};

template <class T>
using GenEigensolverTestMC = GenEigensolverTest<T>;

TYPED_TEST_SUITE(GenEigensolverTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using GenEigensolverTestGPU = GenEigensolverTest<T>;

TYPED_TEST_SUITE(GenEigensolverTestGPU, MatrixElementTypes);
#endif

enum class Allocation { do_allocation, use_preallocated };
using dlaf::eigensolver::internal::Factorization;

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, eigensolver_min_band}
    {0, 2, 100},                                              // m = 0
    {5, 8, 100}, {34, 34, 100},                               // m <= mb
    {4, 3, 100}, {16, 10, 100}, {34, 13, 100}, {32, 5, 100},  // m > mb
    {34, 8, 3},  {32, 6, 3}                                   // m > mb, sub-band
};

template <class T, Backend B, Device D, Allocation allocation, Factorization factorization,
          class... GridIfDistributed>
void testGenEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb,
                        GridIfDistributed&... grid) {
  constexpr bool isDistributed = (sizeof...(grid) == 1);

  const TileElementSize block_size(mb, mb);

  auto create_reference = [&]() {
    if constexpr (isDistributed)
      return Matrix<T, Device::CPU>(GlobalElementSize(m, m), block_size, grid...);
    else
      return Matrix<T, Device::CPU>(LocalElementSize(m, m), block_size);
  };

  Matrix<const T, Device::CPU> reference_a = [&]() {
    auto reference = create_reference();
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<const T, Device::CPU> reference_b = [&]() {
    auto reference = create_reference();
    matrix::util::set_random_hermitian_positive_definite(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(reference_a.distribution());
  copy(reference_a, mat_a_h);
  Matrix<T, Device::CPU> mat_b_h(reference_b.distribution());
  copy(reference_b, mat_b_h);

  EigensolverResult<T, D> ret = [&]() {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);
    MatrixMirror<T, D, Device::CPU> mat_b(mat_b_h);
    if constexpr (allocation == Allocation::do_allocation) {
      if constexpr (isDistributed) {
        if constexpr (factorization == Factorization::do_factorization) {
          return hermitian_generalized_eigensolver<B>(grid..., uplo, mat_a.get(), mat_b.get());
        }
        else {
          cholesky_factorization<B, D, T>(grid..., uplo, mat_b.get());
          return hermitian_generalized_eigensolver_factorized<B>(grid..., uplo, mat_a.get(),
                                                                 mat_b.get());
        }
      }
      else {
        if constexpr (factorization == Factorization::do_factorization) {
          return hermitian_generalized_eigensolver<B>(uplo, mat_a.get(), mat_b.get());
        }
        else {
          cholesky_factorization<B, D, T>(uplo, mat_b.get());
          return hermitian_generalized_eigensolver_factorized<B>(uplo, mat_a.get(), mat_b.get());
        }
      }
    }
    else if constexpr (allocation == Allocation::use_preallocated) {
      const SizeType size = mat_a_h.size().rows();
      Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                         TileElementSize(mat_a_h.blockSize().rows(), 1));
      if constexpr (isDistributed) {
        Matrix<T, D> eigenvectors(GlobalElementSize(size, size), mat_a_h.blockSize(), grid...);
        if constexpr (factorization == Factorization::do_factorization) {
          hermitian_generalized_eigensolver<B>(grid..., uplo, mat_a.get(), mat_b.get(), eigenvalues,
                                               eigenvectors);
        }
        else {
          cholesky_factorization<B, D, T>(grid..., uplo, mat_b.get());
          hermitian_generalized_eigensolver_factorized<B>(grid..., uplo, mat_a.get(), mat_b.get(),
                                                          eigenvalues, eigenvectors);
        }
        return EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
      }
      else {
        Matrix<T, D> eigenvectors(LocalElementSize(size, size), mat_a_h.blockSize());
        if constexpr (factorization == Factorization::do_factorization) {
          hermitian_generalized_eigensolver<B>(uplo, mat_a.get(), mat_b.get(), eigenvalues,
                                               eigenvectors);
        }
        else {
          cholesky_factorization<B, D, T>(uplo, mat_b.get());
          hermitian_generalized_eigensolver_factorized<B>(uplo, mat_a.get(), mat_b.get(), eigenvalues,
                                                          eigenvectors);
        }
        return EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
      }
    }
  }();

  if (mat_a_h.size().isEmpty())
    return;

  testGenEigensolverCorrectness(uplo, reference_a, reference_b, ret, grid...);
}

TYPED_TEST(GenEigensolverTestMC, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    for (auto [m, mb, b_min] : sizes) {
      getTuneParameters().eigensolver_min_band = b_min;
      testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation,
                         Factorization::do_factorization>(uplo, m, mb);
      testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::use_preallocated,
                         Factorization::do_factorization>(uplo, m, mb);
      testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation,
                         Factorization::already_factorized>(uplo, m, mb);
      testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::use_preallocated,
                         Factorization::already_factorized>(uplo, m, mb);
    }
  }
}

TYPED_TEST(GenEigensolverTestMC, CorrectnessDistributed) {
  for (comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        getTuneParameters().eigensolver_min_band = b_min;
        testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation,
                           Factorization::do_factorization>(uplo, m, mb, grid);
        testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::use_preallocated,
                           Factorization::do_factorization>(uplo, m, mb, grid);
        testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation,
                           Factorization::already_factorized>(uplo, m, mb, grid);
        testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::use_preallocated,
                           Factorization::already_factorized>(uplo, m, mb, grid);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GenEigensolverTestGPU, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    for (auto [m, mb, b_min] : sizes) {
      getTuneParameters().eigensolver_min_band = b_min;
      testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation,
                         Factorization::do_factorization>(uplo, m, mb);
      testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::use_preallocated,
                         Factorization::do_factorization>(uplo, m, mb);
      testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation,
                         Factorization::already_factorized>(uplo, m, mb);
      testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::use_preallocated,
                         Factorization::already_factorized>(uplo, m, mb);
    }
  }
}

TYPED_TEST(GenEigensolverTestGPU, CorrectnessDistributed) {
  for (comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        getTuneParameters().eigensolver_min_band = b_min;
        testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation,
                           Factorization::do_factorization>(uplo, m, mb, grid);
        testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::use_preallocated,
                           Factorization::do_factorization>(uplo, m, mb, grid);
        testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation,
                           Factorization::already_factorized>(uplo, m, mb, grid);
        testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::use_preallocated,
                           Factorization::already_factorized>(uplo, m, mb, grid);
      }
    }
  }
}
#endif
