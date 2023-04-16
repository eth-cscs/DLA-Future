//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/eigensolver/api.h"
#include "dlaf/eigensolver/gen_eigensolver.h"

#include <functional>
#include <tuple>

#include <gtest/gtest.h>

#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/tune.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
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

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, eigensolver_min_band}
    {0, 2, 100},                                              // m = 0
    {5, 8, 100}, {34, 34, 100},                               // m <= mb
    {4, 3, 100}, {16, 10, 100}, {34, 13, 100}, {32, 5, 100},  // m > mb
    {34, 8, 3},  {32, 6, 3}                                   // m > mb, sub-band
};

template <class T, Device D, class... GridIfDistributed>
void testGenEigensolverCorrectness(const blas::Uplo uplo, Matrix<const T, Device::CPU>& reference_a,
                                   Matrix<const T, Device::CPU>& reference_b,
                                   eigensolver::EigensolverResult<T, D>& ret,
                                   GridIfDistributed... grid) {
  // Note:
  // Wait for the algorithm to finish all scheduled tasks, because verification has MPI blocking
  // calls that might lead to deadlocks.
  constexpr bool isDistributed = (sizeof...(grid) == 1);
  if constexpr (isDistributed)
    pika::threads::get_thread_manager().wait();

  const SizeType m = reference_a.size().rows();

  auto mat_a_local = allGather(blas::Uplo::General, reference_a, grid...);
  auto mat_b_local = allGather(blas::Uplo::General, reference_b, grid...);
  auto mat_evalues_local = [&]() {
    MatrixMirror<const BaseType<T>, Device::CPU, D> mat_evals(ret.eigenvalues);
    return allGather(blas::Uplo::General, mat_evals.get());
  }();
  auto mat_e_local = [&]() {
    MatrixMirror<const T, Device::CPU, D> mat_e(ret.eigenvectors);
    return allGather(blas::Uplo::General, mat_e.get(), grid...);
  }();

  MatrixLocal<T> mat_be_local({m, m}, reference_a.blockSize());
  // Compute B E which is needed for both checks.
  blas::hemm(blas::Layout::ColMajor, blas::Side::Left, uplo, m, m, T{1}, mat_b_local.ptr(),
             mat_b_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0}, mat_be_local.ptr(),
             mat_be_local.ld());

  MatrixLocal<T> workspace({m, m}, reference_a.blockSize());

  // Check eigenvectors orthogonality (E^H B E == Id)
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, m, m, m, T{1},
             mat_e_local.ptr(), mat_e_local.ld(), mat_be_local.ptr(), mat_be_local.ld(), T{0},
             workspace.ptr(), workspace.ld());

  auto id = [](GlobalElementIndex index) {
    if (index.row() == index.col())
      return T{1};
    return T{0};
  };
  CHECK_MATRIX_NEAR(id, workspace, m * TypeUtilities<T>::error, 10 * m * TypeUtilities<T>::error);

  // Check Ax = lambda B x
  // Compute A E
  blas::hemm(blas::Layout::ColMajor, blas::Side::Left, uplo, m, m, T{1}, mat_a_local.ptr(),
             mat_a_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0}, workspace.ptr(),
             workspace.ld());

  // Compute Lambda E (in place in mat_e_local)
  for (SizeType j = 0; j < m; ++j) {
    blas::scal(m, mat_evalues_local({j, 0}), mat_be_local.ptr({0, j}), 1);
  }

  // Check A E == Lambda E
  CHECK_MATRIX_NEAR(mat_be_local, workspace, m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

template <class T, Backend B, Device D, Allocation allocation, class... GridIfDistributed>
void testGenEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb,
                        GridIfDistributed... grid) {
  constexpr bool isDistributed = (sizeof...(grid) == 1);

  const TileElementSize block_size(mb, mb);

  auto create_reference = [&]() -> auto {
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

  eigensolver::EigensolverResult<T, D> ret = [&]() {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);
    MatrixMirror<T, D, Device::CPU> mat_b(mat_b_h);
    if constexpr (allocation == Allocation::do_allocation) {
      if constexpr (isDistributed) {
        return eigensolver::genEigensolver<B>(grid..., uplo, mat_a.get(), mat_b.get());
      }
      else {
        return eigensolver::genEigensolver<B>(uplo, mat_a.get(), mat_b.get());
      }
    }
    else if constexpr (allocation == Allocation::use_preallocated) {
      const SizeType size = mat_a_h.size().rows();
      Matrix<BaseType<T>, D> eigenvalues(LocalElementSize(size, 1),
                                         TileElementSize(mat_a_h.blockSize().rows(), 1));
      if constexpr (isDistributed) {
        Matrix<T, D> eigenvectors(GlobalElementSize(size, size), mat_a_h.blockSize(), grid...);
        eigensolver::genEigensolver<B>(grid..., uplo, mat_a.get(), mat_b.get(), eigenvalues,
                                       eigenvectors);
        return eigensolver::EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
      }
      else {
        Matrix<T, D> eigenvectors(LocalElementSize(size, size), mat_a_h.blockSize());
        eigensolver::genEigensolver<B>(uplo, mat_a.get(), mat_b.get(), eigenvalues, eigenvectors);
        return eigensolver::EigensolverResult<T, D>{std::move(eigenvalues), std::move(eigenvectors)};
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
      testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation>(uplo, m, mb);
      testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::use_preallocated>(uplo, m, mb);
    }
  }
}

TYPED_TEST(GenEigensolverTestMC, CorrectnessDistributed) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        getTuneParameters().eigensolver_min_band = b_min;
        testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::do_allocation>(uplo, m, mb,
                                                                                           grid);
        testGenEigensolver<TypeParam, Backend::MC, Device::CPU, Allocation::use_preallocated>(uplo, m,
                                                                                              mb, grid);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(GenEigensolverTestGPU, CorrectnessLocal) {
  for (auto uplo : blas_uplos) {
    for (auto [m, mb, b_min] : sizes) {
      getTuneParameters().eigensolver_min_band = b_min;
      testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation>(uplo, m, mb);
      testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::use_preallocated>(uplo, m,
                                                                                             mb);
    }
  }
}

TYPED_TEST(GenEigensolverTestGPU, CorrectnessDistributed) {
  for (const comm::CommunicatorGrid& grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto [m, mb, b_min] : sizes) {
        getTuneParameters().eigensolver_min_band = b_min;
        testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::do_allocation>(uplo, m, mb,
                                                                                            grid);
        testGenEigensolver<TypeParam, Backend::GPU, Device::GPU, Allocation::use_preallocated>(uplo, m,
                                                                                               mb, grid);
      }
    }
  }
}
#endif
