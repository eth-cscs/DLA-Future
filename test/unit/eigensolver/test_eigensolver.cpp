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

template <typename>
inline constexpr bool dependent_false_v = false;

enum class Allocation { use_preallocated, do_allocation };

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower});

const std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {
    // {m, mb, eigensolver_min_band}
    {0, 2, 100},                                              // m = 0
    {5, 8, 100}, {34, 34, 100},                               // m <= mb
    {4, 3, 100}, {16, 10, 100}, {34, 13, 100}, {32, 5, 100},  // m > mb
    {34, 8, 3},  {32, 6, 3}                                   // m > mb, sub-band
};

template <class T, Device D, class... GridIfDistributed>
void testEigensolverCorrectness(const blas::Uplo uplo, Matrix<const T, Device::CPU>& reference,
                                eigensolver::EigensolverResult<T, D>& ret, GridIfDistributed... grid) {
  // Note:
  // Wait for the algorithm to finish all scheduled tasks, because verification has MPI blocking
  // calls that might lead to deadlocks.
  constexpr bool isDistributed = (sizeof...(grid) == 1);
  if constexpr (isDistributed)
    pika::threads::get_thread_manager().wait();

  const SizeType m = reference.size().rows();

  auto mat_a_local = allGather(blas::Uplo::General, reference, grid...);
  auto mat_evalues_local = [&]() {
    MatrixMirror<const BaseType<T>, Device::CPU, D> mat_evals(ret.eigenvalues);
    return allGather(blas::Uplo::General, mat_evals.get());
  }();
  auto mat_e_local = [&]() {
    MatrixMirror<const T, Device::CPU, D> mat_e(ret.eigenvectors);
    return allGather(blas::Uplo::General, mat_e.get(), grid...);
  }();

  MatrixLocal<T> workspace({m, m}, reference.blockSize());

  // Check eigenvectors orthogonality (E^H E == Id)
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, m, m, m, T{1},
             mat_e_local.ptr(), mat_e_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0},
             workspace.ptr(), workspace.ld());

  auto id = [](GlobalElementIndex index) {
    if (index.row() == index.col())
      return T{1};
    return T{0};
  };
  CHECK_MATRIX_NEAR(id, workspace, m * TypeUtilities<T>::error, 10 * m * TypeUtilities<T>::error);

  // Check Ax = lambda x
  // Compute A E
  blas::hemm(blas::Layout::ColMajor, blas::Side::Left, uplo, m, m, T{1}, mat_a_local.ptr(),
             mat_a_local.ld(), mat_e_local.ptr(), mat_e_local.ld(), T{0}, workspace.ptr(),
             workspace.ld());

  // Compute Lambda E (in place in mat_e_local)
  for (SizeType j = 0; j < m; ++j) {
    blas::scal(m, mat_evalues_local({j, 0}), mat_e_local.ptr({0, j}), 1);
  }

  // Check A E == Lambda E
  auto res = [&mat_e_local](GlobalElementIndex index) { return mat_e_local(index); };
  CHECK_MATRIX_NEAR(res, workspace, 2 * m * TypeUtilities<T>::error, 2 * m * TypeUtilities<T>::error);
}

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
    else {
      static_assert(dependent_false_v<Allocation>, "Invalid value for template parameter 'allocation'.");
    }
  }();

  if (mat_a_h.size().isEmpty())
    return;

  testEigensolverCorrectness(uplo, reference, ret);
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
    else {
      static_assert(dependent_false_v<Allocation>, "Invalid value for template parameter 'allocation'.");
    }
  }();

  if (mat_a_h.size().isEmpty())
    return;

  testEigensolverCorrectness(uplo, reference, ret, grid);
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
