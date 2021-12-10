//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/gen_to_std.h"

#include <functional>
#include <tuple>

#include <gtest/gtest.h>
#include <hpx/include/threadmanager.hpp>
#include <hpx/runtime.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T, Device D>
class EigensolverGenToStdTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

template <class T>
using EigensolverGenToStdTestMC = EigensolverGenToStdTest<T, Device::CPU>;

TYPED_TEST_SUITE(EigensolverGenToStdTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <class T>
using EigensolverGenToStdTestMC = EigensolverGenToStdTest<T, Device::GPU>;

TYPED_TEST_SUITE(EigensolverGenToStdTestGPU, MatrixElementTypes);
#endif

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

const std::vector<std::tuple<SizeType, SizeType>> sizes = {
    {0, 2},                              // m = 0
    {5, 8}, {34, 34},                    // m <= mb
    {4, 3}, {16, 10}, {34, 13}, {32, 5}  // m > mb
};

template <class T, Backend B, Device D>
void testGenToStdEigensolver(const blas::Uplo uplo, const SizeType m, const SizeType mb) {
  std::function<T(const GlobalElementIndex&)> el_t, el_a, res_a;

  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<T, Device::CPU> mat_ah(size, block_size);
  Matrix<T, Device::CPU> mat_th(size, block_size);

  std::tie(el_t, el_a, res_a) =
      getGenToStdElementSetters<GlobalElementIndex, T>(m, 1, uplo, BaseType<T>(-2.f), BaseType<T>(1.5f),
                                                       BaseType<T>(.95f));

  set(mat_ah, el_a);
  set(mat_th, el_t);

  {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<T, D, Device::CPU> mat_t(mat_th);

    eigensolver::genToStd<B>(uplo, mat_a.get(), mat_t.get());
  }

  CHECK_MATRIX_NEAR(res_a, mat_ah, 0, 10 * (mat_ah.size().rows() + 1) * TypeUtilities<T>::error);
}

template <class T, Backend B, Device D>
void testGenToStdEigensolver(comm::CommunicatorGrid grid, const blas::Uplo uplo, const SizeType m,
                             const SizeType mb) {
  std::function<T(const GlobalElementIndex&)> el_t, el_a, res_a;

  const GlobalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);
  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  Distribution distr_a(size, block_size, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_ah(std::move(distr_a));

  Distribution distr_t(size, block_size, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_th(std::move(distr_t));

  std::tie(el_t, el_a, res_a) =
      getGenToStdElementSetters<GlobalElementIndex, T>(m, 1, uplo, BaseType<T>(-2.f), BaseType<T>(1.5f),
                                                       BaseType<T>(.95f));

  set(mat_ah, el_a);
  set(mat_th, el_t);

  {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<T, D, Device::CPU> mat_t(mat_th);

    eigensolver::genToStd<B>(grid, uplo, mat_a.get(), mat_t.get());
  }

  CHECK_MATRIX_NEAR(res_a, mat_ah, 0, 10 * (mat_ah.size().rows() + 1) * TypeUtilities<T>::error);
  CHECK_MATRIX_NEAR(el_t, mat_th, 0, TypeUtilities<T>::error);
}

TYPED_TEST(EigensolverGenToStdTestMC, CorrectnessLocal) {
  SizeType m, mb;

  for (auto uplo : blas_uplos) {
    for (auto sz : sizes) {
      std::tie(m, mb) = sz;
      testGenToStdEigensolver<TypeParam, Backend::MC, Device::CPU>(uplo, m, mb);
    }
  }
}

TYPED_TEST(EigensolverGenToStdTestMC, CorrectnessDistributed) {
  SizeType m, mb;

  for (const auto& comm_grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto sz : sizes) {
        std::tie(m, mb) = sz;
        testGenToStdEigensolver<TypeParam, Backend::MC, Device::CPU>(comm_grid, uplo, m, mb);
        hpx::threads::get_thread_manager().wait();
      }
    }
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(EigensolverGenToStdTestGPU, CorrectnessLocal) {
  SizeType m, mb;

  for (auto uplo : blas_uplos) {
    for (auto sz : sizes) {
      std::tie(m, mb) = sz;
      testGenToStdEigensolver<TypeParam, Backend::GPU, Device::GPU>(uplo, m, mb);
    }
  }
}

TYPED_TEST(EigensolverGenToStdTestGPU, CorrectnessDistributed) {
  SizeType m, mb;

  for (const auto& comm_grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto sz : sizes) {
        std::tie(m, mb) = sz;
        testGenToStdEigensolver<TypeParam, Backend::GPU, Device::GPU>(comm_grid, uplo, m, mb);
        hpx::threads::get_thread_manager().wait();
      }
    }
  }
}
#endif
