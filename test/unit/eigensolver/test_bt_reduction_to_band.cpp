//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/bt_reduction_to_band.h"

#include <functional>
#include <sstream>
#include <tuple>

#include <gtest/gtest.h>
#include <pika/modules/threadmanager.hpp>
#include <pika/runtime.hpp>

#include "dlaf/common/index2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::common;
using namespace dlaf::matrix;
using namespace dlaf::matrix::internal;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct BackTransformationReductionToBandEigenSolverTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(BackTransformationReductionToBandEigenSolverTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <class T>
struct BackTransformationReductionToBandEigenSolverTestGPU : public TestWithCommGrids {};

TYPED_TEST_SUITE(BackTransformationReductionToBandEigenSolverTestGPU, MatrixElementTypes);
#endif

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes =
    // m, n, mb, nb
    {
        {3, 0, 1, 1}, {0, 5, 2, 3},                                   // m, n = 0
        {2, 2, 3, 3}, {3, 4, 6, 7},                                   // m < mb
        {3, 3, 1, 1}, {4, 4, 2, 2},  {12, 2, 4, 4},  {24, 36, 6, 6},  // mb = nb
        {5, 8, 3, 2}, {8, 27, 3, 4}, {15, 34, 4, 6},                  // mb != nb
        {3, 3, 2, 2}, {3, 27, 2, 4}                                   // m = mb + 1
};

template <class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().blockSize()};
}

template <class T>
void getTau(T& tau, T dotprod, BaseType<T> /*tau_i*/) {
  tau = static_cast<T>(2.0) / dotprod;
}

template <class T>
void getTau(std::complex<T>& tau, T dotprod, BaseType<T> tau_i) {
  tau = {(T(1) + sqrt(T(1) - dotprod * dotprod * tau_i * tau_i)) / dotprod, tau_i};
}

// Generate the vector with all the taus and compute the reference result in c.
// Note: v is modified as well.
template <class T>
common::internal::vector<T> setUpTest(MatrixLocal<T>& c, MatrixLocal<T>& v) {
  const auto m = c.size().rows();
  const auto mb = c.blockSize().rows();

  const SizeType nr_reflectors = std::max<SizeType>(0, m - mb - 1);

  common::internal::vector<T> taus;
  taus.reserve(nr_reflectors);

  dlaf::matrix::util::internal::getter_random<BaseType<T>> random_value(443);

  // Compute taus such that Q = I - v tau vH is orthogonal.
  // Real case: tau = 2 / (vH v)
  // Complex case: real part of tau = [1 + sqrt(1 - vH v taui^2)]/(vH v)
  for (SizeType k = 0; k < nr_reflectors; ++k) {
    const GlobalElementIndex v_offset{k + mb, k};
    v(v_offset) = T{1};
    auto dotprod = blas::dot(m - mb - k, v.ptr(v_offset), 1, v.ptr(v_offset), 1);

    BaseType<T> tau_i = std::is_same_v<T, ComplexType<T>> ? random_value() : 0;
    T tau;
    getTau(tau, dotprod, tau_i);

    taus.push_back(tau);
  }

  const auto n = c.size().cols();
  if (n > 0) {
    for (SizeType k = nr_reflectors - 1; k >= 0; --k) {
      const GlobalElementIndex v_offset{k + mb, k};
      lapack::larf(lapack::Side::Left, m - mb - k, n, v.ptr(v_offset), 1, taus[k],
                   c.ptr(GlobalElementIndex{k + mb, 0}), c.ld());
    }
  }

  return taus;
}

template <class T>
void testBackTransformationReductionToBand(SizeType m, SizeType n, SizeType mb, SizeType nb) {
  const LocalElementSize size_c(m, n);
  const LocalElementSize size_v(m, m);
  const TileElementSize block_size_c(mb, nb);
  const TileElementSize block_size_v(mb, mb);

  Matrix<T, Device::CPU> mat_c(size_c, block_size_c);
  dlaf::matrix::util::set_random(mat_c);

  Matrix<T, Device::CPU> mat_v(size_v, block_size_v);
  dlaf::matrix::util::set_random(mat_v);

  auto c_loc = dlaf::matrix::test::allGather<T>(blas::Uplo::General, mat_c);
  auto v_loc = dlaf::matrix::test::allGather<T>(blas::Uplo::Lower, mat_v);

  auto taus_loc = setUpTest(c_loc, v_loc);
  auto nr_reflectors = taus_loc.size();

  common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus;
  SizeType nr_reflectors_blocks = std::max<SizeType>(0, dlaf::util::ceilDiv(nr_reflectors, mb));
  taus.reserve(nr_reflectors_blocks);

  for (SizeType k = 0; k < nr_reflectors; k += mb) {
    common::internal::vector<T> tau_tile;
    tau_tile.reserve(mb);
    for (SizeType j = k; j < std::min(k + mb, nr_reflectors); ++j) {
      tau_tile.push_back(taus_loc[j]);
    }
    taus.push_back(pika::make_ready_future(tau_tile));
  }

  eigensolver::backTransformationReductionToBand<Backend::MC>(mat_c, mat_v, taus);

  auto result = [&c_loc](const GlobalElementIndex& index) { return c_loc(index); };

  const auto error = (mat_c.size().rows() + 1) * dlaf::test::TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(result, mat_c, error, error);
}

template <class T>
void testBackTransformationReductionToBand(comm::CommunicatorGrid grid, SizeType m, SizeType n,
                                           SizeType mb, SizeType nb) {
  const GlobalElementSize size_c(m, n);
  const GlobalElementSize size_v(m, m);
  const TileElementSize block_size_c(mb, nb);
  const TileElementSize block_size_v(mb, mb);

  comm::Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));
  const Distribution dist_c(size_c, block_size_c, grid.size(), grid.rank(), src_rank_index);
  const Distribution dist_v(size_v, block_size_v, grid.size(), grid.rank(), src_rank_index);

  Matrix<T, Device::CPU> mat_c(std::move(dist_c));
  dlaf::matrix::util::set_random(mat_c);

  Matrix<T, Device::CPU> mat_v(std::move(dist_v));
  dlaf::matrix::util::set_random(mat_v);

  auto c_loc = dlaf::matrix::test::allGather<T>(blas::Uplo::General, mat_c, grid);
  auto v_loc = dlaf::matrix::test::allGather<T>(blas::Uplo::General, mat_v, grid);

  auto taus_loc = setUpTest(c_loc, v_loc);
  auto nr_reflectors = taus_loc.size();

  common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus;
  SizeType nr_reflectors_blocks = std::max<SizeType>(0, dlaf::util::ceilDiv(m - mb - 1, mb));
  taus.reserve(dlaf::util::ceilDiv<SizeType>(nr_reflectors_blocks, grid.size().cols()));

  for (SizeType k = 0; k < nr_reflectors; k += mb) {
    common::internal::vector<T> tau_tile;
    tau_tile.reserve(mb);
    if (grid.rank().col() == mat_v.distribution().template rankGlobalTile<Coord::Col>(k / mb)) {
      for (SizeType j = k; j < std::min(k + mb, nr_reflectors); ++j) {
        tau_tile.push_back(taus_loc[j]);
      }
      taus.push_back(pika::make_ready_future(tau_tile));
    }
  }

  eigensolver::backTransformationReductionToBand<Backend::MC>(grid, mat_c, mat_v, taus);

  auto result = [&c_loc](const GlobalElementIndex& index) { return c_loc(index); };

  const auto error = (mat_c.size().rows() + 1) * dlaf::test::TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(result, mat_c, error, error);
}

TYPED_TEST(BackTransformationReductionToBandEigenSolverTestMC, CorrectnessLocal) {
  for (const auto& [m, n, mb, nb] : sizes) {
    testBackTransformationReductionToBand<TypeParam>(m, n, mb, nb);
  }
}

TYPED_TEST(BackTransformationReductionToBandEigenSolverTestMC, CorrectnessDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [m, n, mb, nb] : sizes) {
      testBackTransformationReductionToBand<TypeParam>(comm_grid, m, n, mb, nb);
      pika::threads::get_thread_manager().wait();
    }
  }
}
