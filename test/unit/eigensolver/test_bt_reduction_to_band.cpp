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
#include "dlaf/matrix/matrix_mirror.h"
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

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType>> sizes =
    // m, n, mb, nb, b
    {
        {3, 0, 1, 1, 1}, {0, 5, 2, 3, 2},                                         // m, n = 0
        {2, 2, 3, 3, 3}, {3, 4, 6, 7, 6},  {2, 2, 5, 4, 3},   {4, 4, 2, 3, 5},    // m < b
        {3, 3, 2, 2, 2}, {3, 27, 2, 4, 2}, {3, 3, 2, 2, 2},   {3, 3, 2, 2, 2},    // m = b + 1
        {3, 3, 1, 1, 1}, {4, 4, 2, 2, 4},  {12, 2, 4, 4, 2},  {24, 36, 6, 6, 4},  // mb = nb
        {5, 8, 3, 2, 3}, {8, 27, 5, 4, 3}, {15, 34, 4, 6, 6},                     // mb != nb
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
common::internal::vector<T> setUpTest(SizeType b, MatrixLocal<T>& c, MatrixLocal<T>& v) {
  const auto m = c.size().rows();

  const SizeType nr_reflectors = std::max<SizeType>(0, m - b - 1);

  common::internal::vector<T> taus;
  taus.reserve(nr_reflectors);

  dlaf::matrix::util::internal::getter_random<BaseType<T>> random_value(443);

  // Compute taus such that Q = I - v tau vH is orthogonal.
  // Real case: tau = 2 / (vH v)
  // Complex case: real part of tau = [1 + sqrt(1 - vH v taui^2)]/(vH v)
  for (SizeType k = 0; k < nr_reflectors; ++k) {
    const GlobalElementIndex v_offset{k + b, k};
    v(v_offset) = T{1};
    auto dotprod = blas::dot(m - b - k, v.ptr(v_offset), 1, v.ptr(v_offset), 1);

    BaseType<T> tau_i = std::is_same_v<T, ComplexType<T>> ? random_value() : 0;
    T tau;
    getTau(tau, dotprod, tau_i);

    taus.push_back(tau);
  }

  const auto n = c.size().cols();
  if (n > 0) {
    for (SizeType k = nr_reflectors - 1; k >= 0; --k) {
      const GlobalElementIndex v_offset{k + b, k};
      lapack::larf(lapack::Side::Left, m - b - k, n, v.ptr(v_offset), 1, taus[k],
                   c.ptr(GlobalElementIndex{k + b, 0}), c.ld());
    }
  }

  return taus;
}

template <class T, Backend B, Device D>
void testBackTransformationReductionToBand(SizeType m, SizeType n, SizeType mb, SizeType nb,
                                           SizeType b) {
  const LocalElementSize size_c(m, n);
  const LocalElementSize size_v(m, m);
  const TileElementSize block_size_c(mb, nb);
  const TileElementSize block_size_v(mb, mb);

  Matrix<T, Device::CPU> mat_c_h(size_c, block_size_c);
  dlaf::matrix::util::set_random(mat_c_h);

  Matrix<T, Device::CPU> mat_v_h(size_v, block_size_v);
  dlaf::matrix::util::set_random(mat_v_h);

  auto c_loc = dlaf::matrix::test::allGather<T>(blas::Uplo::General, mat_c_h);
  auto v_loc = dlaf::matrix::test::allGather<T>(blas::Uplo::Lower, mat_v_h);

  auto taus_loc = setUpTest(b, c_loc, v_loc);
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

  {
    MatrixMirror<T, D, Device::CPU> mat_c(mat_c_h);
    MatrixMirror<const T, D, Device::CPU> mat_v(mat_v_h);
    eigensolver::backTransformationReductionToBand<B, D, T>(b, mat_c.get(), mat_v.get(), taus);
  }

  auto result = [&c_loc](const GlobalElementIndex& index) { return c_loc(index); };

  const auto error = (mat_c_h.size().rows() + 1) * dlaf::test::TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(result, mat_c_h, error, error);
}

template <class T, Backend B, Device D>
void testBackTransformationReductionToBand(comm::CommunicatorGrid grid, SizeType m, SizeType n,
                                           SizeType mb, SizeType nb) {
  const GlobalElementSize size_c(m, n);
  const GlobalElementSize size_v(m, m);
  const TileElementSize block_size_c(mb, nb);
  const TileElementSize block_size_v(mb, mb);

  comm::Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));
  const Distribution dist_c(size_c, block_size_c, grid.size(), grid.rank(), src_rank_index);
  const Distribution dist_v(size_v, block_size_v, grid.size(), grid.rank(), src_rank_index);

  Matrix<T, Device::CPU> mat_c_h(std::move(dist_c));
  dlaf::matrix::util::set_random(mat_c_h);

  Matrix<T, Device::CPU> mat_v_h(std::move(dist_v));
  dlaf::matrix::util::set_random(mat_v_h);

  auto c_loc = dlaf::matrix::test::allGather<T>(blas::Uplo::General, mat_c_h, grid);
  auto v_loc = dlaf::matrix::test::allGather<T>(blas::Uplo::Lower, mat_v_h, grid);

  auto taus_loc = setUpTest(mb, c_loc, v_loc);
  auto nr_reflectors = taus_loc.size();

  common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus;
  SizeType nr_reflectors_blocks = std::max<SizeType>(0, dlaf::util::ceilDiv(m - mb - 1, mb));
  taus.reserve(dlaf::util::ceilDiv<SizeType>(nr_reflectors_blocks, grid.size().cols()));

  for (SizeType k = 0; k < nr_reflectors; k += mb) {
    common::internal::vector<T> tau_tile;
    tau_tile.reserve(mb);
    if (grid.rank().col() == mat_v_h.distribution().template rankGlobalTile<Coord::Col>(k / mb)) {
      for (SizeType j = k; j < std::min(k + mb, nr_reflectors); ++j) {
        tau_tile.push_back(taus_loc[j]);
      }
      taus.push_back(pika::make_ready_future(tau_tile));
    }
  }

  {
    MatrixMirror<T, D, Device::CPU> mat_c(mat_c_h);
    MatrixMirror<const T, D, Device::CPU> mat_v(mat_v_h);
    eigensolver::backTransformationReductionToBand<B, D, T>(grid, mat_c.get(), mat_v.get(), taus);
  }

  auto result = [&c_loc](const GlobalElementIndex& index) { return c_loc(index); };

  const auto error = (mat_c_h.size().rows() + 1) * dlaf::test::TypeUtilities<T>::error;
  CHECK_MATRIX_NEAR(result, mat_c_h, error, error);
}

TYPED_TEST(BackTransformationReductionToBandEigenSolverTestMC, CorrectnessLocal) {
  for (const auto& [m, n, mb, nb, b] : sizes) {
    testBackTransformationReductionToBand<TypeParam, Backend::MC, Device::CPU>(m, n, mb, nb, b);
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(BackTransformationReductionToBandEigenSolverTestGPU, CorrectnessLocal) {
  for (const auto& [m, n, mb, nb, b] : sizes) {
    testBackTransformationReductionToBand<TypeParam, Backend::GPU, Device::GPU>(m, n, mb, nb, b);
  }
}
#endif

TYPED_TEST(BackTransformationReductionToBandEigenSolverTestMC, CorrectnessDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [m, n, mb, nb, b] : sizes) {
      // b != mb is currently not supported by the implementation yet.
      (void) mb;

      testBackTransformationReductionToBand<TypeParam, Backend::MC, Device::CPU>(comm_grid, m, n, b, nb);
      pika::threads::get_thread_manager().wait();
    }
  }
}
