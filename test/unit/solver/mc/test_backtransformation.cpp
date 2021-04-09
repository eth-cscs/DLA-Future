//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/solver/backtransformation.h"

#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/common/index2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/matrix/matrix_output.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"
//#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
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

//::testing::Environment* const comm_grids_env =
//    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class BackTransformationSolverLocalTest : public ::testing::Test {};
TYPED_TEST_SUITE(BackTransformationSolverLocalTest, MatrixElementTypes);

// template <typename Type>
// class BackTransformationSolverDistributedTest : public ::testing::Test {
// public:
//  const std::vector<CommunicatorGrid>& commGrids() {
//    return comm_grids;
//}
//};
// TYPED_TEST_SUITE(BackTransformationSolverDistributedTest, double);

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    {3, 0, 1, 1}, {0, 5, 2, 3},  // m, n = 0
    {2, 2, 3, 3}, {3, 4, 6, 7},  // m < mb
    {3, 3, 1, 1}, {4, 4, 2, 2}, {6, 3, 3, 3}, {12, 2, 4, 4}, {12, 24, 3, 3}, {24, 36, 6, 6},
    {5, 8, 3, 2}, {4, 6, 2, 3}, {5, 5, 2, 3}, {8, 27, 3, 4}, {15, 34, 4, 6},
};

template <class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().blockSize()};
}

template <class T>
void set_zero(Matrix<T, Device::CPU>& mat) {
  dlaf::matrix::util::set(mat, [](auto&&) { return static_cast<T>(0.0); });
}

template <class T>
void testBacktransformationEigenv(SizeType m, SizeType n, SizeType mb, SizeType nb) {
  comm::CommunicatorGrid comm_grid(MPI_COMM_WORLD, 1, 1, common::Ordering::ColumnMajor);

  LocalElementSize sizeC(m, n);
  TileElementSize blockSizeC(mb, nb);
  Matrix<T, Device::CPU> mat_c(sizeC, blockSizeC);
  dlaf::matrix::util::set_random(mat_c);

  LocalElementSize sizeV(m, m);
  TileElementSize blockSizeV(mb, mb);
  Matrix<T, Device::CPU> mat_v(sizeV, blockSizeV);
  dlaf::matrix::util::set_random(mat_v);

  int tottaus;
  if (m < mb || m == 0 || n == 0)
    tottaus = 0;
  else
    tottaus = (m / mb - 1) * mb + m % mb;

  if (tottaus > 0) {
    // Copy matrices locally
    auto mat_c_loc = dlaf::matrix::test::allGather<T>(mat_c, comm_grid);
    auto mat_v_loc = dlaf::matrix::test::allGather<T>(mat_v, comm_grid);

    // Impose orthogonality: Q = I - v tau v^H is orthogonal (Q Q^H = I)
    // leads to tau = [1 + sqrt(1 - vH v taui^2)]/(vH v) for real
    LocalElementSize sizeTau(m, 1);
    TileElementSize blockSizeTau(1, 1);
    Matrix<T, Device::CPU> mat_tau(sizeTau, blockSizeTau);

    // Reset diagonal and upper values of V
    lapack::laset(lapack::MatrixType::General, std::min(m, mb), mat_v_loc.size().cols(), 0, 0,
                  mat_v_loc.ptr(), mat_v_loc.ld());
    if (m > mb) {
      lapack::laset(lapack::MatrixType::Upper, mat_v_loc.size().rows() - mb, mat_v_loc.size().cols(), 0,
                    1, mat_v_loc.ptr(GlobalElementIndex{mb, 0}), mat_v_loc.ld());
    }

    // TODO: creating a whole matrix, solves issues on the last tile when m%mb != 0 ==> find out how to
    // use only a panel
    LocalElementSize sizeT(tottaus, tottaus);
    TileElementSize blockSizeT(mb, mb);
    Matrix<T, Device::CPU> mat_t(sizeT, blockSizeT);
    set_zero(mat_t);
    auto mat_t_loc = dlaf::matrix::test::allGather<T>(mat_t, comm_grid);

    common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus;

    MatrixLocal<T> tausloc({tottaus, 1}, {mb, mb});
    auto tau_rows = tausloc.nrTiles().rows();

    auto nt = 0;
    for (SizeType i = 0; i < tau_rows; ++i) {
      common::internal::vector<T> t_tile;
      for (SizeType t = 0; t < mb && nt < tottaus; ++t) {
        const GlobalElementIndex v_offset{i * mb + t, i * mb + t};
        auto dotprod = blas::dot(m - t, mat_v_loc.ptr(v_offset), 1, mat_v_loc.ptr(v_offset), 1);
        T taui;
        if (std::is_same<T, ComplexType<T>>::value) {
          auto seed = 10000 * i + 1;
          dlaf::matrix::util::internal::getter_random<T> random_value(seed);
          taui = random_value();
        }
        else {
          taui = static_cast<T>(0.0);
        }
        auto tau = (static_cast<T>(1.0) + sqrt(static_cast<T>(1.0) - dotprod * taui * taui)) / dotprod;
        tausloc({nt, 0}) = tau;
        t_tile.push_back(tau);
        ++nt;
      }
      taus.push_back(hpx::make_ready_future(t_tile));
    }

    for (SizeType i = tottaus - 1; i > -1; --i) {
      const GlobalElementIndex v_offset{i, i};
      auto tau = tausloc({i, 0});
      lapack::larf(lapack::Side::Left, m - i, n, mat_v_loc.ptr(v_offset), 1, tau,
                   mat_c_loc.ptr(GlobalElementIndex{i, 0}), mat_c_loc.ld());
    }

    solver::backTransformation<Backend::MC>(mat_c, mat_v, taus);

    auto result = [& dist = mat_c.distribution(),
                   &mat_local = mat_c_loc](const GlobalElementIndex& element) {
      const auto tile_index = dist.globalTileIndex(element);
      const auto tile_element = dist.tileElementIndex(element);
      return mat_local.tile_read(tile_index)(tile_element);
    };

    double error = 0.01;
    CHECK_MATRIX_NEAR(result, mat_c, error, error);
  }
}

TYPED_TEST(BackTransformationSolverLocalTest, Correctness) {
  SizeType m, n, mb, nb;

  for (auto sz : sizes) {
    std::tie(m, n, mb, nb) = sz;
    testBacktransformationEigenv<TypeParam>(m, n, mb, nb);
  }
}
