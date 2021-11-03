//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/solver/triangular.h"

#include <functional>
#include <tuple>

#include <gtest/gtest.h>
#include <hpx/include/threadmanager.hpp>
#include <hpx/runtime.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_generic_blas.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::util;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class TriangularSolverTestMC : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};
TYPED_TEST_SUITE(TriangularSolverTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <typename Type>
class TriangularSolverTestGPU : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};
TYPED_TEST_SUITE(TriangularSolverTestGPU, MatrixElementTypes);
#endif

const std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<blas::Diag> blas_diags({blas::Diag::NonUnit, blas::Diag::Unit});

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    {0, 0, 1, 1},                                                // m, n = 0
    {0, 2, 1, 2}, {7, 0, 2, 1},                                  // m = 0 or n = 0
    {2, 2, 5, 5}, {10, 10, 2, 3}, {7, 7, 3, 2},                  // m = n
    {3, 2, 7, 7}, {12, 3, 5, 5},  {7, 6, 3, 2}, {15, 7, 3, 5},   // m > n
    {2, 3, 7, 7}, {4, 13, 5, 5},  {7, 8, 2, 9}, {19, 25, 6, 5},  // m < n
};

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

template <class T, Backend B, Device D>
void testTriangularSolver(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                          SizeType m, SizeType n, SizeType mb, SizeType nb) {
  std::function<T(const GlobalElementIndex&)> el_op_a, el_b, res_b;

  LocalElementSize size_a(m, m);
  TileElementSize block_size_a(mb, mb);

  if (side == blas::Side::Right) {
    size_a = {n, n};
    block_size_a = {nb, nb};
  }

  Matrix<T, Device::CPU> mat_ah(size_a, block_size_a);

  LocalElementSize size_b(m, n);
  TileElementSize block_size_b(mb, nb);
  Matrix<T, Device::CPU> mat_bh(size_b, block_size_b);

  if (side == blas::Side::Left)
    std::tie(el_op_a, el_b, res_b) =
        getLeftTriangularSystem<GlobalElementIndex, T>(uplo, op, diag, alpha, m);
  else
    std::tie(el_op_a, el_b, res_b) =
        getRightTriangularSystem<GlobalElementIndex, T>(uplo, op, diag, alpha, n);

  set(mat_ah, el_op_a, op);
  set(mat_bh, el_b);

  {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<T, D, Device::CPU> mat_b(mat_bh);

    solver::triangular<B>(side, uplo, op, diag, alpha, mat_a.get(), mat_b.get());
  }

  CHECK_MATRIX_NEAR(res_b, mat_bh, 40 * (mat_bh.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_bh.size().rows() + 1) * TypeUtilities<T>::error);
}

template <class T, Backend B, Device D>
void testTriangularSolver(comm::CommunicatorGrid grid, blas::Side side, blas::Uplo uplo, blas::Op op,
                          blas::Diag diag, T alpha, SizeType m, SizeType n, SizeType mb, SizeType nb) {
  std::function<T(const GlobalElementIndex&)> el_op_a, el_b, res_b;

  LocalElementSize size_a(m, m);
  TileElementSize block_size_a(mb, mb);
  if (side == blas::Side::Right) {
    size_a = {n, n};
    block_size_a = {nb, nb};
  }

  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));
  GlobalElementSize sz_a = globalTestSize(size_a);
  Distribution distr_a(sz_a, block_size_a, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_ah(std::move(distr_a));

  LocalElementSize size_b(m, n);
  TileElementSize block_size_b(mb, nb);
  GlobalElementSize sz_b = globalTestSize(size_b);
  Distribution distr_b(sz_b, block_size_b, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_bh(std::move(distr_b));

  if (side == blas::Side::Left)
    std::tie(el_op_a, el_b, res_b) =
        getLeftTriangularSystem<GlobalElementIndex, T>(uplo, op, diag, alpha, m);
  else
    std::tie(el_op_a, el_b, res_b) =
        getRightTriangularSystem<GlobalElementIndex, T>(uplo, op, diag, alpha, n);

  set(mat_ah, el_op_a, op);
  set(mat_bh, el_b);

  {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<T, D, Device::CPU> mat_b(mat_bh);

    solver::triangular<B, D, T>(grid, side, uplo, op, diag, alpha, mat_a.get(), mat_b.get());
  }

  CHECK_MATRIX_NEAR(res_b, mat_bh, 20 * (mat_bh.size().rows() + 1) * TypeUtilities<T>::error,
                    20 * (mat_bh.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(TriangularSolverTestMC, CorrectnessLocal) {
  SizeType m, n, mb, nb;

  for (auto side : blas_sides) {
    for (auto uplo : blas_uplos) {
      for (auto op : blas_ops) {
        for (auto diag : blas_diags) {
          for (auto sz : sizes) {
            std::tie(m, n, mb, nb) = sz;
            TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);

            testTriangularSolver<TypeParam, Backend::MC, Device::CPU>(side, uplo, op, diag, alpha, m, n,
                                                                      mb, nb);
          }
        }
      }
    }
  }
}

TYPED_TEST(TriangularSolverTestMC, CorrectnessDistributed) {
  SizeType m, n, mb, nb;

  for (const auto& comm_grid : this->commGrids()) {
    for (auto side : blas_sides) {
      for (auto uplo : blas_uplos) {
        for (auto op : blas_ops) {
          for (auto diag : blas_diags) {
            if (!(op == blas::Op::NoTrans || (side == blas::Side::Left && uplo == blas::Uplo::Lower)))
              continue;

            for (auto sz : sizes) {
              std::tie(m, n, mb, nb) = sz;
              TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);
              testTriangularSolver<TypeParam, Backend::MC, Device::CPU>(comm_grid, side, uplo, op, diag,
                                                                        alpha, m, n, mb, nb);
              hpx::threads::get_thread_manager().wait();
            }
          }
        }
      }
    }
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(TriangularSolverTestGPU, CorrectnessLocal) {
  SizeType m, n, mb, nb;

  for (auto side : blas_sides) {
    for (auto uplo : blas_uplos) {
      for (auto op : blas_ops) {
        for (auto diag : blas_diags) {
          for (auto sz : sizes) {
            std::tie(m, n, mb, nb) = sz;
            TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);

            testTriangularSolver<TypeParam, Backend::GPU, Device::GPU>(side, uplo, op, diag, alpha, m, n,
                                                                       mb, nb);
          }
        }
      }
    }
  }
}

TYPED_TEST(TriangularSolverTestGPU, CorrectnessDistributed) {
  SizeType m, n, mb, nb;

  for (const auto& comm_grid : this->commGrids()) {
    for (auto side : blas_sides) {
      for (auto uplo : blas_uplos) {
        for (auto op : blas_ops) {
          for (auto diag : blas_diags) {
            if (!(op == blas::Op::NoTrans || (side == blas::Side::Left && uplo == blas::Uplo::Lower)))
              continue;

            for (auto sz : sizes) {
              std::tie(m, n, mb, nb) = sz;
              TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);

              testTriangularSolver<TypeParam, Backend::GPU, Device::GPU>(comm_grid, side, uplo, op, diag,
                                                                         alpha, m, n, mb, nb);
              hpx::threads::get_thread_manager().wait();
            }
          }
        }
      }
    }
  }
}
#endif
