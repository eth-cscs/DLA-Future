//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/multiplication/triangular.h"

#include <functional>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/blas/tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

template <typename Type>
class TriangularMultiplicationTestMC : public ::testing::Test {
public:
  //  const std::vector<CommunicatorGrid>& commGrids() {
  //    return comm_grids;
  //  }
};
TYPED_TEST_SUITE(TriangularMultiplicationTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <typename Type>
class TriangularMultiplicationTestGPU : public ::testing::Test {
public:
};
TYPED_TEST_SUITE(TriangularMultiplicationTestGPU, MatrixElementTypes);
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
void testTriangularMultiplication(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag,
                                  T alpha, SizeType m, SizeType n, SizeType mb, SizeType nb) {
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
    std::tie(el_op_a, res_b, el_b) =
        getLeftTriangularSystem<GlobalElementIndex, T>(uplo, op, diag, alpha, m);
  else
    std::tie(el_op_a, res_b, el_b) =
        getRightTriangularSystem<GlobalElementIndex, T>(uplo, op, diag, alpha, n);

  set(mat_ah, el_op_a, op);
  set(mat_bh, el_b);

  {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_ah);
    MatrixMirror<T, D, Device::CPU> mat_b(mat_bh);

    multiplication::triangular<B>(side, uplo, op, diag, alpha, mat_a.get(), mat_b.get());
  }

  Matrix<T, Device::CPU> mat_res({m, n}, {mb, nb});
  set(mat_res, res_b);
  MatrixLocal<T> res({m, n}, {mb, nb});

  GlobalElementSize sz_a(m, m);
  TileElementSize bk_sz_a(mb, mb);
  if (side == blas::Side::Right) {
    sz_a = {n, n};
    bk_sz_a = {nb, nb};
  }
  MatrixLocal<T> mat_loc_a(sz_a, bk_sz_a);
  for (const auto& ij_tile : iterate_range2d(res.nrTiles())) {
    const auto& source_tile = mat_res.read(ij_tile).get();
    copy(source_tile, res.tile(ij_tile));
    if (side == blas::Side::Left) {
      copy(mat_ah.read(GlobalTileIndex{ij_tile.row(), ij_tile.row()}).get(),
           mat_loc_a.tile({ij_tile.row(), ij_tile.row()}));
      tile::gemm(blas::Op::NoTrans, blas::Op::NoTrans, T(0.),
                 mat_loc_a.tile({ij_tile.row(), ij_tile.row()}), res.tile(ij_tile), alpha * alpha,
                 res.tile(ij_tile));
    }
    else {
      copy(mat_ah.read(GlobalTileIndex{ij_tile.col(), ij_tile.col()}).get(),
           mat_loc_a.tile({ij_tile.col(), ij_tile.col()}));
      tile::gemm(blas::Op::NoTrans, blas::Op::NoTrans, T(0.), res.tile(ij_tile),
                 mat_loc_a.tile({ij_tile.col(), ij_tile.col()}), alpha * alpha, res.tile(ij_tile));
    }
  }
  auto result = [& dist = mat_res.distribution(), &mat_local = res](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  CHECK_MATRIX_NEAR(result, mat_bh, 40 * (mat_bh.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_bh.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(TriangularMultiplicationTestMC, CorrectnessLocal) {
  SizeType m, n, mb, nb;

  for (auto side : blas_sides) {
    for (auto uplo : blas_uplos) {
      for (auto op : blas_ops) {
        for (auto diag : blas_diags) {
          for (auto sz : sizes) {
            std::tie(m, n, mb, nb) = sz;
            TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);
            testTriangularMultiplication<TypeParam, Backend::MC, Device::CPU>(side, uplo, op, diag,
                                                                              alpha, m, n, mb, nb);
          }
        }
      }
    }
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(TriangularMultiplicationTestGPU, CorrectnessLocal) {
  SizeType m, n, mb, nb;

  for (auto side : blas_sides) {
    for (auto uplo : blas_uplos) {
      for (auto op : blas_ops) {
        for (auto diag : blas_diags) {
          for (auto sz : sizes) {
            std::tie(m, n, mb, nb) = sz;
            TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);

            testTriangularMultiplication<TypeParam, Backend::GPU, Device::GPU>(side, uplo, op, diag,
                                                                               alpha, m, n, mb, nb);
          }
        }
      }
    }
  }
}
#endif
