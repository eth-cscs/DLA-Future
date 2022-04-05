//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/merge.h"
#include "dlaf/matrix/print_csv.h"
#include "dlaf/util_matrix.h"

#include "gtest/gtest.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::eigensolver::internal;

template <typename Type>
class TridiagEigensolverMergeTest : public ::testing::Test {};
TYPED_TEST_SUITE(TridiagEigensolverMergeTest, RealMatrixElementTypes);

TYPED_TEST(TridiagEigensolverMergeTest, ApplyIndex) {
  SizeType n = 10;
  SizeType nb = 3;

  LocalElementSize sz(n, 1);
  TileElementSize bk(nb, 1);

  Matrix<SizeType, Device::CPU> index(sz, bk);
  Matrix<TypeParam, Device::CPU> in(sz, bk);
  Matrix<TypeParam, Device::CPU> out(sz, bk);
  // reverse order: n-1, n-2, ... ,0
  dlaf::matrix::util::set(index, [n](GlobalElementIndex i) { return n - i.row() - 1; });
  // n, n+1, n+2, ..., 2*n - 1
  dlaf::matrix::util::set(in, [n](GlobalElementIndex i) { return TypeParam(n + i.row()); });

  applyIndex(0, 3, index, in, out);

  // 2*n - 1, 2*n - 2, ..., n
  auto expected_out = [n](GlobalElementIndex i) { return TypeParam(2 * n - 1 - i.row()); };
  CHECK_MATRIX_EQ(expected_out, out);
}

TEST(CopyVector, Index) {
  SizeType n = 10;
  SizeType nb = 3;

  LocalElementSize sz(n, 1);
  TileElementSize bk(nb, 1);

  Matrix<SizeType, Device::CPU> in(sz, bk);
  Matrix<SizeType, Device::CPU> out(sz, bk);
  // reverse order: n-1, n-2, ... ,0
  dlaf::matrix::util::set(in, [](GlobalElementIndex i) { return i.row(); });

  copyVector(0, 3, in, out);

  auto expected_out = [](GlobalElementIndex i) { return i.row(); };
  CHECK_MATRIX_EQ(expected_out, out);
}

TYPED_TEST(TridiagEigensolverMergeTest, SortIndex) {
  SizeType n = 10;
  SizeType nb = 3;
  SizeType split = 4;

  LocalElementSize sz(n, 1);
  TileElementSize bk(nb, 1);

  Matrix<TypeParam, Device::CPU> vec(sz, bk);
  Matrix<SizeType, Device::CPU> in(sz, bk);
  Matrix<SizeType, Device::CPU> out(sz, bk);

  std::vector<SizeType> vec_arr{7, 2, 4, 8, 12, 1, 17, 32, 9, 6};
  DLAF_ASSERT(vec_arr.size() == to_sizet(n), n);
  dlaf::matrix::util::set(vec, [&vec_arr](GlobalElementIndex i) { return vec_arr[to_sizet(i.row())]; });

  // `in` orders `vec` in two sorted ranges : [2, 4, 7, 8] and [1, 6, 9, 12, 17, 32] (note split = 4)
  std::vector<SizeType> in_arr{1, 2, 0, 3, 5, 9, 8, 4, 6, 7};
  DLAF_ASSERT(in_arr.size() == to_sizet(n), n);
  dlaf::matrix::util::set(in, [&in_arr](GlobalElementIndex i) { return in_arr[to_sizet(i.row())]; });

  // Sort `vec` in ascending order
  sortIndex(0, 3, pika::make_ready_future(split), vec, in, out);

  // Merges the two sorted ranges in `vec` to get the indices of the sorted array [1, 2, 4, 6, 7, 8, 9, 12, 17, 32]
  std::vector<SizeType> expected_out_arr{1, 2, 0, 3, 5, 9, 8, 4, 6, 7};
  auto expected_out = [&expected_out_arr](GlobalElementIndex i) {
    return expected_out_arr[to_sizet(i.row())];
  };
  CHECK_MATRIX_EQ(expected_out, out);
}

TEST(StablePartitionIndexOnDeflated, FullRange) {
  SizeType n = 10;
  SizeType nb = 3;

  LocalElementSize sz(n, 1);
  TileElementSize bk(nb, 1);

  Matrix<ColType, Device::CPU> c(sz, bk);
  Matrix<SizeType, Device::CPU> in(sz, bk);
  Matrix<SizeType, Device::CPU> out(sz, bk);

  std::vector<ColType> c_arr{ColType::LowerHalf, ColType::Dense,     ColType::Deflated,
                             ColType::Deflated,  ColType::UpperHalf, ColType::UpperHalf,
                             ColType::LowerHalf, ColType::Dense,     ColType::Deflated,
                             ColType::LowerHalf};
  DLAF_ASSERT(c_arr.size() == to_sizet(n), n);
  dlaf::matrix::util::set(c, [&c_arr](GlobalElementIndex i) { return c_arr[to_sizet(i.row())]; });

  std::vector<SizeType> in_arr{1, 4, 2, 3, 0, 5, 6, 7, 8, 9};
  dlaf::matrix::util::set(in, [&in_arr](GlobalElementIndex i) { return in_arr[to_sizet(i.row())]; });

  SizeType i_begin = 0;
  SizeType i_end = 3;
  pika::future<SizeType> k_fut = stablePartitionIndexForDeflation(i_begin, i_end, c, in, out);

  ASSERT_TRUE(k_fut.get() == 7);

  std::vector<SizeType> expected_out_arr{1, 4, 0, 5, 6, 7, 9, 2, 3, 8};
  auto expected_out = [&expected_out_arr](GlobalElementIndex i) {
    return expected_out_arr[to_sizet(i.row())];
  };
  CHECK_MATRIX_EQ(expected_out, out);
}

TEST(PartitionIndexBasedOnColTypes, FullRange) {
  SizeType n = 10;
  SizeType nb = 3;

  LocalElementSize sz(n, 1);
  TileElementSize bk(nb, 1);

  Matrix<ColType, Device::CPU> c(sz, bk);
  Matrix<SizeType, Device::CPU> index(sz, bk);

  std::vector<ColType> c_arr{ColType::LowerHalf, ColType::Dense,     ColType::Deflated,
                             ColType::Deflated,  ColType::UpperHalf, ColType::UpperHalf,
                             ColType::LowerHalf, ColType::Dense,     ColType::Deflated,
                             ColType::LowerHalf};
  DLAF_ASSERT(c_arr.size() == to_sizet(n), n);
  dlaf::matrix::util::set(c, [&c_arr](GlobalElementIndex i) { return c_arr[to_sizet(i.row())]; });

  std::vector<SizeType> index_arr{1, 4, 0, 5, 6, 7, 9, 2, 3, 8};
  dlaf::matrix::util::set(index,
                          [&index_arr](GlobalElementIndex i) { return index_arr[to_sizet(i.row())]; });

  SizeType i_begin = 0;
  SizeType i_end = 3;
  auto clens_fut = partitionIndexForMatrixMultiplication(i_begin, i_end, c, index);

  //  non-stable partitioning may change the order of elements, that is why the output index is applied
  //  to the vector and then checked for correctness
  Matrix<ColType, Device::CPU> c_out(sz, bk);
  applyIndex(i_begin, i_end, index, c, c_out);

  std::vector<ColType> expected_out_arr{ColType::UpperHalf, ColType::UpperHalf, ColType::Dense,
                                        ColType::Dense,     ColType::LowerHalf, ColType::LowerHalf,
                                        ColType::LowerHalf, ColType::Deflated,  ColType::Deflated,
                                        ColType::Deflated};

  auto expected_out_fn = [&expected_out_arr](GlobalElementIndex i) {
    return expected_out_arr[to_sizet(i.row())];
  };
  CHECK_MATRIX_EQ(expected_out_fn, c_out);

  ColTypeLens clens = clens_fut.get();
  ASSERT_TRUE(clens.num_deflated == 3);
  ASSERT_TRUE(clens.num_dense == 2);
  ASSERT_TRUE(clens.num_lowhalf == 3);
  ASSERT_TRUE(clens.num_uphalf == 2);
}

TYPED_TEST(TridiagEigensolverMergeTest, Deflation) {
  SizeType n = 10;
  SizeType nb = 3;

  LocalElementSize sz(n, 1);
  TileElementSize bk(nb, 1);

  constexpr ColType deflated = ColType::Deflated;
  constexpr ColType up = ColType::UpperHalf;
  constexpr ColType low = ColType::LowerHalf;
  constexpr ColType dense = ColType::Dense;

  Matrix<SizeType, Device::CPU> index_mat(sz, bk);
  Matrix<TypeParam, Device::CPU> d_mat(sz, bk);
  Matrix<TypeParam, Device::CPU> z_mat(sz, bk);
  Matrix<ColType, Device::CPU> c_mat(sz, bk);

  // the index array that sorts `d`
  std::vector<SizeType> index_arr{0, 6, 7, 1, 9, 5, 2, 4, 3, 8};
  dlaf::matrix::util::set(index_mat,
                          [&index_arr](GlobalElementIndex i) { return index_arr[to_sizet(i.row())]; });

  // 11 11 11 13 13 17 18 18 34 34
  std::vector<TypeParam> d_arr{11, 13, 18, 34, 18, 17, 11, 11, 34, 13};
  dlaf::matrix::util::set(d_mat, [&d_arr](GlobalElementIndex i) { return d_arr[to_sizet(i.row())]; });

  // 12 72 102 31 9 0 16 0 0 0
  std::vector<TypeParam> z_arr{12, 31, 16, 0, 0, 0, 72, 102, 0, 9};
  dlaf::matrix::util::set(z_mat, [&z_arr](GlobalElementIndex i) { return z_arr[to_sizet(i.row())]; });

  // u l l u l l u u u l
  std::vector<ColType> c_arr{up, up, up, up, up, low, low, low, low, low};
  dlaf::matrix::util::set(c_mat, [&c_arr](GlobalElementIndex i) { return c_arr[to_sizet(i.row())]; });

  TypeParam tol = 0.01;
  TypeParam rho = 1;
  SizeType i_begin = 0;
  SizeType i_end = 3;
  auto rots_fut =
      applyDeflation<TypeParam>(i_begin, i_end, pika::make_ready_future(rho),
                                pika::make_ready_future(tol), index_mat, d_mat, z_mat, c_mat);

  Matrix<TypeParam, Device::CPU> d_mat_sorted(sz, bk);
  Matrix<TypeParam, Device::CPU> z_mat_sorted(sz, bk);
  Matrix<ColType, Device::CPU> c_mat_sorted(sz, bk);
  applyIndex(i_begin, i_end, index_mat, d_mat, d_mat_sorted);
  applyIndex(i_begin, i_end, index_mat, z_mat, z_mat_sorted);
  applyIndex(i_begin, i_end, index_mat, c_mat, c_mat_sorted);

  // Check sorted `d`
  std::vector<TypeParam> expected_d_arr{11, 11, 11, 13, 13, 17, 18, 18, 34, 34};
  auto expected_d_fn = [&expected_d_arr](GlobalElementIndex i) {
    return expected_d_arr[to_sizet(i.row())];
  };
  CHECK_MATRIX_NEAR(expected_d_fn, d_mat_sorted, tol, tol);

  // Check sorted `z`
  std::vector<TypeParam> expected_z_arr{125.427, 0, 0, 32.28, 0, 0, 16, 0, 0, 0};
  auto expected_z_fn = [&expected_z_arr](GlobalElementIndex i) {
    return expected_z_arr[to_sizet(i.row())];
  };
  CHECK_MATRIX_NEAR(expected_z_fn, z_mat_sorted, tol, tol);

  // Check sorted `c`
  std::vector<ColType> expected_c_arr{dense,    deflated, deflated, dense,    deflated,
                                      deflated, up,       deflated, deflated, deflated};
  auto expected_c_fn = [&expected_c_arr](GlobalElementIndex i) {
    return expected_c_arr[to_sizet(i.row())];
  };
  CHECK_MATRIX_EQ(expected_c_fn, c_mat_sorted);
}
