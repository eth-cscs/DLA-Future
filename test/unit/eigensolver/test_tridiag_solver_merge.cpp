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
  std::vector<SizeType> expected_out_arr{5, 1, 2, 9, 0, 3, 8, 4, 6, 7};
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

  // f, u, d, d, l, u, l, f, d, l
  std::vector<SizeType> in_arr{1, 4, 2, 3, 0, 5, 6, 7, 8, 9};
  dlaf::matrix::util::set(in, [&in_arr](GlobalElementIndex i) { return in_arr[to_sizet(i.row())]; });

  SizeType i_begin = 0;
  SizeType i_end = 3;
  pika::future<SizeType> k_fut = stablePartitionIndexForDeflation(i_begin, i_end, c, in, out);

  ASSERT_TRUE(k_fut.get() == 7);

  // f u l u l f l d d d
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

// import numpy as np
// from scipy.linalg import eigh, norm
//
// n = 10
// d = np.log(4.2 + np.arange(n))
// z = 0.42 + np.sin(np.arange(n))
// z = z / norm(z)
// rho = 1.5
// defl = np.diag(d) + rho * np.outer(z, np.transpose(z))
// eigh(defl)
//
TYPED_TEST(TridiagEigensolverMergeTest, SolveRank1Problem) {
  using T = TypeParam;

  SizeType n = 10;
  SizeType nb = 3;

  SizeType i_begin = 0;  // first tile
  SizeType i_end = 3;    // last tile
  pika::shared_future<SizeType> k_fut = pika::make_ready_future<SizeType>(n);
  pika::shared_future<T> rho_fut = pika::make_ready_future<T>(1.5);
  Matrix<T, Device::CPU> d_defl(LocalElementSize(n, 1), TileElementSize(nb, 1));
  Matrix<T, Device::CPU> z_defl(LocalElementSize(n, 1), TileElementSize(nb, 1));

  dlaf::matrix::util::set(d_defl, [](GlobalElementIndex i) { return std::log(4.2 + i.row()); });
  constexpr T z_norm = 2.84813054;
  dlaf::matrix::util::set(z_defl,
                          [](GlobalElementIndex i) { return (0.42 + std::sin(i.row())) / z_norm; });

  Matrix<T, Device::CPU> evals(LocalElementSize(n, 1), TileElementSize(nb, 1));
  Matrix<T, Device::CPU> evecs(LocalElementSize(n, n), TileElementSize(nb, nb));
  Matrix<T, Device::CPU> ws(LocalElementSize(n, n), TileElementSize(nb, nb));

  dlaf::eigensolver::internal::solveRank1Problem(i_begin, i_end, k_fut, rho_fut, d_defl, z_defl, evals,
                                                 evecs);
  dlaf::eigensolver::internal::formEvecs(i_begin, i_end, k_fut, d_defl, z_defl, ws, evecs);

  std::vector<T> expected_evals{1.44288664, 1.70781225, 1.93425131, 2.03886453, 2.11974489,
                                2.24176809, 2.32330826, 2.44580313, 2.56317737, 3.70804892};
  auto expected_evals_fn = [&expected_evals](GlobalElementIndex i) {
    return expected_evals[to_sizet(i.row())];
  };
  CHECK_MATRIX_NEAR(expected_evals_fn, evals, 1e-6, 1e-6);

  // Note: results obtained with numpy have eigenvectors at column indices 0, 1 and 6 negated! That is OK
  //       as eigenvectors are unique up to a sign!
  // clang-format off
  std::vector<T> expected_evecs{
    -0.99062157, -0.06289579, -0.04282673, -0.05184314, -0.02573696, -0.02007637, -0.0030797 , -0.00930346, -0.00683292, -0.09477875,
     0.1128137 , -0.87095855, -0.22482338, -0.24093737, -0.11234659, -0.08201277, -0.01217812, -0.03542954, -0.02531549, -0.31419042,
     0.0640934 ,  0.4650665 , -0.61676376, -0.46226377, -0.18892758, -0.12285628, -0.01735852, -0.04790472, -0.0330292 , -0.36200199,
     0.01943896,  0.08606708,  0.71706706, -0.64552354, -0.16161698, -0.08082905, -0.01046475, -0.02663145, -0.0174812 , -0.16598507,
    -0.00937307, -0.03470794, -0.10091066, -0.38457841,  0.90517984,  0.0943602 ,  0.01000846,  0.02206963,  0.01346553,  0.1077081,
    -0.01277496, -0.04304038, -0.0962647 , -0.22271985, -0.22733617,  0.9209565 ,  0.03371619,  0.05324683,  0.02875438,  0.18566604,
     0.00294152,  0.00934252,  0.0184359 ,  0.03695469,  0.02910645,  0.06724124, -0.99466407, -0.02550312, -0.01071523, -0.05203969,
     0.02036836,  0.06211784,  0.11380957,  0.21287906,  0.15256466,  0.23847159,  0.07574514, -0.806713  , -0.13421982, -0.42752026,
     0.02450085,  0.07252844,  0.12647603,  0.22707187,  0.15491465,  0.20928701,  0.05153133,  0.56717303, -0.41893489, -0.59911163,
     0.01346385,  0.0389555 ,  0.06556727,  0.11455813,  0.07581674,  0.09480502,  0.02109541,  0.13860117,  0.89625418, -0.37843828
  };
  // clang-format on
  auto expected_evecs_fn = [&expected_evecs, n](GlobalElementIndex i) {
    return expected_evecs[to_sizet(i.col() + n * i.row())];
  };
  CHECK_MATRIX_NEAR(expected_evecs_fn, evecs, 1e-6, 1e-6);
}
