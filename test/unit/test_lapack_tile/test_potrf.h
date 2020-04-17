//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/tile.h"
#include "dlaf/util_blas.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

using dlaf::util::size_t::mul;

template <class T, bool return_info>
void testPotrf(blas::Uplo uplo, SizeType n, SizeType extra_lda) {
  TileElementSize size_a = TileElementSize(n, n);

  SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;

  std::stringstream s;
  s << "POTRF: " << uplo;
  s << ", n = " << n << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));

  // Create tiles.
  Tile<T, Device::CPU> a(size_a, std::move(mem_a), lda);

  // Note: The tile elements are chosen such that:
  // - res_ij = 1 / 2^(|i-j|) * exp(I*(-i+j)),
  // where I = 0 for real types or I is the complex unit for complex types.
  // Therefore the result should be:
  // a_ij = Sum_k(res_ik * ConjTrans(res)_kj) =
  //      = Sum_k(1 / 2^(|i-k| + |j-k|) * exp(I*(-i+j))),
  // where k = 0 .. min(i,j)
  // Therefore,
  // a_ij = (4^(min(i,j)+1) - 1) / (3 * 2^(i+j)) * exp(I*(-i+j))
  auto el_a = [uplo](const TileElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    double i = index.row();
    double j = index.col();

    return TypeUtilities<T>::polar(std::exp2(-(i + j)) / 3 * (std::exp2(2 * (std::min(i, j) + 1)) - 1),
                                   -i + j);
  };

  auto res_a = [uplo](const TileElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    double i = index.row();
    double j = index.col();

    return TypeUtilities<T>::polar(std::exp2(-std::abs(i - j)), -i + j);
  };

  // Set tile elements.
  set(a, el_a);

  if (return_info) {
    EXPECT_EQ(0, tile::potrfInfo(uplo, a));
  }
  else {
    tile::potrf(uplo, a);
  }

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_a, a, 4 * (n + 1) * TypeUtilities<T>::error,
                  4 * (n + 1) * TypeUtilities<T>::error);
}

template <class T, bool return_info>
void testPotrfArgExceptions(blas::Uplo uplo, TileElementSize size_a, SizeType extra_lda) {
  SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;

  std::stringstream s;
  s << "POTRF Arguments Exceptions: " << uplo;
  s << ", size_a = " << size_a << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));

  // Create tiles.
  Tile<T, Device::CPU> a(size_a, std::move(mem_a), lda);

  if (return_info) {
    EXPECT_THROW(tile::potrfInfo(uplo, a), std::invalid_argument);
  }
  else {
    EXPECT_THROW(tile::potrf(uplo, a), std::invalid_argument);
  }
}

template <class T, bool return_info>
void testPotrfNonPosDef(blas::Uplo uplo, SizeType n, SizeType extra_lda) {
  TileElementSize size_a = TileElementSize(n, n);

  SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;

  std::stringstream s;
  s << "POTRF Non Positive Definite: " << uplo;
  s << ", n = " << n << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));

  // Create tiles.
  Tile<T, Device::CPU> a(size_a, std::move(mem_a), lda);

  // Use null matrix
  auto el_a = [](const TileElementIndex&) { return TypeUtilities<T>::element(0, 0); };

  // Set tile elements.
  set(a, el_a);

  if (return_info) {
    EXPECT_EQ(1, tile::potrfInfo(uplo, a));
  }
  else {
    EXPECT_THROW(tile::potrf(uplo, a), std::runtime_error);
  }
}
