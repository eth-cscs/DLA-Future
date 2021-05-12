//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/blas/enum_output.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

template <class T, bool return_info>
void testPotrf(const blas::Uplo uplo, const SizeType n, const SizeType extra_lda) {
  const TileElementSize size_a = TileElementSize(n, n);
  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;

  std::stringstream s;
  s << "POTRF: " << uplo;
  s << ", n = " << n << ", lda = " << lda;
  SCOPED_TRACE(s.str());

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

    const double i = index.row();
    const double j = index.col();

    return TypeUtilities<T>::polar(std::exp2(-(i + j)) / 3 * (std::exp2(2 * (std::min(i, j) + 1)) - 1),
                                   -i + j);
  };

  auto res_a = [uplo](const TileElementIndex& index) {
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    const double i = index.row();
    const double j = index.col();

    return TypeUtilities<T>::polar(std::exp2(-std::abs(i - j)), -i + j);
  };

  auto a = createTile<T>(el_a, size_a, lda);

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
void testPotrfNonPosDef(const blas::Uplo uplo, SizeType n, SizeType extra_lda) {
  const TileElementSize size_a = TileElementSize(n, n);
  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;

  std::stringstream s;
  s << "POTRF Non Positive Definite: " << uplo;
  s << ", n = " << n << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  // Use null matrix
  auto el_a = [](const TileElementIndex&) { return TypeUtilities<T>::element(0, 0); };

  auto a = createTile<T>(el_a, size_a, lda);

  if (return_info) {
    EXPECT_EQ(1, tile::potrfInfo(uplo, a));
  }
}

}
}
