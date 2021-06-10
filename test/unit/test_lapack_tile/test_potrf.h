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
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf_test/lapack/invoke.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

template <class T, Device D, bool return_info>
void testPotrf(const blas::Uplo uplo, const SizeType n, const SizeType extra_lda) {
  std::function<T(const TileElementIndex&)> el_a, res_a;
  const TileElementSize size_a = TileElementSize(n, n);
  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;

  std::tie(el_a, res_a) = getCholeskySetters<TileElementIndex, T>(uplo);

  auto a = createTile<T, D>(el_a, size_a, lda);

  if (return_info) {
    EXPECT_EQ(0, invokeLapackInfo<D>(tile::potrfInfo_o, uplo, a));
  }
  else {
    invokeLapack<D>(tile::potrf_o, uplo, a);
  }

  std::stringstream s;
  s << "POTRF: " << uplo;
  s << ", n = " << n << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_a, a, 4 * (n + 1) * TypeUtilities<T>::error,
                  4 * (n + 1) * TypeUtilities<T>::error);
}

template <class T, Device D>
void testPotrfNonPosDef(const blas::Uplo uplo, SizeType n, SizeType extra_lda) {
  const TileElementSize size_a = TileElementSize(n, n);
  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;

  // Use null matrix
  auto el_a = [](const TileElementIndex&) { return TypeUtilities<T>::element(0, 0); };

  auto a = createTile<T, D>(el_a, size_a, lda);

  auto info = invokeLapackInfo<D>(tile::potrfInfo_o, uplo, a);

  std::stringstream s;
  s << "POTRF Non Positive Definite: " << uplo;
  s << ", n = " << n << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  EXPECT_EQ(1, info);
}

}
}
