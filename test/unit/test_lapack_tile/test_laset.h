//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <functional>
#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/blas/enum_output.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf_test/blas/invoke.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

template <class T, Device D>
void testLaset(blas::Uplo uplo, const SizeType m, const SizeType n, T alpha, T beta,
               SizeType extra_lda) {
  const SizeType lda = std::max<SizeType>(1, m) + extra_lda;

  auto el = [](const TileElementIndex& idx) {
    return TypeUtilities<T>::element(idx.row() + idx.col(), idx.row() - idx.col());
  };
  auto res = [uplo, alpha, beta, el](const TileElementIndex& idx) {
    const double i = idx.row();
    const double j = idx.col();
    if (i == j)
      return beta;
    else if (uplo == blas::Uplo::General || (uplo == blas::Uplo::Lower && i > j) ||
             (uplo == blas::Uplo::Upper && i < j))
      return alpha;
    return el(idx);
  };

  auto tile = createTile<T, D>(el, TileElementSize(m, n), lda);

  invoke<D>(tile::internal::laset_o, uplo, alpha, beta, tile);

  std::stringstream s;
  s << "LASET: uplo = " << uplo << ", m = " << m << ", n = " << n;
  s << ", alpha = " << alpha << ", beta = " << beta << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  CHECK_TILE_EQ(res, tile);
}

template <class T, Device D>
void testSet0(const SizeType m, const SizeType n, SizeType extra_lda) {
  const SizeType lda = std::max<SizeType>(1, m) + extra_lda;

  auto el = [](const TileElementIndex& idx) {
    return TypeUtilities<T>::element(idx.row() + idx.col(), idx.row() - idx.col());
  };
  auto res = [](const TileElementIndex&) { return TypeUtilities<T>::element(0.0, 0.0); };

  auto tile = createTile<T, D>(el, TileElementSize(m, n), lda);

  invoke<D>(tile::internal::set0_o, tile);

  std::stringstream s;
  s << "SET0: m = " << m << ", n = " << n << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  CHECK_TILE_EQ(res, tile);
}
}
}
