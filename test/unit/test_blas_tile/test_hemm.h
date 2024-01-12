//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <sstream>

#include <dlaf/blas/enum_output.h>
#include <dlaf/blas/tile.h>
#include <dlaf/matrix/tile.h>

#include <gtest/gtest.h>

#include <dlaf_test/blas/invoke.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/matrix/util_tile_blas.h>
#include <dlaf_test/util_types.h>

namespace dlaf {
namespace test {

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

using dlaf::util::size_t::mul;

template <Device D, class T, class CT = const T>
void testHemm(const blas::Side side, const blas::Uplo uplo, const SizeType m, const SizeType n,
              const SizeType extra_lda, const SizeType extra_ldb, const SizeType extra_ldc) {
  const SizeType k = (side == blas::Side::Left) ? m : n;

  const TileElementSize size_a(k, k);
  const TileElementSize size_b(m, n);
  const TileElementSize size_c(m, n);

  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  const SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;
  const SizeType ldc = std::max<SizeType>(1, size_c.rows()) + extra_ldc;

  const T alpha = TypeUtilities<T>::element(1.2, .7);
  const T beta = TypeUtilities<T>::element(1.1, .4);

  auto [el_a, el_b, el_c, res_c] =
      getHermitianMatrixMultiplication<TileElementIndex, T>(side, uplo, k, alpha, beta);

  // Read-only tiles become constant if CT is const T.
  auto a = createTile<CT, D>(el_a, size_a, lda);
  auto b = createTile<CT, D>(el_b, size_b, ldb);
  auto c = createTile<T, D>(el_c, size_c, ldc);

  invokeBlas<D>(tile::internal::hemm_o, side, uplo, alpha, a, b, beta, c);

  std::stringstream s;
  s << "HEMM: " << side << ", " << uplo;
  s << ", m = " << m << ", n = " << n;
  s << ", lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_c, c, 2 * (k + 1) * TypeUtilities<T>::error,
                  2 * (k + 1) * TypeUtilities<T>::error);
}

}
}
