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
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/blas/enum_output.h"
#include "dlaf/blas/tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf_test/blas/invoke.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/matrix/util_tile_blas.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

template <Device D, class T, class CT = const T>
void testTrmm(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
              const SizeType m, const SizeType n, const SizeType extra_lda, const SizeType extra_ldb) {
  const TileElementSize size_a =
      side == blas::Side::Left ? TileElementSize(m, m) : TileElementSize(n, n);
  const TileElementSize size_b(m, n);

  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  const SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;

  const T alpha = TypeUtilities<T>::element(-1.2, .7);

  auto [el_op_a, res_b, el_b] =
      getTriangularSystem<TileElementIndex, T>(side, uplo, op, diag, static_cast<T>(1.0) / alpha, m, n);

  auto a = createTile<CT, D>(el_op_a, size_a, lda, op);
  auto b = createTile<T, D>(el_b, size_b, ldb);

  invokeBlas<D>(tile::internal::trmm_o, side, uplo, op, diag, alpha, a, b);

  std::stringstream s;
  s << "TRMM: " << side << ", " << uplo << ", " << op << ", " << diag << ", m = " << m << ", n = " << n
    << ", lda = " << lda << ", ldb = " << ldb;
  SCOPED_TRACE(s.str());

  CHECK_TILE_NEAR(res_b, b, 10 * (m + 1) * TypeUtilities<T>::error,
                  10 * (m + 1) * TypeUtilities<T>::error);
}

template <Device D, class T, class CT = const T>
void testTrmm3(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
               const SizeType m, const SizeType n, const SizeType extra_lda, const SizeType extra_ldb,
               const SizeType extra_ldc) {
  const TileElementSize size_a =
      side == blas::Side::Left ? TileElementSize(m, m) : TileElementSize(n, n);
  const TileElementSize size_b(m, n);

  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  const SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;
  const SizeType ldc = std::max<SizeType>(1, size_b.rows()) + extra_ldc;

  const T alpha = TypeUtilities<T>::element(-1.2, .7);

  auto [el_op_a, res_c, el_b] =
      getTriangularSystem<TileElementIndex, T>(side, uplo, op, diag, static_cast<T>(1.0) / alpha, m, n);

  auto a = createTile<CT, D>(el_op_a, size_a, lda, op);
  auto b = createTile<CT, D>(el_b, size_b, ldb);

  auto el_c = [](auto) { return TypeUtilities<T>::element(-999.2, 888.7); };
  auto c = createTile<T, D>(el_c, size_b, ldc);

  invokeBlas<D>(tile::internal::trmm3_o, side, uplo, op, diag, alpha, a, b, c);

  std::stringstream s;
  s << "TRMM3: " << side << ", " << uplo << ", " << op << ", " << diag << ", m = " << m << ", n = " << n
    << ", lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  CHECK_TILE_NEAR(res_c, c, 10 * (m + 1) * TypeUtilities<T>::error,
                  10 * (m + 1) * TypeUtilities<T>::error);
}

}
}
