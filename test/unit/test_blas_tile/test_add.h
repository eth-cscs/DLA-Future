//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
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
#include <dlaf_test/matrix/util_generic_blas.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/matrix/util_tile_blas.h>
#include <dlaf_test/util_types.h>

namespace dlaf {
namespace test {

// Computes A += alpha * B

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

template <Device D, class T, class CT = const T>
void testGemm(const blas::Op op_a, const blas::Op op_b, const SizeType m, const SizeType n,
              const SizeType k, const SizeType extra_lda, const SizeType extra_ldb,
              const SizeType extra_ldc) {
  const TileElementSize size_a =
      (op_a == blas::Op::NoTrans) ? TileElementSize(m, k) : TileElementSize(k, m);
  const TileElementSize size_b =
      (op_b == blas::Op::NoTrans) ? TileElementSize(k, n) : TileElementSize(n, k);
  const TileElementSize size_c(m, n);

  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  const SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;
  const SizeType ldc = std::max<SizeType>(1, size_c.rows()) + extra_ldc;

  const T alpha = TypeUtilities<T>::element(-1.2, .7);
  const T beta = TypeUtilities<T>::element(1.1, .4);

  auto [el_a, el_b, el_c, res_c] =
      getMatrixMatrixMultiplication<TileElementIndex, T>(op_a, op_b, k, alpha, beta);

  auto a = createTile<CT, D>(el_a, size_a, lda);
  auto b = createTile<CT, D>(el_b, size_b, ldb);
  auto c = createTile<T, D>(el_c, size_c, ldc);

  invokeBlas<D>(tile::internal::gemm_o, op_a, op_b, alpha, a, b, beta, c);

  std::stringstream s;
  s << "GEMM: " << op_a << ", " << op_b;
  s << ", m = " << m << ", n = " << n << ", k = " << k;
  s << ", lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_c, c, 2 * (k + 1) * TypeUtilities<T>::error,
                  2 * (k + 1) * TypeUtilities<T>::error);
}

}
}
