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
#include "dlaf/blas/tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/matrix/util_tile_blas.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

template <class T, class CT = const T>
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

  std::stringstream s;
  s << "GEMM: " << op_a << ", " << op_b;
  s << ", m = " << m << ", n = " << n << ", k = " << k;
  s << ", lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  // Note: The tile elements are chosen such that:
  // - op_a(a)_ik = .9 * (i+1) / (k+.5) * exp(I*(2*i-k)),
  // - op_b(b)_kj = .8 * (k+.5) / (j+2) * exp(I*(k+j)),
  // - c_ij = 1.2 * i / (j+1) * exp(I*(-i+j)),
  // where I = 0 for real types or I is the complex unit for complex types.
  // Therefore the result should be:
  // res_ij = beta * c_ij + Sum_k(alpha * op_a(a)_ik * op_b(b)_kj)
  //        = beta * c_ij + gamma * (i+1) / (j+2) * exp(I*(2*i+j)),
  // where gamma = .72 * k * alpha.
  auto el_op_a = [](const TileElementIndex& index) {
    const double i = index.row();
    const double k = index.col();
    return TypeUtilities<T>::polar(.9 * (i + 1) / (k + .5), 2 * i - k);
  };
  auto el_op_b = [](const TileElementIndex& index) {
    const double k = index.row();
    const double j = index.col();
    return TypeUtilities<T>::polar(.8 * (k + .5) / (j + 2), k + j);
  };
  auto el_c = [](const TileElementIndex& index) {
    const double i = index.row();
    const double j = index.col();
    return TypeUtilities<T>::polar(1.2 * i / (j + 1), -i + j);
  };

  const T alpha = TypeUtilities<T>::element(-1.2, .7);
  const T beta = TypeUtilities<T>::element(1.1, .4);

  const T gamma = TypeUtilities<T>::element(.72 * k, 0) * alpha;
  auto res_c = [beta, el_c, gamma](const TileElementIndex& index) {
    const double i = index.row();
    const double j = index.col();
    return beta * el_c(index) + gamma * TypeUtilities<T>::polar((i + 1) / (j + 2), 2 * i + j);
  };

  auto a = createTile<CT>(el_op_a, size_a, lda, op_a);
  auto b = createTile<CT>(el_op_b, size_b, ldb, op_b);
  auto c = createTile<T>(el_c, size_c, ldc);

  tile::gemm(op_a, op_b, alpha, a, b, beta, c);

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_c, c, 2 * (k + 1) * TypeUtilities<T>::error,
                  2 * (k + 1) * TypeUtilities<T>::error);
}

}
}
