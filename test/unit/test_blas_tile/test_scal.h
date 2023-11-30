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
#include <dlaf_test/matrix/util_generic_blas_extensions.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/matrix/util_tile_blas.h>
#include <dlaf_test/util_types.h>

namespace dlaf {
namespace test {

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

template <Device D, class T, class CT = const T>
void testScal(const blas::Op op_a, const SizeType m,
              const SizeType k, const SizeType extra_lda) {
  const TileElementSize size_a =
      (op_a == blas::Op::NoTrans) ? TileElementSize(m, k) : TileElementSize(k, m);


  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;

  const T beta = TypeUtilities<T>::element(1.1, .4);

  auto [el_a, res_a] =
      getMatrixScal<TileElementIndex, T>(op_a, beta);

  auto a = createTile<CT, D>(el_a, size_a, lda);

  invokeBlas<D>(tile::internal::gemm_o, op_a, beta);

  std::stringstream s;
  s << "GEMM: " << op_a;
  s << ", m = " << m << ", n = " << n << ", k = " << k;
  s << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_c, c, 2 * (k + 1) * TypeUtilities<T>::error,
                  2 * (k + 1) * TypeUtilities<T>::error);
}

}
}
