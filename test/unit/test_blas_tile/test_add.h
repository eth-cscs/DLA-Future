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
#include <dlaf/blas/tile_extensions.h>
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
void testAdd(const SizeType m, const SizeType n, const SizeType extra_lda, const SizeType extra_ldb) {
  const TileElementSize size(m, n);

  const SizeType lda = std::max<SizeType>(1, size.rows()) + extra_lda;
  const SizeType ldb = std::max<SizeType>(1, size.rows()) + extra_ldb;

  const T alpha = TypeUtilities<T>::element(-1.2, .7);

  auto [el_a, el_b, res_a] = getMatrixAdd<TileElementIndex, T>(alpha);

  auto a = createTile<T, D>(el_a, size, lda);
  auto b = createTile<CT, D>(el_b, size, ldb);

  invoke<D>(tile::internal::add_o, alpha, b, a);

  std::stringstream s;
  s << "Add: ";
  s << ", m = " << m << ", n = " << n;
  s << ", lda = " << lda << ", ldb = " << ldb;
  SCOPED_TRACE(s.str());

  CHECK_TILE_NEAR(res_a, a, 2 * TypeUtilities<T>::error, 2 * TypeUtilities<T>::error);
}

}
}
