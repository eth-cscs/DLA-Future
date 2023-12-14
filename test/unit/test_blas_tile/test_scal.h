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
#include <dlaf/blas/tile_extensions.h>

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
void testScal(const SizeType m,
              const SizeType n, const SizeType extra_lda) {
  const TileElementSize size_a(m, n);

  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;

  const T beta = TypeUtilities<T>::element(1.1, .4);

  auto [el_a, res_a] =
      getMatrixScal<TileElementIndex, T>(beta);

  auto a = createTile<T, D>(el_a, size_a, lda);

  invokeBlas<D>(tile::internal::scal_o, beta, a);

  std::stringstream s;
  s << "Scal: ";
  s << ", m = " << m << ", n = " << n;
  s << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_a, a, 2 * TypeUtilities<T>::error,
                  2 * TypeUtilities<T>::error);
}

}
}
