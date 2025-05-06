//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <sstream>

#include <dlaf/blas/enum_output.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/tile.h>

#include <gtest/gtest.h>

#include <dlaf_test/blas/invoke.h>
#include <dlaf_test/matrix/util_generic_lapack.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/util_types.h>

namespace dlaf::test {
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

template <class T, Device D>
void test_trtri_workspace(const blas::Uplo uplo, const blas::Diag diag, const SizeType n,
                          const SizeType extra_lda) {
  const TileElementSize size_a = TileElementSize(n, n);
  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;

  auto [el_a, res_a] = get_triangular_inverse_setters<TileElementIndex, T>(uplo, diag);

  auto a = createTile<T, D>(el_a, size_a, lda);
  auto ws = createTile<T, D>(size_a, lda);

  invokeBlas<D>(tile::internal::trtri_workspace_o, uplo, diag, a, ws);

  std::stringstream s;
  s << "TRTRI: " << uplo << " " << diag;
  s << ", n = " << n << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_a, a, 4 * (n + 1) * TypeUtilities<T>::error,
                  4 * (n + 1) * TypeUtilities<T>::error);
}

}
