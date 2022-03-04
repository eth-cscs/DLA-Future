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
#include "dlaf_test/lapack/invoke.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

template <class T, Device D>
void testHegst(const int itype, const blas::Uplo uplo, const SizeType m, const SizeType extra_lda,
               const SizeType extra_ldb) {
  const TileElementSize size(m, m);
  const SizeType lda = std::max<SizeType>(1, size.rows()) + extra_lda;
  const SizeType ldb = std::max<SizeType>(1, size.rows()) + extra_ldb;

  auto [el_t, el_a, res_a] =
      getGenToStdElementSetters<TileElementIndex, T>(m, itype, uplo, 1.2f, 1.5f, 1.1f);

  auto a = createTile<T, D>(el_a, size, lda);
  auto t = createTile<T, D>(el_t, size, ldb);

  invokeLapack<D>(tile::internal::hegst_o, itype, uplo, a, t);

  std::stringstream s;
  s << "HEGST: itype = " << itype << ", uplo = " << uplo;
  s << ", m = " << m << ", lda = " << lda << ", ldb = " << ldb;
  SCOPED_TRACE(s.str());

  CHECK_TILE_NEAR(res_a, a, 10 * (m + 1) * TypeUtilities<T>::error,
                  10 * (m + 1) * TypeUtilities<T>::error);
}

}
}
