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

#include <functional>
#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/blas/enum_output.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

using dlaf::util::size_t::mul;

template <class ElementIndex, class T>
void testHegst(const int itype, const blas::Uplo uplo, const SizeType m, const SizeType extra_ld) {
  const TileElementSize size(m, m);

  const SizeType ld = std::max<SizeType>(1, size.rows()) + extra_ld;

  std::stringstream s;
  s << "HEGST: itype = " << itype << ", uplo = " << uplo << ", m = " << m << ", ld = " << ld;
  SCOPED_TRACE(s.str());

  const BaseType<T> alpha = 1.2f;
  const BaseType<T> beta = 1.5f;
  const BaseType<T> gamma = -1.1f;

  std::function<T(const TileElementIndex&)> el_t, el_a, res_a;

  std::tie(el_t, el_a, res_a) =
      getGenToStdElementSetters<ElementIndex, BaseType<T>>(m, itype, uplo, alpha, beta, gamma);

  auto a = createTile<T>(el_a, size, ld);
  auto t = createTile<T>(el_t, size, ld);

  tile::hegst(itype, uplo, a, t);

  CHECK_TILE_NEAR(res_a, a, 10 * (m + 1) * TypeUtilities<T>::error,
                  10 * (m + 1) * TypeUtilities<T>::error);
}

}
}
