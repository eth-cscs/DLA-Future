//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <exception>
#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/blas_tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/tile.h"
#include "dlaf/util_blas.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

using dlaf::util::size_t::mul;

template <class ElementIndex, class T>
void testHegst(int itype, blas::Uplo uplo, SizeType m, SizeType extra_ld) {
  const TileElementSize size(m, m);

  const SizeType ld = std::max<SizeType>(1, size.rows()) + extra_ld;

  std::stringstream s;
  s << "HEGST: itype = " << itype << ", uplo = " << uplo << ", m = " << m << ", ld = " << ld;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(ld, size.cols()));
  memory::MemoryView<T, Device::CPU> mem_t(mul(ld, size.cols()));

  Tile<T, Device::CPU> a(size, std::move(mem_a), ld);
  Tile<T, Device::CPU> t(size, std::move(mem_t), ld);

  const BaseType<T> alpha = 1.2f;
  const BaseType<T> beta = 1.5f;
  const BaseType<T> gamma = -1.1f;

  std::function<T(const TileElementIndex&)> el_t, el_a, res_a;

  std::tie(el_t, el_a, res_a) =
      dlaf::matrix::test::getGenToStdElementSetters<ElementIndex, BaseType<T>>(m, itype, uplo, alpha,
                                                                               beta, gamma);

  set(t, el_t);
  set(a, el_a);

  tile::hegst(itype, uplo, a, t);

  CHECK_TILE_NEAR(res_a, a, 10 * (m + 1) * TypeUtilities<T>::error,
                  10 * (m + 1) * TypeUtilities<T>::error);
}
