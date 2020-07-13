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
#include "dlaf/tile_output.h"
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
void testLowerHegst(SizeType m, SizeType extra_ld) {
  TileElementSize size(m, m);
  SizeType ld = std::max<SizeType>(1, size.rows()) + extra_ld;

  std::stringstream s;
  s << "HEGST: inv(L) * A * inv(L**H) "
    << ", m = " << m << ", ld = " << ld;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(ld, size.cols()));
  memory::MemoryView<T, Device::CPU> mem_l(mul(ld, size.cols()));

  Tile<T, Device::CPU> a(size, std::move(mem_a), ld);
  Tile<T, Device::CPU> l(size, std::move(mem_l), ld);

  memory::MemoryView<T, Device::CPU> mem_b(mul(ld, size.cols()));
  Tile<T, Device::CPU> b(size, std::move(mem_b), ld);

  BaseType<T> alpha = 1.2f;
  BaseType<T> beta  = 1.5f;
  BaseType<T> gamma = -1.1f;

  std::function<T(const TileElementIndex&)> el_l, el_a, res_b;

  std::tie(el_l, el_a, res_b) =
      dlaf::matrix::test::getLowerHermitianSystem<ElementIndex, BaseType<T>>(alpha, beta, gamma);

  set(l, el_l);
  set(a, el_a);
  set(b, res_b);

  tile::hegst(1, blas::Uplo::Lower, a, l);

  CHECK_TILE_NEAR(res_b, a, 10 * (m + 1) * TypeUtilities<T>::error,
                  10 * (m + 1) * TypeUtilities<T>::error);
}
