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

  T alpha = TypeUtilities<T>::element(1.0, 0.0);
  T beta = TypeUtilities<T>::element(1.0, 0.0);
  T gamma = TypeUtilities<T>::element(1.0, 0.0);

  std::function<T(const TileElementIndex&)> el_l, el_a, res_b;

  std::tie(el_l, el_a, res_b) =
      dlaf::matrix::test::getLowerHermitianSystem<ElementIndex, T>(alpha, beta, gamma);

  set(l, el_l);
  set(a, el_a);
  set(b, res_b);

  std::cout << "L" << std::endl;
  printElementTile(l);
  std::cout << "A" << std::endl;
  printElementTile(a);

  tile::hegst(1, blas::Uplo::Lower, a, l);

  std::cout << "B" << std::endl;
  printElementTile(a);
  std::cout << "Bres" << std::endl;
  printElementTile(b);

  CHECK_TILE_NEAR(res_b, a, 10 * (m + 1) * TypeUtilities<T>::error,
                  10 * (m + 1) * TypeUtilities<T>::error);
}

// TO DO rows != col?
// template <class T, class CT = const T>
// void testTrsmExceptions(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag,
//                        const TileElementSize& size_a, const TileElementSize& size_b, SizeType
//                        extra_lda, SizeType extra_ldb) {
//  SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
//  SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;
//
//  std::stringstream s;
//  s << "TRSM: " << side << ", " << uplo << ", " << op << ", " << diag << ", size_a = " << size_a
//    << ", size_b = " << size_b << ", lda = " << lda << ", ldb = " << ldb;
//  SCOPED_TRACE(s.str());
//
//  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));
//  memory::MemoryView<T, Device::CPU> mem_b(mul(ldb, size_b.cols()));
//
//  Tile<CT, Device::CPU> a(size_a, std::move(mem_a), lda);
//  Tile<T, Device::CPU> b(size_b, std::move(mem_b), ldb);
//
//  T alpha = TypeUtilities<T>::element(-1.2, .7);
//
//  EXPECT_THROW(tile::trsm(side, uplo, op, diag, alpha, a, b), std::invalid_argument);
//}
//
