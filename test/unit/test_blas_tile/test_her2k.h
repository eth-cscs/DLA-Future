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

#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/blas_tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/tile.h"
#include "dlaf/util_blas.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/matrix/util_tile_blas.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

using dlaf::util::size_t::mul;

template <class T, class CT = const T>
void testHer2k(const blas::Uplo uplo, const blas::Op op, const SizeType n, const SizeType k,
               const SizeType extra_lda, const SizeType extra_ldc) {
  const TileElementSize size_a =
      (op == blas::Op::NoTrans) ? TileElementSize(n, k) : TileElementSize(k, n);
  const TileElementSize size_b =
      (op == blas::Op::NoTrans) ? TileElementSize(n, k) : TileElementSize(k, n);
  const TileElementSize size_c(n, n);

  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  const SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_lda;
  const SizeType ldc = std::max<SizeType>(1, size_c.rows()) + extra_ldc;

  std::stringstream s;
  s << "HER2K: " << uplo << ", " << op;
  s << ", n = " << n << ", k = " << k;
  s << ", lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));
  memory::MemoryView<T, Device::CPU> mem_b(mul(ldb, size_b.cols()));
  memory::MemoryView<T, Device::CPU> mem_c(mul(ldc, size_c.cols()));

  Tile<T, Device::CPU> a0(size_a, std::move(mem_a), lda);
  Tile<T, Device::CPU> b0(size_b, std::move(mem_b), lda);
  Tile<T, Device::CPU> c(size_c, std::move(mem_c), ldc);

  // Returns op(a)_ik
  auto el_op_a = [](const TileElementIndex& index) {
    const double i = index.row();
    const double k = index.col();
    return TypeUtilities<T>::polar(.9 * (i + 1) / (k + .5), i - k);
  };
  // Returns op(b)_kj
  auto el_op_b = [](const TileElementIndex& index) {
    const double k = index.row();
    const double j = index.col();
    return TypeUtilities<T>::polar(.7 * (k + .5) / (j + 1), k - j);
  };
  auto el_c = [uplo](const TileElementIndex& index) {
    // Return -1 for elements not referenced
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-1, 0);

    const double i = index.row();
    const double j = index.col();
    return TypeUtilities<T>::polar(1.2 * i / (j + 1), -i + j);
  };

  const T alpha = TypeUtilities<T>::element(-1.2, 0.2);
  const BaseType<T> beta = 1.1f;

  auto res_c = [uplo, k, alpha, el_op_a, el_op_b, beta, el_c](const TileElementIndex& index) {
    // Return el_c(index) for elements not referenced
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return el_c(index);

    T tmp = TypeUtilities<T>::element(0, 0);
    // Compute result of cij
    for (SizeType kk = 0; kk < k; ++kk) {
      tmp += alpha * el_op_a({index.row(), kk}) * TypeUtilities<T>::conj(el_op_b({index.col(), kk})) +
             TypeUtilities<T>::conj(alpha) * el_op_b({index.row(), kk}) *
                 TypeUtilities<T>::conj(el_op_a({index.col(), kk}));
    }
    return beta * el_c(index) + tmp;
  };

  set(a0, el_op_a, op);
  set(b0, el_op_b, op);
  set(c, el_c);

  Tile<CT, Device::CPU> a(std::move(a0));
  Tile<CT, Device::CPU> b(std::move(b0));

  tile::her2k(uplo, op, alpha, a, b, beta, c);

  CHECK_TILE_NEAR(res_c, c, (k + 1) * TypeUtilities<T>::error, (k + 1) * TypeUtilities<T>::error);
}
