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
void testHerk(blas::Uplo uplo, blas::Op op_a, SizeType n, SizeType k, SizeType extra_lda,
              SizeType extra_ldc) {
  TileElementSize size_a(n, k);
  if (op_a != blas::Op::NoTrans)
    size_a.transpose();
  TileElementSize size_c(n, n);

  SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  SizeType ldc = std::max<SizeType>(1, size_c.rows()) + extra_ldc;

  std::stringstream s;
  s << "HERK: " << uplo << ", " << op_a;
  s << ", n = " << n << ", k = " << k;
  s << ", lda = " << lda << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));
  memory::MemoryView<T, Device::CPU> mem_c(mul(ldc, size_c.cols()));

  Tile<T, Device::CPU> a0(size_a, std::move(mem_a), lda);
  Tile<T, Device::CPU> c(size_c, std::move(mem_c), ldc);

  // Returns op_a(a)_ik
  auto el_op_a = [](const TileElementIndex& index) {
    double i = index.row();
    double k = index.col();
    return TypeUtilities<T>::polar(.9 * (i + 1) / (k + .5), i - k);
  };
  auto el_c = [uplo](const TileElementIndex& index) {
    // Return -1 for elements not referenced
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return TypeUtilities<T>::element(-1, 0);

    double i = index.row();
    double j = index.col();
    return TypeUtilities<T>::polar(1.2 * i / (j + 1), -i + j);
  };

  BaseType<T> alpha = -1.2f;
  BaseType<T> beta = 1.1f;

  auto res_c = [uplo, k, alpha, el_op_a, beta, el_c](const TileElementIndex& index) {
    // Return el_c(index) for elements not referenced
    if ((uplo == blas::Uplo::Lower && index.row() < index.col()) ||
        (uplo == blas::Uplo::Upper && index.row() > index.col()))
      return el_c(index);

    T tmp = TypeUtilities<T>::element(0, 0);
    // Compute result of cij
    for (SizeType kk = 0; kk < k; ++kk) {
      tmp += el_op_a({index.row(), kk}) * TypeUtilities<T>::conj(el_op_a({index.col(), kk}));
    }
    return beta * el_c(index) + alpha * tmp;
  };

  set(a0, el_op_a, op_a);
  set(c, el_c);

  Tile<CT, Device::CPU> a(std::move(a0));

  tile::herk(uplo, op_a, alpha, a, beta, c);

  CHECK_TILE_NEAR(res_c, c, (k + 1) * TypeUtilities<T>::error, (k + 1) * TypeUtilities<T>::error);
}

template <class T, class CT = const T>
void testHerkExceptions(blas::Uplo uplo, blas::Op op_a, const TileElementSize& size_op_a,
                        const TileElementSize& size_c, SizeType extra_lda, SizeType extra_ldc) {
  TileElementSize size_a = size_op_a;
  if (op_a != blas::Op::NoTrans)
    size_a.transpose();

  SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  SizeType ldc = std::max<SizeType>(1, size_c.rows()) + extra_ldc;

  std::stringstream s;
  s << "HERK Exceptions: " << uplo << ", " << op_a;
  s << ", size_a = " << size_a << ", size_c = " << size_c;
  s << ", lda = " << lda << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));
  memory::MemoryView<T, Device::CPU> mem_c(mul(ldc, size_c.cols()));

  Tile<CT, Device::CPU> a(size_a, std::move(mem_a), lda);
  Tile<T, Device::CPU> c(size_c, std::move(mem_c), ldc);

  BaseType<T> alpha = -1.2f;
  BaseType<T> beta = 1.1f;

  EXPECT_THROW(tile::herk(uplo, op_a, alpha, a, beta, c), std::invalid_argument);
}
