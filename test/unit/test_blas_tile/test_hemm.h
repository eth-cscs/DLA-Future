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
void testHemm(blas::Side side, blas::Uplo uplo, SizeType m, SizeType n, SizeType extra_lda,
              SizeType extra_ldb, SizeType extra_ldc) {
  DLAF_ASSERT(side == blas::Side::Left || side == blas::Side::Right,
              "Only Left and Right side supported", side);

  SizeType k;

  if (side == blas::Side::Left)
    k = m;
  else
    k = n;

  TileElementSize size_a(k, k);

  TileElementSize size_b(m, n);
  TileElementSize size_c(m, n);

  SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;
  SizeType ldc = std::max<SizeType>(1, size_c.rows()) + extra_ldc;

  std::stringstream s;
  s << "HEMM: " << side << ", " << uplo;
  s << ", m = " << m << ", n = " << n;
  s << ", lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  T alpha = TypeUtilities<T>::element(1.2, .7);
  T beta = TypeUtilities<T>::element(1.1, .4);
  BaseType<T> gamma = 1.3f;

  memory::MemoryView<T, Device::CPU> mem_a(mul(lda, size_a.cols()));
  memory::MemoryView<T, Device::CPU> mem_b(mul(ldb, size_b.cols()));
  memory::MemoryView<T, Device::CPU> mem_c(mul(ldc, size_c.cols()));

  // Create tiles.
  Tile<T, Device::CPU> a0(size_a, std::move(mem_a), lda);
  Tile<T, Device::CPU> b0(size_b, std::move(mem_b), ldb);
  Tile<T, Device::CPU> c(size_c, std::move(mem_c), ldc);

  // Note: The tile elements are chosen such that:
  // Cij = 1.2 * i / (j+1) * exp(I * (j-i))
  // where I is the imaginary number
  //
  // if side == Left
  // Aik = 0.9 * (i+1) * (k+1) * exp(gamma * I * (i-k))
  // Bkj = 0.7 / ((j+1) * (k+1)) * exp(gamma * I * (k+j))
  //
  // if side == Right
  // Bik = 0.7 / ((i+1) * (k+1)) * exp(gamma * I * (i+k))
  // Akj = 0.9 * (j+1) * (k+1) * exp(gamma * I * (j-k))
  //
  // Hence the solution (S) will be
  // if side == Left
  // Sij = beta * Cij + 0.63 * k * gamma * (i+1)/(j+1) exp(I * alpha * (i+j))
  // if side == Right
  // Sij = beta * Cij + 0.63 * k * gamma * (j+1)/(i+1) exp(I * alpha * (i+j))
  auto el_a = [side, gamma](const TileElementIndex& index) {
    if (side == blas::Side::Left) {
      double i = index.row();
      double k = index.col();
      return TypeUtilities<T>::polar(.9 * (i + 1) * (k + 1), gamma * (i - k));
    }
    else {
      double k = index.row();
      double j = index.col();
      return TypeUtilities<T>::polar(.9 * (j + 1) * (k + 1), gamma * (j - k));
    }
  };

  auto el_b = [side, gamma](const TileElementIndex& index) {
    if (side == blas::Side::Left) {
      double k = index.row();
      double j = index.col();
      return TypeUtilities<T>::polar(.7 / ((j + 1) * (k + 1)), gamma * (k + j));
    }
    else {
      double i = index.row();
      double k = index.col();
      return TypeUtilities<T>::polar(.7 / ((i + 1) * (k + 1)), gamma * (i + k));
    }
  };

  auto el_c = [](const TileElementIndex& index) {
    double i = index.row();
    double j = index.col();
    return TypeUtilities<T>::polar(1.2 * i / (j + 1), -i + j);
  };

  auto res_c = [side, k, alpha, beta, gamma, el_c](const TileElementIndex& index) {
    double i = index.row();
    double j = index.col();

    if (side == blas::Side::Left) {
      return beta * el_c(index) + TypeUtilities<T>::element(0.63 * k, 0) * alpha *
                                      TypeUtilities<T>::polar((i + 1) / (j + 1), gamma * (i + j));
    }
    else {
      return beta * el_c(index) + TypeUtilities<T>::element(0.63 * k, 0) * alpha *
                                      TypeUtilities<T>::polar((j + 1) / (i + 1), gamma * (i + j));
    }
  };

  // Set tile elements.
  set(a0, el_a);
  set(b0, el_b);
  set(c, el_c);

  // Read-only tiles become constant if CT is const T.
  const Tile<CT, Device::CPU> a(std::move(a0));
  const Tile<CT, Device::CPU> b(std::move(b0));

  tile::hemm(side, uplo, alpha, a, b, beta, c);

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_c, c, 2 * (k + 1) * TypeUtilities<T>::error,
                  2 * (k + 1) * TypeUtilities<T>::error);
}
