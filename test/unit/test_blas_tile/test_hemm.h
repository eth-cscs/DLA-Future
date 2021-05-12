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

#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/blas/enum_output.h"
#include "dlaf/blas/tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/matrix/util_tile_blas.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {

using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

using dlaf::util::size_t::mul;

template <class T, class CT = const T>
void testHemm(const blas::Side side, const blas::Uplo uplo, const SizeType m, const SizeType n,
              const SizeType extra_lda, const SizeType extra_ldb, const SizeType extra_ldc) {
  const SizeType k = (side == blas::Side::Left) ? m : n;

  const TileElementSize size_a(k, k);
  const TileElementSize size_b(m, n);
  const TileElementSize size_c(m, n);

  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  const SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;
  const SizeType ldc = std::max<SizeType>(1, size_c.rows()) + extra_ldc;

  std::stringstream s;
  s << "HEMM: " << side << ", " << uplo;
  s << ", m = " << m << ", n = " << n;
  s << ", lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  const T alpha = TypeUtilities<T>::element(1.2, .7);
  const T beta = TypeUtilities<T>::element(1.1, .4);
  const BaseType<T> gamma = 1.3f;

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
      const double i = index.row();
      const double k = index.col();
      return TypeUtilities<T>::polar(.9 * (i + 1) * (k + 1), gamma * (i - k));
    }
    else {
      const double k = index.row();
      const double j = index.col();
      return TypeUtilities<T>::polar(.9 * (j + 1) * (k + 1), gamma * (j - k));
    }
  };

  auto el_b = [side, gamma](const TileElementIndex& index) {
    if (side == blas::Side::Left) {
      const double k = index.row();
      const double j = index.col();
      return TypeUtilities<T>::polar(.7 / ((j + 1) * (k + 1)), gamma * (k + j));
    }
    else {
      const double i = index.row();
      const double k = index.col();
      return TypeUtilities<T>::polar(.7 / ((i + 1) * (k + 1)), gamma * (i + k));
    }
  };

  auto el_c = [](const TileElementIndex& index) {
    const double i = index.row();
    const double j = index.col();
    return TypeUtilities<T>::polar(1.2 * i / (j + 1), -i + j);
  };

  auto res_c = [side, k, alpha, beta, gamma, el_c](const TileElementIndex& index) {
    const double i = index.row();
    const double j = index.col();

    if (side == blas::Side::Left) {
      return beta * el_c(index) + TypeUtilities<T>::element(0.63 * k, 0) * alpha *
                                      TypeUtilities<T>::polar((i + 1) / (j + 1), gamma * (i + j));
    }
    else {
      return beta * el_c(index) + TypeUtilities<T>::element(0.63 * k, 0) * alpha *
                                      TypeUtilities<T>::polar((j + 1) / (i + 1), gamma * (i + j));
    }
  };

  // Read-only tiles become constant if CT is const T.
  auto a = createTile<CT>(el_a, size_a, lda);
  auto b = createTile<CT>(el_b, size_b, ldb);
  auto c = createTile<T>(el_c, size_c, ldc);

  tile::hemm(side, uplo, alpha, a, b, beta, c);

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_c, c, 2 * (k + 1) * TypeUtilities<T>::error,
                  2 * (k + 1) * TypeUtilities<T>::error);
}

}
}
