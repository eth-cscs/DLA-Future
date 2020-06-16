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

#include <gtest/gtest.h>

#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/index.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/tile.h"
#include "dlaf/types.h"

#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {
namespace lantr {

using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::TileElementIndex;
using dlaf::Tile;
using dlaf::Device;

using dlaf::tile::lantr;
using dlaf::util::size_t::mul;
using dlaf::matrix::test::set;

template <class T>
using NormT = dlaf::BaseType<T>;

// This setter returns values of an abstract matrix like this
//
// if (uplo == Lower)
//
// 2v  0   0   0
// 0   0   0   0
// 0   0   0   0
// 3v  0   0   2v
// 0   0   0   0
// 0   0   0   0
// 0   0   0   0
// 0   0   0   0
//
// if (uplo == Upper)
//
// 2v  0   0   3v  0   0   0   0
// 0   0   0   0   0   0   0   0
// 0   0   0   0   0   0   0   0
// 0   0   0   2v  0   0   0   0
//
// If the matrix is 1x1, its only value will be 2v
template <class T>
struct TileSetter {
  static const T value;

  TileSetter(TileElementSize size, blas::Uplo uplo) : size_(size), uplo_(uplo) {}

  T operator()(const TileElementIndex& index) const {
    const auto tr_size = std::min(size_.rows(), size_.cols());

    // top left corner
    if (TileElementIndex{0, 0} == index)
      return T(2) * value;
    // bottom right corner
    else if (TileElementIndex{tr_size - 1, tr_size - 1} == index)
      return T(2) * value;

    // out of diagonal corner
    switch (uplo_) {
      case blas::Uplo::Lower:
        if (TileElementIndex{tr_size - 1, 0} == index)
          return T(3) * value;
        break;
      case blas::Uplo::Upper:
        if (TileElementIndex{0, tr_size - 1} == index)
          return T(3) * value;
        break;
      case blas::Uplo::General:
      default:
        break;
    }

    // all the rest
    return T(0);
  }

private:
  TileElementSize size_;
  blas::Uplo uplo_;
};

template <class T>
const T TileSetter<T>::value = dlaf_test::TypeUtilities<T>::element(13, -13);

template <class T>
void test_lantr(lapack::Norm norm, blas::Uplo uplo, blas::Diag diag, const Tile<T, Device::CPU>& a,
                NormT<T> norm_expected) {
  set(a, TileSetter<T>{a.size(), uplo});

  SCOPED_TRACE(::testing::Message() << "LANTR: " << lapack::norm2str(norm) << ", " << a.size()
                                    << ", ld = " << a.ld() << " uplo = " << blas::uplo2str(uplo)
                                    << " diag = " << blas::diag2str(diag));

  EXPECT_FLOAT_EQ(norm_expected, lantr(norm, uplo, diag, a));
}

template <class T>
void run(lapack::Norm norm, blas::Uplo uplo, blas::Diag diag, const Tile<T, Device::CPU>& a) {
  const TileElementSize size = a.size();

  NormT<T> value = std::abs(TileSetter<T>::value);
  NormT<T> norm_expected = -1;

  switch (norm) {
    case lapack::Norm::Max:
      switch (diag) {
        case blas::Diag::Unit:
          norm_expected = size != TileElementSize{1, 1} ? 3 * value : 1;
          break;
        case blas::Diag::NonUnit:
          norm_expected = size != TileElementSize{1, 1} ? 3 * value : 2 * value;
          break;
      }
      break;
    case lapack::Norm::One:
    case lapack::Norm::Inf:
      switch (diag) {
        case blas::Diag::Unit:
          norm_expected = size != TileElementSize{1, 1} ? 3 * value + 1 : 1;
          break;
        case blas::Diag::NonUnit:
          norm_expected = size != TileElementSize{1, 1} ? 5 * value : 2 * value;
          break;
      }
      break;
    case lapack::Norm::Fro:
      switch (diag) {
        case blas::Diag::Unit:
          norm_expected = size != TileElementSize{1, 1}
                              ? std::sqrt(std::pow(3 * value, 2) + std::min(size.rows(), size.cols()))
                              : 1;
          break;
        case blas::Diag::NonUnit:
          norm_expected = size != TileElementSize{1, 1} ? std::sqrt(17) * value : std::sqrt(4) * value;
          break;
      }
      break;
    case lapack::Norm::Two:
      FAIL() << "norm " << lapack::norm2str(norm) << " is not supported by lantr";
  }

  // by LAPACK documentation, if it is an empty matrix return 0
  norm_expected = (a.size().isEmpty()) ? 0 : norm_expected;

  test_lantr<T>(norm, uplo, diag, a, norm_expected);
}

}
}
}
