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
namespace lange {

using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::TileElementIndex;
using dlaf::Tile;
using dlaf::Device;

using dlaf::tile::lange;
using dlaf::util::size_t::mul;
using dlaf::matrix::test::set;

template <class T>
using NormT = dlaf::BaseType<T>;

// This setter returns values of an abstract matrix like this
//
// 0v  0   0   3v
// 0   0   0   0
// 0   0   0   0
// 2v  0   0   2v
// 0   0   0   0
// 0   0   0   0
// 0   0   0   0
// 0   0   0   0
//
// If the matrix is 1x1, its only value will be 2v
template <class T>
struct TileSetter {
  static const T value;

  TileSetter(TileElementSize size) : size_(size) {}

  T operator()(const TileElementIndex& index) const {
    // bottom right corner
    if (TileElementIndex{size_.rows() - 1, size_.cols() - 1} == index)
      return T(2) * value;
    // bottom left corner
    else if (TileElementIndex{size_.rows() - 1, 0} == index)
      return T(2) * value;
    // top right corner
    else if (TileElementIndex{0, size_.cols() - 1} == index)
      return T(3) * value;
    // all the rest
    return T(0);
  }

private:
  TileElementSize size_;
};

template <class T>
const T TileSetter<T>::value = dlaf_test::TypeUtilities<T>::element(13, -13);

template <class T>
void test_lange(lapack::Norm norm, const Tile<T, Device::CPU>& a, NormT<T> norm_expected) {
  set(a, TileSetter<T>{a.size()});

  SCOPED_TRACE(::testing::Message() << "LANGE: " << lapack::norm2str(norm) << ", " << a.size()
                                    << ", ld = " << a.ld());

  EXPECT_FLOAT_EQ(norm_expected, lange(norm, a));
}

template <class T>
void run(lapack::Norm norm, const Tile<T, Device::CPU>& a) {
  const TileElementSize size = a.size();

  NormT<T> value = std::abs(TileSetter<T>::value);
  NormT<T> norm_expected = -1;

  switch (norm) {
    case lapack::Norm::One:
      norm_expected = size != TileElementSize{1, 1} ? 5 : 2;
      break;
    case lapack::Norm::Max:
      norm_expected = size != TileElementSize{1, 1} ? 3 : 2;
      break;
    case lapack::Norm::Inf:
      norm_expected = size != TileElementSize{1, 1} ? 4 : 2;
      break;
    case lapack::Norm::Fro:
      norm_expected = size != TileElementSize{1, 1} ? std::sqrt(17) : std::sqrt(4);
      break;
    case lapack::Norm::Two:
      FAIL() << "norm " << lapack::norm2str(norm) << " is not supported by lange";
  }

  norm_expected *= value;

  // by LAPACK documentation, if it is an empty matrix return 0
  norm_expected = (a.size().isEmpty()) ? 0 : norm_expected;

  test_lange<T>(norm, a, norm_expected);
}

}
}
}
