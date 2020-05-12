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

#include <gtest/gtest.h>
#include <lapack.hh>

#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/index.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/tile.h"
#include "dlaf/types.h"

#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

namespace {

using dlaf::SizeType;

template <class T>
void testLange(lapack::Norm norm, SizeType m, SizeType n, SizeType extra_lda) {
  using dlaf::TileElementSize;
  using dlaf::TileElementIndex;
  using dlaf::Tile;
  using dlaf::Device;
  using dlaf::memory::MemoryView;

  using dlaf::tile::lange;
  using dlaf::util::size_t::mul;
  using dlaf::matrix::test::set;

  using NormT = dlaf::BaseType<T>;

  TileElementSize size = TileElementSize(m, n);

  SizeType lda = std::max<SizeType>(1, size.rows()) + extra_lda;

  std::stringstream s;
  s << "LANGE: " << lapack::norm2str(norm);
  s << ", " << size << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  MemoryView<T, Device::CPU> mem_a(mul(lda, size.cols()));

  // Create tiles.
  Tile<T, Device::CPU> a(size, std::move(mem_a), lda);

  const T max_value = dlaf_test::TypeUtilities<T>::element(13, -13);

  // by LAPACK documentation, if it is an empty matrix return 0
  const NormT norm_expected = (size.isEmpty()) ? 0 : std::abs(max_value);

  {
    SCOPED_TRACE("Max in Lower Triangular");

    auto el_L = [size, max_value](const TileElementIndex& index) {
      if (TileElementIndex{size.rows() - 1, 0} == index)
        return max_value;
      return T(0);
    };

    set(a, el_L);
    EXPECT_FLOAT_EQ(norm_expected, lange(norm, a));
  }

  {
    SCOPED_TRACE("Max in Upper Triangular");

    auto el_U = [size, max_value](const TileElementIndex& index) {
      if (TileElementIndex{0, size.cols() - 1} == index)
        return max_value;
      return T(0);
    };

    set(a, el_U);
    EXPECT_FLOAT_EQ(norm_expected, lange(norm, a));
  }

  {
    SCOPED_TRACE("Max on Diagonal");

    auto el_D = [size, max_value](const TileElementIndex& index) {
      if (TileElementIndex{size.rows() - 1, size.cols() - 1} == index)
        return max_value;
      return T(0);
    };

    set(a, el_D);
    EXPECT_FLOAT_EQ(norm_expected, lange(norm, a));
  }
}

}
