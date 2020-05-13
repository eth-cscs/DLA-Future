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

#include <blas_util.hh>
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
void testLantr(lapack::Norm norm, blas::Uplo uplo, blas::Diag diag, SizeType m, SizeType n,
               SizeType extra_lda) {
  using dlaf::TileElementSize;
  using dlaf::TileElementIndex;
  using dlaf::Tile;
  using dlaf::Device;
  using dlaf::memory::MemoryView;

  using dlaf::tile::lantr;
  using dlaf::util::size_t::mul;
  using dlaf::matrix::test::set;

  using NormT = dlaf::BaseType<T>;

  TileElementSize size = TileElementSize(m, n);

  SizeType lda = std::max<SizeType>(1, size.rows()) + extra_lda;

  std::stringstream s;
  s << "LANTR: " << lapack::norm2str(norm);
  s << ", " << blas::uplo2str(uplo);
  s << ", " << blas::diag2str(diag);
  s << ", " << size << ", lda = " << lda;
  SCOPED_TRACE(s.str());

  MemoryView<T, Device::CPU> mem_a(mul(lda, size.cols()));

  // Create tiles.
  Tile<T, Device::CPU> a(size, std::move(mem_a), lda);

  const T max_value = dlaf_test::TypeUtilities<T>::element(13, -13);

  // by LAPACK documentation, if it is an empty matrix return 0
  const NormT norm_expected = (size.isEmpty()) ? 0 : std::abs(max_value);

  {
    SCOPED_TRACE("Max in Triangular");

    auto el_T = [size, max_value, uplo, diag](const TileElementIndex& index) {
      auto max_in_range = max_value;
      auto max_out_range = max_value + max_value;

      if (TileElementSize{1, 1} == size)
        return blas::Diag::Unit == diag ? 1 : max_in_range;

      if (TileElementIndex{size.rows() - 1, 0} == index)
        return blas::Uplo::Lower == uplo ? max_in_range : max_out_range;
      else if (TileElementIndex{0, size.cols() - 1} == index)
        return blas::Uplo::Upper == uplo ? max_in_range : max_out_range;
      return T(0);
    };

    set(a, el_T);

    const NormT result = lantr(norm, uplo, diag, a);

    // if unit-diagonal and without any element out of the diagonal
    if (blas::Diag::Unit == diag && TileElementSize{1, 1} == size)
      EXPECT_FLOAT_EQ(1, result);
    else
      EXPECT_FLOAT_EQ(norm_expected, result);
  }

  // it does not make sense to test a max on the diagonal when it is a unit diagonal
  if (blas::Diag::NonUnit == diag) {
    SCOPED_TRACE("Max on Diagonal");

    auto el_D = [size, max_value](const TileElementIndex& index) {
      if (TileElementIndex{size.rows() - 1, size.cols() - 1} == index)
        return max_value;
      return T(0);
    };

    set(a, el_D);
    EXPECT_FLOAT_EQ(norm_expected, lantr(norm, uplo, diag, a));
  }
}

}
