//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <blas.hh>

#include <dlaf/common/assert.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/types.h>
#include <dlaf/util_tile.h>

namespace dlaf {
namespace tile {
namespace internal {

struct gemmSizes {
  const SizeType m;
  const SizeType n;
  const SizeType k;
};

template <typename T, Device D>
gemmSizes getGemmSizes(const blas::Op op_a, const blas::Op op_b, const dlaf::matrix::Tile<const T, D>& a,
                       const dlaf::matrix::Tile<const T, D>& b,
                       [[maybe_unused]] const dlaf::matrix::Tile<T, D>& c) {
  SizeType m;
  SizeType n;
  SizeType k;

  if (op_a == blas::Op::NoTrans) {
    m = a.size().rows();
    k = a.size().cols();
  }
  else {
    m = a.size().cols();
    k = a.size().rows();
  }

  [[maybe_unused]] SizeType k2;
  if (op_b == blas::Op::NoTrans) {
    k2 = b.size().rows();
    n = b.size().cols();
  }
  else {
    k2 = b.size().cols();
    n = b.size().rows();
  }

  DLAF_ASSERT(m == c.size().rows(), op_a, a, c);
  DLAF_ASSERT(n == c.size().cols(), op_b, b, c);
  DLAF_ASSERT(k == k2, op_a, a, op_b, b);

  return {m, n, k};
}

struct hemmSizes {
  const SizeType m;
  const SizeType n;
};

template <typename T, Device D>
hemmSizes getHemmSizes(const blas::Side side, [[maybe_unused]] const dlaf::matrix::Tile<const T, D>& a,
                       [[maybe_unused]] const dlaf::matrix::Tile<const T, D>& b,
                       const dlaf::matrix::Tile<T, D>& c) {
  const hemmSizes s{c.size().rows(), c.size().cols()};

  if (side == blas::Side::Left) {
    DLAF_ASSERT(s.m == a.size().rows(), c, a);
    DLAF_ASSERT(s.n == b.size().cols(), c, b);
    DLAF_ASSERT(a.size().cols() == b.size().rows(), a, b);
  }
  else if (side == blas::Side::Right) {
    DLAF_ASSERT(s.m == b.size().rows(), c, b);
    DLAF_ASSERT(s.n == a.size().cols(), c, a);
    DLAF_ASSERT(a.size().rows() == b.size().cols(), a, b);
  }

  return s;
}

struct her2kSizes {
  const SizeType n;
  const SizeType k;
};

template <typename T, Device D>
her2kSizes getHer2kSizes(blas::Op op, const dlaf::matrix::Tile<const T, D>& a,
                         [[maybe_unused]] const dlaf::matrix::Tile<const T, D>& b,
                         [[maybe_unused]] const dlaf::matrix::Tile<T, D>& c) {
  const SizeType rows = a.size().rows();
  const SizeType cols = a.size().cols();
  const auto s = (op == blas::Op::NoTrans) ? her2kSizes{rows, cols} : her2kSizes{cols, rows};

  DLAF_ASSERT(tile_complex_trans<T>(op), op);

  DLAF_ASSERT(square_size(c), c);
  DLAF_ASSERT(a.size() == b.size(), a, b);
  DLAF_ASSERT(c.size().rows() == s.n, c, op, a);

  return s;
}

struct herkSizes {
  const SizeType n;
  const SizeType k;
};

template <typename T, Device D>
herkSizes getHerkSizes(const blas::Op op, const dlaf::matrix::Tile<const T, D>& a,
                       [[maybe_unused]] const dlaf::matrix::Tile<T, D>& c) {
  const SizeType rows = a.size().rows();
  const SizeType cols = a.size().cols();
  const auto s = (op == blas::Op::NoTrans) ? herkSizes{rows, cols} : herkSizes{cols, rows};

  DLAF_ASSERT(tile_complex_trans<T>(op), op);
  DLAF_ASSERT(square_size(c), c);
  DLAF_ASSERT(c.size().rows() == s.n, c, op, a);

  return s;
}

struct trmmSizes {
  const SizeType m;
  const SizeType n;
};

template <typename T, Device D>
trmmSizes getTrmmSizes(const blas::Side side, [[maybe_unused]] const dlaf::matrix::Tile<const T, D>& a,
                       const dlaf::matrix::Tile<T, D>& b) {
  trmmSizes s{b.size().rows(), b.size().cols()};
  [[maybe_unused]] const auto b_side = (side == blas::Side::Left ? s.m : s.n);

  DLAF_ASSERT(square_size(a), a);
  DLAF_ASSERT(a.size().rows() == b_side, a, side, b);

  return s;
}

template <typename T, Device D>
trmmSizes getTrmm3Sizes(const blas::Side side, const dlaf::matrix::Tile<const T, D>& a,
                        [[maybe_unused]] const dlaf::matrix::Tile<const T, D>& b,
                        const dlaf::matrix::Tile<T, D>& c) {
  DLAF_ASSERT(b.size() == c.size(), b, c);
  return getTrmmSizes(side, a, c);
}

struct trsmSizes {
  const SizeType m;
  const SizeType n;
};

template <typename T, Device D>
trsmSizes getTrsmSizes(const blas::Side side, [[maybe_unused]] const dlaf::matrix::Tile<const T, D>& a,
                       const dlaf::matrix::Tile<T, D>& b) {
  trsmSizes s{b.size().rows(), b.size().cols()};

  DLAF_ASSERT(square_size(a), a);

  [[maybe_unused]] const auto left_side = (side == blas::Side::Left ? s.m : s.n);
  DLAF_ASSERT(a.size().rows() == left_side, a, side, b);

  return s;
}

}
}
}
