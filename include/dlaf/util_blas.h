//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <blas.hh>

#include <dlaf/common/assert.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/types.h>

namespace dlaf {
namespace tile {
namespace internal {

struct gemmSizes {
  const SizeType m;
  const SizeType n;
  const SizeType k;
};

template <typename T, Device device>
gemmSizes getGemmSizes(blas::Op op_a, blas::Op op_b, const dlaf::matrix::Tile<const T, device>& a,
                       const dlaf::matrix::Tile<const T, device>& b,
                       const dlaf::matrix::Tile<T, device>& c) {
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

  SizeType k2;
  if (op_b == blas::Op::NoTrans) {
    k2 = b.size().rows();
    n = b.size().cols();
  }
  else {
    k2 = b.size().cols();
    n = b.size().rows();
  }

  DLAF_ASSERT(m == c.size().rows(), "`m` cannot be determined!", m, c);
  DLAF_ASSERT(n == c.size().cols(), "`n` cannot be determined!", n, c);
  DLAF_ASSERT(k == k2, "`k` cannot be determined!", k, k2);

  return {m, n, k};
}

struct hemmSizes {
  const SizeType m;
  const SizeType n;
};

template <typename T, Device device>
hemmSizes getHemmSizes(blas::Side side, const dlaf::matrix::Tile<const T, device>& a,
                       const dlaf::matrix::Tile<const T, device>& b,
                       const dlaf::matrix::Tile<T, device>& c) {
  const hemmSizes s{c.size().rows(), c.size().cols()};

  if (side == blas::Side::Left) {
    DLAF_ASSERT(s.m == a.size().rows(), "`m` cannot be determined!", s.m, a);
    DLAF_ASSERT(s.n == b.size().cols(), "`n` cannot be determined!", s.n, b);
    DLAF_ASSERT(a.size().cols() == b.size().rows(),
                "columns of matrix `a` does not correspond to rows of matrix `b`", a, b);
  }
  else if (side == blas::Side::Right) {
    DLAF_ASSERT(s.m == b.size().rows(), "`m` cannot be determined!", s.m, b);
    DLAF_ASSERT(s.n == a.size().cols(), "`n` cannot be determined!", s.n, a);
    DLAF_ASSERT(a.size().rows() == b.size().cols(),
                "rows of matrix `a` does not correspond to columns of matrix `b`", a, b);
  }

  return s;
}

struct her2kSizes {
  const SizeType n;
  const SizeType k;
};

template <typename T, Device device>
her2kSizes getHer2kSizes(blas::Op op, const dlaf::matrix::Tile<const T, device>& a,
                         const dlaf::matrix::Tile<const T, device>&,
                         const dlaf::matrix::Tile<T, device>& c) {
  const SizeType rows = a.size().rows();
  const SizeType cols = a.size().cols();
  const auto s = (op == blas::Op::NoTrans) ? her2kSizes{rows, cols} : her2kSizes{cols, rows};

  DLAF_ASSERT((!std::is_same<T, ComplexType<T>>::value || op != blas::Op::Trans),
              "`op = Trans` is not allowed in complex HER2K.");
  DLAF_ASSERT(c.size().rows() == c.size().cols(), "`c` is not square!", c);
  DLAF_ASSERT(c.size().rows() == s.n, "`c` has an invalid size!", c, s.n);

  return s;
}

struct herkSizes {
  const SizeType n;
  const SizeType k;
};

template <typename T, Device device>
herkSizes getHerkSizes(blas::Op op, const dlaf::matrix::Tile<const T, device>& a,
                       const dlaf::matrix::Tile<T, device>& c) {
  const SizeType rows = a.size().rows();
  const SizeType cols = a.size().cols();
  const auto s = (op == blas::Op::NoTrans) ? herkSizes{rows, cols} : herkSizes{cols, rows};

  DLAF_ASSERT((!std::is_same<T, ComplexType<T>>::value || op != blas::Op::Trans),
              "op = Trans is not allowed for Complex values!");
  DLAF_ASSERT(c.size().rows() == c.size().cols(), "`c` is not square!", c);
  DLAF_ASSERT(c.size().rows() == s.n, "`c` has an invalid size!", c, s.n);

  return s;
}

struct trsmSizes {
  const SizeType m;
  const SizeType n;
};

template <typename T, Device device>
trsmSizes getTrsmSizes(blas::Side side, const dlaf::matrix::Tile<const T, device>& a,
                       const dlaf::matrix::Tile<T, device>& b) {
  trsmSizes s{b.size().rows(), b.size().cols()};

  DLAF_ASSERT(a.size().rows() == a.size().cols(), "`a` is not square!", a);

  const auto left_side = (side == blas::Side::Left ? s.m : s.n);
  DLAF_ASSERT(a.size().rows() == left_side, "`a` has an invalid size!", a, left_side);

  return s;
}

}
}
}
