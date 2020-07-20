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

/// @file
///
/// The combination of `IteratorRange2D`, `IterableRange2D` and `iterate_range2d()`
/// allow us to write nested loops as a simple range-based for loop. For
/// example, instead of this:
///
/// ```
/// Size2D sz(5, 6);
///
/// for (SizeType idx_j = 0; idx_j < sz.cols(); ++idx_j) {
///    for (SizeType idx_i = 0; idx_i < sz.rows(); ++idx_i) {
///      Index2D idx{idx_i, idx_j};
///      ....
///    }
/// }
/// ```
///
/// we can write this:
///
/// ```
/// for (Index2D idx : iterate_range2d(sz)) {
///   ....
/// }
/// ```

#include <cstddef>

#include "dlaf/common/assert.h"
#include "dlaf/common/index2d.h"

namespace dlaf {
namespace common {

/// An Iterator returning indices in column-major order.
template <typename IndexT, class Tag>
class IteratorRange2D {
  using index2d_t = Index2D<IndexT, Tag>;

public:
  IteratorRange2D(index2d_t begin, IndexT ld, std::ptrdiff_t i) : begin_(begin), ld_(ld), i_(i) {}

  void operator++() noexcept {
    ++i_;
  }

  bool operator!=(const IteratorRange2D& o) const noexcept {
    return i_ != o.i_;
  }

  index2d_t operator*() const noexcept {
    return index2d_t(begin_.row() + static_cast<IndexT>(i_ % ld_),
                     begin_.col() + static_cast<IndexT>(i_ / ld_));
  }

protected:
  index2d_t begin_;
  IndexT ld_;
  std::ptrdiff_t i_;
};

/// An Iterable representing a 2D range.
template <typename IndexT, class Tag>
class IterableRange2D {
  using size2d_t = Size2D<IndexT, Tag>;
  using index2d_t = Index2D<IndexT, Tag>;
  using iter2d_t = IteratorRange2D<IndexT, Tag>;

public:
  IterableRange2D(index2d_t begin_idx, size2d_t sz)
      : begin_idx_(begin_idx), ld_(sz.rows()), i_max_(sz.rows() * sz.cols()) {}

  IterableRange2D(index2d_t begin_idx, index2d_t end_idx)
      : begin_idx_(begin_idx), ld_(end_idx.row() - begin_idx.row()),
        i_max_(ld_ * (end_idx.col() - begin_idx_.col())) {
    DLAF_ASSERT(begin_idx.row() < end_idx.row(), begin_idx, end_idx);
    DLAF_ASSERT(begin_idx.col() < end_idx.col(), begin_idx, end_idx);
  }

  iter2d_t begin() const noexcept {
    return iter2d_t(begin_idx_, ld_, 0);
  }
  iter2d_t end() const noexcept {
    return iter2d_t(begin_idx_, ld_, i_max_);
  }

private:
  index2d_t begin_idx_;
  IndexT ld_;
  std::ptrdiff_t i_max_;  // the maximum linear index
};

/// Function wrappers to deduce types in constructor calls to IterableRange2D.
///
/// The overload set returns indices in column-major order with the begin index included and the end
/// index/size excluded.
///
/// The following one-argument variants of `iterate_range2d()` iterate over the
/// same range of indices:
///
/// (0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)
///
/// ```
/// for(auto idx : iterate_range2d(Index2D(2, 3))) { ... }
/// for(auto idx : iterate_range2d(Size2D(2, 3))) { ... }
/// ```
///
/// The following two-argument variants of `iterate_range2d()` iterate over the
/// same range of indices:
///
/// (5, 6), (6, 6), (5, 7), (6, 7), (5, 8), (6, 8)
///
/// ```
/// for(auto idx : iterate_range2d(Index2D(5, 6), Index2D(7, 9))) { ... }
/// for(auto idx : iterate_range2d(Index2D(5, 6), Size2D(2, 3))) { ... }
/// ```
///
/// - Note: Size2D(2, 3) is added to Index2D(5, 6) to determine the end index of
///         the range.

template <typename IndexT, class Tag>
IterableRange2D<IndexT, Tag> iterate_range2d(Index2D<IndexT, Tag> begin_idx,
                                             Index2D<IndexT, Tag> end_idx) noexcept {
  return IterableRange2D<IndexT, Tag>(begin_idx, end_idx);
}

template <typename IndexT, class Tag>
IterableRange2D<IndexT, Tag> iterate_range2d(Index2D<IndexT, Tag> begin_idx,
                                             Size2D<IndexT, Tag> sz) noexcept {
  return IterableRange2D<IndexT, Tag>(begin_idx, sz);
}

template <typename IndexT, class Tag>
IterableRange2D<IndexT, Tag> iterate_range2d(Size2D<IndexT, Tag> sz) noexcept {
  return IterableRange2D<IndexT, Tag>(Index2D<IndexT, Tag>(0, 0), sz);
}

template <typename IndexT, class Tag>
IterableRange2D<IndexT, Tag> iterate_range2d(Index2D<IndexT, Tag> end_idx) noexcept {
  return IterableRange2D<IndexT, Tag>(Index2D<IndexT, Tag>(0, 0), end_idx);
}

}
}
