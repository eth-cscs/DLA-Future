//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include "dlaf/common/index2d.h"
#include "dlaf/types.h"

namespace dlaf {
namespace common {

namespace internal {

// A base class for column/row-major Iterators.
class Iterator2DBase {
public:
  Iterator2DBase(SizeType ld, SizeType i) : ld(ld), i(i) {}
  void operator++() noexcept {
    ++i;
  }

  bool operator!=(Iterator2DBase const& o) const noexcept {
    return i != o.i;
  }

protected:
  SizeType ld;
  SizeType i;
};

}

// An Iterator returning indices in row-major order
template <typename IndexT, class Tag>
class RowMajorIterator : public internal::Iterator2DBase {
  using index2d_t = Index2D<IndexT, Tag>;
  using Iterator2DBase::i;
  using Iterator2DBase::ld;

public:
  RowMajorIterator(SizeType ld, SizeType i) : internal::Iterator2DBase(ld, i) {}
  index2d_t operator*() const noexcept {
    return index2d_t(i / ld, i % ld);
  }
};

// An Iterator returning indices in column-major order
template <typename IndexT, class Tag>
class ColMajorIterator : public internal::Iterator2DBase {
  using index2d_t = Index2D<IndexT, Tag>;
  using Iterator2DBase::i;
  using Iterator2DBase::ld;

public:
  ColMajorIterator(SizeType ld, SizeType i) : internal::Iterator2DBase(ld, i) {}
  index2d_t operator*() const noexcept {
    return index2d_t(i % ld, i / ld);
  }
};

/// An Iterable representing a 2D range in column-major order.
template <typename IndexT, class Tag>
class ColMajorRange {
  using size2d_t = Size2D<IndexT, Tag>;
  using iter2d_t = ColMajorIterator<IndexT, Tag>;

public:
  ColMajorRange(size2d_t sz) : sz_(sz) {}

  iter2d_t begin() const noexcept {
    return iter2d_t(sz_.rows(), 0);
  }
  iter2d_t end() const noexcept {
    return iter2d_t(sz_.rows(), sz_.rows() * sz_.cols());
  }

private:
  size2d_t sz_;
};

/// Function wrapper to deduce types in constructor call for ColMajorRange
template <typename IndexT, class Tag>
ColMajorRange<IndexT, Tag> iterateColMajor(Size2D<IndexT, Tag> sz) noexcept {
  return ColMajorRange<IndexT, Tag>(sz);
}

/// An iterable representing a 2D range in row-major order.
template <typename IndexT, class Tag>
class RowMajorRange {
  using size2d_t = Size2D<IndexT, Tag>;
  using iter2d_t = RowMajorIterator<IndexT, Tag>;

public:
  RowMajorRange(size2d_t sz) : sz_(sz) {}

  iter2d_t begin() const noexcept {
    return iter2d_t(sz_.cols(), 0);
  }
  iter2d_t end() const noexcept {
    return iter2d_t(sz_.cols(), sz_.rows() * sz_.cols());
  }

private:
  size2d_t sz_;
};

/// Function wrapper to deduce types in constructor call for RowMajorRange
template <typename IndexT, class Tag>
RowMajorRange<IndexT, Tag> iterateRowMajor(Size2D<IndexT, Tag> sz) noexcept {
  return RowMajorRange<IndexT, Tag>(sz);
}

}
}
